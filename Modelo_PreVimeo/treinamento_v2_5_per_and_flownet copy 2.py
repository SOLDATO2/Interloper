#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import random
import argparse
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch import autocast, GradScaler

import matplotlib.pyplot as plt
import pickle

##############################################################################
# LiteFlowNet e Funções Auxiliares
##############################################################################
from LiteFlowNet.liteflownet_run import LiteFlowNet  # Ajuste ao seu projeto

def warp_image(frame, flow):
    """
    Executa o warping de 'frame' [C,H,W] usando 'flow' [2,H,W].
    flow[0] = deslocamento vertical (dy)
    flow[1] = deslocamento horizontal (dx)
    """
    device = frame.device
    C, H, W = frame.shape

    frame = frame.unsqueeze(0)  # => [1,C,H,W]
    flow  = flow.unsqueeze(0)   # => [1,2,H,W]

    # Meshgrid base
    y_base, x_base = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    x_base = x_base.float()
    y_base = y_base.float()

    flow_y = flow[:, 0, :, :]
    flow_x = flow[:, 1, :, :]

    new_y = y_base + flow_y[0]
    new_x = x_base + flow_x[0]

    # Normaliza para grid_sample => [-1,1]
    new_y = 2.0 * (new_y / (H - 1)) - 1.0
    new_x = 2.0 * (new_x / (W - 1)) - 1.0

    grid = torch.stack((new_x, new_y), dim=-1).unsqueeze(0)  # [1,H,W,2]

    warped = F.grid_sample(frame, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped.squeeze(0)  # => [C,H,W]

def tensor_to_image(tensor):
    """
    Converte um tensor normalizado [C,H,W] -> imagem NumPy [H,W,C].
    Assumindo normalização em [-1,1].
    """
    tensor = tensor.detach().cpu().clone()
    tensor = 0.5 * (tensor + 1.0)  # de [-1,1] para [0,1]
    tensor = torch.clamp(tensor, 0, 1)
    arr = tensor.numpy().transpose(1, 2, 0)  # [H,W,C]
    arr = (arr * 255).astype(np.uint8)
    return arr

def flow_to_color(flow):
    """
    Converte fluxo óptico [2,H,W] para BGR colorido (estilo OpenCV).
    """
    flow_np = flow.detach().cpu().numpy().transpose(1, 2, 0)  # [H,W,2]
    dx = flow_np[..., 1]
    dy = flow_np[..., 0]
    mag, ang = cv2.cartToPolar(dx, dy)
    hsv = np.zeros((flow_np.shape[0], flow_np.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue
    hsv[..., 1] = 255                    # Saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def resize_image_max_keep_ratio(image, max_w=1280, max_h=720):
    """
    Redimensiona a imagem [H,W,C] para caber em (max_w x max_h), mantendo a proporção.
    NÃO amplia se for menor do que (max_w x max_h).
    """
    h, w = image.shape[:2]
    if (w > max_w) or (h > max_h):
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

##############################################################################
# Perceptual Loss (VGG)
##############################################################################
import torchvision.models as models

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_name='relu3_3'):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        # até a relu3_3 => índice 17
        self.vgg = nn.Sequential()
        for i in range(17):
            self.vgg.add_module(str(i), vgg[i])
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.vgg(x)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_extractor = VGGFeatureExtractor()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        pf = self.vgg_extractor(pred)
        tf = self.vgg_extractor(target)
        return self.l1(pf, tf)

##############################################################################
# Modelo de Interpolação (10 canais -> 3 canais)
##############################################################################
class FrameInterpolationModel(nn.Module):
    """
    Recebe 10 canais (ex.: warped1, warped3, flow12, flow32) => [B,10,H,W].
    Sai [B,3,H,W].
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=7, padding=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1, stride=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

##############################################################################
# Dataset On-The-Fly
##############################################################################
class FrameDatasetOnTheFly(Dataset):
    """
    Cada item do dataset é uma tripla (frame1, frame2, frame3).
    Carregamos as 3 imagens, depois computamos o fluxo (1->2) e (3->2) usando LiteFlowNet,
    executamos warp e retornamos (input_10c, frame2, frame1, frame3, flow_12, flow_32, warped1, warped3).
    """
    def __init__(self, triple_paths, lite_flow_net, device):
        super().__init__()
        self.triple_paths = triple_paths
        self.lite_flow_net = lite_flow_net
        self.device = device

        self.to_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    def __len__(self):
        return len(self.triple_paths)

    @torch.no_grad()
    def compute_flow(self, imgA, imgB):
        """
        Computa o fluxo de A->B usando LiteFlowNet.
        imgA, imgB => tensores [1,3,H,W].
        Retorna [2,H,W].
        """
        flow = self.lite_flow_net(imgA, imgB)  # => [1,2,H,W]
        return flow.squeeze(0)

    def __getitem__(self, idx):
        f1_path, f2_path, f3_path = self.triple_paths[idx]

        # Carrega frames BGR
        frame1_bgr = cv2.imread(f1_path)
        frame2_bgr = cv2.imread(f2_path)
        frame3_bgr = cv2.imread(f3_path)

        # Redimensionar (ex: 1280x720 fixo)
        desired_w, desired_h = 1280, 720
        frame1_bgr = cv2.resize(frame1_bgr, (desired_w, desired_h), interpolation=cv2.INTER_AREA)
        frame2_bgr = cv2.resize(frame2_bgr, (desired_w, desired_h), interpolation=cv2.INTER_AREA)
        frame3_bgr = cv2.resize(frame3_bgr, (desired_w, desired_h), interpolation=cv2.INTER_AREA)

        # BGR->RGB
        frame1_rgb = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2RGB)
        frame3_rgb = cv2.cvtColor(frame3_bgr, cv2.COLOR_BGR2RGB)

        # to_tensor => [-1..1]
        t1 = self.to_tensor(frame1_rgb).to(self.device)  # [3,H,W]
        t2 = self.to_tensor(frame2_rgb).to(self.device)
        t3 = self.to_tensor(frame3_rgb).to(self.device)

        # Fluxos
        with torch.no_grad():
            f1_t = t1.unsqueeze(0).to(self.device)
            f2_t = t2.unsqueeze(0).to(self.device)
            f3_t = t3.unsqueeze(0).to(self.device)

            flow_12 = self.compute_flow(f1_t, f2_t)  # => [2,H,W]
            flow_32 = self.compute_flow(f3_t, f2_t)

        # Warp
        warped1 = warp_image(t1.to(self.device), flow_12)
        warped3 = warp_image(t3.to(self.device), flow_32)

        # Monta o input com 16 canais:
        #  - f1 (3 canais)
        #  - f3 (3 canais)
        #  - warped1 (3 canais)
        #  - warped3 (3 canais)
        #  - flow_12 (2 canais)
        #  - flow_32 (2 canais)
        input_16c = torch.cat([
            t1,              # (3 canais)
            t3,              # (3 canais)
            warped1,         # (3 canais)
            warped3,         # (3 canais)
            flow_12.float(), # (2 canais)
            flow_32.float()  # (2 canais)
        ], dim=0)  # total: 3+3+3+3+2+2 = 16

        return (
        input_16c,   # [16,H,W]
        t2,          # Ground Truth do frame2
        t1,          # frame1_orig
        t3,          # frame3_orig
        flow_12,     # flow1->2
        flow_32,     # flow3->2
        warped1,     # warped1->2
        warped3      # warped3->2
    )

##############################################################################
# Checkpoint Helpers
##############################################################################
def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)
    print(f"[Checkpoint] Salvo em '{filename}'.")

def save_best_model(model, filename="best_model.pth.tar"):
    torch.save({'model_state_dict': model.state_dict()}, filename)
    print(f"[Best Model] Salvo em '{filename}'.")

def load_checkpoint(model, optimizer, device, checkpoint_path="checkpoint.pth"):
    if os.path.exists(checkpoint_path):
        print(f"Deseja carregar o checkpoint '{checkpoint_path}'? (s/n): ", end="")
        resp = input().strip().lower()
        if resp == 's':
            print(f"Carregando checkpoint de '{checkpoint_path}'...")
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            epoch = ckpt['epoch']
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            print(f"Checkpoint carregado: Retomando da época {epoch}, best_val_loss={best_val_loss:.4f}")
            return epoch, best_val_loss
    return 0, float('inf')

def load_best_model(model, device, best_model_path="best_model_on_the_fly.pth"):
    if os.path.exists(best_model_path):
        print(f"Carregando o melhor modelo de '{best_model_path}'...")
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print("Melhor modelo carregado.")
        return True
    return False

##############################################################################
# Trainer
##############################################################################
class Trainer:
    def __init__(self, model, device, lr=3e-5, alpha=0.3, beta=0.1, gamma=0.6):
        """
        alpha => peso SSIM
        beta  => peso L1
        gamma => peso Perceptual
        """
        self.model = model
        self.device = device

        self.optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.perceptual_loss = PerceptualLoss().to(device)
        self.l1_loss = nn.L1Loss()

        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

        self.scaler = GradScaler()

        self.window_initialized = False

    def ssim_loss(self, pred, target):
        ssim_val = self.ssim(pred, target)
        return (1.0 - ssim_val)

    def train_one_epoch(self, dataloader, epoch, epochs,
                        show_window=True, display_step=1,
                        best_val_loss=None, checkpoint_path="checkpoint.pth",
                        max_width=720, max_height=1280):
        self.model.train()
        epoch_loss = 0.0
        interrupted = False

        for batch_idx, (input_16c, t2, t1, t3, flow12, flow32, warped1, warped3) in enumerate(dataloader):
            input_16c = input_16c.to(self.device)
            t2        = t2.to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type=str(self.device.type)):
                output = self.model(input_16c)
                if output.shape[2:] != t2.shape[2:]:
                    output = F.interpolate(output, size=t2.shape[2:], mode='bilinear', align_corners=True)

                loss_ssim = self.ssim_loss(output, t2)
                loss_l1   = self.l1_loss(output, t2)
                loss_p    = self.perceptual_loss(output, t2)

                loss = self.alpha*loss_ssim + self.beta*loss_l1 + self.gamma*loss_p

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()

            if show_window and (batch_idx+1) % display_step == 0:
                try:
                    f1_img = tensor_to_image(t1[0])
                    f2_img = tensor_to_image(t2[0])
                    f3_img = tensor_to_image(t3[0])
                    out_img = tensor_to_image(output[0])

                    fl12_img = flow_to_color(flow12[0])
                    fl32_img = flow_to_color(flow32[0])

                    warp12_img = tensor_to_image(warped1[0])
                    warp32_img = tensor_to_image(warped3[0])

                    # Layout
                    top_row = np.hstack([f1_img, out_img, f3_img])
                    mid_row = np.hstack([fl12_img, f2_img, fl32_img])

                    blank_middle = np.zeros_like(f2_img)
                    bot_row = np.hstack([warp12_img, blank_middle, warp32_img])

                    merged_top = np.vstack([top_row, mid_row])
                    merged = np.vstack([merged_top, bot_row])

                    merged = resize_image_max_keep_ratio(merged, max_w=max_width, max_h=max_height)

                    if not self.window_initialized:
                        cv2.namedWindow("Treinamento On-The-Fly", cv2.WINDOW_AUTOSIZE)
                        self.window_initialized = True

                    cv2.imshow("Treinamento On-The-Fly", cv2.cvtColor(merged, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n[Usuário] Pressionou 'q'. Salvando checkpoint e interrompendo...")

                        save_checkpoint({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'best_val_loss': best_val_loss
                        }, filename=checkpoint_path)

                        interrupted = True
                        break
                except Exception as e:
                    print(f"[Visual] Erro: {e}")

            print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{len(dataloader)}", end="\r")
            if interrupted:
                break

        avg_epoch_loss = epoch_loss / (batch_idx+1)
        return avg_epoch_loss, interrupted

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        val_loss = 0.0
        count = 0
        for input_16c, t2, _, _, _, _, _, _ in dataloader:
            input_16c = input_16c.to(self.device)
            t2        = t2.to(self.device)
            count += 1

            with autocast(device_type=str(self.device.type)):
                out_v = self.model(input_16c)
                if out_v.shape[2:] != t2.shape[2:]:
                    out_v = F.interpolate(out_v, size=t2.shape[2:], mode='bilinear', align_corners=True)

                ssim_val = self.ssim_loss(out_v, t2)
                l1_val   = self.l1_loss(out_v, t2)
                p_val    = self.perceptual_loss(out_v, t2)
                loss     = self.alpha*ssim_val + self.beta*l1_val + self.gamma*p_val

            val_loss += loss.item()

        if count == 0:
            return 0.0
        return val_loss / count

##############################################################################
# Função principal
##############################################################################
def main():
    parser = argparse.ArgumentParser(description="Treinamento de Interpolação com Treino e Val Separados")
    parser.add_argument("--train_frames_dir", type=str, default="train_frames",
                        help="Diretório com subpastas de frames de TREINO.")
    parser.add_argument("--val_frames_dir", type=str, default="val_frames",
                        help="Diretório com subpastas de frames de VALIDAÇÃO.")
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas de treinamento.")
    parser.add_argument("--batch_size", type=int, default=5, help="Tamanho do batch.")
    parser.add_argument("--hide_window", action='store_true',
                        help="Se definido, NÃO exibe a janela de visualização (modo headless).")
    parser.add_argument("--max_width", type=int, default=1280, help="Largura máxima da janela de exibição.")
    parser.add_argument("--max_height", type=int, default=720, help="Altura máxima da janela de exibição.")
    args = parser.parse_args()

    show_window = not args.hide_window
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Info] Train Frames Dir: {args.train_frames_dir}")
    print(f"[Info] Val   Frames Dir: {args.val_frames_dir}")
    print(f"[Info] Using device = {device}")

    # -- 1) Gera lista de triplas para train
    train_triples = []
    train_subfolders = [
        f for f in os.listdir(args.train_frames_dir)
        if os.path.isdir(os.path.join(args.train_frames_dir, f))
    ]
    for sub in train_subfolders:
        subpath = os.path.join(args.train_frames_dir, sub)
        frame_files = sorted([
            ff for ff in os.listdir(subpath)
            if ff.lower().endswith(( '.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        ])
        if len(frame_files) < 3:
            print(f"[Treino] Pasta '{sub}' com menos de 3 frames. Pulando...")
            continue

        for i in range(len(frame_files) - 2):
            f1 = os.path.join(subpath, frame_files[i])
            f2 = os.path.join(subpath, frame_files[i + 1])
            f3 = os.path.join(subpath, frame_files[i + 2])
            train_triples.append((f1, f2, f3))

    # -- 2) Gera lista de triplas para val
    val_triples = []
    val_subfolders = [
        f for f in os.listdir(args.val_frames_dir)
        if os.path.isdir(os.path.join(args.val_frames_dir, f))
    ]
    for sub in val_subfolders:
        subpath = os.path.join(args.val_frames_dir, sub)
        frame_files = sorted([
            ff for ff in os.listdir(subpath)
            if ff.lower().endswith(( '.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        ])
        if len(frame_files) < 3:
            print(f"[Val] Pasta '{sub}' com menos de 3 frames. Pulando...")
            continue

        for i in range(len(frame_files) - 2):
            f1 = os.path.join(subpath, frame_files[i])
            f2 = os.path.join(subpath, frame_files[i + 1])
            f3 = os.path.join(subpath, frame_files[i + 2])
            val_triples.append((f1, f2, f3))

    print(f"[Info] Total de triplas (TREINO): {len(train_triples)}")
    print(f"[Info] Total de triplas (VAL):    {len(val_triples)}")

    # -- 3) Inicializa LiteFlowNet
    lite_flow_net = LiteFlowNet().to(device).eval()

    # -- 4) Monta datasets e dataloaders
    train_dataset = FrameDatasetOnTheFly(train_triples, lite_flow_net, device)
    val_dataset   = FrameDatasetOnTheFly(val_triples,   lite_flow_net, device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # -- 5) Modelo e Trainer
    model = FrameInterpolationModel().to(device)
    trainer = Trainer(model, device, lr=3e-5)

    # -- 6) Scheduler
    scheduler = ReduceLROnPlateau(
        trainer.optimizer,
        factor=0.5,
        patience=5,
        threshold=1e-4,
        min_lr=1e-7,
        verbose=True
    )

    # -- 7) Carregar best_model e checkpoint
    best_model_path = "best_model.pth.tar"
    if os.path.exists(best_model_path):
        print(f"\n[Info] Detected '{best_model_path}'. Carregando melhor modelo...")
        best_ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_ckpt['model_state_dict'])
        print("[Info] Melhor modelo carregado com sucesso.")

    checkpoint_path = "checkpoint.pth"
    start_epoch, best_val_loss = load_checkpoint(
        model, trainer.optimizer, device, checkpoint_path=checkpoint_path
    )

    # Listas para armazenar o train_loss e val_loss
    train_losses = []
    val_losses = []

    # Se existir um histórico salvo, carregue-o
    history_file = "loss_history.pkl"
    if os.path.exists(history_file):
        with open(history_file, 'rb') as f:
            history_data = pickle.load(f)
            train_losses = history_data.get('train_losses', [])
            val_losses = history_data.get('val_losses', [])
            print("[Info] Histórico de treinamento carregado.")

    # -- 8) Loop de Treino
    epochs = args.epochs
    patience = 99
    epochs_no_improve = 0

    for epoch_idx in range(start_epoch, epochs):
        # Exibe LR
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch_idx+1}/{epochs}] LR = {current_lr}")

        # Treino
        train_loss, interrupted = trainer.train_one_epoch(
            dataloader=train_loader,
            epoch=epoch_idx,
            epochs=epochs,
            show_window=show_window,
            display_step=1,
            best_val_loss=best_val_loss,
            checkpoint_path=checkpoint_path,
            max_width=args.max_width,
            max_height=args.max_height
        )
        if interrupted:
            print(f"[Interrompido] Treino interrompido na época {epoch_idx}")
            plot_losses(train_losses, val_losses)

            break

        # Val
        val_loss = trainer.validate(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"\n[Epoch {epoch_idx+1}/{epochs}]"
              f" Train Loss: {train_loss:.4f},"
              f" Val Loss: {val_loss:.4f}")

        # Ajusta scheduler e EarlyStopping
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_best_model(model, filename=best_model_path)
            print(f"** Novo melhor modelo salvo! Val Loss = {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("[Early Stopping] Paciência esgotada.")
                break

        # Salva checkpoint
        save_checkpoint({
            'epoch': epoch_idx + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, filename=checkpoint_path)

        # Salvar histórico a cada época
        with open(history_file, 'wb') as f:
            pickle.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)

    print("[Treinamento finalizado]")
    if show_window:
        cv2.destroyAllWindows()

    # Após o término do loop (ou Early Stopping), exibe o gráfico final
    if not interrupted:
        plot_losses(train_losses, val_losses)


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='s')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Exibe o gráfico de forma bloqueante
    plt.show(block=True)

    # Pausa manual para evitar que o script termine e feche a janela
    input("Pressione Enter para encerrar o gráfico...")


if __name__ == "__main__":
    main()
