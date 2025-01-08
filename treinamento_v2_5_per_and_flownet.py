#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import os
from torch import GradScaler, autocast
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import glob
import re

###############################################################################
# 1) Warp Image
###############################################################################
def warp_image(frame, flow):
    """
    Executa o warping de 'frame' [C,H,W] usando 'flow' [2,H,W].
      * flow[0] = deslocamento vertical (dy)
      * flow[1] = deslocamento horizontal (dx)
    Retorna tensor [C,H,W].
    """
    device = frame.device
    C, H, W = frame.shape

    # Precisamos de batch=1 para usar grid_sample
    frame = frame.unsqueeze(0)  # => [1,C,H,W]
    flow  = flow.unsqueeze(0)   # => [1,2,H,W]

    y_base, x_base = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    x_base = x_base.float()
    y_base = y_base.float()

    flow_y = flow[:, 0, :, :]  # [1,H,W]
    flow_x = flow[:, 1, :, :]  # [1,H,W]

    new_y = y_base + flow_y[0]
    new_x = x_base + flow_x[0]

    # Converter coords p/ [-1,1]
    new_y = 2.0 * (new_y / (H - 1)) - 1.0
    new_x = 2.0 * (new_x / (W - 1)) - 1.0

    grid = torch.stack((new_x, new_y), dim=-1).unsqueeze(0)  # => [1,H,W,2]

    warped = F.grid_sample(frame, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped.squeeze(0)  # => [C,H,W]

###############################################################################
# 2) Funções de Conversão e Visualização
###############################################################################
def tensor_to_image(tensor):
    """
    Converte um tensor normalizado [C, H, W] para uma imagem NumPy [H, W, C].
    """
    tensor = tensor.cpu().clone().detach()
    tensor = tensor * 0.5 + 0.5  # Desnormaliza (assumindo normalização [-1,1])
    tensor = torch.clamp(tensor, 0, 1)
    arr = tensor.numpy().transpose(1, 2, 0)  # => [H,W,C]
    arr = (arr * 255).astype(np.uint8)
    return arr

def flow_to_color(flow):
    """
    Converte um fluxo óptico [2,H,W] em uma imagem de cor (HSV->BGR).
    Atenção: cartToPolar espera (x,y) => (dx, dy).
    """
    flow_np = flow.cpu().numpy().transpose(1, 2, 0)  # => [H,W,2]
    dx = flow_np[...,1]  # canal 1 => desloc. horizontal
    dy = flow_np[...,0]  # canal 0 => desloc. vertical
    mag, ang = cv2.cartToPolar(dx, dy)
    hsv = np.zeros((flow_np.shape[0], flow_np.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def resize_image_max_600(image, window_width, window_height):
    """
    Redimensiona a imagem (colagem) para caber na janela especificada.
    Mantém a proporção e não aumenta se for menor que a janela.
    """
    h, w = image.shape[:2]
    max_w, max_h = window_width, window_height
    if (h > max_h) or (w > max_w):
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
    return image

###############################################################################
# 3) Perceptual Loss
###############################################################################
class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_name='relu3_3'):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.vgg = nn.Sequential()
        for i in range(17):  # até a relu3_3
            self.vgg.add_module(str(i), vgg[i])
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.vgg(x)

class PerceptualLoss(nn.Module):
    def __init__(self, layer_name='relu3_3'):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGFeatureExtractor(layer_name=layer_name)
        self.l1 = nn.L1Loss()
    def forward(self, pred, target):
        pf = self.vgg(pred)
        tf = self.vgg(target)
        return self.l1(pf, tf)

###############################################################################
# 4) Modelo com 10 canais no Input
###############################################################################
class FrameInterpolationModel(nn.Module):
    """
    Recebe 10 canais: [3(warped_1), 3(warped_3), 2(flow_12), 2(flow_32)] => [B,10,H,W].
    Sai 3 canais => frame2_fake.
    """
    def __init__(self):
        super(FrameInterpolationModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=7, padding=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=7, padding=3, stride=2, output_padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1, stride=1),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # => [B,3,H,W]

###############################################################################
# 5) Dataset com Fluxos Diretos e Dados para Visualização
###############################################################################
class FrameDatasetDirectFlows(Dataset):
    """
    frames: lista de frames (np.array)
    flows_12: lista [frame1->frame2]
    flows_32: lista [frame3->frame2]
    Retorna: (input_tensor, t2, t1, t3, flow_12, flow_32)
             - input_tensor => [10,H,W]
             - t2 => [3,H,W]
             - t1 => [3,H,W]
             - t3 => [3,H,W]
             - flow_12 => [2,H,W]
             - flow_32 => [2,H,W]
    """
    def __init__(self, frames, flows_12, flows_32, mean_flow=None, std_flow=None):
        super().__init__()
        self.frames = frames
        self.flows_12 = flows_12
        self.flows_32 = flows_32
        self.mean_flow = mean_flow
        self.std_flow = std_flow

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    def __len__(self):
        return len(self.frames) - 2

    def __getitem__(self, idx):
        frame1 = self.frames[idx]
        frame2 = self.frames[idx + 1]
        frame3 = self.frames[idx + 2]

        flow_12 = self.flows_12[idx]  # [2,H,W]
        flow_32 = self.flows_32[idx]  # [2,H,W]

        # Converte frames p/ tensor normalizado
        t1 = self.transform(frame1)  # => [3,H,W]
        t2 = self.transform(frame2)
        t3 = self.transform(frame3)

        # Normaliza fluxo se quiser
        if self.mean_flow is not None and self.std_flow is not None:
            flow_12 = (flow_12 - self.mean_flow[:,None,None]) / self.std_flow[:,None,None]
            flow_32 = (flow_32 - self.mean_flow[:,None,None]) / self.std_flow[:,None,None]

        # Ajusta tamanho do fluxo para bater com t2
        flow_12 = F.interpolate(flow_12.unsqueeze(0), size=t2.shape[1:], mode='bilinear', align_corners=True).squeeze(0)
        flow_32 = F.interpolate(flow_32.unsqueeze(0), size=t2.shape[1:], mode='bilinear', align_corners=True).squeeze(0)

        # Faz warp
        warped_1 = warp_image(t1, flow_12)  # => [3,H,W]
        warped_3 = warp_image(t3, flow_32)  # => [3,H,W]

        # Concat
        input_tensor = torch.cat([warped_1, warped_3, flow_12, flow_32], dim=0)  # => [10,H,W]

        return input_tensor, t2, t1, t3, flow_12, flow_32

###############################################################################
# 6) SSIM, checkpoint, etc.
###############################################################################
def ssim_loss(predicted_frame, target_frame, ssim_metric):
    ssim_value = ssim_metric(predicted_frame, target_frame)
    return ssim_value

def calculate_flow_mean_std(flows):
    all_flows = torch.stack(flows)
    mean = torch.mean(all_flows, dim=[0,2,3])
    std  = torch.std(all_flows, dim=[0,2,3])
    return mean, std

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    print(f"Checkpoint salvo em {filename}")

def load_checkpoint(model, optimizer, device, filename="checkpoint.pth.tar"):
    if os.path.isfile(filename):
        print(f"Carregando checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location=device)
        print(f"Chaves no checkpoint: {checkpoint.keys()}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        current_video = checkpoint.get('current_video', None)
        used_videos = set(checkpoint.get('used_videos', []))
        print(f"Checkpoint carregado: Época {epoch}, Melhor Loss: {best_loss}, Vídeo: {current_video}")
        return epoch, best_loss, current_video, used_videos
    else:
        print(f"Checkpoint '{filename}' não encontrado.")
        return 0, float('inf'), None, set()

###############################################################################
# 7) Treinamento com visualização
###############################################################################
def train_model(model, dataloader, val_loader, device, epochs, lr, save_path, optimizer, 
               patience=10, display_step=1, checkpoint_path="checkpoint.pth.tar", 
               start_epoch=0, best_loss=float('inf'), current_video=None, used_videos=set(),
               visualization_enabled=False, window_width=600, window_height=600):
    """
    Treina o modelo; exibe os frames se visualization_enabled=True.
    Tamanho máximo da janela de exibição: especificado por window_width x window_height.
    """
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    scaler = GradScaler()
    model.train()

    alpha = 0.3   # peso SSIM
    beta  = 0.1   # peso L1
    gamma = 0.6   # peso Perceptual
    perceptual_loss_fn = PerceptualLoss().to(device)
    l1_fn = nn.L1Loss()

    epochs_without_improvement = 0
    print("Iniciando treinamento...")

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0.0
            print(f"Época {epoch+1}/{epochs}")

            for batch_idx, (inputs, target, frame1_t, frame3_t, flow12_t, flow32_t) in enumerate(dataloader):
                inputs, target = inputs.to(device), target.to(device)
                frame1_t, frame3_t = frame1_t.to(device), frame3_t.to(device)
                flow12_t, flow32_t = flow12_t.to(device), flow32_t.to(device)

                optimizer.zero_grad()

                with autocast(device_type=str(device)):
                    # Forward
                    output = model(inputs)  # => [B,3,H,W]
                    # Ajustar resolução de 'output' p/ bater com 'target'
                    output = F.interpolate(output, size=target.shape[2:], mode='bilinear', align_corners=True)

                    # Cálculo das losses
                    ssim_val = ssim_loss(output, target, ssim_metric)
                    ssim_norm = (1 - ssim_val)/2
                    l1_val    = l1_fn(output, target)
                    p_val     = perceptual_loss_fn(output, target)
                    loss      = alpha*ssim_norm + beta*l1_val + gamma*p_val

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

                # Visualização
                if visualization_enabled and (batch_idx + 1) % display_step == 0:
                    try:
                        # Primeiro item do batch
                        frame1_img = tensor_to_image(frame1_t[0])
                        frame3_img = tensor_to_image(frame3_t[0])
                        output_img = tensor_to_image(output[0])
                        frame2_img = tensor_to_image(target[0])
                        flow12_img = flow_to_color(flow12_t[0])
                        flow32_img = flow_to_color(flow32_t[0])

                        # Top row: [frame1, output, frame3]
                        top_row = np.hstack((frame1_img, output_img, frame3_img))
                        # Bottom row: [flow12, frame2_original, flow32]
                        bottom_row = np.hstack((flow12_img, frame2_img, flow32_img))
                        visualization = np.vstack((top_row, bottom_row))

                        # Ajustar para janela especificada
                        visualization = resize_image_max_600(visualization, window_width, window_height)

                        cv2.imshow("Treinamento - Frame Interpolation", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("Interrompendo pelo usuário.")
                            save_checkpoint({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_loss': best_loss,
                                'current_video': current_video,
                                'used_videos': list(used_videos)
                            }, filename=checkpoint_path)
                            return

                    except Exception as e:
                        print(f"[Visualização] Erro: {e}")

                print(f"Progresso: {batch_idx+1}/{len(dataloader)}", end="\r")

            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"\n[Train] Epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}")

            # Validação
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs_v, target_v, _, _, _, _ in val_loader:
                    inputs_v, target_v = inputs_v.to(device), target_v.to(device)
                    with autocast(device_type=str(device)):
                        out_v = model(inputs_v)
                        out_v = F.interpolate(out_v, size=target_v.shape[2:], mode='bilinear', align_corners=True)

                        ssim_valv = ssim_loss(out_v, target_v, ssim_metric)
                        ssim_normv = (1 - ssim_valv)/2
                        l1_valv    = l1_fn(out_v, target_v)
                        p_valv     = perceptual_loss_fn(out_v, target_v)
                        loss_v     = alpha*ssim_normv + beta*l1_valv + gamma*p_valv
                    val_loss += loss_v.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"[Val]  Epoch {epoch+1}, Loss: {avg_val_loss:.4f}")

            # Early stopping e checkpoint
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                epochs_without_improvement = 0
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'current_video': current_video,
                    'used_videos': list(used_videos)
                }, filename=save_path)
                print(f"** Novo melhor modelo salvo! Val Loss = {avg_val_loss:.4f}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print("Early stopping ativado.")
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_loss': best_loss,
                        'current_video': current_video,
                        'used_videos': list(used_videos)
                    }, filename=checkpoint_path)
                    break

    finally:
        if visualization_enabled:
            cv2.destroyAllWindows()

###############################################################################
# 8) Extrair frames e limitá-los a 720p
###############################################################################
def extract_frames(video_path, max_seconds=15):
    """
    Extrai frames de um vídeo até 'max_seconds'.
    Se a resolução for maior que 1280x720, redimensiona para 720p.
    Mantém a proporção.
    Retorna lista de frames (cada frame em RGB np.array).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    max_frames = int(max_seconds * fps)

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Se a imagem for maior que 720p, reduzir
        # 720p => 1280x720
        h, w = frame.shape[:2]
        if (w > 1280) or (h > 720):
            scale = min(1280 / w, 720 / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        frames.append(frame)
    cap.release()
    return frames

###############################################################################
# 9) Função principal
###############################################################################
def main(videos_dir, flows_dir, model_path="best_model.pth.tar", checkpoint_path="checkpoint.pth.tar"):
    """
    Ao rodar, permite configurar visualização e tamanho da janela.
    """
    # Parse de argumentos para visualização e tamanho da janela
    parser = argparse.ArgumentParser(description="Treinamento de Interpolação de Frames com Fluxos Diretos")
    parser.add_argument('--show_window', action='store_true', help='Habilita a visualização da janela de treinamento')
    parser.add_argument('--window_width', type=int, default=600, help='Largura máxima da janela de visualização')
    parser.add_argument('--window_height', type=int, default=600, help='Altura máxima da janela de visualização')
    args = parser.parse_args()

    show_window = args.show_window
    window_width = args.window_width
    window_height = args.window_height

    if show_window:
        print(f"Visualização habilitada com tamanho máximo de {window_width}x{window_height}")
    else:
        print("Visualização desabilitada.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando {device}")

    model = FrameInterpolationModel().to(device)
    
    epochs = 100
    lr = 0.00003
    batch_size = 5

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Perguntar se carrega checkpoint
    load_checkpoint_flag = input("Deseja carregar o checkpoint salvo? (s/n): ").strip().lower() == 's'
    start_epoch = 0
    best_loss = float('inf')
    current_video = None
    used_videos = set()

    # Carrega checkpoint
    if load_checkpoint_flag and os.path.exists(checkpoint_path):
        start_epoch, best_loss, current_video, used_videos = load_checkpoint(model, optimizer, device, filename=checkpoint_path)
    elif load_checkpoint_flag and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_loss = checkpoint.get('best_loss', float('inf'))
            current_video = checkpoint.get('current_video', None)
            used_videos = set(checkpoint.get('used_videos', []))
            print(f"Modelo carregado de {model_path} a partir da época {start_epoch} com best_loss {best_loss}")
        else:
            print(f"Arquivo '{model_path}' não contém um checkpoint válido.")
    elif not load_checkpoint_flag and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = 0
            best_loss = float('inf')
            current_video = None
            used_videos = set()
            print(f"Modelo existente carregado de {model_path}. Continuando do início.")
        else:
            print(f"Arquivo '{model_path}' não contém um checkpoint válido.")
    else:
        print("Iniciando treinamento do zero.")

    # Se quiser normalizar fluxos (opcional)
    calculate_normalization = False
    mean_flow = None
    std_flow  = None
    if calculate_normalization:
        pass  # Implementar se necessário

    while True:
        if current_video:
            video_file = current_video
            video_path = os.path.join(videos_dir, video_file)
            print(f"Continuando no vídeo: {video_file}")
        else:
            video_files = [f for f in os.listdir(videos_dir) if f.lower().endswith(('.mp4','.avi','.mov')) and f not in used_videos]
            if not video_files:
                print("Não há mais vídeos para treinar na pasta.")
                break
            video_file = random.choice(video_files)
            video_path = os.path.join(videos_dir, video_file)
            print(f"Usando vídeo: {video_file}")
            current_video = video_file

        # Extrair frames - limitando em 720p
        frames = extract_frames(video_path, max_seconds=15)
        print(f"Total frames extraídos: {len(frames)}")

        video_name = os.path.splitext(video_file)[0]
        flows_video_dir = os.path.join(flows_dir, video_name)

        flows_12 = []
        flows_32 = []
        for i in range(len(frames) - 2):
            flow_12_path = os.path.join(flows_video_dir, f"flow_12_{i}.pt")
            flow_32_path = os.path.join(flows_video_dir, f"flow_32_{i}.pt")
            if not (os.path.exists(flow_12_path) and os.path.exists(flow_32_path)):
                print(f"[ERRO] Faltam fluxos: {flow_12_path} ou {flow_32_path}")
                return
            f12 = torch.load(flow_12_path)
            f32 = torch.load(flow_32_path)
            flows_12.append(f12)
            flows_32.append(f32)

        print(f"Fluxos carregados: {len(flows_12)} (1->2), {len(flows_32)} (3->2)")

        dataset = FrameDatasetDirectFlows(frames, flows_12, flows_32, mean_flow=mean_flow, std_flow=std_flow)
        train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        print("Treinando modelo...")
        train_model(model,
                    train_loader,
                    val_loader,
                    device,
                    epochs=epochs,
                    lr=lr,
                    save_path=model_path,
                    optimizer=optimizer,
                    checkpoint_path=checkpoint_path,
                    start_epoch=start_epoch,
                    best_loss=best_loss,
                    current_video=current_video,
                    used_videos=used_videos,
                    visualization_enabled=show_window,
                    window_width=window_width,
                    window_height=window_height)

        used_videos.add(video_file)
        current_video = None
        start_epoch = 0
        best_loss = float('inf')

###############################################################################
# Execução
###############################################################################
if __name__ == "__main__":
    videos_dir = "videos3"
    flows_dir  = "flows_dir"
    main(videos_dir, flows_dir)

    
    
    
