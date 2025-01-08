import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import os
from torch import GradScaler, autocast
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class FrameInterpolationModel(nn.Module):
    def __init__(self):
        super(FrameInterpolationModel, self).__init__()

        # Encoder com convoluções 3D
        self.encoder = nn.Sequential(
            # Primeira camada com kernel_size=2 na profundidade
            nn.ReplicationPad3d((3, 3, 3, 3, 1, 0)),
            nn.Conv3d(3, 64, kernel_size=(2, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.PReLU(),

            # Camadas subsequentes com kernel_size=1 na profundidade
            nn.ReplicationPad3d((2, 2, 2, 2, 0, 0)),
            nn.Conv3d(64, 128, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2)),
            nn.PReLU(),

            nn.ReplicationPad3d((1, 1, 1, 1, 0, 0)),
            nn.Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.PReLU(),
        )

        # Decoder com convoluções 3D
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.PReLU(),

            nn.ConvTranspose3d(128, 64, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2), output_padding=(0, 1, 1)),
            nn.PReLU(),

            nn.ConvTranspose3d(64, 3, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), output_padding=(0, 1, 1)),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.squeeze(2)
        return x


    
#Prepara os frames para o modelo
class FrameDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.frames) - 2

    def __getitem__(self, idx):
        frame1 = self.frames[idx]
        frame2 = self.frames[idx + 1]  # Ground truth
        frame3 = self.frames[idx + 2]

        frame1 = self.transform(frame1)
        frame2 = self.transform(frame2)
        frame3 = self.transform(frame3)

        # Empilhar frame1 e frame3 na dimensão temporal
        input_frames = torch.stack([frame1, frame3], dim=1)  # [3, 2, H, W]

        return input_frames, frame2  # Retorna o tensor de entrada e o frame intermediário
    
   


def extract_frames(video_path, max_seconds):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    max_frames = max_seconds * fps
    
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    return frames


def ssim_loss(predicted_frame, target_frame, ssim_metric):
    ssim_value = ssim_metric(predicted_frame, target_frame)
    return ssim_value

def train_model(model, dataloader, val_loader, device, epochs, lr, save_path, patience=5):
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler()
    model.to(device)
    model.train()

    best_loss = float('inf')
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []
    alpha = 0.8

    print("Iniciando treinamento...")
    for epoch in range(epochs):
        qtd_frames = 0
        model.train()
        epoch_loss = 0.0
        print(f"Época {epoch+1}/{epochs}")
        for input_frames, frame2 in dataloader:
            input_frames, frame2 = input_frames.to(device), frame2.to(device)
            qtd_frames += 1
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                output = model(input_frames)
                # Verificar se o output tem as mesmas dimensões do frame2
                if output.shape != frame2.shape:
                    # Use frame2.shape[2:] para obter as dimensões espaciais (H, W)
                    output = F.interpolate(output, size=frame2.shape[2:], mode='bilinear', align_corners=True)
                ssim_component = ssim_loss(output, frame2, ssim_metric)
                ssim_normalizado = (1 - ssim_component) / 2
                l1_component = nn.L1Loss()(output, frame2)
                loss = alpha * ssim_normalizado + (1 - alpha) * l1_component

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            print(f"Progresso: {qtd_frames}/{len(dataloader)}", end="\r")

        avg_epoch_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_epoch_loss)
        print(f'Época [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

        # Avaliação no conjunto de validação
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_frames, frame2 in val_loader:
                input_frames, frame2 = input_frames.to(device), frame2.to(device)
                with autocast(device_type=device.type):
                    output = model(input_frames)
                    if output.shape != frame2.shape:
                        output = F.interpolate(output, size=frame2.shape[2:], mode='bilinear', align_corners=True)
                    ssim_component = ssim_loss(output, frame2, ssim_metric)
                    ssim_normalizado = (1 - ssim_component) / 2
                    l1_component = nn.L1Loss()(output, frame2)
                    loss = alpha * ssim_normalizado + (1 - alpha) * l1_component
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f'Modelo salvo com validation loss {avg_val_loss:.4f}')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print('Early stopping triggered')
                break


def main(videos_dir, model_path="frame_interpolation_model_v2_4.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando {device}")
    model = FrameInterpolationModel().to(device)

    epochs = 7
    lr = 0.00003
    batch_size = 2

    load_model = input("Deseja carregar o modelo salvo? (s/n): ").strip().lower() == 's'
    if load_model and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Modelo carregado de", model_path)

    used_videos = set()

    while True:
        video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov')) and f not in used_videos]

        if not video_files:
            print("Não há mais vídeos disponíveis para treinar na pasta.")
            break

        video_file = random.choice(video_files)
        video_path = os.path.join(videos_dir, video_file)
        print(f"Usando o vídeo: {video_file}")

        used_videos.add(video_file)

        frames = extract_frames(video_path, max_seconds=15)
        dataset = FrameDataset(frames)
        train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        print("Treinando modelo...")
        train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr, save_path=model_path)

if __name__ == "__main__":
    diretorio_videos = 'videos2'
    main(videos_dir=diretorio_videos)
