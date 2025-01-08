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

# Classe para extrair features da VGG16 e calcular a Perceptual Loss
class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_name='relu3_3'):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.vgg = nn.Sequential()
        # Copiar camadas da VGG até a relu3_3 (por exemplo, a relu da 3ª camada convolucional do bloco 3)
        # Indice aproximado: conv1_1:0, conv1_2:2, relu1_2:3 ...
        # conv3_3 geralmente em torno do indice 16 no features (pode variar dependendo da versão)
        for i in range(17):  # até a relu3_3
            self.vgg.add_module(str(i), vgg[i])
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.vgg(x)  # retorna as features extraídas

class PerceptualLoss(nn.Module):
    def __init__(self, layer_name='relu3_3'):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGFeatureExtractor(layer_name=layer_name)
        self.l1 = nn.L1Loss()
        
    def forward(self, pred, target):
        # Extrair features
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        # L1 entre features
        return self.l1(pred_features, target_features)

class FrameInterpolationModel(nn.Module):
    def __init__(self):
        super(FrameInterpolationModel, self).__init__() 
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 128, kernel_size=7, padding=1, stride=2),
            nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size=5, padding=1, stride=2),
            nn.PReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, padding=1, stride=2, output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=7, padding=1, stride=2, output_padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1, stride=1),
        )

    def forward(self, frame1, frame3):
        x = torch.cat([frame1, frame3], dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
        frame3 = self.frames[idx + 2]
        frame2 = self.frames[idx + 1]
        frame1 = self.transform(frame1)
        frame3 = self.transform(frame3)
        frame2 = self.transform(frame2)
        return frame1, frame3, frame2

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

def train_model(model, dataloader, val_loader, device, epochs, lr, save_path, patience=10):
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler()
    model.train()

    best_loss = float('inf')
    epochs_without_improvement = 0

    # Perdas: SSIM + L1 + Perceptual
    alpha = 0.3   # peso SSIM
    beta = 0.1    # peso L1
    gamma = 0.6   # peso Perceptual
    perceptual_loss_fn = PerceptualLoss().to(device)
    l1_fn = nn.L1Loss()

    print("Iniciando treinamento...")
    for epoch in range(epochs):
        qtd_frames = 0
        model.train()
        epoch_loss = 0.0
        print(f"Epoca {epoch+1}/{epochs}")
        for frame1, frame3, frame2 in dataloader:
            frame1, frame3, frame2 = frame1.to(device), frame3.to(device), frame2.to(device)
            qtd_frames += 1
            optimizer.zero_grad()
            with autocast(device_type=str(device)):
                output = model(frame1, frame3)
                output = F.interpolate(output, size=frame2.shape[2:], mode='bilinear', align_corners=True)
                
                # SSIM
                ssim_component = ssim_loss(output, frame2, ssim_metric)
                ssim_normalizado = (1 - ssim_component) / 2

                # L1
                l1_component = l1_fn(output, frame2)

                # Perceptual
                perceptual_component = perceptual_loss_fn(output, frame2)

                # Combinação das perdas
                loss = alpha * ssim_normalizado + beta * l1_component + gamma * perceptual_component

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            print("Progresso:", qtd_frames, "/", len(dataloader), end="\r")

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

        # Avaliação no conjunto de validação
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frame1, frame3, frame2 in val_loader:
                frame1, frame3, frame2 = frame1.to(device), frame3.to(device), frame2.to(device)
                with autocast(device_type=str(device)):
                    output = model(frame1, frame3)
                    output = F.interpolate(output, size=frame2.shape[2:], mode='bilinear', align_corners=True)
                    
                    ssim_component = ssim_loss(output, frame2, ssim_metric)
                    ssim_normalizado = (1 - ssim_component) / 2
                    l1_component = l1_fn(output, frame2)
                    perceptual_component = perceptual_loss_fn(output, frame2)
                    loss = alpha * ssim_normalizado + beta * l1_component + gamma * perceptual_component
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f'Model saved with validation loss {avg_val_loss:.4f}')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print('Early stopping triggered')
                break

def main(videos_dir, model_path="frame_interpolation_model_v2_4_with_perceptual.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando {device}")
    model = FrameInterpolationModel().to(device)
    
    epochs = 100
    lr = 0.00003
    batch_size = 5

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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        
        print("Treinando modelo...")
        train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr, save_path=model_path)

if __name__ == "__main__":
    diretorio_videos = 'videos_30fps_2'
    main(videos_dir=diretorio_videos)
