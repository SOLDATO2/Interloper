import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split

class FrameInterpolationModel(nn.Module):
    def __init__(self):
        super(FrameInterpolationModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 128, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Dropout(p=0.2) #O Dropout funciona desligando (ou "dropando") aleatoriamente uma fração das unidades (neurônios) na rede neural durante a fase de treinamento. Isso significa que, durante cada iteração de treinamento, algumas unidades não são atualizadas. A fração de unidades a serem dropadas é determinada por um parâmetro p, que representa a probabilidade de qualquer unidade ser dropada.
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
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
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # sem isso a imagem simplesmente tem inconsistencia de cores, fica piscando cores solidas
        ]) # ou semi solidas

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

def extract_frames(video_path, max_seconds=10):
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

def train_model(model, dataloader, val_loader, device, epochs, lr, save_path, patience=5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler()
    model.train()

    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for frame1, frame3, frame2 in dataloader:
            frame1, frame3, frame2 = frame1.to(device), frame3.to(device), frame2.to(device)
            optimizer.zero_grad()
            with autocast():
                output = model(frame1, frame3)
                loss = criterion(output, frame2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            #torch.cuda.empty_cache()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frame1, frame3, frame2 in val_loader:
                frame1, frame3, frame2 = frame1.to(device), frame3.to(device), frame2.to(device)
                with autocast():
                    output = model(frame1, frame3)
                    loss = criterion(output, frame2)
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

def main(video_path, model_path="frame_interpolation_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrameInterpolationModel().to(device)
    frames = extract_frames(video_path, max_seconds=10)
    
    epochs = 3
    lr = 0.00003
    batch_size = 4

    load_model = input("Deseja carregar o modelo salvo? (s/n): ").strip().lower() == 's'
    if load_model and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Modelo carregado de", model_path)
        continuar_treinando = input("Continuar treinando modelo salvo? (s/n): ").strip().lower() == 's'
        if continuar_treinando:
            dataset = FrameDataset(frames)
            train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr, save_path=model_path)
    else:
        print("Treinando um novo modelo")
        dataset = FrameDataset(frames)
        train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr, save_path=model_path)

if __name__ == "__main__":
    video_path = 'videos\\cyberpunk_V_arasaka.mp4'
    main(video_path)
