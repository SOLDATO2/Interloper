import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from torch import GradScaler, autocast
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class FrameInterpolationModel(nn.Module):
    def __init__(self):
        super(FrameInterpolationModel, self).__init__()
        # Usando Conv3d
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 2, 2)),  # Reduz a resolução espacial pela metade
            nn.PReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 2, 2)),  # Reduz novamente a resolução espacial
            nn.PReLU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 2, 2)),  # Reduz mais uma vez
            nn.PReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 2, 2), output_padding=(0, 1, 1)),
            nn.PReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 2, 2), output_padding=(0, 1, 1)),
            nn.PReLU(),
            nn.ConvTranspose3d(64, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 2, 2), output_padding=(0, 1, 1)),
        )

    def forward(self, frame1, frame3):
        # Empilha os frames na dimensão temporal e mantém os 3 canais de cor
        # A forma resultante deve ser [batch_size, 3, 2, height, width], onde 3 é o número de canais de cor e 2 é o número de frames.
        x = torch.stack([frame1, frame3], dim=2)  # Combina os dois frames na dimensão temporal
        # Permuta a forma para [batch_size, 2, 3, height, width] para que a dimensão temporal seja tratada como canal de entrada
        x = x.permute(0, 2, 1, 3, 4)
        # Agora, `x` tem a forma esperada para a `Conv3d` com 2 canais de entrada e 3 canais de cor.
        x = self.encoder(x)
        x = self.decoder(x)
        # Seleciona a dimensão temporal intermediária para gerar o frame interpolado.
        return x[:, :, 1, :, :]


#Prepara os frames para o modelo
class FrameDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames
        #serão aplicadas varias transformações para enviar os frames para o modelo
        self.transform = transforms.Compose([
            #Converte o array NumPy (formato em que os frames são inicialmente lidos) em uma imagem PIL (Python Imaging Library).
            #Essa conversão é necessária porque muitas operações do PyTorch trabalham diretamente com imagens PIL.
            transforms.ToPILImage(),
            
            #Converte a imagem PIL em um tensor do PyTorch, normalizando automaticamente os valores dos pixels para o intervalo de [0, 1].
            transforms.ToTensor(),
            
            
            #Esta normalização é usada para ajustar os valores dos pixels de [0, 1] para o intervalo de [-1, 1], o que pode ajudar no treinamento do modelo.
            #(0.5, 0.5, 0.5): Estas são as médias usadas para normalizar os canais (R, G, B) da imagem.
            
            #(0.5, 0.5, 0.5): Estas são as desvios-padrão para os três canais. Dividir pelo desvio-padrão ajuda a manter as entradas na mesma escala, 
            #melhorando a convergência durante o treinamento.
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # sem isso a imagem simplesmente tem inconsistencia de cores, fica piscando cores solidas
        ])                                                         # ou semi solidas
    
    #retorna o numero total de frames disponiveis no dataset
    def __len__(self):
        
    
        #O dataset é baseado em trios de frames consecutivos (frame1, frame2 e frame3). 
        #Portanto, se tivermos N frames, o número de trios possíveis será N - 2, pois o último trio possível começa em N-2 e termina em N-1.
        #ex:
        #Se tivermos 10 frames, podemos formar 8 trios: (frame1, frame2, frame3), (frame2, frame3, frame4), ..., (frame8, frame9, frame10).
        return len(self.frames) - 2

    def __getitem__(self, idx):
        #Recuperamos um set de frames com base em um index e então transformamos e retornamoos o set de frames
        frame1 = self.frames[idx]
        frame3 = self.frames[idx + 2]
        frame2 = self.frames[idx + 1]
        frame1 = self.transform(frame1)
        frame3 = self.transform(frame3)
        frame2 = self.transform(frame2)
        return frame1, frame3, frame2

def extract_frames(video_path, max_seconds=3):
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
    criterion = nn.MSELoss() #SSIM 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler()
    model.train()

    best_loss = float('inf')
    epochs_without_improvement = 0

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
                    # Ajusta a saída para a dimensão dos frames originais
                output = F.interpolate(output, size=frame2.shape[2:], mode='bilinear', align_corners=True) #era false
                loss = criterion(output, frame2)
            #
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            print("Progresso:", qtd_frames, "/", len(dataloader), end="\r")
            

        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frame1, frame3, frame2 in val_loader:
                frame1, frame3, frame2 = frame1.to(device), frame3.to(device), frame2.to(device)
                with autocast(device_type=str(device)):
                    output = model(frame1, frame3)
                    output = F.interpolate(output, size=frame2.shape[2:], mode='bilinear', align_corners=True)
                    loss = criterion(output, frame2)
                #
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

def main(videos_dir, model_path="frame_interpolation_model_v3.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando {device}")
    model = FrameInterpolationModel().to(device)
    
    epochs = 3
    lr = 0.00003
    batch_size = 2

    load_model = input("Deseja carregar o modelo salvo? (s/n): ").strip().lower() == 's'
    if load_model and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Modelo carregado de", model_path)

    # Lista para armazenar vídeos já utilizados
    used_videos = set()

    while True:
        video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov')) and f not in used_videos]
        
        if not video_files:
            print("Não há mais vídeos disponíveis para treinar na pasta.")
            break

        video_file = random.choice(video_files)
        video_path = os.path.join(videos_dir, video_file)
        print(f"Usando o vídeo: {video_file}")
        
        # Adiciona o vídeo à lista de vídeos utilizados
        used_videos.add(video_file)

        # Extrai frames do vídeo selecionado e prepara os datasets
        frames = extract_frames(video_path, max_seconds=3)
        dataset = FrameDataset(frames)
        train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        print("Treinando modelo...")
        train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr, save_path=model_path)

        
        

if __name__ == "__main__":
    diretorio_videos = 'videos_all'
    main(videos_dir=diretorio_videos)
