import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch import autocast
import os
from time import perf_counter

class FrameInterpolationModel3D(nn.Module):
    def __init__(self):
        super(FrameInterpolationModel3D, self).__init__() 
        # Testar Conv3d
        self.encoder = nn.Sequential(
            nn.Conv3d(6, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2)),  # Reduz a resolução espacial pela metade
            nn.PReLU(),
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2)),  # Reduz novamente pela metade
            nn.PReLU(),
            nn.Conv3d(128, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2)),  # Reduz mais uma vez pela metade
            nn.PReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2), output_padding=(0, 1, 1)),  # Aumenta a resolução
            nn.PReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2), output_padding=(0, 1, 1)),  # Aumenta novamente
            nn.PReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2), output_padding=(0, 1, 1)),  # Aumenta mais uma vez
            nn.PReLU(),
            nn.Conv3d(32, 3, kernel_size=(1, 3, 3), padding=(0, 1, 1)),  # Última camada para ajustar para 3 canais (RGB)
        )

    def forward(self, frames_concat):
        x = frames_concat
        x = self.encoder(x)
        x = self.decoder(x)
        return x[:, :, 0]  # Retorna o frame intermediário

class FrameDataset3D(Dataset):
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
        frame1 = self.transform(frame1)
        frame3 = self.transform(frame3)

        # Concatena os frames 1 e 3 ao longo da dimensão do canal (dim=0) para criar uma entrada com 6 canais
        frames_concat = torch.cat([frame1, frame3], dim=0)
        
        # Ajustar os frames para adicionar uma dimensão de profundidade
        frames_concat = frames_concat.unsqueeze(1)  # Adiciona uma dimensão de profundidade (dim=1)
        
        return frames_concat

def extract_frames(video_path, max_seconds=15):
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
    return frames, fps

def create_video(frames, output_path, fps):
    height, width, layers = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

def interpolate_frames(model, frames, output_path, fps, device):
    dataset = FrameDataset3D(frames)
    interpolated_frames = []

    model.eval()
    counter1 = perf_counter()
    for i in range(0, len(frames) - 2, 2):
        frames_concat = dataset[i].unsqueeze(0).to(device)
        with torch.no_grad():
            with autocast(device_type=str(device)):
                frame2_interpolated = model(frames_concat).squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        frame2_interpolated = (frame2_interpolated * 0.5 + 0.5) * 255
        frame2_interpolated = frame2_interpolated.astype(np.uint8)
        
        interpolated_frames.append(frames[i])
        interpolated_frames.append(frame2_interpolated)
        print("Frame '" + str(i) + "' interpolado.")
    counter2 = perf_counter()
    print(f'Tempo total de interpolação: {counter2-counter1:.2f} segundos')
    interpolated_frames.append(frames[-1])
    create_video(interpolated_frames, output_path, fps)

def main(video_path, output_path, model_path="frame_interpolation_model_v3.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrameInterpolationModel3D().to(device)
    
    print(f"Usando {device}")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Modelo carregado de", model_path)
    else:
        print("Modelo salvo não encontrado em", model_path)
        return

    frames, fps = extract_frames(video_path, max_seconds=15)
    interpolate_frames(model, frames, output_path, fps, device)
    print(f'Vídeo interpolado salvo em {output_path}')

if __name__ == "__main__":
    video_path = 'videos3\\thelastofus.mp4'
    output_path = 'interlope_videos\\thelastofus_interlope_v3.mp4'
    main(video_path=video_path, output_path=output_path)
