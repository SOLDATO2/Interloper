import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch import autocast
import os
import torch.nn.functional as F
from time import perf_counter

class FrameInterpolationModel(nn.Module):
    def __init__(self):
        super(FrameInterpolationModel, self).__init__()
        #Testar Conv3d
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 128, kernel_size=7, padding=1, stride=2),  # Reduz a resolução pela metade
            nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size=5, padding=1, stride=2),  # Reduz novamente pela metade
            nn.PReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),  # Camada final do encoder
            nn.PReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),  # Primeira camada do decoder
            nn.PReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, padding=1, stride=2, output_padding=1),  # Upsample
            nn.PReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=7, padding=1, stride=2, output_padding=1),  # Upsample final
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1, stride=1),  # Ajusta para 3 canais RGB
        )

    def forward(self, frame1, frame2):
        x = torch.cat([frame1, frame2], dim=1)
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
        return len(self.frames) - 1

    def __getitem__(self, idx):
        frame1 = self.frames[idx]
        frame2 = self.frames[idx + 1]
        frame1 = self.transform(frame1)
        frame2 = self.transform(frame2)
        return frame1, frame2

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

def create_video(frames, output_path, original_fps):
    height, width, layers = frames[0].shape
    size = (width, height)
    
    # Ajuste do FPS para manter o mesmo tempo do vídeo original
    interpolated_fps = original_fps * 2  # Como adicionamos um frame entre cada par, o FPS é dobrado.
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), interpolated_fps, size)
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

def interpolate_frames(model, frames, output_path, fps, device):
    dataset = FrameDataset(frames)
    interpolated_frames = []

    model.eval()
    counter1 = perf_counter()
    for i in range(len(dataset)):
        frame1 = dataset[i][0].unsqueeze(0).to(device)
        frame2 = dataset[i][1].unsqueeze(0).to(device)
        with torch.no_grad():
            with autocast(device_type=str(device)):
                output = model(frame1, frame2)
                # Redimensiona a saída para corresponder ao tamanho dos frames originais
                output = F.interpolate(output, size=frame1.shape[2:], mode='bilinear', align_corners=True)
                frame1_5_interpolated = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame1_5_interpolated = (frame1_5_interpolated * 0.5 + 0.5) * 255
        frame1_5_interpolated = frame1_5_interpolated.astype(np.uint8)
        
        interpolated_frames.append(frames[i])
        interpolated_frames.append(frame1_5_interpolated)
        print(f"Frame '{i}' interpolado.")
    
    counter2 = perf_counter()
    print(f"Tempo de interpolação: {counter2 - counter1} segundos")

    interpolated_frames.append(frames[-1])
    create_video(interpolated_frames, output_path, fps)
    
def main(video_path, output_path, model_path="frame_interpolation_model_v2_4_with_perceptual.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrameInterpolationModel().to(device)
    
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
    video_path = 'rollercoaster.mp4'
    output_path = 'testes_videos\\rollercoaster_v2_4_GEN6_perceptual.mp4'
    main(video_path, output_path)
