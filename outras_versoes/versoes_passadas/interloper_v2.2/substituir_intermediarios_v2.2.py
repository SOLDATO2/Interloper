import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch import autocast
import os
from time import perf_counter

class FrameInterpolationModel(nn.Module):
    def __init__(self):
        super(FrameInterpolationModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 128, kernel_size=3, padding=1, stride=2),  # Reduz a resolução pela metade
            nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # Reduz novamente pela metade
            nn.PReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),  # Reduz mais uma vez pela metade
            nn.PReLU(),
        )
        #Auto encoding
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),  # Aumenta a resolução
            nn.PReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),  # Aumenta novamente
            nn.PReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),  # Aumenta mais uma vez
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1, stride=1),  # Última camada para ajustar para 3 canais (RGB)
        )

    def forward(self, frame1, frame3):
        x = torch.cat([frame1, frame3],  dim=1)
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
    dataset = FrameDataset(frames)
    interpolated_frames = []

    model.eval()#will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.
    counter1 = perf_counter()
    for i in range(0, len(frames) - 2, 2):
        frame1 = dataset.transform(frames[i]).unsqueeze(0).to(device)
        frame3 = dataset.transform(frames[i + 2]).unsqueeze(0).to(device)
        with torch.no_grad():
            with autocast(device_type=str(device)):
                frame2_interpolated = model(frame1, frame3).squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame2_interpolated = (frame2_interpolated * 0.5 + 0.5) * 255
        frame2_interpolated = frame2_interpolated.astype(np.uint8)
        
        # Impressão das resoluções
        #print(f"Resolução de Frame1: {frames[i].shape[1]} x {frames[i].shape[0]}")  # largura x altura
        #print(f"Resolução de Frame Interpolado: {frame2_interpolated.shape[1]} x {frame2_interpolated.shape[0]}")
        #print(f"Resolução de Frame3: {frames[i + 2].shape[1]} x {frames[i + 2].shape[0]}")
        
        interpolated_frames.append(frames[i])
        interpolated_frames.append(frame2_interpolated)
        print("Frame '" + str(i) + "' interpolado.")
    counter2 = perf_counter()
    print(counter2-counter1)
    interpolated_frames.append(frames[-1])
    create_video(interpolated_frames, output_path, fps)

def main(video_path, output_path, model_path="frame_interpolation_model_v2_2.pth"):
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
    video_path = 'videos_all_30_60\\ifrit.mp4'
    output_path = 'testes_videos\\ifrit_v2_2_GEN4.mp4'
    main(video_path, output_path)
