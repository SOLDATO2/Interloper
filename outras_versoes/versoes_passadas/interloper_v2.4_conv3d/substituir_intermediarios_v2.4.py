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

        # Encoder com convoluções 3D
        self.encoder = nn.Sequential(
            nn.ReplicationPad3d((3, 3, 3, 3, 0, 0)),  # Padding espacial
            nn.Conv3d(3, 64, kernel_size=(2, 7, 7), stride=(1, 2, 2)),
            nn.PReLU(),
            nn.ReplicationPad3d((2, 2, 2, 2, 0, 0)),
            nn.Conv3d(64, 128, kernel_size=(1, 5, 5), stride=(1, 2, 2)),
            nn.PReLU(),
            nn.ReplicationPad3d((1, 1, 1, 1, 0, 0)),
            nn.Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            nn.PReLU(),
        )

        # Decoder com convoluções 3D (ajustado)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            nn.PReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=(1, 5, 5), stride=(1, 2, 2), output_padding=(0, 1, 1)),
            nn.PReLU(),
            nn.ConvTranspose3d(64, 3, kernel_size=(1, 7, 7), stride=(1, 2, 2), output_padding=(0, 1, 1)),
            nn.Tanh(),
        )

    def forward(self, x):
        # x tem formato [batch_size, channels, depth, height, width]
        x = self.encoder(x)
        x = self.decoder(x)
        # Remover a dimensão de profundidade (depth) que agora é 1
        x = x.squeeze(2)  # Resultado em [batch_size, channels, height, width]
        return x

class FrameDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.frames) - 2

    def __getitem__(self, idx):
        frame1 = self.frames[idx]
        frame3 = self.frames[idx + 2]
        frame1 = self.transform(frame1)
        frame3 = self.transform(frame3)
        # Empilhar frame1 e frame3 na dimensão temporal
        input_frames = torch.stack([frame1, frame3], dim=1)  # [C, D, H, W]
        return input_frames

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

    model.eval()
    counter1 = perf_counter()
    for i in range(len(frames) - 2):
        input_frames = dataset[i].unsqueeze(0).to(device)  # [1, C, D, H, W]
        with torch.no_grad():
            with autocast(device_type=device.type):
                output = model(input_frames)
                if output.shape[2:] != input_frames.shape[3:]:
                    output = F.interpolate(output, size=input_frames.shape[3:], mode='bilinear', align_corners=True)
                frame_interpolated = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame_interpolated = (frame_interpolated * 0.5 + 0.5) * 255
        frame_interpolated = np.clip(frame_interpolated, 0, 255).astype(np.uint8)
        
        # Adiciona frames originais e interpolados
        interpolated_frames.append(frames[i])
        interpolated_frames.append(frame_interpolated)
        print(f"Frame '{i}' interpolado.")
    counter2 = perf_counter()
    print(f"Tempo total de inferência: {counter2 - counter1:.2f} segundos")
    interpolated_frames.append(frames[-1])
    create_video(interpolated_frames, output_path, fps * 2)  # Multiplica o FPS por 2 para compensar os frames adicionados

def main(video_path, output_path, model_path="frame_interpolation_model_v2_4.pth"):
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
    video_path = 'videos_all_30_60\\thelastofus.mp4'
    output_path = 'testes_videos\\thelastofus_v2_4_GEN1.mp4'
    main(video_path, output_path)
