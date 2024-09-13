import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.cuda.amp import autocast
import os
from time import perf_counter

class FrameInterpolationModel(nn.Module):
    def __init__(self):
        super(FrameInterpolationModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 128, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.2) #O Dropout funciona desligando (ou "dropando") aleatoriamente uma fração das unidades (neurônios) na rede neural durante a fase de treinamento. Isso significa que, durante cada iteração de treinamento, algumas unidades não são atualizadas. A fração de unidades a serem dropadas é determinada por um parâmetro p, que representa a probabilidade de qualquer unidade ser dropada.
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            #nn.Upsample(scale_factor=3, mode='nearest', align_corners=False),  # Aumenta a resolução
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
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
            with autocast():
                frame2_interpolated = model(frame1, frame3).squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame2_interpolated = (frame2_interpolated * 0.5 + 0.5) * 255
        frame2_interpolated = frame2_interpolated.astype(np.uint8)
        interpolated_frames.append(frames[i])
        interpolated_frames.append(frame2_interpolated)
    counter2 = perf_counter()
    print(counter2-counter1)
    interpolated_frames.append(frames[-1])
    create_video(interpolated_frames, output_path, fps)

def main(video_path, output_path, model_path="frame_interpolation_model_BETA.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrameInterpolationModel().to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Modelo carregado de", model_path)
    else:
        print("Modelo salvo não encontrado em", model_path)
        return

    frames, fps = extract_frames(video_path, max_seconds=60)
    interpolate_frames(model, frames, output_path, fps, device)
    print(f'Vídeo interpolado salvo em {output_path}')

if __name__ == "__main__":
    video_path = 'videos\\boss_puppy.mp4'
    output_path = 'interlope_videos\\boss_puppy_interlope_BETA.mp4'
    main(video_path, output_path)
