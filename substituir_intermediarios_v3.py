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

def main(video_path, output_path, model_path="frame_interpolation_model_v3.pth"):
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
    video_path = 'test_video.mp4'
    output_path = 'test_video_v3.mp4'
    main(video_path, output_path)
