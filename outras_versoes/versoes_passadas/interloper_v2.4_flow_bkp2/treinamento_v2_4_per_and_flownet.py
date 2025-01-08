# treinamento.py
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
from depthwise import DepthwiseSeparableConv
import torch.nn.functional as F
import glob
import re

# Função para converter tensor para imagem
def tensor_to_image(tensor):
    """
    Converte um tensor normalizado [C, H, W] para uma imagem NumPy [H, W, C].
    """
    tensor = tensor.cpu().clone().detach()
    tensor = tensor * 0.5 + 0.5  # Desnormalizar (assumindo normalização de (-1, 1))
    tensor = torch.clamp(tensor, 0, 1)
    image = tensor.numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
    image = (image * 255).astype(np.uint8)
    return image

# Função para converter fluxo óptico para imagem de cor
def flow_to_color(flow):
    """
    Converte um fluxo óptico [2, H, W] para uma imagem de cor HSV.
    """
    flow = flow.cpu().numpy().transpose(1, 2, 0)  # [H, W, 2]
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

# Função para redimensionar imagens mantendo a proporção
def resize_image(image, max_width=1280, max_height=720):
    """
    Redimensiona a imagem para que a largura não exceda max_width e a altura não exceda max_height.
    Mantém a proporção.
    """
    h, w, c = image.shape
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h))
        return resized_image
    return image

# Classe para extrair features da VGG16 e calcular a Perceptual Loss
class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_name='relu3_3'):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.vgg = nn.Sequential()
        # Copiar camadas da VGG até a relu3_3
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
            nn.Conv2d(8, 64, kernel_size=7, padding=3, stride=1),  # Reduzido de 128 para 64
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),  # Reduzido de 256 para 128
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),  # Reduzido de 512 para 256
            nn.PReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1),  # Reduzido de 256 para 128
            nn.PReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=7, padding=3, stride=2, output_padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1, stride=1),
        )
    
    def forward(self, frame1, frame3, flow):
        x = torch.cat([frame1, frame3, flow], dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class FrameDataset(Dataset):
    def __init__(self, frames, flows, mean_flow=None, std_flow=None):
        self.frames = frames
        self.flows = flows
        self.mean_flow = mean_flow
        self.std_flow = std_flow
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
        flow = self.flows[idx]  # Carrega o fluxo correspondente

        # Redimensionar frames se necessário
        frame1 = resize_image(frame1)
        frame3 = resize_image(frame3)
        frame2 = resize_image(frame2)

        frame1 = self.transform(frame1)
        frame3 = self.transform(frame3)
        frame2 = self.transform(frame2)

        # Normalizar o fluxo óptico, se os parâmetros estiverem definidos
        if self.mean_flow is not None and self.std_flow is not None:
            flow = (flow - self.mean_flow[:, None, None]) / self.std_flow[:, None, None]

        # Ajustar tamanho do fluxo para coincidir com os frames
        flow = torch.nn.functional.interpolate(flow.unsqueeze(0), size=frame2.shape[1:], mode='bilinear', align_corners=True).squeeze(0)

        return frame1, frame3, flow, frame2

def extract_frames(video_path, max_seconds):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Valor padrão caso não seja possível obter FPS
    max_frames = int(max_seconds * fps)
    
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_image(frame, max_width=1280, max_height=720)  # Redimensionar para 720p se necessário
        frames.append(frame)
    
    cap.release()
    return frames

def ssim_loss(predicted_frame, target_frame, ssim_metric):
    ssim_value = ssim_metric(predicted_frame, target_frame)
    return ssim_value

def calculate_flow_mean_std(flows):
    all_flows = torch.stack(flows)  # [N, 2, H, W]
    mean = torch.mean(all_flows, dim=[0, 2, 3])  # [2]
    std = torch.std(all_flows, dim=[0, 2, 3])    # [2]
    return mean, std

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """
    Salva o estado atual do treinamento.
    
    Args:
        state (dict): Estado do treinamento a ser salvo.
        filename (str): Nome do arquivo para salvar o checkpoint.
    """
    torch.save(state, filename)
    print(f"Checkpoint salvo em {filename}")

def load_checkpoint(model, optimizer, device, filename="checkpoint.pth.tar"):
    """
    Carrega o estado salvo do treinamento.

    Args:
        model (nn.Module): Modelo a ser carregado.
        optimizer (optim.Optimizer): Otimizador a ser carregado.
        device (torch.device): Dispositivo onde o modelo e otimizador devem ser carregados.
        filename (str): Nome do arquivo do checkpoint.

    Returns:
        int: Época a partir da qual o treinamento será retomado.
        float: Melhor perda de validação registrada até o momento.
        str or None: Nome do vídeo que estava sendo treinado.
        set: Conjunto de vídeos já treinados.
    """
    if os.path.isfile(filename):
        print(f"Carregando checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location=device)
        print(f"Chaves no checkpoint: {checkpoint.keys()}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        current_video = checkpoint.get('current_video', None)
        used_videos = set(checkpoint.get('used_videos', []))
        print(f"Checkpoint carregado: Época {epoch}, Melhor Validação Loss: {best_loss}, Vídeo Atual: {current_video}")
        return epoch, best_loss, current_video, used_videos
    else:
        print(f"Checkpoint '{filename}' não encontrado.")
        return 0, float('inf'), None, set()

def train_model(model, dataloader, val_loader, device, epochs, lr, save_path, optimizer, 
               patience=10, display_step=1, checkpoint_path="checkpoint.pth.tar", 
               start_epoch=0, best_loss=float('inf'), current_video=None, used_videos=set()):
    """
    Função para treinar o modelo.

    Args:
        model (nn.Module): Modelo a ser treinado.
        dataloader (DataLoader): DataLoader para treinamento.
        val_loader (DataLoader): DataLoader para validação.
        device (torch.device): Dispositivo de treinamento.
        epochs (int): Número de épocas.
        lr (float): Taxa de aprendizado.
        save_path (str): Caminho para salvar o melhor modelo.
        optimizer (optim.Optimizer): Otimizador.
        patience (int): Paciência para early stopping.
        display_step (int): Frequência para exibir visualizações.
        checkpoint_path (str): Caminho para salvar checkpoints.
        start_epoch (int): Época inicial.
        best_loss (float): Melhor perda de validação.
        current_video (str or None): Vídeo atualmente sendo treinado.
        used_videos (set): Conjunto de vídeos já treinados.
    """
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    scaler = GradScaler()
    model.train()

    epochs_without_improvement = 0

    # Perdas: SSIM + L1 + Perceptual
    alpha = 0.3   # peso SSIM
    beta = 0.1    # peso L1
    gamma = 0.6   # peso Perceptual
    perceptual_loss_fn = PerceptualLoss().to(device)
    l1_fn = nn.L1Loss()

    print("Iniciando treinamento...")
    try:
        for epoch in range(start_epoch, epochs):
            qtd_frames = 0
            model.train()
            epoch_loss = 0.0
            print(f"Época {epoch+1}/{epochs}")
            for batch_idx, (frame1, frame3, flow, frame2) in enumerate(dataloader):
                frame1, frame3, flow, frame2 = frame1.to(device), frame3.to(device), flow.to(device), frame2.to(device)
                qtd_frames += 1
                optimizer.zero_grad()
                with autocast(device_type=str(device)):
                    output = model(frame1, frame3, flow)
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

                # Visualização
                if (batch_idx + 1) % display_step == 0:
                    try:
                        # Selecionar o primeiro exemplo do batch
                        frame1_img = tensor_to_image(frame1[0])
                        frame3_img = tensor_to_image(frame3[0])
                        output_img = tensor_to_image(output[0])
                        frame2_img = tensor_to_image(frame2[0])
                        flow_img = flow_to_color(flow[0])

                        # Redimensionar imagens para visualização
                        frame1_img = resize_image(frame1_img, max_width=1280, max_height=720)
                        frame3_img = resize_image(frame3_img, max_width=1280, max_height=720)
                        output_img = resize_image(output_img, max_width=1280, max_height=720)
                        frame2_img = resize_image(frame2_img, max_width=1280, max_height=720)
                        flow_img = resize_image(flow_img, max_width=1280, max_height=720)

                        # Criar a visualização
                        # Linha Superior: frame1, output (frame2_fake), frame3
                        top_row = np.hstack((frame1_img, output_img, frame3_img))

                        # Linha Intermediária: Espaço vazio, frame2_original, Espaço vazio
                        blank_frame = np.zeros_like(frame1_img)
                        middle_row = np.hstack((blank_frame, frame2_img, blank_frame))

                        # Linha Inferior: Espaço vazio, flow, Espaço vazio
                        blank_flow = np.zeros_like(flow_img)
                        bottom_row = np.hstack((blank_flow, flow_img, blank_flow))

                       # Combinar todas as linhas verticalmente
                        visualization = np.vstack((top_row, middle_row, bottom_row))

                        # **Redimensionar a visualização para exibição**
                        # Definir o tamanho máximo da visualização
                        max_display_width = 1280
                        max_display_height = 720

                        h_vis, w_vis = visualization.shape[:2]
                        scale = min(max_display_width / w_vis, max_display_height / h_vis, 1.0)  # Não aumenta se menor que o max

                        if scale < 1.0:
                            new_w = int(w_vis * scale)
                            new_h = int(h_vis * scale)
                            visualization = cv2.resize(visualization, (new_w, new_h), interpolation=cv2.INTER_AREA)
                            
                        

                        # Exibir a visualização
                        cv2.imshow('Treinamento - Frame Interpolation', cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("Interrompendo o treinamento pelo usuário.")
                            save_checkpoint({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_loss': best_loss,
                                'current_video': current_video,
                                'used_videos': list(used_videos)
                            }, filename=checkpoint_path)
                            return

                        #print(f'Visualização exibida no batch {batch_idx+1}')
                    except Exception as e:
                        print(f"Erro na visualização: {e}")

                print(f"Progresso: {batch_idx+1}/{len(dataloader)}", end="\r")

            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f'\nEpoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

            # Avaliação no conjunto de validação
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for frame1, frame3, flow, frame2 in val_loader:
                    frame1, frame3, flow, frame2 = frame1.to(device), frame3.to(device), flow.to(device), frame2.to(device)
                    with autocast(device_type=str(device)):
                        output = model(frame1, frame3, flow)
                        output = F.interpolate(output, size=frame2.shape[2:], mode='bilinear', align_corners=True)
                        
                        ssim_component = ssim_loss(output, frame2, ssim_metric)
                        ssim_normalizado = (1 - ssim_component) / 2
                        l1_component = l1_fn(output, frame2)
                        perceptual_component = perceptual_loss_fn(output, frame2)
                        loss = alpha * ssim_normalizado + beta * l1_component + gamma * perceptual_component
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f'Validation Loss: {avg_val_loss:.4f}')

            # Salvar checkpoint se a validação melhorar
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                epochs_without_improvement = 0
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'current_video': current_video,
                    'used_videos': list(used_videos)
                }, filename=save_path)  # Salva o melhor modelo
                print(f'Model salvo com Validation Loss {avg_val_loss:.4f}')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print('Early stopping ativado')
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_loss': best_loss,
                        'current_video': current_video,
                        'used_videos': list(used_videos)
                    }, filename=checkpoint_path)  # Salva o checkpoint
                    break
    finally:
        cv2.destroyAllWindows()

def main(videos_dir, flows_dir, model_path="best_model.pth.tar", checkpoint_path="checkpoint.pth.tar"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando {device}")
    model = FrameInterpolationModel().to(device)
    
    epochs = 100
    lr = 0.00003
    batch_size = 5

    # Definir o otimizador aqui para poder carregar o estado corretamente
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Flag para decidir se carrega o checkpoint
    load_checkpoint_flag = input("Deseja carregar o checkpoint salvo? (s/n): ").strip().lower() == 's'
    start_epoch = 0
    best_loss = float('inf')
    current_video = None
    used_videos = set()

    if load_checkpoint_flag and os.path.exists(checkpoint_path):
        start_epoch, best_loss, current_video, used_videos = load_checkpoint(model, optimizer, device, filename=checkpoint_path)
    elif load_checkpoint_flag and os.path.exists(model_path):
        # Opcional: Carregar o melhor modelo existente como ponto de partida
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_loss = checkpoint.get('best_loss', float('inf'))
            current_video = checkpoint.get('current_video', None)
            used_videos = set(checkpoint.get('used_videos', []))
            print(f"Modelo carregado de {model_path} a partir da época {start_epoch} com best_loss {best_loss}")
        else:
            print(f"Arquivo '{model_path}' não contém um checkpoint válido. Certifique-se de que o arquivo foi salvo usando a função 'save_checkpoint'.")
            
    elif not load_checkpoint_flag and os.path.exists(model_path):
        # Caso o usuário não queira carregar o checkpoint, mas um modelo existente exista
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = 0  # Reiniciar a contagem de épocas
            best_loss = float('inf')  # Reiniciar a melhor perda
            current_video = None  # Nenhum vídeo atual em andamento
            used_videos = set()    # Nenhum vídeo usado ainda
            print(f"Modelo existente carregado de {model_path}. Continuando treinamento a partir do início.")
        else:
            print(f"Arquivo '{model_path}' não contém um checkpoint válido. Certifique-se de que o arquivo foi salvo usando a função 'save_checkpoint'.")
    else:
        print("Iniciando treinamento do zero.")

    # Preparar para calcular média e desvio padrão (opcional)
    calculate_normalization = False  # precisa de testes, por enquanto fica desligado
    mean_flow = None
    std_flow = None

    if calculate_normalization:
        all_flows = []
        video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        for video_file in video_files:
            video_name = os.path.splitext(video_file)[0]
            flows_video_dir = os.path.join(flows_dir, video_name)
            flow_files = sorted(glob.glob(os.path.join(flows_video_dir, "flow_*.pt")), key=lambda x: int(re.findall(r'flow_(\d+).pt', x)[0]))
            flows = [torch.load(flow_file) for flow_file in flow_files]
            all_flows.extend(flows)
        
        if all_flows:
            mean_flow, std_flow = calculate_flow_mean_std(all_flows)
            print("Média dos fluxos:", mean_flow)
            print("Desvio padrão dos fluxos:", std_flow)
        else:
            print("Nenhum fluxo óptico encontrado para calcular média e desvio padrão.")

    # Inicializar `used_videos` a partir do checkpoint ou vazio
    used_videos = used_videos  # já carregado a partir do checkpoint ou iniciado como vazio

    while True:
        if current_video:
            # Continuar treinando o vídeo atual
            video_file = current_video
            video_path = os.path.join(videos_dir, video_file)
            print(f"Continuando o treinamento no vídeo: {video_file}")
        else:
            # Selecionar um novo vídeo
            video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov')) and f not in used_videos]
            
            if not video_files:
                print("Não há mais vídeos disponíveis para treinar na pasta.")
                break

            video_file = random.choice(video_files)
            video_path = os.path.join(videos_dir, video_file)
            print(f"Usando o vídeo: {video_file}")
            current_video = video_file

        # Extrair frames
        frames = extract_frames(video_path, max_seconds=15)  # Ajuste max_seconds conforme necessário
        print(f"Total de frames extraídos: {len(frames)}")

        # Carregar os fluxos ópticos correspondentes
        video_name = os.path.splitext(video_file)[0]
        flows_video_dir = os.path.join(flows_dir, video_name)
        flow_files = sorted(glob.glob(os.path.join(flows_video_dir, "flow_*.pt")), key=lambda x: int(re.findall(r'flow_(\d+).pt', x)[0]))
        flows = [torch.load(flow_file) for flow_file in flow_files]
        print(f"Total de fluxos carregados: {len(flows)}")

        # Criar o dataset, incluindo a normalização dos fluxos se calculada
        dataset = FrameDataset(frames, flows, mean_flow, std_flow)
        train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        
        print("Treinando modelo...")
        train_model(
            model, 
            train_loader, 
            val_loader, 
            device, 
            epochs=epochs, 
            lr=lr, 
            save_path=model_path,  # Caminho para salvar o melhor modelo
            optimizer=optimizer, 
            checkpoint_path=checkpoint_path, 
            start_epoch=start_epoch, 
            best_loss=best_loss, 
            current_video=current_video, 
            used_videos=used_videos
        )

        # Após concluir o treinamento no vídeo atual, marcá-lo como usado
        used_videos.add(video_file)
        current_video = None
        # Resetar start_epoch e best_loss para a próxima iteração
        start_epoch = 0
        best_loss = float('inf')

if __name__ == "__main__":
    
    videos_dir = "videos2"
    flows_dir = "flows_dir"
    
    main(videos_dir, flows_dir)
    
    #parei no videos2