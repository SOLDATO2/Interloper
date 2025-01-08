# precompute_flows.py

import os
import torch
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms

from LiteFlowNet.liteflownet_run import LiteFlowNet  # Ajuste conforme seu projeto
from treinamento_v2_4_per_and_flownet import extract_frames  # import da função de extrair frames

###########################
# Configuração de transform
###########################
transform = transforms.Compose([
    transforms.ToTensor()  # Converte [H,W,3] em [3,H,W] e valores [0..1]
])

###########################
# Função para pré-computar fluxos de UM vídeo
###########################
def compute_and_save_flows(frames, output_dir, lite_flow_net):
    """
    frames: lista de frames (cada um em formato numpy/PIL)
    output_dir: pasta onde salvaremos os fluxos flow_0.pt, flow_1.pt, ...
    lite_flow_net: modelo LiteFlowNet carregado
    """
    lite_flow_net.eval()
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(frames) - 2):
        frame1 = frames[i]
        frame3 = frames[i + 2]

        # Converter frames em tensores [1,3,H,W]
        f1_t = transform(frame1).unsqueeze(0).cuda()
        f3_t = transform(frame3).unsqueeze(0).cuda()

        with torch.no_grad():
            flow = lite_flow_net(f1_t, f3_t)  # => [1,2,H,W]

        # Salvar em disco, ex: "flow_0.pt", "flow_1.pt", ...
        flow_path = os.path.join(output_dir, f"flow_{i}.pt")
        flow = flow.squeeze(0)  # => [2,H,W]
        torch.save(flow.cpu(), flow_path)


        print(f"Salvo fluxo para índice {i} em {flow_path}")

###########################
# Função principal que processa TODOS os vídeos de um diretório
###########################
def precompute_all_videos(input_videos_dir, flows_base_dir, max_seconds=15):
    """
    input_videos_dir: pasta contendo vários vídeos .mp4/.avi/etc.
    flows_base_dir: pasta onde iremos criar subpastas para cada vídeo
    max_seconds: quantos segundos de cada vídeo extrair
    """

    # Cria/instancia a rede
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lite_flow_net = LiteFlowNet().to(device)
    lite_flow_net.eval()

    # Listar todos os arquivos de vídeo
    video_files = [f for f in os.listdir(input_videos_dir)
                   if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    if not video_files:
        print("Nenhum arquivo de vídeo encontrado em", input_videos_dir)
        return

    os.makedirs(flows_base_dir, exist_ok=True)

    for video_file in video_files:
        video_path = os.path.join(input_videos_dir, video_file)
        print(f"\n### Processando vídeo: {video_file} ###")

        # Extrair frames
        frames = extract_frames(video_path, max_seconds=max_seconds)
        print(f"Total de frames extraídos: {len(frames)}")

        # Nome da subpasta para fluxos: usar o nome do vídeo (sem extensão)
        video_name = os.path.splitext(video_file)[0]
        output_dir = os.path.join(flows_base_dir, video_name)

        # Computar e salvar fluxos
        compute_and_save_flows(frames, output_dir, lite_flow_net)

def main():
    # Exemplo de uso
    diretorio_videos = "videos"     # Pasta contendo seus vídeos
    flows_base_dir = "./flows_dir"        # Pasta onde cada vídeo terá sua subpasta de fluxos
    max_seconds = 15

    precompute_all_videos(diretorio_videos, flows_base_dir, max_seconds)

if __name__ == "__main__":
    main()
