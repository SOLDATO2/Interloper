# precompute_flows.py

import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2

from LiteFlowNet.liteflownet_run import LiteFlowNet  # Ajuste ao seu projeto
from treinamento_v2_5_per_and_flownet import extract_frames  # Função para extrair frames

transform = transforms.Compose([
    transforms.ToTensor()  # [H,W,3] -> [3,H,W], valores [0..1]
])

def compute_and_save_flows(frames, output_dir, lite_flow_net):
    """
    frames: lista de frames (numpy/PIL)
    output_dir: pasta onde salvaremos fluxos (flow_12_i.pt, flow_32_i.pt)
    lite_flow_net: instância do LiteFlowNet (rede de fluxo)
    """
    lite_flow_net.eval()
    os.makedirs(output_dir, exist_ok=True)

    # i => frame1, i+1 => frame2, i+2 => frame3
    for i in range(len(frames) - 2):
        frame1 = frames[i]
        frame2 = frames[i + 1]
        frame3 = frames[i + 2]

        # Converte p/ tensores [1,3,H,W]
        f1_t = transform(frame1).unsqueeze(0).cuda()
        f2_t = transform(frame2).unsqueeze(0).cuda()
        f3_t = transform(frame3).unsqueeze(0).cuda()

        with torch.no_grad():
            # Fluxo de frame1 -> frame2
            flow_1_2 = lite_flow_net(f1_t, f2_t)  # [1,2,H,W]
            # Fluxo de frame3 -> frame2
            flow_3_2 = lite_flow_net(f3_t, f2_t)  # [1,2,H,W]

        flow_1_2 = flow_1_2.squeeze(0).cpu()  # [2,H,W]
        flow_3_2 = flow_3_2.squeeze(0).cpu()

        # Salvar
        flow_12_path = os.path.join(output_dir, f"flow_12_{i}.pt")
        flow_32_path = os.path.join(output_dir, f"flow_32_{i}.pt")
        torch.save(flow_1_2, flow_12_path)
        torch.save(flow_3_2, flow_32_path)

        print(f"[{i}] salvo flow_12_{i}.pt e flow_32_{i}.pt")

def precompute_all_videos(input_videos_dir, flows_base_dir, max_seconds=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lite_flow_net = LiteFlowNet().to(device).eval()

    video_files = [f for f in os.listdir(input_videos_dir) if f.lower().endswith(('.mp4','.avi','.mov'))]
    if not video_files:
        print("Nenhum arquivo de vídeo encontrado!")
        return

    os.makedirs(flows_base_dir, exist_ok=True)

    for video_file in video_files:
        video_path = os.path.join(input_videos_dir, video_file)
        print(f"\n### Processando vídeo: {video_file} ###")

        # Extrai frames
        frames = extract_frames(video_path, max_seconds=max_seconds)
        print(f"Total de frames extraídos: {len(frames)}")

        # Cria a subpasta de fluxo para esse vídeo
        video_name = os.path.splitext(video_file)[0]
        out_dir = os.path.join(flows_base_dir, video_name)

        # Computa e salva fluxos
        compute_and_save_flows(frames, out_dir, lite_flow_net)

def main():
    videos_dir = "videos3"
    flows_dir = "flows_dir"
    max_seconds = 15

    precompute_all_videos(videos_dir, flows_dir, max_seconds)

if __name__ == "__main__":
    main()
