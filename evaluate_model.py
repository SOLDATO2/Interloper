import os
import argparse
import math
import pickle

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure

import numpy as np
import cv2
import matplotlib.pyplot as plt


from LiteFlowNet.liteflownet_run import LiteFlowNet
from treinamento_v2_5_per_and_flownet import FrameDatasetOnTheFly, FrameInterpolationModel


def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calcula o PSNR entre duas imagens (tensores) com valores em [0,1].
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * math.log10(max_val**2 / mse.item())
    return psnr

def main():
    parser = argparse.ArgumentParser(description="Avaliação do modelo de interpolação: cálculo de SSIM e PSNR")
    parser.add_argument("--vimeo_dir", type=str, default="D:\\vimeo_triplet",
                        help="Diretório root do Vimeo (contendo 'sequences', 'tri_trainlist.txt' e 'tri_testlist.txt')")
    parser.add_argument("--batch_size", type=int, default=4, help="Tamanho do batch para avaliação")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth.tar", help="Caminho para o checkpoint do modelo")
    parser.add_argument("--test_list", type=str, default="tri_testlist.txt", help="Arquivo de lista de teste")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Usando dispositivo: {device}")

    test_list_path = os.path.join(args.vimeo_dir, args.test_list)
    if not os.path.isfile(test_list_path):
        print(f"[Erro] Arquivo {test_list_path} não encontrado!")
        return
    with open(test_list_path, 'r') as f:
        test_lines = [line.strip() for line in f if line.strip()]
    print(f"[Info] Total de triplas de teste: {len(test_lines)}")

    root_sequence = os.path.join(args.vimeo_dir, "sequences")
    if not os.path.isdir(root_sequence):
        print(f"[Erro] Diretório de sequências não encontrado: {root_sequence}")
        return

    lite_flow_net = LiteFlowNet().to(device).eval()
    test_dataset = FrameDatasetOnTheFly(test_lines, root_sequence, lite_flow_net, device)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FrameInterpolationModel().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("[Info] Modelo de interpolação carregado.")

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            print(f"[Info] Processando batch {count+1} de 3781")
            # (input_16c, t2, t1, t3, flow_12, flow_32, warped1, warped3)
            input_16c, gt_frame, _, _, _, _, _, _ = batch
            input_16c = input_16c.to(device)
            gt_frame = gt_frame.to(device)


            output = model(input_16c)
            if output.shape[2:] != gt_frame.shape[2:]:
                output = F.interpolate(output, size=gt_frame.shape[2:], mode='bilinear', align_corners=True)
            
            for i in range(output.size(0)):
                out_img = output[i]
                gt_img = gt_frame[i]
                psnr_val = calculate_psnr(out_img, gt_img, max_val=1.0)
                ssim_val = ssim_metric(out_img.unsqueeze(0), gt_img.unsqueeze(0)).item()
                total_psnr += psnr_val
                total_ssim += ssim_val
                count += 1

    mean_psnr = total_psnr / count if count > 0 else 0.0
    mean_ssim = total_ssim / count if count > 0 else 0.0

    print(f"[Result] PSNR médio: {mean_psnr:.4f} dB")
    print(f"[Result] SSIM médio: {mean_ssim:.4f}")

if __name__ == "__main__":
    main()
