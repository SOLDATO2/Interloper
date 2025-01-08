import cv2

def calcular_fps_resolucao(video_path):
    # Abrir o vídeo
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obter a taxa de quadros (FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Obter a largura e altura do vídeo (resolução)
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Liberar o vídeo
    cap.release()

    return fps, largura, altura

# Exemplo de uso
video_path = 'darkurge_intro.mp4'
fps, largura, altura = calcular_fps_resolucao(video_path)
print(f'A quantidade de FPS do vídeo é: {fps}')
print(f'A resolução do vídeo é: {largura}x{altura}')
