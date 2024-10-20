import cv2

def calcular_fps(video_path):
    # Abrir o vídeo
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obter a taxa de quadros (FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

# Exemplo de uso
video_path = 'videos_30fps\\gta5.mp4'
fps = calcular_fps(video_path)
print(f'A quantidade de FPS do vídeo é: {fps}')
