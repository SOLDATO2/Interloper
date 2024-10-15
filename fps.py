import cv2

# Caminho para o vídeo MP4
video_path = 'interlope_videos\\darkurge_talk_interlope.mp4'

# Abre o vídeo
cap = cv2.VideoCapture(video_path)

# Verifica se o vídeo foi aberto corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
else:
    # Obtém a quantidade de FPS do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Quantidade de FPS: {fps}")

# Libera o objeto de captura de vídeo
cap.release()
