import cv2
try:
    from fer import FER
except ImportError:
    # Algumas versões do pacote expõem a classe no submódulo fer.fer
    from fer.fer import FER

# ============================================
# Detector de emoções em tempo real com webcam
# Biblioteca: FER
# Detector facial: MTCNN
# Classificador: CNN treinada no FER-2013
# ============================================

# Inicializa detector
# mtcnn=True -> usa MTCNN para detecção facial
detector = FER(mtcnn=True)

# Mapeamento de emoções para Português
EMOTION_MAP = {
    "angry": "Raiva",
    "disgust": "Nojo",
    "fear": "Medo",
    "happy": "Feliz",
    "sad": "Triste",
    "surprise": "Surpresa",
    "neutral": "Neutro",
}

# Cores BGR para cada emoção (usadas ao desenhar as porcentagens)
EMOTION_COLORS = {
    "angry": (0, 0, 255),       # vermelho
    "disgust": (0, 128, 0),     # verde escuro
    "fear": (128, 0, 128),      # roxo
    "happy": (0, 215, 255),     # dourado/laranja
    "sad": (255, 0, 0),         # azul
    "surprise": (255, 0, 255),  # magenta
    "neutral": (200, 200, 200), # cinza
}

# Inicializa webcam
cap = cv2.VideoCapture(0)

# Ajusta resolução para melhor performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)

if not cap.isOpened():
    print("Erro ao acessar webcam")
    exit()

print("Pressione 'q' para sair.")

frame_count = 0
last_results = []

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erro ao capturar frame")
        break

    frame_count += 1
    detect_emotions_this_frame = (frame_count % 2 == 0)

    # Detecta emoções apenas a cada 2 frames para ganho de performance
    if detect_emotions_this_frame:
        results = detector.detect_emotions(frame)
        last_results = results
    else:
        results = last_results

    # Preparar para coletar emoções da primeira detecção (para listagem à esquerda)
    first_emotions = None

    for result in results:
        # Coordenadas do rosto
        x, y, w, h = result["box"]

        # Emoções detectadas
        emotions = result["emotions"]

        # Emoção dominante (traduzida) e confiança em %
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]
        dominant_label = EMOTION_MAP.get(dominant_emotion, dominant_emotion)
        dominant_text = f"{dominant_label}: {int(confidence * 100)}%"

        # Desenha retângulo ao redor do rosto
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Texto principal melhorado (DUPLEX, tamanho 1.2, sem contorno duplicado)
        cv2.putText(frame, dominant_text, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

        # Guardar emoções da primeira detecção para desenhar à esquerda do frame
        if first_emotions is None:
            first_emotions = emotions


    # Exibe janela
    # Desenha as porcentagens no lado esquerdo do frame (se houver detecção)
    if first_emotions is not None:
        left_x = 10
        left_y = 30
        offset = 0

        for emotion, score in first_emotions.items():
            label = EMOTION_MAP.get(emotion, emotion)
            percent_text = f"{label}: {int(score * 100)}%"
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))

            # Estilo melhorado: DUPLEX (mais elegante) e tamanho 0.75
            cv2.putText(frame, percent_text, (left_x, left_y + offset), cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2, cv2.LINE_AA)

            offset += 25

    cv2.imshow("Detector de Emoções", frame)

    # Sai ao pressionar q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()