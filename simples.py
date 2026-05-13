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

# Inicializa webcam
cap = cv2.VideoCapture(0)

# Ajusta resolução (opcional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Erro ao acessar webcam")
    exit()

print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erro ao capturar frame")
        break

    # Detecta emoções
    results = detector.detect_emotions(frame)

    for result in results:
        # Coordenadas do rosto
        x, y, w, h = result["box"]

        # Emoções detectadas
        emotions = result["emotions"]

        # Emoção dominante
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]

        # Desenha retângulo
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Texto principal
        text = f"{dominant_emotion}: {confidence:.2f}"

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Lista probabilidades das emoções
        offset = 25

        for emotion, score in emotions.items():
            emotion_text = f"{emotion}: {score:.2f}"

            cv2.putText(
                frame,
                emotion_text,
                (x, y + h + offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            offset += 20

    # Exibe janela
    cv2.imshow("Detector de Emoções", frame)

    # Sai ao pressionar q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()