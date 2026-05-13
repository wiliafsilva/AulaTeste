# ============================================================
# DETECÇÃO DE EMOÇÕES EM TEMPO REAL COM PY-FEAT + WEBCAM
# ============================================================
# MODELOS UTILIZADOS:
#
# Detector de rosto:
#   - RetinaFace
#
# Landmark detector:
#   - PFLD
#
# Emotion model:
#   - ResMaskNet
#
# O sistema mostra:
#   - Bounding box do rosto
#   - Emoção dominante
#   - Percentual de TODAS emoções
#   - FPS
#
# ============================================================

import cv2
import time
import numpy as np
try:
    from feat import Detector
except ImportError as e:
    raise ImportError(
        "Não foi possível importar `Detector` do pacote `feat`. "
        "Provavelmente está instalado outro pacote chamado 'feat'.\n"
        "No ambiente virtual, execute:\n"
        "  pip uninstall feat -y\n"
        "  pip install py-feat\n"
        "Depois, tente executar novamente `python completo.py`."
    ) from e

# ============================================================
# CONFIGURAÇÃO DOS MODELOS
# ============================================================

detector = Detector(
    face_model="retinaface",
    landmark_model="pfld",
    au_model="xgb",
    emotion_model="resmasknet",
    device="cpu"  # use "cuda" se tiver GPU NVIDIA
)

# ============================================================
# EMOÇÕES PADRÃO DO PY-FEAT / AFFECTNET
# ============================================================

EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
    "neutral"
]

# ============================================================
# INICIAR WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir webcam")
    exit()

# ============================================================
# LOOP PRINCIPAL
# ============================================================

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # --------------------------------------------------------
    # Conversão BGR -> RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------------
    # DETECÇÃO
    # --------------------------------------------------------

    try:

        predictions = detector.detect_image(rgb)

        if len(predictions) > 0:

            for idx in range(len(predictions)):

                row = predictions.iloc[idx]

                # ------------------------------------------------
                # Bounding box
                # ------------------------------------------------

                x = int(row["FaceRectX"])
                y = int(row["FaceRectY"])
                w = int(row["FaceRectWidth"])
                h = int(row["FaceRectHeight"])

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                # ------------------------------------------------
                # EMOÇÕES
                # ------------------------------------------------

                emotion_values = {}

                for emotion in EMOTIONS:
                    value = float(row[emotion]) * 100
                    emotion_values[emotion] = value

                # Emoção dominante
                dominant_emotion = max(
                    emotion_values,
                    key=emotion_values.get
                )

                dominant_value = emotion_values[dominant_emotion]

                # ------------------------------------------------
                # TEXO PRINCIPAL
                # ------------------------------------------------

                cv2.putText(
                    frame,
                    f"{dominant_emotion}: {dominant_value:.2f}%",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # ------------------------------------------------
                # LISTA COMPLETA DAS EMOÇÕES
                # ------------------------------------------------

                offset = 20

                for emotion, value in emotion_values.items():

                    text = f"{emotion}: {value:.2f}%"

                    cv2.putText(
                        frame,
                        text,
                        (x, y + h + offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

                    offset += 20

    except Exception as e:
        print("Erro:", e)

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # --------------------------------------------------------
    # EXIBIR
    # --------------------------------------------------------

    cv2.imshow("Py-Feat Emotion Detection", frame)

    # ESC para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ============================================================
# FINALIZAR
# ============================================================

cap.release()
cv2.destroyAllWindows()