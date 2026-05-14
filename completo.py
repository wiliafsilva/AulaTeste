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
import io
try:
    import torch
except Exception:
    torch = None
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

device = "cpu"
if torch is not None:
    try:
        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        device = "cpu"

print(f"Using device: {device}")

detector = Detector(
    face_model="retinaface",
    landmark_model="pfld",
    au_model="xgb",
    emotion_model="resmasknet",
    device=device
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

# Mapeamento de emoções para Português (compatível com os nomes do py-feat)
EMOTION_MAP = {
    "anger": "Raiva",
    "disgust": "Nojo",
    "fear": "Medo",
    "happiness": "Feliz",
    "sadness": "Triste",
    "surprise": "Surpresa",
    "neutral": "Neutro",
}

# Cores BGR para cada emoção (usadas ao desenhar as porcentagens)
EMOTION_COLORS = {
    "anger": (0, 0, 255),       # vermelho
    "disgust": (0, 128, 0),     # verde escuro
    "fear": (128, 0, 128),      # roxo
    "happiness": (0, 215, 255), # dourado/laranja
    "sadness": (255, 0, 0),     # azul
    "surprise": (255, 0, 255),  # magenta
    "neutral": (200, 200, 200), # cinza
}

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
frame_count = 0
last_emotions = None
last_face_boxes = None

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # --------------------------------------------------------
        # Conversão BGR -> RGB
        # --------------------------------------------------------

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --------------------------------------------------------
        # DETECÇÃO (redimensionar para acelerar e escalar de volta)
        # Processa emoções a cada 2 frames para ganho de FPS
        # --------------------------------------------------------

        try:
            frame_count += 1
            detect_emotions_this_frame = (frame_count % 2 == 0)

            # redimensionar imagem para largura máxima (mantém proporção)
            det_img = rgb
            scale = 1.0
            max_w = 320
            h, w = rgb.shape[:2]
            if w > max_w:
                scale = w / max_w
                det_img = cv2.resize(rgb, (max_w, int(h / scale)))

            # Sempre detecta rostos (rápido)
            faces = detector.detect_faces(det_img)

            # Detecta emoções apenas a cada 2 frames para ganho de performance
            first_emotions = None
            emotion_cols = detector.info.get("emotion_model_columns", EMOTIONS)
            
            if len(faces) > 0 and len(faces[0]) > 0:
                if detect_emotions_this_frame:
                    # Detectar landmarks e emoções apenas a cada 2 frames
                    landmarks = detector.detect_landmarks(det_img, faces)
                    emotions_list = detector.detect_emotions(det_img, faces, landmarks)
                    last_emotions = emotions_list
                    last_face_boxes = faces
                else:
                    # Reutilizar emoções do frame anterior
                    emotions_list = last_emotions

                # Desenhar rostos e emoções
                if emotions_list is not None:
                    for face_idx, face in enumerate(last_face_boxes[0] if last_face_boxes else faces[0]):
                        x1, y1, x2, y2, conf = face

                        # escalar coordenadas de volta para o frame original
                        x1o = int(x1 * scale)
                        y1o = int(y1 * scale)
                        x2o = int(x2 * scale)
                        y2o = int(y2 * scale)
                        x, y, w_box, h_box = x1o, y1o, x2o - x1o, y2o - y1o

                        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

                        emo_vals = emotions_list[0][face_idx]  # numpy array
                        emotion_values = {
                            emotion_cols[i]: float(emo_vals[i]) * 100
                            for i in range(len(emotion_cols))
                        }
                        # guardar para desenhar a lista à esquerda (apenas a primeira face)
                        if first_emotions is None:
                            first_emotions = emotion_values

                        # Emoção dominante (estilo melhorado: maior, sem contorno duplicado)
                        dominant_emotion = max(emotion_values, key=lambda k: emotion_values[k])
                        dominant_value = emotion_values[dominant_emotion]

                        dominant_label = EMOTION_MAP.get(dominant_emotion, dominant_emotion)
                        dominant_text = f"{dominant_label}: {int(dominant_value)}%"

                        # Texto maior e mais legível (DUPLEX, tamanho 1.2)
                        cv2.putText(frame, dominant_text, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

                    # Desenhar lista de porcentagens no lado esquerdo (para a primeira face detectada)
                    if first_emotions is not None:
                        left_x = 10
                        left_y = 30
                        offset = 0
                        # desenhar nas mesmas colunas definidas em emotion_cols (ordem consistente)
                        for emotion in emotion_cols:
                            value = first_emotions.get(emotion, 0)
                            label = EMOTION_MAP.get(emotion, emotion)
                            pct = max(0, min(int(value), 100))
                            percent_text = f"{label}: {pct}%"
                            color = EMOTION_COLORS.get(emotion, (255, 255, 255))

                            # Estilo melhorado: DUPLEX (mais elegante) e tamanho 0.75
                            cv2.putText(frame, percent_text, (left_x, left_y + offset), cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2, cv2.LINE_AA)

                            offset += 25
        except Exception as e:
            print("Erro:", e)

        # --------------------------------------------------------
        # FPS (canto superior direito)
        # --------------------------------------------------------

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
        prev_time = current_time

        fps_text = f"FPS: {fps:.2f}"
        fps_x = frame.shape[1] - 150
        # contorno escuro
        cv2.putText(frame, fps_text, (fps_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        # texto em amarelo
        cv2.putText(frame, fps_text, (fps_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # --------------------------------------------------------
        # EXIBIR
        # --------------------------------------------------------

        cv2.imshow("Py-Feat Emotion Detection", frame)

        # ESC ou 'q' para sair
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

except KeyboardInterrupt:
    print("Interrompido pelo usuário")

# ============================================================
# FINALIZAR
# ============================================================

cap.release()
cv2.destroyAllWindows()