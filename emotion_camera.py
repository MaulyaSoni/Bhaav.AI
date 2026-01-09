# =========================================================
# REAL-TIME EMOTION DETECTION (TF 2.10 – FINAL STABLE)
# =========================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = r"D:\Emotion-Detection\saved-models\emotion_model-opt1.h5"
FACE_CASCADE_PATH = r"D:\Emotion-Detection\saved-models\haarcascade_frontalface_default.xml"

IMG_SIZE = 48
CONFIDENCE_THRESHOLD = 0.30
SMOOTHING_FRAMES = 7

emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

# -----------------------------
# LOAD MODEL (FULL MODEL)
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Emotion model not found")

tf.keras.backend.clear_session()

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("✅ Emotion model loaded")

# -----------------------------
# LOAD HAAR CASCADE
# -----------------------------
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
if face_cascade.empty():
    raise IOError("❌ Haar Cascade failed to load")

print("✅ Haar Cascade Loaded")

# -----------------------------
# SMOOTHING BUFFER
# -----------------------------
emotion_queue = deque(maxlen=SMOOTHING_FRAMES)

# -----------------------------
# CAMERA SETUP
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise IOError("❌ Webcam not accessible")

print("🎥 Webcam started")

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=(0, -1))

        preds = model.predict(face, verbose=0)
        confidence = float(np.max(preds))
        label_index = int(np.argmax(preds))

        if confidence < CONFIDENCE_THRESHOLD:
            emotion = "Uncertain"
        else:
            emotion_queue.append(label_index)
            emotion = emotion_labels[
                max(set(emotion_queue), key=emotion_queue.count)
            ]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{emotion} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Real-Time Emotion Detection | Press Q to Exit", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()
print("🛑 Camera closed safely")

