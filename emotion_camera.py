# =========================================================
# REAL-TIME EMOTION DETECTION
# VISUAL + TERMINAL + FPS OPTIMIZED
# =========================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = r"D:\Emotion-Detection\saved-models\emotion_model-opt1.h5"
FACE_CASCADE_PATH = r"D:\Emotion-Detection\saved-models\haarcascade_frontalface_default.xml"

IMG_SIZE = 48
CONFIDENCE_THRESHOLD = 0.30
SMOOTHING_FRAMES = 5

DETECT_EVERY = 5
PREDICT_EVERY = 3
RESIZE_SCALE = 0.5

emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

# -----------------------------
# LOAD MODEL
# -----------------------------
tf.keras.backend.clear_session()
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Warm-up (FPS boost)
dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
model.predict(dummy, verbose=0)

print("✅ Emotion model loaded")

# -----------------------------
# LOAD FACE CASCADE
# -----------------------------
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
if face_cascade.empty():
    raise IOError("❌ Haar Cascade failed to load")

print("✅ Haar Cascade Loaded")

# -----------------------------
# CAMERA SETUP
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise IOError("❌ Webcam not accessible")

print("🎥 Webcam started")

# -----------------------------
# RUNTIME VARIABLES
# -----------------------------
frame_count = 0
faces = []
emotion_queue = deque(maxlen=SMOOTHING_FRAMES)
last_emotion = None

fps_timer = time.time()
fps_counter = 0

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    fps_counter += 1

    # ---------- FPS COUNTER ----------
    if time.time() - fps_timer >= 1.0:
        print(f"⚡ FPS: {fps_counter}")
        fps_counter = 0
        fps_timer = time.time()

    # ---------- Resize for speed ----------
    small = cv2.resize(frame, None, fx=RESIZE_SCALE, fy=RESIZE_SCALE)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # ---------- Face Detection ----------
    if frame_count % DETECT_EVERY == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60)
        )

    # ---------- Process Faces ----------
    for (x, y, w, h) in faces:
        # scale back to original frame
        x = int(x / RESIZE_SCALE)
        y = int(y / RESIZE_SCALE)
        w = int(w / RESIZE_SCALE)
        h = int(h / RESIZE_SCALE)

        face = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE))
        gray_face = gray_face.astype("float32") / 255.0
        gray_face = np.expand_dims(gray_face, axis=(0, -1))

        if frame_count % PREDICT_EVERY == 0:
            preds = model.predict(gray_face, verbose=0)
            confidence = float(np.max(preds))
            label_index = int(np.argmax(preds))

            if confidence >= CONFIDENCE_THRESHOLD:
                emotion_queue.append(label_index)
                emotion = emotion_labels[
                    max(set(emotion_queue), key=emotion_queue.count)
                ]

                if emotion != last_emotion:
                    print(f"🎭 Emotion → {emotion} ({confidence:.2f})")
                    last_emotion = emotion
            else:
                emotion = "Uncertain"
        else:
            emotion = last_emotion if last_emotion else "Detecting"

        # ---------- DRAW UI ----------
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{emotion}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Real-Time Emotion Detection (Optimized)", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()
print("🛑 Camera closed")
