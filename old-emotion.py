import os
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - handled at runtime
    tf = None


class EmotionCameraDetector:
    """
    Reusable emotion detector that can run both as a CLI webcam tool and
    as a library used by the FastAPI backend. All TensorFlow / OpenCV
    initialization happens once inside this class so downstream callers
    simply pass frames and get predictions out.
    """

    EMOTION_LABELS: List[str] = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def __init__(
        self,
        model_path: str = r"D:\Emotion-Detection\saved-models\emotion_model-opt1.h5",
        cascade_path: str = r"D:\Emotion-Detection\saved-models\haarcascade_frontalface_default.xml",
        img_size: int = 48,
        confidence_threshold: float = 0.25,
        smoothing_frames: int = 7,
    ) -> None:
        if tf is None:
            raise ImportError("TensorFlow is required for EmotionCameraDetector but is not installed.")

        self.model_path = model_path
        self.cascade_path = cascade_path
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        self.emotion_queue = deque(maxlen=smoothing_frames)
        self.model = self._load_model()
        self.face_cascade = self._load_cascade()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Emotion model not found at {self.model_path}")
        model = tf.keras.models.load_model(self.model_path, compile=False)
        return model

    def _load_cascade(self):
        cascade = cv2.CascadeClassifier(self.cascade_path)
        if cascade.empty():
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if cascade.empty():
            raise IOError("Failed to load Haar Cascade for face detection.")
        return cascade

    def _detect_face(self, gray_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        faces = self.face_cascade.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    def _preprocess_face(self, gray_image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        pad = int(0.15 * w)
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, gray_image.shape[1])
        y2 = min(y + h + pad, gray_image.shape[0])

        face = gray_image[y1:y2, x1:x2]
        face = cv2.resize(face, (self.img_size, self.img_size))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=(0, -1))
        return face

    def _predict(self, face: np.ndarray) -> Tuple[str, float, List[float]]:
        preds = self.model.predict(face, verbose=0)[0]
        confidence = float(np.max(preds))
        label_index = int(np.argmax(preds))

        if confidence < self.confidence_threshold:
            return "Uncertain", confidence, preds.tolist()

        self.emotion_queue.append(label_index)
        smoothed = max(set(self.emotion_queue), key=self.emotion_queue.count)
        return self.EMOTION_LABELS[smoothed], confidence, preds.tolist()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_bgr_frame(
        self, frame: np.ndarray, apply_smoothing: bool = True
    ) -> Tuple[str, float, List[float], bool, Optional[Tuple[int, int, int, int]]]:
        """Analyze a single BGR frame (OpenCV format)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbox = self._detect_face(gray)

        if bbox is None:
            return "Neutral", 1 / len(self.EMOTION_LABELS), [1 / len(self.EMOTION_LABELS)] * len(self.EMOTION_LABELS), False, None

        face = self._preprocess_face(gray, bbox)

        if apply_smoothing:
            emotion, confidence, predictions = self._predict(face)
        else:
            preds = self.model.predict(face, verbose=0)[0]
            confidence = float(np.max(preds))
            label_index = int(np.argmax(preds))
            emotion = (
                self.EMOTION_LABELS[label_index] if confidence >= self.confidence_threshold else "Uncertain"
            )
            predictions = preds.tolist()

        return emotion, confidence, predictions, True, tuple(int(v) for v in bbox)

    def run_webcam(self, camera_index: int = 0) -> None:
        """Standalone webcam runner (similar to the original script)."""
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            raise IOError("❌ Webcam not accessible")

        print("🎥 Webcam started")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                emotion, confidence, _, face_detected, bbox = self.analyze_bgr_frame(frame, apply_smoothing=True)

                if face_detected and bbox is not None:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{emotion} ({confidence:.2f})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("Real-Time Emotion Detection (Press Q or ESC to Exit)", frame)

                if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("🛑 Camera closed safely")


if __name__ == "__main__":
    detector = EmotionCameraDetector()
    detector.run_webcam()
