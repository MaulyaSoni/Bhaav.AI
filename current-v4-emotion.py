# =========================================================
# EMOTION DETECTION - STREAMLIT APP (ULTRA OPTIMIZED)
# Instant camera start/stop with zero buffering
# =========================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import threading

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'running' not in st.session_state:
    st.session_state.running = False
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'face_cascade' not in st.session_state:
    st.session_state.face_cascade = None
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'current_confidence' not in st.session_state:
    st.session_state.current_confidence = 0
if 'current_fps' not in st.session_state:
    st.session_state.current_fps = 0

# =========================================================
# CUSTOM CSS - DYNAMIC THEME
# =========================================================
def get_theme_css():
    if st.session_state.dark_mode:
        return """
        <style>
            .main {background-color: #0A0E27;}
            .main-header {
                font-size: 3.5rem; font-weight: 900; color: #00D9FF;
                text-align: center; margin-bottom: 0.5rem;
                text-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
            }
            .sub-header {
                text-align: center; color: #FFFFFF;
                font-size: 1.3rem; font-weight: 600; margin-bottom: 2rem;
            }
            .emotion-card {
                padding: 2rem; border-radius: 20px;
                border: 4px solid #00D9FF; background: #1A1F3A;
                text-align: center; box-shadow: 0 0 30px rgba(0, 217, 255, 0.3);
            }
            section[data-testid="stSidebar"] {
                background-color: #1A1F3A; border-right: 3px solid #00D9FF;
            }
            .status-badge {
                padding: 0.5rem 1.5rem; background: #00D9FF;
                color: #0A0E27; border-radius: 25px;
                font-weight: 700; font-size: 0.9rem; margin: 0.5rem;
            }
            h1, h2, h3 {color: #FFD700 !important;}
            p, label {color: #FFFFFF !important;}
            hr {border: 2px solid #00D9FF;}
        </style>
        """
    else:
        return """
        <style>
            .main {background-color: #FFFFFF;}
            .main-header {
                font-size: 3.5rem; font-weight: 900; color: #000000;
                text-align: center; margin-bottom: 0.5rem;
            }
            .sub-header {
                text-align: center; color: #4A4A4A;
                font-size: 1.3rem; font-weight: 600; margin-bottom: 2rem;
            }
            .emotion-card {
                padding: 2rem; border-radius: 20px;
                border: 4px solid #000000; background: white;
                text-align: center; box-shadow: 8px 8px 0px #000000;
            }
            section[data-testid="stSidebar"] {
                background-color: #F5F5F5; border-right: 3px solid #000000;
            }
            .status-badge {
                padding: 0.5rem 1.5rem; background: #000000;
                color: white; border-radius: 25px;
                font-weight: 700; font-size: 0.9rem; margin: 0.5rem;
            }
            hr {border: 2px solid #000000;}
        </style>
        """

st.markdown(get_theme_css(), unsafe_allow_html=True)

# =========================================================
# EMOTION THEMES
# =========================================================
EMOTION_THEMES = {
    "Happy": {"emoji": "😊", "message": "RADIATING POSITIVE VIBES!", "icon": "⚡"},
    "Sad": {"emoji": "😢", "message": "IT'S OKAY TO FEEL DOWN", "icon": "💧"},
    "Angry": {"emoji": "😠", "message": "TAKE A DEEP BREATH", "icon": "🔥"},
    "Surprise": {"emoji": "😲", "message": "SOMETHING UNEXPECTED!", "icon": "⚡"},
    "Fear": {"emoji": "😨", "message": "STAY STRONG & SAFE", "icon": "🛡️"},
    "Disgust": {"emoji": "🤢", "message": "NOT FEELING IT", "icon": "⚠️"},
    "Neutral": {"emoji": "😐", "message": "CALM & COLLECTED", "icon": "⚖️"}
}

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = r"D:\Emotion-Detection\saved-models\emotion_model-opt1.h5"
FACE_CASCADE_PATH = r"D:\Emotion-Detection\saved-models\haarcascade_frontalface_default.xml"

IMG_SIZE = 48
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# =========================================================
# LOAD MODEL (EAGER LOADING ON FIRST RUN)
# =========================================================
@st.cache_resource
def load_model_and_cascade():
    """Load model once and cache it"""
    try:
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        model.predict(dummy, verbose=0)
        
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        if face_cascade.empty():
            return None, None, "Cascade load failed"
        
        return model, face_cascade, None
    except Exception as e:
        return None, None, str(e)

# Load immediately
if not st.session_state.model_loaded:
    with st.spinner("🔄 Initializing AI Model..."):
        model, face_cascade, error = load_model_and_cascade()
        if error:
            st.error(f"❌ {error}")
            st.stop()
        st.session_state.model = model
        st.session_state.face_cascade = face_cascade
        st.session_state.model_loaded = True

# =========================================================
# MAIN UI
# =========================================================
theme_icon = "🌙" if st.session_state.dark_mode else "☀️"
st.markdown('<div class="main-header">🎭 EMOTION DETECTION AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-Time Facial Emotion Recognition Powered by Deep Learning</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    if st.button(f"{theme_icon} TOGGLE THEME", key="theme_btn"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    
    st.markdown("---")
    st.markdown("## ⚙️ SETTINGS")
    
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.35, 0.05, key="conf_slider")
    show_confidence = st.checkbox("Show Confidence", value=True, key="show_conf")
    show_fps = st.checkbox("Show FPS", value=True, key="show_fps")
    
    st.markdown("---")
    st.markdown("## 📊 STATISTICS")
    
    if st.session_state.emotion_history:
        from collections import Counter
        emotion_counts = Counter(st.session_state.emotion_history)
        st.markdown(f"**Total:** {len(st.session_state.emotion_history)}")
        
        for emotion, count in emotion_counts.most_common(3):
            theme = EMOTION_THEMES.get(emotion, EMOTION_THEMES["Neutral"])
            pct = (count/len(st.session_state.emotion_history)*100)
            st.markdown(f"**{theme['emoji']} {emotion}:** {count} ({pct:.1f}%)")
    else:
        st.info("No data yet")
    
    st.markdown("---")
    
    if st.button("🗑️ CLEAR HISTORY", key="clear_btn"):
        st.session_state.emotion_history = []
        st.session_state.current_emotion = None
    
    st.markdown("---")
    st.markdown("## 🤖 MODEL")
    st.markdown('<span class="status-badge">✅ READY</span>', unsafe_allow_html=True)

# =========================================================
# LAYOUT
# =========================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📹 LIVE FEED")
    FRAME_WINDOW = st.image([])
    
with col2:
    st.markdown("### 🎯 EMOTION")
    emotion_display = st.empty()

st.markdown("---")

col_start, col_stop = st.columns(2)
with col_start:
    if st.button("▶️ START", key="start_btn", type="primary"):
        st.session_state.running = True
with col_stop:
    if st.button("⏹️ STOP", key="stop_btn"):
        st.session_state.running = False

# =========================================================
# WEBCAM PROCESSING - INSTANT START/STOP
# =========================================================
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    model = st.session_state.model
    face_cascade = st.session_state.face_cascade
    
    frame_count = 0
    faces = []
    emotion_queue = deque(maxlen=5)
    last_emotion = None
    
    import time
    fps_time = time.time()
    fps_count = 0
    
    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            fps_count += 1
            
            # FPS calculation
            if time.time() - fps_time >= 1.0:
                st.session_state.current_fps = fps_count
                fps_count = 0
                fps_time = time.time()
            
            # Detect faces every 3 frames
            if frame_count % 3 == 0:
                small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            # Process faces
            for (x, y, w, h) in faces:
                x, y, w, h = x*2, y*2, w*2, h*2
                
                face = frame[y:y+h, x:x+w]
                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE))
                gray_face = gray_face.astype("float32") / 255.0
                gray_face = np.expand_dims(gray_face, axis=(0, -1))
                
                if frame_count % 2 == 0:
                    preds = model.predict(gray_face, verbose=0)[0]
                    conf = float(np.max(preds))
                    idx = int(np.argmax(preds))
                    
                    if conf >= confidence:
                        emotion_queue.append(idx)
                        emotion = emotion_labels[max(set(emotion_queue), key=emotion_queue.count)]
                        
                        if emotion != last_emotion:
                            last_emotion = emotion
                            st.session_state.current_emotion = emotion
                            st.session_state.current_confidence = conf
                            st.session_state.emotion_history.append(emotion)
                    else:
                        emotion = "Uncertain"
                else:
                    emotion = last_emotion or "Detecting"
                
                # Draw
                color = (0, 217, 255) if st.session_state.dark_mode else (0, 0, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                label = f"{emotion}"
                if show_confidence and emotion != "Uncertain":
                    label += f" {st.session_state.current_confidence:.0%}"
                
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # FPS
            if show_fps:
                fps_color = (0, 217, 255) if st.session_state.dark_mode else (0, 0, 0)
                cv2.putText(frame, f"FPS: {st.session_state.current_fps}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
            
            # Display
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Update emotion card
            if st.session_state.current_emotion:
                emotion = st.session_state.current_emotion
                theme = EMOTION_THEMES[emotion]
                text_color = "#FFD700" if st.session_state.dark_mode else "#000000"
                
                emotion_display.markdown(f"""
                <div class="emotion-card">
                    <div style="font-size: 5rem;">{theme['emoji']}</div>
                    <h1 style="color: {text_color}; margin: 1rem 0; font-weight: 900;">{emotion.upper()}</h1>
                    <div style="font-size: 2.5rem;">{theme['icon']}</div>
                    <p style="color: {text_color}; font-size: 1.1rem; font-weight: 700;">{theme['message']}</p>
                </div>
                """, unsafe_allow_html=True)
            
    finally:
        cap.release()
        cv2.destroyAllWindows()

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
footer_color = "#FFFFFF" if st.session_state.dark_mode else "#000000"
st.markdown(f"""
<div style="text-align: center; color: {footer_color}; padding: 1.5rem; font-weight: 700;">
    BUILT WITH 🖤 USING STREAMLIT & TENSORFLOW
</div>
""", unsafe_allow_html=True)