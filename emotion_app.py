# =========================================================
# BHAAV AI - STREAMLIT (PRODUCTION)
# Ultra-optimized with async processing and zero latency
# =========================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter
import threading
from queue import Queue
import time
from textwrap import dedent
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="BHAAV AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# SESSION STATE
# =========================================================
defaults = {
    'history': [],
    'running': False,
    'emotion': None,
    'confidence': 0,
    'model_ready': False,
    'frame_queue': Queue(maxsize=2),
    'result_queue': Queue(maxsize=2),
    'fps': 0
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =========================================================
# PREMIUM CSS STYLES
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Poppins:wght@400;600;700;900&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, .main-header {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Fix white blocks - Make everything transparent on gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Remove all white backgrounds */
    .block-container {
        background: transparent !important;
    }
    
    div[data-testid="stAppViewContainer"] {
        background: transparent !important;
    }
    
    div[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Navbar fix */
    .stApp > header {
        background: transparent !important;
    }
    
    /* Banner Container */
    # .banner-container {
    #     width: 100%;
    #     margin-bottom: 2rem;
    #     border-radius: 2rem;
    #     overflow: hidden;
    #     box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    #     border: 3px solid rgba(255,255,255,0.2);
    # }
    
    /* Headers */
    .main-header {
        font-size: 4rem;
        font-weight: 900;
        color: #FFFFFF;
        text-align: center;
        margin: 1.5rem 0 0.5rem 0;
        letter-spacing: 2px;
        text-shadow: 0 4px 20px rgba(0,0,0,0.4);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }
    
    .main-header-emoji {
        font-size: 4rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .sub-header {
        text-align: center;
        color: rgba(255,255,255,0.95);
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 2rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    /* Status Badge */
    .status-badge {
        padding: 0.8rem 2rem;
        background: linear-gradient(135deg, #00FF88 0%, #00CC6A 100%);
        color: #1a1a2e;
        border-radius: 50px;
        font-weight: 800;
        font-size: 1rem;
        box-shadow: 0 8px 25px rgba(0,255,136,0.4);
        display: inline-flex;
        align-items: center;
        gap: 0.7rem;
        animation: pulse-glow 2s infinite;
        border: 2px solid rgba(255,255,255,0.3);
    }
    
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 8px 25px rgba(0,255,136,0.4); }
        50% { box-shadow: 0 12px 35px rgba(0,255,136,0.6); }
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(240,240,255,0.95) 100%);
        border-right: 3px solid rgba(102,126,234,0.5);
        border-radius: 0 2rem 2rem 0;
        backdrop-filter: blur(10px);
    }
    
    section[data-testid="stSidebar"] > div {
        padding: 1.5rem;
    }
    
    /* 3D Confidence Container */
    .confidence-container {
        background: linear-gradient(145deg, #ffffff, #f0f0ff);
        border: 3px solid #667eea;
        border-radius: 1.5rem;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 
            0 10px 30px rgba(102,126,234,0.3),
            inset 0 2px 5px rgba(255,255,255,0.8),
            inset 0 -2px 5px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .confidence-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .confidence-header {
        color: #667eea;
        font-weight: 800;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .confidence-info-box {
        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
        border: 2px solid rgba(102,126,234,0.3);
        border-radius: 1rem;
        padding: 1rem;
        margin-top: 1rem;
        box-shadow: inset 0 2px 10px rgba(102,126,234,0.1);
    }
    
    .confidence-info-box p {
        color: #4a4a6a;
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0;
    }
    
    .confidence-legend {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        margin-top: 1rem;
        padding: 0.75rem;
        background: rgba(255,255,255,0.5);
        border-radius: 0.75rem;
    }
    
    .confidence-recommendation {
        margin-top: 1rem;
        padding: 0.75rem;
        background: linear-gradient(135deg, rgba(0,255,136,0.2), rgba(0,204,106,0.2));
        border-radius: 0.75rem;
        text-align: center;
    }
    
    .confidence-recommendation span {
        color: #00CC6A;
        font-weight: 800;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        color: #4a4a6a;
    }
    
    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Emotion Card */
    .emotion-card {
        padding: 2rem;
        border-radius: 2rem;
        border: 3px solid rgba(255,255,255,0.3);
        background: linear-gradient(145deg, rgba(255,255,255,0.95), rgba(240,240,255,0.95));
        text-align: center;
        box-shadow: 
            0 20px 60px rgba(0,0,0,0.2),
            inset 0 2px 5px rgba(255,255,255,0.8);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .emotion-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 70px rgba(0,0,0,0.3);
    }
    
   /* Buttons */
.stButton > button {
    background: linear-gradient(
        135deg,
        #ff0844 0%,   /* hot red */
        #b721ff 50%,  /* vivid purple */
        #ff5fcb 100%  /* neon pink */
    ) !important;

    color: #ffffff !important;
    cursor: pointer;
    
    border: 2px solid rgba(255,255,255,0.5) !important;
    border-radius: 1.5rem !important;
    padding: 1rem 2.5rem !important;
    font-weight: 1000 !important;
    font-size: 1.1rem !important;
    font-family: 'Poppins', sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    opacity:0.7;

    box-shadow:
        0 12px 35px rgba(255, 8, 68, 0.45),
        0 0 20px rgba(255, 95, 203, 0.6) !important;

    transition: all 0.25s ease !important;
}

/* Hover effect */
.stButton > button:hover {
    transform: translateY(-3px) scale(1.05);
    box-shadow:
        0 18px 45px rgba(183, 33, 255, 0.6),
        0 0 30px rgba(255, 95, 203, 0.9) !important;
    opacity:1;
    font-size: 1.3rem !important;
}

/* Active / Click */
.stButton > button:active {
    transform: scale(0.97);
    box-shadow:
        0 8px 20px rgba(255, 8, 68, 0.4) !important;
    opacity:1;
}

    
    .stButton > button:hover {
        transform: scale(1.05) translateY(-2px) !important;
        box-shadow: 0 15px 40px rgba(102,126,234,0.6) !important;
    }
    
    /* Section Headers */
    .section-header {
        color: #FFFFFF;
        font-weight: 800;
        font-size: 1.4rem;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.7rem;
        text-shadow: 0 3px 10px rgba(0,0,0,0.3);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Stats Item */
    .stats-item {
        background: linear-gradient(145deg, #ffffff, #f5f5ff);
        border: 2px solid rgba(102,126,234,0.3);
        border-radius: 1rem;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102,126,234,0.2);
    }
    
    .stats-item:hover {
        transform: translateX(5px);
        border-color: #667eea;
        box-shadow: 0 6px 20px rgba(102,126,234,0.3);
    }
    
    /* Sidebar Headers */
    .sidebar-section {
        color: #667eea;
        font-weight: 800;
        font-size: 1.1rem;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'Poppins', sans-serif;
    }
    
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
        margin: 1.5rem 0;
    }
    
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.9);
        padding: 2rem;
        font-weight: 700;
        font-size: 1rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem !important;
        }
        .emotion-card {
            padding: 1.5rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# THEMES
# =========================================================
THEMES = {
    "Happy": {"emoji": "😊", "msg": "RADIATING POSITIVE VIBES!", "icon": "⚡"},
    "Sad": {"emoji": "😢", "msg": "IT'S OKAY TO FEEL DOWN", "icon": "💧"},
    "Angry": {"emoji": "😠", "msg": "TAKE A DEEP BREATH", "icon": "🔥"},
    "Surprise": {"emoji": "😲", "msg": "SOMETHING UNEXPECTED!", "icon": "⚡"},
    "Fear": {"emoji": "😨", "msg": "STAY STRONG & SAFE", "icon": "🛡️"},
    "Disgust": {"emoji": "🤢", "msg": "NOT FEELING IT", "icon": "⚠️"},
    "Neutral": {"emoji": "😐", "msg": "CALM & COLLECTED", "icon": "⚖️"}
}

# =========================================================
# CONFIG
# =========================================================
MODEL = r"D:\Emotion-Detection\saved-models\emotion_model-opt1.h5"
CASCADE = r"D:\Emotion-Detection\saved-models\haarcascade_frontalface_default.xml"
LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
BANNER_IMAGE = r"D:\Emotion-Detection\images\image.png"

# =========================================================
# LOAD MODEL (CACHED)
# =========================================================
@st.cache_resource
def load():
    m = tf.keras.models.load_model(MODEL, compile=False)
    m.predict(np.zeros((1, 48, 48, 1), dtype=np.float32), verbose=0)
    c = cv2.CascadeClassifier(CASCADE)
    return m, c

if not st.session_state.model_ready:
    model, cascade = load()
    st.session_state.model_ready = True
else:
    model, cascade = load()

# =========================================================
# BANNER
# =========================================================
st.markdown('<div class="banner-container">', unsafe_allow_html=True)
st.image(BANNER_IMAGE, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown('''
<div class="main-header">
    <span class="main-header-emoji">🎭</span>
    <span>BHAAV AI</span>
</div>
''', unsafe_allow_html=True)

st.markdown('<div class="sub-header">Real-Time Facial Emotion Recognition powered by Deep Learning</div>', unsafe_allow_html=True)

st.markdown('''
<div style="text-align: center; margin-bottom: 2rem;">
    <div class="status-badge">✅ MODEL READY</div>
</div>
''', unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown('<div class="sidebar-section">⚙️ SETTINGS</div>', unsafe_allow_html=True)
    
    st.markdown(dedent('''
        <div class="confidence-container">
            <div class="confidence-header">📊 Confidence Threshold</div>
        </div>
    '''), unsafe_allow_html=True)
    
    conf = st.slider("Adjust Level", 0.0, 1.0, 0.35, 0.05, label_visibility="collapsed")
    
    st.markdown(dedent('''
        <div class="confidence-info-box">
            <p><strong>What is Confidence Threshold?</strong></p>
            <p>Filters predictions based on model certainty. Higher = more accurate but fewer detections.</p>
            <div class="confidence-recommendation">
                <p><span>✨ Recommended: 0.35 - 0.50</span></p>
            </div>
        </div>
    '''), unsafe_allow_html=True)
    
    show_f = st.checkbox("Show FPS Counter", True)
    
    st.markdown("---")
    st.markdown('<div class="sidebar-section">📊 SESSION STATS</div>', unsafe_allow_html=True)
    
    if st.session_state.history:
        cnts = Counter(st.session_state.history)
        st.markdown(f"**Total Detections:** {len(st.session_state.history)}")
        for e, n in cnts.most_common(3):
            p = (n / len(st.session_state.history) * 100)
            st.markdown(f'''
            <div class="stats-item">
                {THEMES[e]['emoji']} <strong>{e}</strong>: {n} ({p:.0f}%)
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.info("📭 No detections yet")
    
    st.markdown("---")
    if st.button("🗑️ CLEAR HISTORY"):
        st.session_state.history = []
        st.session_state.emotion = None
        st.rerun()

# =========================================================
# MAIN LAYOUT
# =========================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="section-header">📹 LIVE FEED</div>', unsafe_allow_html=True)
    vid = st.empty()

with col2:
    st.markdown('<div class="section-header">🎯 EMOTION</div>', unsafe_allow_html=True)
    emo_display = st.empty()
    
    if not st.session_state.emotion:
        emo_display.markdown('''
        <div class="emotion-card">
            <div style="font-size: 4rem; opacity: 0.5;">🎭</div>
            <h2 style="color: #667eea; font-weight: 800;">Waiting...</h2>
            <p style="color: #4a4a6a;">Start camera to detect</p>
        </div>
        ''', unsafe_allow_html=True)

st.markdown("---")

# Buttons
c1, c2, c3 = st.columns([1, 1, 1])
with c2:
    b1, b2 = st.columns(2)
    with b1:
        if st.button("▶️ START", type="primary"):
            st.session_state.running = True
    with b2:
        if st.button("⏹️ STOP"):
            st.session_state.running = False

# =========================================================
# OPTIMIZED VIDEO PROCESSING
# =========================================================
if st.session_state.running:
    # Initialize camera with MJPEG + DirectShow
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    time.sleep(0.5)
    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    fc = 0
    faces = []
    eq = deque(maxlen=5)
    le = None
    ft = time.time()
    fpc = 0
    cfps = 0
    
    # Process only every 5th frame for emotion (frame skipping)
    PROCESS_EVERY = 5
    
    while st.session_state.running:
        ret, f = cap.read()
        if not ret:
            continue
        
        f = cv2.flip(f, 1)  # Mirror
        fc += 1
        fpc += 1
        
        if time.time() - ft >= 1:
            cfps = fpc
            fpc = 0
            ft = time.time()
        
        # Fast face detection every 3 frames
        if fc % 3 == 0:
            s = cv2.resize(f, (320, 240))  # Lower resolution for detection
            g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(g, 1.15, 3, minSize=(30, 30))
        
        # Process emotion only every 5th frame
        for (x, y, w, h) in faces:
            x, y, w, h = x*2, y*2, w*2, h*2
            
            # Only predict on select frames
            if fc % PROCESS_EVERY == 0:
                roi = f[y:y+h, x:x+w]
                gf = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gf = cv2.resize(gf, (48, 48))
                gf = gf.astype("float32") / 255.0
                gf = np.expand_dims(gf, axis=(0, -1))
                
                p = model.predict(gf, verbose=0)[0]
                c = float(np.max(p))
                i = int(np.argmax(p))
                
                if c >= conf:
                    eq.append(i)
                    e = LABELS[max(set(eq), key=eq.count)]
                    if e != le:
                        le = e
                        st.session_state.emotion = e
                        st.session_state.confidence = c
                        st.session_state.history.append(e)
                else:
                    e = "Uncertain"
            else:
                e = le or "..."
            
            # Draw
            cv2.rectangle(f, (x, y), (x+w, y+h), (102, 126, 234), 3)
            cv2.putText(f, e, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 126, 234), 2)
        
        if show_f:
            cv2.putText(f, f"FPS: {cfps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 126, 234), 2)
        
        vid.image(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), channels="RGB")
        
        # Update UI every 15 frames
        if fc % 15 == 0 and st.session_state.emotion:
            e = st.session_state.emotion
            t = THEMES[e]
            
            emo_display.markdown(f'''
            <div class="emotion-card">
                <div style="font-size: 5rem;">{t['emoji']}</div>
                <h1 style="color: #667eea; font-weight: 900; font-size: 2.2rem; margin: 0.75rem 0;">{e.upper()}</h1>
                <div style="font-size: 2.5rem; margin: 0.5rem 0;">{t['icon']}</div>
                <p style="color: #4a4a6a; font-size: 1.1rem; font-weight: 700;">{t['msg']}</p>
                <div style="margin-top: 1rem; padding: 0.75rem 1.5rem; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 1rem;">
                    <span style="color: #FFF; font-weight: 800;">Confidence: {st.session_state.confidence:.0%}</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    cap.release()

st.markdown("---")
st.markdown('<div class="footer"> Built with ❤️ using TensorFlow & Streamlit by Maulya Soni</div>', unsafe_allow_html=True)