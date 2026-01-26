# =========================================================
# BHAAV AI - STREAMLIT (PRODUCTION)
# Zero-latency real-time detection with proper streaming
# Premium UI with Enhanced Design
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
import time

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="BHAAV AI",
    page_icon="🎭",
    layout="wide"
)

# =========================================================
# SESSION STATE
# =========================================================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'running' not in st.session_state:
    st.session_state.running = False
if 'emotion' not in st.session_state:
    st.session_state.emotion = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0
if 'dark' not in st.session_state:
    st.session_state.dark = False
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False

# =========================================================
# PREMIUM CSS STYLES
# =========================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Poppins:wght@400;600;700;900&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, .main-header {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Main Background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Banner Container */
    .banner-container {
        width: 100%;
        margin-bottom: 2rem;
        border-radius: 2rem;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        border: 3px solid rgba(255,255,255,0.2);
    }
    
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
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(240,240,255,0.95) 100%);
        border-right: 3px solid rgba(102,126,234,0.5);
        border-radius: 0 2rem 2rem 0;
        backdrop-filter: blur(10px);
    }
    
    section[data-testid="stSidebar"] > div {
        padding: 1.5rem;
    }
    
    /* 3D Confidence Slider Container */
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
        background: linear-gradient(45deg, 
            transparent, 
            rgba(255,255,255,0.1), 
            transparent);
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #FFFFFF !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        border-radius: 1.5rem !important;
        padding: 1rem 2.5rem !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        font-family: 'Poppins', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 30px rgba(102,126,234,0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
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
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255,255,255,0.5), 
            transparent);
        margin: 1.5rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.9);
        padding: 2rem;
        font-weight: 700;
        font-size: 1rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    /* Confidence Meter Visual */
    .conf-meter-container {
        background: linear-gradient(145deg, #ffffff, #f0f0ff);
        border: 2px solid rgba(102,126,234,0.4);
        border-radius: 1.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 
            0 10px 25px rgba(102,126,234,0.2),
            inset 0 2px 5px rgba(255,255,255,0.8);
    }
    
    /* Mobile Responsive */
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
# THEMES DATA
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

# =========================================================
# LOAD MODEL (CACHED)
# =========================================================
@st.cache_resource
def load():
    m = tf.keras.models.load_model(MODEL, compile=False)
    m.predict(np.zeros((1, 48, 48, 1), dtype=np.float32), verbose=0)
    c = cv2.CascadeClassifier(CASCADE)
    return m, c

# =========================================================
# LOADING SCREEN
# =========================================================
if not st.session_state.model_ready:
    model, cascade = load()
    st.session_state.model_ready = True
else:
    model, cascade = load()

# =========================================================
# BANNER IMAGE
# =========================================================
st.markdown('<div class="banner-container">', unsafe_allow_html=True)
st.image(r"D:\Emotion-Detection\images\20260125_2024_Image Generation_remix_01kftt9pc3fs892v2jnmea8s1c.png", use_column_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# MAIN HEADER
# =========================================================
st.markdown('''
<div class="main-header">
    <span class="main-header-emoji">🎭</span>
    <span>BHAAV AI</span>
</div>
''', unsafe_allow_html=True)

st.markdown('<div class="sub-header">Real-Time Facial Emotion Recognition powered by Deep Learning</div>', unsafe_allow_html=True)

# Status Badge
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
    
    # 3D Confidence Container
    st.markdown('''
    <div class="confidence-container">
        <div class="confidence-header">
            📊 Confidence Threshold
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    conf = st.slider("Adjust Confidence Level", 0.0, 1.0, 0.35, 0.05, label_visibility="collapsed")
    
    st.markdown('''
    <div class="confidence-info-box">
        <p><strong>What is Confidence Threshold?</strong></p>
        <p>This setting filters emotion predictions based on the model's certainty. Higher values show only confident detections.</p>
        
        <div class="confidence-legend">
            <div class="legend-item">
                <div class="legend-dot" style="background: #00FF00;"></div>
                <span><strong>0.7+</strong> → High Accuracy</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: #FFD700;"></div>
                <span><strong>0.4-0.6</strong> → Balanced</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: #FF4444;"></div>
                <span><strong>0.2-0.3</strong> → More Detections</span>
            </div>
        </div>
        
        <div style="margin-top: 1rem; padding: 0.75rem; background: linear-gradient(135deg, rgba(0,255,136,0.2), rgba(0,204,106,0.2)); border-radius: 0.75rem; text-align: center;">
            <span style="color: #00CC6A; font-weight: 800;">✨ Recommended: 0.35 - 0.50</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    show_c = st.checkbox("Show Confidence %", True)
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
        st.info("📭 No detections yet. Start the camera!")
    
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
    st.markdown('<div class="section-header">📹 LIVE CAMERA FEED</div>', unsafe_allow_html=True)
    vid = st.empty()

with col2:
    st.markdown('<div class="section-header">🎯 DETECTED EMOTION</div>', unsafe_allow_html=True)
    emo_display = st.empty()
    
    if not st.session_state.emotion:
        emo_display.markdown('''
        <div class="emotion-card">
            <div style="font-size: 4rem; opacity: 0.5;">🎭</div>
            <h2 style="color: #667eea; font-weight: 800;">Waiting...</h2>
            <p style="color: #4a4a6a; font-size: 1rem;">Start camera to detect emotions</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header">📊 CONFIDENCE METER</div>', unsafe_allow_html=True)
    conf_meter = st.empty()
    
    conf_percent = int(st.session_state.confidence * 100)
    conf_color = "#00FF00" if conf_percent >= 70 else ("#FFD700" if conf_percent >= 40 else "#FF4444")
    
    conf_meter.markdown(f'''
    <div class="conf-meter-container">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem;">
            <span style="color: #667eea; font-weight: 700;">Confidence</span>
            <span style="color: {conf_color}; font-weight: 900; font-size: 1.2rem;">{conf_percent}%</span>
        </div>
        <div style="background: rgba(200,200,220,0.3); border-radius: 1rem; height: 14px; overflow: hidden; box-shadow: inset 0 2px 5px rgba(0,0,0,0.2);">
            <div style="background: linear-gradient(90deg, {conf_color}, {conf_color}80); height: 100%; width: {conf_percent}%; transition: width 0.5s ease; border-radius: 1rem; box-shadow: 0 0 10px {conf_color};"></div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("---")

# Control Buttons
c1, c2, c3 = st.columns([1, 1, 1])
with c2:
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("▶️ START", type="primary"):
            st.session_state.running = True
    with btn_col2:
        if st.button("⏹️ STOP"):
            st.session_state.running = False

# =========================================================
# VIDEO PROCESSING (UNCHANGED LOGIC)
# =========================================================
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    fc = 0
    faces = []
    eq = deque(maxlen=5)
    le = None
    ft = time.time()
    fpc = 0
    cfps = 0
    
    while st.session_state.running:
        ret, f = cap.read()
        if not ret:
            continue
        
        fc += 1
        fpc += 1
        
        if time.time() - ft >= 1:
            cfps = fpc
            fpc = 0
            ft = time.time()
        
        if fc % 4 == 0:
            s = cv2.resize(f, (0, 0), fx=0.4, fy=0.4)
            g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(g, 1.1, 4, minSize=(25, 25))
        
        for (x, y, w, h) in faces:
            x, y, w, h = int(x / 0.4), int(y / 0.4), int(w / 0.4), int(h / 0.4)
            
            roi = f[y:y + h, x:x + w]
            gf = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gf = cv2.resize(gf, (48, 48))
            gf = gf.astype("float32") / 255.0
            gf = np.expand_dims(gf, axis=(0, -1))
            
            if fc % 3 == 0:
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
            
            clr = (102, 126, 234)
            cv2.rectangle(f, (x, y), (x + w, y + h), clr, 3)
            
            lbl = e
            if show_c and e != "Uncertain":
                lbl += f" {st.session_state.confidence:.0%}"
            
            cv2.putText(f, lbl, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2)
        
        if show_f:
            cv2.putText(f, f"FPS: {cfps}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (102, 126, 234), 2)
        
        vid.image(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), channels="RGB")
        
        if fc % 6 == 0 and st.session_state.emotion:
            e = st.session_state.emotion
            t = THEMES[e]
            
            emo_display.markdown(f'''
            <div class="emotion-card">
                <div style="font-size: 4.5rem;">{t['emoji']}</div>
                <h1 style="color: #667eea; font-weight: 900; font-size: 2rem; margin: 0.75rem 0;">{e.upper()}</h1>
                <div style="font-size: 2rem; margin: 0.5rem 0;">{t['icon']}</div>
                <p style="color: #4a4a6a; font-size: 1rem; font-weight: 700;">{t['msg']}</p>
                <div style="margin-top: 1rem; padding: 0.75rem 1.5rem; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 1rem;">
                    <span style="color: #FFF; font-weight: 800;">Confidence: {st.session_state.confidence:.0%}</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            live_conf = int(st.session_state.confidence * 100)
            live_color = "#00FF00" if live_conf >= 70 else ("#FFD700" if live_conf >= 40 else "#FF4444")
            
            conf_meter.markdown(f'''
            <div class="conf-meter-container">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem;">
                    <span style="color: #667eea; font-weight: 700;">Confidence</span>
                    <span style="color: {live_color}; font-weight: 900; font-size: 1.2rem;">{live_conf}%</span>
                </div>
                <div style="background: rgba(200,200,220,0.3); border-radius: 1rem; height: 14px; overflow: hidden; box-shadow: inset 0 2px 5px rgba(0,0,0,0.2);">
                    <div style="background: linear-gradient(90deg, {live_color}, {live_color}80); height: 100%; width: {live_conf}%; transition: width 0.5s ease; border-radius: 1rem; box-shadow: 0 0 10px {live_color};"></div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    cap.release()

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown('<div class="footer">Built with 💜 using TensorFlow & Streamlit</div>', unsafe_allow_html=True)