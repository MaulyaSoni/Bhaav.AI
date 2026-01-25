# =========================================================
#BHAAV AI - STREAMLIT (PRODUCTION)
# Zero-latency real-time detection with proper streaming
# Premium UI with Dark/Light Themes
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
    st.session_state.dark = True  # Default to dark mode
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False

# =========================================================
# THEME COLORS
# =========================================================
if st.session_state.dark:
    bg = "#000000"
    sidebar_bg = "#0a0a0a"
    primary_gradient = "linear-gradient(135deg, #FF00CC 0%, #333399 100%)"
    text_color = "#FFFFFF"
    sub_text = "#e0e0e0"
    card_bg = "#121212"
    border_color = "#FF00CC"
    accent = "#FF00CC"
    btn_gradient = "linear-gradient(135deg, #FF00CC 0%, #6633CC 100%)"
    shadow = "0 8px 32px rgba(255, 0, 204, 0.3)"
else:
    bg = "#FAFAFA"
    sidebar_bg = "#FFFFFF"
    primary_gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    text_color = "#1a1a2e"
    sub_text = "#4a4a6a"
    card_bg = "#FFFFFF"
    border_color = "#667eea"
    accent = "#667eea"
    btn_gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    shadow = "0 8px 24px rgba(102, 126, 234, 0.2)"

# =========================================================
# PREMIUM CSS STYLES
# =========================================================
st.markdown(f"""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    /* ========================================== */
    /* 🔴 FIX 1: White Block Removal (Dark Mode) */
    /* ========================================== */
    
    /* Main Background */
    .main {{
        background-color: {bg};
    }}
    
    .stApp {{
        background: {bg};
    }}
    
    /* Fix white blocks in dark mode - CRITICAL */
    .block-container {{
        background-color: transparent !important;
    }}
    
    div[data-testid="stVerticalBlock"] {{
        background: transparent !important;
    }}
    
    div[data-testid="stAppViewContainer"] {{
        background-color: {bg} !important;
    }}
    
    div[data-testid="stHorizontalBlock"] {{
        background: transparent !important;
    }}
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background: {sidebar_bg};
        border-right: 2px solid {border_color};
        border-radius: 0 1.5rem 1.5rem 0;
    }}
    
    section[data-testid="stSidebar"] > div {{
        padding: 1rem;
    }}
    
    /* ========================================== */
    /* 🎭 FIX 2: Emoji & Icon Rendering          */
    /* ========================================== */
    
    /* Emoji & icon rendering fix */
    .emoji, .icon {{
        filter: none !important;
        opacity: 1 !important;
        text-shadow: none !important;
    }}
    
    .emotion-card div:first-child {{
        filter: drop-shadow(0 0 8px rgba(255,0,204,0.6));
    }}
    
    /* Headers */
    .main-header {{
        font-size: 3rem;
        font-weight: 900;
        background: {primary_gradient};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }}
    
    .sub-header {{
        text-align: center;
        color: {sub_text};
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }}
    
    /* Status Badge */
    .status-container {{
        display: flex;
        justify-content: center;
        margin-bottom: 1.5rem;
    }}
    
    .status-badge {{
        padding: 0.6rem 1.8rem;
        background: {btn_gradient};
        color: #FFFFFF;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.9rem;
        box-shadow: {shadow};
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        animation: pulse 2s infinite;
    }}
    
    .status-badge.idle {{
        background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
        animation: none;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); opacity: 1; }}
        50% {{ transform: scale(1.02); opacity: 0.9; }}
    }}
    
    /* Emotion Card */
    .emotion-card {{
        padding: 2rem;
        border-radius: 1.5rem;
        border: 2px solid {border_color};
        background: {card_bg};
        text-align: center;
        box-shadow: {shadow};
        transition: all 0.3s ease;
    }}
    
    .emotion-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(255, 0, 204, 0.4);
    }}
    
    /* Confidence Meter Section */
    .confidence-info {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 1rem;
        padding: 1rem;
        margin: 1rem 0;
    }}
    
    .confidence-info h4 {{
        color: {accent};
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }}
    
    .confidence-info p {{
        color: {sub_text};
        font-size: 0.85rem;
        line-height: 1.5;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: {btn_gradient} !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 1rem !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: {shadow} !important;
    }}
    
    .stButton > button:hover {{
        transform: scale(1.05) !important;
        box-shadow: 0 12px 35px rgba(255, 0, 204, 0.5) !important;
    }}
    
    .stButton > button:active {{
        transform: scale(0.98) !important;
    }}
    
    /* ========================================== */
    /* 🎚️ FIX 3: Premium Volume-Style Slider     */
    /* ========================================== */
    
    /* Premium Confidence Slider (Dark Mode) */
    .stSlider {{
        padding: 0.5rem 0.2rem;
    }}
    
    .stSlider > div {{
        height: 1.1rem;
        border-radius: 1rem;
        background: rgba(255,255,255,0.1);
        border: 1px solid {border_color};
    }}
    
    .stSlider > div > div {{
        border-radius: 1rem !important;
        background: {primary_gradient} !important;
    }}
    
    .stSlider > div > div > div > div {{
        width: 1.2rem !important;
        height: 1.2rem !important;
        border-radius: 50% !important;
        background: #ffffff !important;
        border: 3px solid {accent} !important;
        box-shadow: 0 0 12px rgba(255,0,204,0.8);
        transform: translateY(-2px);
    }}
    
    /* ========================================== */
    /* 🌞 FIX 4: Light Mode Slider Enhancement   */
    /* ========================================== */
    
    /* Light mode slider enhancement */
    {"" if st.session_state.dark else """
    .stSlider > div {{
        background: #eaeaf5 !important;
    }}
    
    .stSlider > div > div {{
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
    }}
    
    .stSlider > div > div > div > div {{
        border-color: #667eea !important;
        box-shadow: 0 0 8px rgba(102,126,234,0.6);
    }}
    """}
    
    /* Checkbox */
    .stCheckbox > label {{
        color: {text_color} !important;
    }}
    
    /* Text Colors for Dark Mode */
    {"h1, h2, h3, h4, h5, h6 { color: " + text_color + " !important; }" if st.session_state.dark else ""}
    {"p, span, label, .stMarkdown { color: " + sub_text + " !important; }" if st.session_state.dark else ""}
    
    /* Divider */
    hr {{
        border: none;
        height: 2px;
        background: {primary_gradient};
        border-radius: 1px;
        margin: 1.5rem 0;
    }}
    
    /* Section Headers */
    .section-header {{
        color: {text_color};
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    /* Loading Spinner */
    .loading-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
        background: {card_bg};
        border-radius: 1.5rem;
        border: 2px solid {border_color};
        box-shadow: {shadow};
    }}
    
    .gear-spinner {{
        width: 60px;
        height: 60px;
        border: 4px solid transparent;
        border-top: 4px solid {accent};
        border-right: 4px solid #6633CC;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }}
    
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    .loading-text {{
        margin-top: 1.5rem;
        color: {text_color};
        font-weight: 600;
        font-size: 1.1rem;
    }}
    
    .loading-sub {{
        color: {sub_text};
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }}
    
    /* Video Container */
    .video-container {{
        border-radius: 1.5rem;
        overflow: hidden;
        border: 2px solid {border_color};
        box-shadow: {shadow};
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        color: {sub_text};
        padding: 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
    }}
    
    /* Stats Card */
    .stats-item {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 1rem;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }}
    
    .stats-item:hover {{
        transform: translateX(5px);
        border-color: {accent};
    }}
    
    /* ========================================== */
    /* 🧠 BONUS: Slider Container Card           */
    /* ========================================== */
    
    .slider-container {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 1rem;
        padding: 1rem;
        box-shadow: {shadow};
        margin-bottom: 1rem;
    }}
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
    loading_placeholder = st.empty()
    loading_placeholder.markdown(f"""
    <div class="loading-container">
        <div class="gear-spinner"></div>
        <div class="loading-text">⚙️ Initializing AI Model...</div>
        <div class="loading-sub">Please wait while we prepare your experience</div>
    </div>
    """, unsafe_allow_html=True)
    
    model, cascade = load()
    st.session_state.model_ready = True
    loading_placeholder.empty()
else:
    model, cascade = load()

# =========================================================
# MAIN UI
# =========================================================
st.markdown('<div class="main-header"><span class="emoji">🎭</span> BHAAV AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-Time Facial Emotion Recognition powered by Deep Learning</div>', unsafe_allow_html=True)

# Model Status Badge at Top
status_text = "✅ MODEL READY" if st.session_state.model_ready else "⏳ LOADING..."
status_class = "" if st.session_state.model_ready else "idle"
st.markdown(f"""
<div class="status-container">
    <div class="status-badge {status_class}">{status_text}</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    # Theme Toggle
    theme_icon = "🌙" if st.session_state.dark else "☀️"
    theme_label = "LIGHT MODE" if st.session_state.dark else "DARK MODE"
    if st.button(f"{theme_icon} {theme_label}", key="theme_btn"):
        st.session_state.dark = not st.session_state.dark
        st.rerun()
    
    st.markdown("---")
    st.markdown("""<div class="section-header">⚙️ SETTINGS</div>""", unsafe_allow_html=True)
    
    # Confidence Slider with Premium Container
    st.markdown(f"""<div class="slider-container">""", unsafe_allow_html=True)
    conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.35, 0.05, key="confidence_slider")
    st.markdown("""</div>""", unsafe_allow_html=True)
    
    # Confidence Meter Brief - Enhanced Version
    st.markdown(f"""
    <div class="confidence-info">
        <h4>📊 About Confidence Meter</h4>
        <p>The confidence threshold filters out uncertain predictions. 
        Higher values (0.7+) show only high-confidence detections, 
        while lower values (0.3-0.5) allow more predictions but may include less accurate results. 
        Recommended: <strong>0.35 - 0.50</strong></p>
    </div>
    <div style="background: linear-gradient(135deg, rgba(255,0,204,0.15) 0%, rgba(102,51,204,0.15) 100%); 
                border: 2px solid {accent}; 
                border-radius: 1rem; 
                padding: 1.25rem; 
                margin: 1rem 0;
                box-shadow: 0 4px 15px rgba(255,0,204,0.2);">
        <h4 style="color: {accent}; margin: 0 0 0.75rem 0; font-size: 1rem; display: flex; align-items: center; gap: 0.5rem;">
            📊 About Confidence Meter
        </h4>
        <p style="color: {sub_text}; font-size: 0.85rem; line-height: 1.6; margin: 0;">
            The <strong style="color: {text_color};">confidence threshold</strong> filters predictions:
        </p>
        <ul style="color: {sub_text}; font-size: 0.8rem; margin: 0.5rem 0 0 0; padding-left: 1.25rem; line-height: 1.8;">
            <li><span style="color: #00FF00; font-weight: 700;">0.7+</span> → High accuracy only</li>
            <li><span style="color: #FFD700; font-weight: 700;">0.4-0.6</span> → Balanced detection</li>
            <li><span style="color: #FF4444; font-weight: 700;">0.2-0.3</span> → More detections, less accurate</li>
        </ul>
        <div style="margin-top: 0.75rem; padding: 0.5rem; background: rgba(0,255,0,0.1); border-radius: 0.5rem; text-align: center;">
            <span style="color: #00FF00; font-weight: 700; font-size: 0.8rem;">✨ Recommended: 0.35 - 0.50</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    show_c = st.checkbox("Show Confidence %", True)
    show_f = st.checkbox("Show FPS Counter", True)
    
    st.markdown("---")
    st.markdown('<div class="section-header">📊 SESSION STATS</div>', unsafe_allow_html=True)
    
    if st.session_state.history:
        cnts = Counter(st.session_state.history)
        st.markdown(f"**Total Detections:** {len(st.session_state.history)}")
        for e, n in cnts.most_common(3):
            p = (n / len(st.session_state.history) * 100)
            st.markdown(f"""
            <div class="stats-item">
                {THEMES[e]['emoji']} <strong>{e}</strong>: {n} ({p:.0f}%)
            </div>
            """, unsafe_allow_html=True)
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
    
    # Initial placeholder
    if not st.session_state.emotion:
        emo_display.markdown(f"""
        <div class="emotion-card">
            <div class="emoji" style="font-size: 3.5rem; opacity: 0.5;">🎭</div>
            <h2 style="color: {sub_text}; font-weight: 700;">Waiting...</h2>
            <p style="color: {sub_text}; font-size: 0.9rem;">Start camera to detect emotions</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Confidence Meter Visual Display (Dynamic)
    st.markdown("---")
    st.markdown('<div class="section-header">📊 CONFIDENCE METER</div>', unsafe_allow_html=True)
    conf_meter_display = st.empty()
    
    # Initial confidence meter state
    conf_percent = int(st.session_state.confidence * 100)
    conf_color = "#00FF00" if conf_percent >= 70 else ("#FFD700" if conf_percent >= 40 else "#FF4444")
    
    conf_meter_display.markdown(f"""
    <div style="background: {card_bg}; border: 2px solid {border_color}; border-radius: 1rem; padding: 1.5rem; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <span style="color: {text_color}; font-weight: 700; font-size: 1rem;">Current Confidence</span>
            <span style="color: {conf_color}; font-weight: 900; font-size: 1.5rem;">{conf_percent}%</span>
        </div>
        <div style="background: #333; border-radius: 10px; height: 12px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, {accent}, {conf_color}); height: 100%; width: {conf_percent}%; border-radius: 10px; transition: width 0.3s ease;"></div>
        </div>
        <div style="margin-top: 1rem; padding: 0.75rem; background: rgba(255,0,204,0.1); border-radius: 0.5rem; border-left: 3px solid {accent};">
            <p style="color: {sub_text}; font-size: 0.8rem; margin: 0; line-height: 1.4;">
                <strong style="color: {accent};">What is this?</strong><br>
                The confidence meter shows how certain the AI is about its emotion prediction. 
                <strong>70%+</strong> = High accuracy, <strong>40-70%</strong> = Moderate, <strong>&lt;40%</strong> = Low certainty.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Control Buttons
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    pass
with c2:
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("▶️ START", type="primary", use_container_width=True):
            st.session_state.running = True
    with btn_col2:
        if st.button("⏹️ STOP", use_container_width=True):
            st.session_state.running = False
with c3:
    pass

# =========================================================
# VIDEO PROCESSING
# =========================================================
if st.session_state.running:
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
            
            # Draw rectangle with theme-appropriate color
            clr = (204, 0, 255) if st.session_state.dark else (102, 126, 234)  # BGR format
            cv2.rectangle(f, (x, y), (x + w, y + h), clr, 3)
            
            lbl = e
            if show_c and e != "Uncertain":
                lbl += f" {st.session_state.confidence:.0%}"
            
            cv2.putText(f, lbl, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2)
        
        if show_f:
            fps_color = (204, 0, 255) if st.session_state.dark else (102, 126, 234)
            cv2.putText(f, f"FPS: {cfps}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        
        vid.image(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        
        if fc % 6 == 0 and st.session_state.emotion:
            e = st.session_state.emotion
            t = THEMES[e]
            
            # Update Emotion Card
            emo_display.markdown(f"""
            <div class="emotion-card">
                <div class="emoji" style="font-size: 4rem;">{t['emoji']}</div>
                <h1 style="background: {primary_gradient}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; font-size: 1.8rem; margin: 0.5rem 0;">{e.upper()}</h1>
                <div class="icon" style="font-size: 1.8rem; margin: 0.5rem 0;">{t['icon']}</div>
                <p style="color: {sub_text}; font-size: 0.95rem; font-weight: 600;">{t['msg']}</p>
                <div style="margin-top: 1rem; padding: 0.5rem 1rem; background: {btn_gradient}; border-radius: 0.75rem; display: inline-block;">
                    <span style="color: #FFF; font-weight: 700; font-size: 0.9rem;">Confidence: {st.session_state.confidence:.0%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Update Confidence Meter (Real-time)
            live_conf_percent = int(st.session_state.confidence * 100)
            live_conf_color = "#00FF00" if live_conf_percent >= 70 else ("#FFD700" if live_conf_percent >= 40 else "#FF4444")
            conf_status = "🟢 HIGH" if live_conf_percent >= 70 else ("🟡 MODERATE" if live_conf_percent >= 40 else "🔴 LOW")
            
            conf_meter_display.markdown(f"""
            <div style="background: {card_bg}; border: 2px solid {border_color}; border-radius: 1rem; padding: 1.5rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <span style="color: {text_color}; font-weight: 700; font-size: 1rem;">Current Confidence</span>
                    <span style="color: {live_conf_color}; font-weight: 900; font-size: 1.5rem;">{live_conf_percent}%</span>
                </div>
                <div style="background: #333; border-radius: 10px; height: 14px; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);">
                    <div style="background: linear-gradient(90deg, {accent}, {live_conf_color}); height: 100%; width: {live_conf_percent}%; border-radius: 10px; transition: width 0.3s ease; box-shadow: 0 0 10px {live_conf_color};"></div>
                </div>
                <div style="text-align: center; margin-top: 0.75rem;">
                    <span style="color: {live_conf_color}; font-weight: 700; font-size: 0.9rem;">{conf_status}</span>
                </div>
                <div style="margin-top: 0.75rem; padding: 0.5rem; background: rgba(255,0,204,0.1); border-radius: 0.5rem; border-left: 3px solid {accent};">
                    <p style="color: {sub_text}; font-size: 0.75rem; margin: 0; line-height: 1.3;">
                        <strong style="color: {accent};">📊 Live Status:</strong> AI is {live_conf_percent}% confident about detecting <strong>{e}</strong>
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    cap.release()

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(f'<div class="footer">Built with 💜 using TensorFlow & Streamlit</div>', unsafe_allow_html=True)