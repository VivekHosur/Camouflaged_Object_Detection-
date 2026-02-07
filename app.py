import streamlit as st
import cv2
import os
import numpy as np
import base64
import tempfile

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Camouflaged Object Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------ BACKGROUND + ANIMATIONS ------------------
def add_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>

        /* ===== KEYFRAMES ===== */
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        @keyframes slideUp {{
            from {{ transform: translateY(50px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}

        @keyframes glowWave {{
            0% {{ text-shadow: 0 0 10px rgba(120,255,200,0.2); }}
            50% {{ text-shadow: 0 0 25px rgba(120,255,200,0.8); }}
            100% {{ text-shadow: 0 0 10px rgba(120,255,200,0.2); }}
        }}

        @keyframes floatBG {{
            0% {{ background-position: center top; }}
            100% {{ background-position: center bottom; }}
        }}

        @keyframes breathe {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.02); }}
            100% {{ transform: scale(1); }}
        }}

        @keyframes wipe {{
            from {{ clip-path: inset(0 100% 0 0); }}
            to {{ clip-path: inset(0 0 0 0); }}
        }}

        @keyframes shimmer {{
            0% {{ box-shadow: 0 0 10px rgba(255,0,0,0.2); }}
            50% {{ box-shadow: 0 0 35px rgba(255,0,0,0.9); }}
            100% {{ box-shadow: 0 0 10px rgba(255,0,0,0.2); }}
        }}

        /* ===== GLOBAL ===== */
        .stApp {{
            animation: fadeIn 1.2s ease-in-out;
            background:
                linear-gradient(rgba(10,35,25,0.9), rgba(10,35,25,0.9)),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            animation: floatBG 15s ease-in-out infinite alternate;
        }}

        .glass {{
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(18px);
            border-radius: 26px;
            padding: 30px;
            box-shadow: 0 30px 60px rgba(0,0,0,0.45);
        }}

        .hero {{
            animation: slideUp 1s ease-out;
            text-align: center;
            margin-bottom: 45px;
        }}

        .hero h1 {{
            font-size: 54px;
            color: #eafff5;
            animation: glowWave 3s infinite;
        }}

        .hero p {{
            font-size: 20px;
            color: #c9f7e1;
        }}

        .upload {{
            animation: slideUp 1.1s ease-out;
            margin-bottom: 35px;
        }}

        .card {{
            animation: breathe 4s ease-in-out infinite;
            transition: transform 0.4s ease;
        }}

        .card:hover {{
            transform: scale(1.06);
        }}

        .wipe {{
            animation: wipe 1.2s ease-out;
        }}

        .shimmer {{
            animation: shimmer 2.5s infinite;
        }}

        h2, h3, label {{
            color: #eafff5 !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

add_bg("bg.jpg")

# ------------------ HERO ------------------
st.markdown("""
<div class="glass hero">
    <h1>🦎 Camouflaged Object Detection</h1>
    <p>Enhanced animated boundary visualization</p>
</div>
""", unsafe_allow_html=True)

# ------------------ EDGE FOLDER ------------------
EDGE_DIR = "edges"
if not os.path.exists(EDGE_DIR):
    st.error("❌ 'edges' folder not found")
    st.stop()

# ------------------ UPLOAD ------------------
st.markdown("<div class='glass upload'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "📂 Upload a Camouflaged Image",
    type=["jpg", "png", "jpeg"]
)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is None:
    st.stop()

# ------------------ SAVE TEMP ------------------
with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    tmp.write(uploaded_file.read())
    img_path = tmp.name

img_name = uploaded_file.name
edge_path = os.path.join("edges", os.path.splitext(img_name)[0] + ".png")

img = cv2.imread(img_path)
edge = cv2.imread(edge_path, 0)

img = cv2.resize(img, (256, 256))
edge = cv2.resize(edge, (256, 256))

kernel = np.ones((3, 3), np.uint8)
edge = cv2.dilate(edge, kernel, iterations=1)

overlay = img.copy()
overlay[edge > 0] = [0, 0, 255]

# ------------------ DISPLAY ------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='glass card wipe'>", unsafe_allow_html=True)
    st.subheader("📷 Original Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass card wipe shimmer'>", unsafe_allow_html=True)
    st.subheader("🔴 Detected Boundary")
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    st.markdown("</div>", unsafe_allow_html=True)

st.success("✅ Visualization completed")

