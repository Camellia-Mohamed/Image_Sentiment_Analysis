import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
import re
import joblib

# Set page configuration
st.set_page_config(
    page_title="OCR Chat Analyzer",
    page_icon="📷",
    layout="wide"
)

# Google Fonts and Custom CSS for a modern, attractive look
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif !important;
        min-height: 100vh;
        background: linear-gradient(135deg, #23243a 0%, #3a2e5f 100%) !important;
        color: #f3f6fa;
        position: relative;
        overflow-x: hidden;
    }
    body::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        z-index: -1;
        background: linear-gradient(120deg, rgba(127,90,240,0.08) 0%, rgba(0,198,251,0.08) 100%);
        animation: gradientMove 8s ease-in-out infinite alternate;
    }
    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    .stApp {
        max-width: 900px;
        margin: 0 auto;
        min-height: 90vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .center-content {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 80vh;
    }
    .upload-section, .result-section, .chat-section {
        background: rgba(34, 36, 58, 0.95);
        border-radius: 18px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin-bottom: 2rem;
        padding: 2rem 2.5rem;
        border: 1.5px solid rgba(60, 60, 120, 0.25);
    }
    .upload-section {
        border-left: 6px solid #7f5af0;
    }
    .result-section {
        border-left: 6px solid #00c6fb;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700;
        color: #a084f3;
        letter-spacing: 1px;
    }
    .main-header {
        font-size: 3.2rem;
        margin-bottom: 0.5rem;
        color: #a084f3;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        text-shadow: 0 4px 24px rgba(127,90,240,0.15);
    }
    .main-tagline {
        font-size: 1.35rem;
        color: #f3f6fa;
        text-align: center;
        margin-bottom: 2.5rem;
        opacity: 0.85;
    }
    .section-title {
        display: flex;
        align-items: center;
        font-size: 1.5rem;
        margin-bottom: 1.2rem;
        color: #00c6fb;
        letter-spacing: 1px;
    }
    .section-title span {
        margin-right: 0.7rem;
        font-size: 2rem;
    }
    .sentiment-animated {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.2rem;
        font-weight: bold;
        color: #fff;
        background: linear-gradient(90deg, #7f5af0 0%, #00c6fb 100%);
        border-radius: 12px;
        margin: 1.2rem 0 1.5rem 0;
        padding: 1.5rem 0;
        box-shadow: 0 4px 24px 0 rgba(127,90,240,0.15);
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(127,90,240,0.3); }
        70% { box-shadow: 0 0 0 16px rgba(127,90,240,0); }
        100% { box-shadow: 0 0 0 0 rgba(127,90,240,0); }
    }
    .extracted-text {
        background: #23243a;
        color: #f3f6fa;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
        border: 1px solid #393a5a;
        font-family: 'Fira Mono', 'Consolas', monospace;
    }
    .stTextArea textarea {
        background: #23243a !important;
        color: #f3f6fa !important;
        border-radius: 8px !important;
        font-family: 'Fira Mono', 'Consolas', monospace !important;
        font-size: 1.05rem !important;
    }
    .footer {
        width: 100vw;
        text-align: center;
        padding: 1.2rem 0 1.2rem 0;
        color: #bdbdfc;
        font-size: 1.05rem;
        letter-spacing: 0.5px;
        opacity: 0.85;
        position: fixed;
        left: 0; right: 0; bottom: 0;
        background: linear-gradient(90deg, rgba(34,36,58,0.95) 60%, rgba(127,90,240,0.12) 100%);
        z-index: 100;
        border-top: 1px solid #393a5a;
    }
    </style>
    """, unsafe_allow_html=True)

# Set the path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = 'tesseract_ocr/tesseract.exe'
sentiment_model = joblib.load('model/twitter_sentiment_model_LR.joblib')
le_sentiment = joblib.load('model\label_encoder_LR.joblib')
bow = joblib.load('model\\bow.joblib')

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub('[^A-Za-z0-9 ]+', ' ', text)
    return text

def predict_sentiment(text):
    processed_text = preprocess_text(text)
    text_vectorized = bow.transform([processed_text])
    sentiment = sentiment_model.predict(text_vectorized)
    sentiment = le_sentiment.inverse_transform(sentiment)[0]
    return sentiment

def read_text_from_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    image_array = np.array(image)
    image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(filtered, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    scale_percent = 150
    width = int(thresh.shape[1] * scale_percent / 100)
    height = int(thresh.shape[0] * scale_percent / 100)
    resized = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_LINEAR)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(resized, config=custom_config, lang='eng')
    return text

# Streamlit UI
st.markdown("""
    <div class='main-header'>
        <span>📷</span> OCR Chat Analyzer
    </div>
    <div class='main-tagline'>Upload an image to extract text and analyze its sentiment using AI 🤖</div>
""", unsafe_allow_html=True)

# Upload Section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown('<div class="section-title"><span>📤</span>Upload Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span>🖼️</span>Preview</div>', unsafe_allow_html=True)
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    st.markdown('<div class="section-title"><span>🔍</span>Analysis</div>', unsafe_allow_html=True)
    with st.spinner("Processing image..."):
        extracted_text = read_text_from_image(uploaded_file)
    if extracted_text.strip():
        st.success("✅ Text extracted successfully!")
        st.markdown('<div class="section-title"><span>📝</span>Extracted Text</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="extracted-text">{extracted_text}</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span>🎯</span>Sentiment Analysis</div>', unsafe_allow_html=True)
        sentiment = predict_sentiment(extracted_text)
        sentiment_emoji = {
            "Positive": "🤗",
            "Negative": "😞",
            "Neutral": "😊",
            "Irrelevant": "💁‍♀️"
        }
        sentiment_colors = {
            "Positive": "linear-gradient(90deg, #43e97b 0%, #38f9d7 100%)",
            "Negative": "linear-gradient(90deg, #fa709a 0%, #fee140 100%)",
            "Neutral": "linear-gradient(90deg, #a1c4fd 0%, #c2e9fb 100%)",
            "Irrelevant": "linear-gradient(90deg, #f7971e 0%, #ffd200 100%)"
        }
        sentiment_bg = sentiment_colors.get(sentiment, "linear-gradient(90deg, #7f5af0 0%, #00c6fb 100%)")
        st.markdown(f"""
            <div class='sentiment-animated' style='background: {sentiment_bg};'>
                <span>{sentiment_emoji.get(sentiment, '')}</span>
                <span style='margin-left: 1rem;'>{sentiment}</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No readable text found in the image.")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("""
        <div style='text-align: center; padding: 2rem; background: rgba(34,36,58,0.95); border-radius: 18px; border-left: 6px solid #7f5af0;'>
            <h3 style='color: #7f5af0;'>👆 Upload an image to begin analysis</h3>
            <p style='color: #f3f6fa;'>Supported formats: JPG, JPEG, PNG</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class='footer'>
        Made with ❤️ by Habiba Ahmed & Camellia Mohamed &middot; 2025 &middot; <a href='https://github.com/' style='color:#7f5af0;text-decoration:none;' target='_blank'>GitHub</a>
    </div>
""", unsafe_allow_html=True)
    