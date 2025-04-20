import streamlit as st
import joblib
from PIL import Image
import base64

# Load models
clf_model = joblib.load('imdb_sentiment_classifier.pkl')
reg_model = joblib.load('imdb_rating_regressor.pkl')

# For history
if 'history' not in st.session_state:
    st.session_state.history = []

# Background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("bg.jpg")

# IMDb Logo
st.image("imdb_logo.png", width=150)

# Title
st.markdown("<h1 style='color:gold;text-shadow:1px 1px 3px black;'>🎬 IMDB Review Analyzer</h1>", unsafe_allow_html=True)
st.markdown("Paste a movie review and find out what the model thinks!")

# Text input
review = st.text_area("Enter your review", height=200)

# Submit button (Styled)
custom_btn = """
    <style>
    div.stButton > button:first-child {
        background-color: #FFD700;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 0.6em 1.2em;
        font-size: 18px;
        box-shadow: 2px 2px 5px grey;
    }
    </style>
"""
st.markdown(custom_btn, unsafe_allow_html=True)


# Prediction
if st.button("🎯 Analyze Review"):
    if review.strip() == "":
        st.warning("Enter a review first, bruhh.")
    else:
        sentiment = clf_model.predict([review])[0]
        rating = reg_model.predict([review])[0]
        rating = max(1.0, min(10.0, rating))  # Clamp

        label = "Positive 😊" if sentiment == 1 else "Negative 😞"

        # Save to history
        st.session_state.history.insert(0, {
            "text": review[:100] + "...",
            "sentiment": label,
            "rating": round(rating, 1)
        })

        # Display results
        sentiment_color = "lime" if sentiment == 1 else "red"
        st.markdown(
            f"### Sentiment: <span style='color:{sentiment_color}; font-weight:bold;'>{label}</span>",
            unsafe_allow_html=True
    )   

        # Rating slider
        st.markdown("### Predicted Rating")
        st.markdown("<h3 style='color:#FFD700; font-weight:bold;'>🎯 Predicted Rating</h3>", unsafe_allow_html=True)

        # Disabled slider (just for visual display)
        st.slider(
            label="",
            min_value=1.0,
            max_value=10.0,
            value=rating,
            step=0.1,
            disabled=True
        )


        # Star rating
        full_stars = int(round(rating))
        stars = "⭐" * full_stars + "☆" * (10 - full_stars)
        st.markdown(f"<h2 style='color:#FFD700'>{stars}</h2>", unsafe_allow_html=True)

# History
if st.session_state.history:
    st.markdown("---")
    st.markdown("## 🔄 Previous Predictions")
    for item in st.session_state.history[:5]:
        st.markdown(f"• *{item['text']}* → **{item['sentiment']}**, Rating: {item['rating']}/10")
