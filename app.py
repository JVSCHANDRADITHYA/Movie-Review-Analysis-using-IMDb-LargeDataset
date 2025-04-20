import streamlit as st
import joblib

# Load trained models
clf_model = joblib.load('imdb_sentiment_classifier.pkl')
reg_model = joblib.load('imdb_rating_regressor.pkl')

# Streamlit UI
st.title("🎬 IMDB Movie Review Sentiment & Rating Predictor")
st.markdown("Paste an IMDB-style movie review below and click **Submit** to analyze.")

# Text input
review = st.text_area("Enter your movie review:", height=200)

# Button to submit
if st.button("Submit"):
    if review.strip() == "":
        st.warning("Bruhh enter a review before clicking.")
    else:
        # Predict sentiment (0/1)
        sentiment = clf_model.predict([review])[0]
        sentiment_label = "Positive 😊" if sentiment == 1 else "Negative 😞"

        # Predict rating (regression)
        rating = reg_model.predict([review])[0]
        rating = max(1.0, min(10.0, rating))  # Clip between 1 and 10

        # Output
        st.subheader("🧠 Prediction:")
        st.write(f"**Sentiment:** {sentiment_label}")
        st.write(f"**Estimated Rating:** {rating:.1f} / 10")
