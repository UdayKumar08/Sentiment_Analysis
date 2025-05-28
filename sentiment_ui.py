import streamlit as st
from predictor import predict_sentiment

st.set_page_config(page_title="Drug Review Sentiment Analyzer", layout="centered")

st.title("💊 Healthcare Opinion Sentiment Analyzer")
st.write("Enter a patient drug review below and get the predicted sentiment.")

user_input = st.text_area("Patient Review Text", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        sentiment, confidence = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{sentiment.capitalize()}**")
        st.info(f"Confidence: {confidence}%")