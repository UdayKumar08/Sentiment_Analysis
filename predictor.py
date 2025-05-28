"""
Script to perform sentiment prediction on clinical drug review text using a fine-tuned Bio_ClinicalBERT model.
It loads a saved model, tokenizes new input, and returns predicted sentiment with confidence score.
"""

import tensorflow as tf
from transformers import TFBertForSequenceClassification, AutoTokenizer
import numpy as np
import logging

""" 
Suppress transformers library warnings to keep output clean.
"""
logging.getLogger("transformers").setLevel(logging.ERROR)


""" 
Configuration:
- MODEL_PATH: path to the saved transformer model
- MODEL_NAME: Hugging Face model used for tokenization
- LABELS: list of sentiment classes in correct order
"""
MODEL_PATH = "./saved_model"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
LABELS = ["negative", "neutral", "positive"]


""" 
Load the trained model from disk and initialize tokenizer for prediction.
"""
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def predict_sentiment(text):
    """
    Takes raw input text and returns:
    - Predicted sentiment label (negative, neutral, positive)
    - Confidence score as a percentage

    Steps:
    - Tokenizes input
    - Passes through the trained model
    - Applies softmax to get probabilities
    - Extracts the label with the highest probability
    """
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    outputs = model(inputs)
    logits = outputs.logits
    probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
    predicted_label = np.argmax(probabilities)
    sentiment = LABELS[predicted_label]
    confidence = round(probabilities[predicted_label] * 100, 2)
    return sentiment, confidence


""" 
Run this module as a script to test on an example input.
Prints the predicted sentiment and its confidence score.
"""
if __name__ == "__main__":
    example = "This medication worked really well for my condition and I had no side effects."
    sentiment, confidence = predict_sentiment(example)
    print(f"Predicted Sentiment: {sentiment} ({confidence}% confidence)")
