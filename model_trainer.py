""" 
Module to build, fine-tune, and evaluate a transformer model for sentiment classification.
Uses Bio_ClinicalBERT for clinical drug review sentiment analysis.
Includes data preparation, training, saving, and prediction utilities.
"""

import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np


""" 
Global model configuration:
- MODEL_NAME: Hugging Face model for clinical text
- num_labels: number of sentiment classes (positive, neutral, negative)
"""
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
num_labels = 3


""" 
Load the pre-trained transformer model and tokenizer.
The model is configured for sequence classification with 3 output labels.
"""
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def encode_labels(labels):
    """ 
    Encodes string sentiment labels into integer values using LabelEncoder.
    Returns the encoded labels and the fitted encoder.
    """
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return labels, le


def prepare_tf_dataset(encodings, labels, batch_size=32):
    """ 
    Prepares a TensorFlow dataset from Hugging Face-style encodings and integer labels.
    Shuffles and batches the dataset for efficient training.
    """
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    dataset = dataset.shuffle(len(labels)).batch(batch_size)
    return dataset


def train_model(train_dataset, val_dataset, epochs=3):
    """ 
    Compiles and trains the model using:
    - Adam optimizer
    - Sparse categorical cross-entropy loss (from logits)
    - Accuracy as the evaluation metric

    Trains the model for the specified number of epochs and returns the trained model.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
    return model


def save_model(model, path="./saved_model"):
    """ 
    Saves the trained model to the specified directory using Hugging Face's save_pretrained method.
    """
    model.save_pretrained(path)


def load_trained_model(path="./saved_model"):
    """ 
    Loads a previously saved transformer model for inference or further training.
    """
    return TFAutoModelForSequenceClassification.from_pretrained(path)


def predict_sentiment(text):
    """ 
    Predicts the sentiment class for a given input text:
    - Tokenizes input using the tokenizer
    - Passes it through the model to get logits
    - Returns the predicted class index (0, 1, or 2)
    """
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    outputs = model(inputs)
    logits = outputs.logits
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    return predicted_class
