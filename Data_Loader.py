import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import load_dataset


""" 
Define a global tokenizer using the Bio_ClinicalBERT model 
This tokenizer will convert clinical review text into token IDs
"""
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_data():
    """
    Loads the drug reviews dataset from Hugging Face (Zakia/drugscom_reviews).
    Returns a pandas DataFrame containing the dataset.
    """
    dataset = load_dataset("Zakia/drugscom_reviews", split="train")
    df = pd.DataFrame(dataset)
    return df


def clean_text(text):
    """
    Cleans review text by:
    - converting to lowercase
    - removing HTML tags
    - removing special characters
    - removing extra whitespace
    """
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def label_sentiment(row):
    """
    Maps numerical ratings to sentiment labels:
    - rating >= 8: positive
    - rating >= 4: neutral
    - rating < 4: negative
    """
    rating = row['rating']
    if rating >= 8:
        return "positive"
    elif rating >= 4:
        return "neutral"
    else:
        return "negative"


def preprocess_data(df):
    """
    Applies preprocessing steps to raw data:
    - drops rows with missing review or rating
    - cleans text
    - adds sentiment labels
    Returns a DataFrame with 'cleaned_review' and 'sentiment' columns.
    """
    df = df.dropna(subset=['review', 'rating'])
    df['cleaned_review'] = df['review'].apply(clean_text)
    df['sentiment'] = df.apply(label_sentiment, axis=1)
    return df[['cleaned_review', 'sentiment']]


def tokenize_data(texts, labels, max_length=128):
    """
    Tokenizes input texts using the specified Hugging Face tokenizer.
    - Applies truncation, padding, and max length control
    Returns a dictionary of tokenized encodings and the corresponding labels.
    """
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_length)
    return encodings, labels


def load_and_prepare():
    """
    Complete data processing pipeline:
    - Load raw dataset
    - Preprocess: clean text and label sentiment
    - Balance classes to avoid bias
    - Split into train and test sets
    - Tokenize using Bio_ClinicalBERT tokenizer
    Returns tokenized train/test encodings and their corresponding labels.
    """
    df = load_data()
    df = preprocess_data(df)

    """ 
    Balancing the dataset:
    Samples 3000 examples from each sentiment class to ensure class balance 
    """
    df = df.groupby('sentiment').sample(n=3000, random_state=42).reset_index(drop=True)

    """ 
    Splitting the dataset:
    Uses stratified split to maintain label distribution in train/test sets 
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_review'], df['sentiment'], test_size=0.2, stratify=df['sentiment']
    )

    """ 
    Tokenizing the train and test sets:
    Converts text to input IDs, attention masks, etc., required for transformer models 
    """
    train_encodings, train_labels = tokenize_data(X_train, y_train)
    test_encodings, test_labels = tokenize_data(X_test, y_test)

    return train_encodings, train_labels, test_encodings, test_labels