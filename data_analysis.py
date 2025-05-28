"""
Exploratory Data Analysis (EDA) and visualization tools for the drug reviews dataset.
Includes plots for rating distribution, sentiment class distribution, and word clouds per sentiment.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from Data_Loader import load_data, clean_text, label_sentiment


def plot_rating_distribution(df):
    """
    Plots the distribution of original numerical review ratings using a histogram.
    Helps understand user rating behavior and data imbalance in rating values.
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(x='rating', data=df)
    plt.title("Distribution of Review Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.show()


def plot_sentiment_distribution(df):
    """
    Plots the count of sentiment labels (positive, neutral, negative).
    Useful for checking label balance after preprocessing and mapping.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x='sentiment', data=df)
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()


def generate_wordcloud(df, sentiment_label):
    """
    Generates and displays a word cloud for a given sentiment class.
    Combines all cleaned reviews for that class into one text blob.

    Args:
        df (DataFrame): DataFrame with cleaned reviews and sentiment labels.
        sentiment_label (str): One of "positive", "neutral", or "negative".
    """
    text = " ".join(df[df['sentiment'] == sentiment_label]['cleaned_review'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for {sentiment_label.capitalize()} Reviews")
    plt.show()


if __name__ == "__main__":
    """ 
    Executes EDA steps when script is run directly:
    - Loads raw data
    - Cleans review text
    - Maps numeric ratings to sentiment classes
    - Visualizes ratings, sentiment distribution, and review content via word clouds
    """
    df = load_data()
    df = df.dropna(subset=['review', 'rating'])
    df['cleaned_review'] = df['review'].apply(clean_text)
    df['sentiment'] = df.apply(label_sentiment, axis=1)

    plot_rating_distribution(df)
    plot_sentiment_distribution(df)
    generate_wordcloud(df, "positive")
    generate_wordcloud(df, "negative")
