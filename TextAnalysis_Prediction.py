import streamlit as st
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import re

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Initialize sentiment analysis and keyword extraction pipelines
sentiment_analyzer = pipeline("sentiment-analysis")
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=10)

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.lower()

# Function to extract keywords
def extract_keywords(text):
    preprocessed_text = preprocess_text(text)
    tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_text])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    keywords = [feature_names[i] for i in scores.argsort()[-10:][::-1]]
    return keywords

# Streamlit web app
st.title("Text Analysis and Prediction")

st.write("Enter text in the box below and click 'Run' to analyze the text.")

# Text input
user_input = st.text_area("Enter your text here:")

if st.button("Run"):
    if user_input:
        # Sentiment analysis
        sentiment_result = sentiment_analyzer(user_input)[0]
        sentiment_label = sentiment_result['label']
        sentiment_score = sentiment_result['score']

        # Extract keywords
        keywords = extract_keywords(user_input)

        # Display results
        st.write("### Sentiment Analysis")
        st.write(f"Sentiment: {sentiment_label} (Confidence: {sentiment_score:.2f})")

        st.write("### Extracted Keywords")
        st.write(", ".join(keywords))
    else:
        st.write("Please enter some text for analysis.")
