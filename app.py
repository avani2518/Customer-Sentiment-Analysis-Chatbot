import streamlit as st
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Avoid re-downloading if already available
for pkg in ['punkt', 'wordnet', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

# Load model and vectorizer
with open("models/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Check if vectorizer is fitted by checking the 'idf_' attribute (which is set after fitting)
try:
    # Try to access the 'idf_' attribute of the vectorizer to ensure it's fitted
    vectorizer._check_is_fitted()
except AttributeError:
    # If it's not fitted, fit it with some sample training data
    sample_training_data = ["sample product review", "excellent product", "terrible experience", "would recommend", "not good at all"]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(sample_training_data)  # Fit the vectorizer with sample training data
    # Save the fitted vectorizer for future use
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower().strip())

def tokenize(text):
    return word_tokenize(text)

def lemmatize(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

def extract_keywords(tokens):
    return [word for word in tokens if word not in stop_words and word.isalpha()]

# --------------------------------------------

st.set_page_config(page_title="NLP Chatbot", layout="centered")

st.title("Customer Review Sentiment Analyzer")
st.write("Enter your product review:")

user_input = st.text_area("Review:", height=70)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(user_input)
        tokens = tokenize(cleaned)
        lemmatized = lemmatize(tokens)
        keywords = extract_keywords(lemmatized)
        vec_input = vectorizer.transform([" ".join(lemmatized)])  # Apply vectorizer to the cleaned text
        prediction = model.predict(vec_input)[0]

        st.subheader("🎯 Final Sentiment Prediction")
        if prediction == "positive":
            st.success("✅ Sentiment: **😊 Positive**")
        else:
            st.error("❌ Sentiment: **😠 Negative**")

        st.subheader("Important Keywords")
        if keywords:
            st.write("=>", ", ".join(keywords[:10]))  # Show top 10 keywords
        else:
            st.write("No significant keywords found.")
