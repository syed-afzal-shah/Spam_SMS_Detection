import streamlit as st
import pickle
import pandas as pd
import re
import string

# === Load all saved files ===

# Load TF-IDF Vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Load Label Encoder
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load Spam Classifier Model
with open('spam_classifier_model.pkl', 'rb') as f:
    spam_classifier_model = pickle.load(f)

# === Data Cleaning Function ===
def data_cleaning(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)                      # remove digits
    text = re.sub(r'https?://\S+|www\.\S+', '', text)    # remove URLs
    text = re.sub(r'\W', ' ', text)                      # remove special characters
    text = re.sub(r'\s+', ' ', text).strip()             # remove extra spaces
    return text

# === Streamlit UI ===
st.title("📩 SMS Spam Classifier")
user_input = st.text_input("Enter your message:")

if user_input:
    # Preprocess and Predict
    cleaned = data_cleaning(user_input)
    transformed = tfidf.transform([cleaned])
    prediction = spam_classifier_model.predict(transformed)
    final_label = le.inverse_transform(prediction)[0]

    # Display Result
    if final_label == 'spam':
        st.error("⚠️ This message is SPAM!")
    else:
        st.success("✅ This message is NOT spam.")
