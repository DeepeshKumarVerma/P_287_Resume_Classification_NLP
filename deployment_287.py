# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:16:49 2023
@author: DELL
"""
import io
import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from streamlit import set_page_config
import textract
import tempfile
import os

# Load the saved XGBoost classifier model
pickle_in = open(r"C:\Users\DELL\Desktop\P_287\xgb_classifier.pkl", "rb")
xgb_model = pickle.load(pickle_in)

# Load the TF-IDF vectorizer if used during training
tfidf_vectorizer = pickle.load(open(r"C:\Users\DELL\Desktop\P_287\tfidf_vectorizer.pkl", "rb"))

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation and non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove stop words
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    
    processed_text = ' '.join(words)
    return processed_text

def vectorize_text(text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)
    
    # Transform the preprocessed text using TF-IDF
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
    
    return text_tfidf

def classify_text(text):
    # Vectorize the text
    text_tfidf = vectorize_text(text)
    
    # Make predictions using the XGBoost model
    prediction = xgb_model.predict(text_tfidf)
    
    return prediction

def main():
    # Set page configuration
    set_page_config(
        page_title="Resume Classification App",
        page_icon="📰",
        layout="wide"
    )

    # Use CSS to set background image
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://ychef.files.bbci.co.uk/976x549/p0218c97.jpg');
            background-size: cover;
            background-repeat: no-repeat;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Resume Classification")

    # Allow user to upload a file
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "doc", "docx"])

    if uploaded_file is not None:

        # Create a temporary directory to store the uploaded file
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)

        # Save the uploaded file to the temporary directory
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Process the temporary file using textract
        text = textract.process(temp_file_path).decode("utf-8")
    
        # Clean up: remove the temporary file
        os.remove(temp_file_path)

        result = classify_text(text)
        
        target_class = int(result[0])

        result_title = {0:'Peoplesoft Resume', 1: 'SQL Developer', 2: 'Workday Resume', 3: 'React Developer'}

        # Assuming result_title[target_class] contains the text you want to display
        text_to_display = f'The predicted category is {result_title[target_class]}'

        # Apply CSS styling to customize the appearance
        styled_text = f'<div style="background-color: rgba(0, 128, 0, 0.4); padding: 10px; margin: 10px 0; border-radius: 5px; font-size: 30px;">\
                 <span style="font-weight: bold; color: white; text-shadow: -1px -1px 0 black, 1px -1px 0 black, -1px 1px 0 black, 1px 1px 0 black;">{text_to_display}</span>\
               </div>'
        st.write(styled_text, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
