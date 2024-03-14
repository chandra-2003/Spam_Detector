# Spam_Detector

import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  # Typo corrected here

ps = PorterStemmer()

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
st.title("Email/SMS Spam Classifier")  # Corrected spelling of "Classifier"
input_sms = st.text_area("Enter the message ")  # Corrected spelling of "text_area"
if st.button('Predict'):  # Changed 'predict' to 'Predict' for consistency and readability
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")  # Corrected capitalization
    else:
        st.header("Not Spam")  # Corrected capitalization
