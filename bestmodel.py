import streamlit as st
import re
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Accuracy, Precision, Recall
import joblib
import tensorflow as tf

# Load necessary files
alay_dict = pd.read_csv('new_kamusalay.csv', names=['original', 'replacement'], encoding='latin-1')
stopword_dict = pd.read_csv('stopwordbahasa.csv', names=['stopword'], encoding='latin-1')

alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))

# Preprocessing functions
def lowercase(text):
    return text.lower()

def remove_unnecessary_char(text):
    text = re.sub(r'\\+n', ' ', text)
    text = re.sub(r'\n', " ", text)
    text = re.sub(r'rt', ' ', text)
    text = re.sub(r'RT', ' ', text)
    text = re.sub(r'user', ' ', text)
    text = re.sub(r'USER', ' ', text)
    text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', text)
    text = re.sub(r':', ' ', text)
    text = re.sub(r';', ' ', text)
    text = re.sub(r'\\+n', ' ', text)
    text = re.sub(r'\n', " ", text)
    text = re.sub(r'\\+', ' ', text)
    text = re.sub(r'  +', ' ', text)
    return text

def remove_nonaplhanumeric(text):
    text = re.sub(r'[^0-9a-zA-Z]+', ' ', text)
    return text

def normalize_alay(text):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

def remove_stopword(text):
    text = ' '.join(['' if word in stopword_dict.stopword.values else word for word in text.split(' ')])
    text = re.sub(r'  +', ' ', text)
    text = text.strip()
    return text

def preprocess(text):
    text = lowercase(text)
    text = remove_nonaplhanumeric(text)
    text = remove_unnecessary_char(text)
    text = normalize_alay(text)
    text = remove_stopword(text)
    return text

@st.cache_resource(ttl=600)
def load_lstm_model():
    return load_model('model.h5', custom_objects={'Accuracy': Accuracy, 'Precision': Precision, 'Recall': Recall})

@st.cache_resource(ttl=600)
def load_tokenizer():
    X_train = pd.read_pickle('X_train.pkl')
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    return tokenizer

@st.cache_resource(ttl=600)
def load_logistic_model():
    return joblib.load('best_model.pkl')

# Streamlit UI
st.title('Aplikasi Klasifikasi Tweet')

input_text = st.text_area('Masukkan tweet untuk diklasifikasikan')

if st.button('Proses'):
    if input_text:
        preprocessed_text = preprocess(input_text)

        lstm_model = load_lstm_model()
        tokenizer = load_tokenizer()
        best_model = load_logistic_model()

        tokenized_text = tokenizer.texts_to_sequences([preprocessed_text])
        padded_text = pad_sequences(tokenized_text, maxlen=100)

        lstm_prediction = lstm_model.predict(padded_text)[0]
        lr_prediction = best_model.predict([preprocessed_text])[0]

        # Interpret LSTM prediction
        lstm_result = "Isi Tweet "
        if lstm_prediction[0] < 0.5 and lstm_prediction[1] < 0.5:
            lstm_result += "bukan hate speech maupun abusive."
        elif lstm_prediction[0] >= 0.5 and lstm_prediction[1] < 0.5:
            lstm_result += "hate speech tetapi tidak abusive."
        elif lstm_prediction[0] < 0.5 and lstm_prediction[1] >= 0.5:
            lstm_result += "bukan hate speech tetapi abusive."
        else:
            lstm_result += "mengandung hate speech dan abusive."
        
        # Interpret Logistic Regression prediction
        lr_result = "Isi Tweet "
        if lr_prediction == 0:
            lr_result += "bukan hate speech maupun abusive."
        elif lr_prediction == 1:
            lr_result += "hate speech tetapi tidak abusive."
        elif lr_prediction == 2:
            lr_result += "bukan hate speech tetapi abusive."
        else:
            lr_result += "mengandung hate speech dan abusive."
        
        st.write("Tweet yang diinputkan: ", input_text)
        st.write("Prediksi Model LSTM: ", lstm_result)
        st.write("Prediksi Model Logistic Regression: ", lr_result)
    else:
        st.write("Masukkan tweet untuk diklasifikasikan")
