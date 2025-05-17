import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import BertForSequenceClassification
import pickle
import gdown
import os

# 0. กำหนด device1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. โหลด BERT model และ tokenizer จากไฟล์พิกเคิล พร้อม dataset
@st.cache_resource
def load_model_and_data():
    model_path = "bert_fakenews_model_state_dict.pkl"
    model_file_id = "1ypkrVqTVqwbV3JkaWgsv-CYKycWs-lO6"
    if not os.path.exists(model_path):
        gdown.download(f"https://drive.google.com/uc?id={model_file_id}", model_path, quiet=False)
    # 1.1 โหลด state_dict ของโมเดล
    with open("bert_fakenews_model_state_dict.pkl", "rb") as f_mod:
        state_dict = pickle.load(f_mod)

    # 1.2 โหลด tokenizer
    with open("bert_fakenews_tokenizer.pkl", "rb") as f_tok:
        tokenizer = pickle.load(f_tok)

    # 1.3 สร้างโมเดลเปล่าและโหลด state_dict
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, tokenizer

model, tokenizer = load_model_and_data()

# 2. สร้าง UI
st.title('🧠 Fake/Real News Detector (BERT)')

st.header('📝 Predict a News Article')
title_input = st.text_input('Enter the news title:')
text_input  = st.text_area('Enter the news body text:')

if st.button('Predict'):
    if title_input.strip() and text_input.strip():
        content = f"{title_input} {text_input}"
        inputs = tokenizer(
            content,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).item()
            prob = torch.softmax(logits, dim=-1)[0][pred].item()

        label_text = 'Real' if pred == 0 else 'Fake'
        st.subheader('🔍 Prediction Result:')
        st.success(f'📰 This news is: **{label_text}**')
        st.write(f'📊 Confidence: `{prob:.2f}`')
    else:
        st.warning('Please enter both a title and body text.')