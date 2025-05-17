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

    # 1.4 โหลด dataset สำหรับ visualization
    data = pd.read_csv("news.csv")
    data["content"] = data["title"] + " " + data["text"]
    # แผนผัง: REAL → 0 (Real), FAKE → 1 (Fake)
    data["label"] = data["label"].map({"REAL": 0, "FAKE": 1})

    return model, tokenizer, data

model, tokenizer, data = load_model_and_data()

# 2. สร้าง UI
st.title('🧠 Fake/Real News Detector (BERT)')

tabs = ['Predict News', 'Visualize Dataset', 'Predict from CSV']
selected_tab = st.radio('Select Tab:', tabs)

# --- Tab 1: Predict Single News ---
if selected_tab == 'Predict News':
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

            # 0 → Real, 1 → Fake
            label_text = 'Real' if pred == 0 else 'Fake'
            st.subheader('🔍 Prediction Result:')
            st.success(f'📰 This news is: **{label_text}**')
            st.write(f'📊 Confidence: {prob:.2f}')
        else:
            st.warning('Please enter both a title and body text.')

# --- Tab 2: Visualize Dataset ---
elif selected_tab == 'Visualize Dataset':
    st.header('📊 Dataset Visualization')
    # ให้ mapping สอดคล้องกับ above: 0→Real, 1→Fake
    label_map = {0: 'Real', 1: 'Fake'}
    data_vis = data.copy()
    data_vis['label_text'] = data_vis['label'].map(label_map)

    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(x='label_text', data=data_vis, palette='viridis', ax=ax)
    ax.set_title('Distribution of Real vs Fake News')
    ax.set_xlabel('News Type')
    ax.set_ylabel('Number of Articles')
    st.pyplot(fig)

# --- Tab 3: Predict from CSV ---
elif selected_tab == 'Predict from CSV':
    st.header('📂 Predict News from CSV File')
    uploaded_file = st.file_uploader("Upload a CSV with 'title' and 'text' columns", type=['csv'])

    if uploaded_file is not None:
        csv_df = pd.read_csv(uploaded_file)
        if {'title','text'}.issubset(csv_df.columns):
            csv_df['content'] = csv_df['title'] + " " + csv_df['text']
            inputs = tokenizer(
                csv_df['content'].tolist(),
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).tolist()
                probs = torch.softmax(logits, dim=-1).max(dim=1).values.tolist()

            csv_df['prediction'] = preds
            csv_df['confidence'] = probs
            # 0→Real, 1→Fake
            csv_df['prediction_text'] = csv_df['prediction'].map({0:'Real',1:'Fake'})

            st.subheader('📋 Prediction Results:')
            st.write(csv_df[['title','prediction_text','confidence']])

            fig, ax = plt.subplots(figsize=(8,5))
            sns.countplot(x='prediction_text', data=csv_df, palette='viridis', ax=ax)
            ax.set_title('Fake vs Real Prediction in Uploaded CSV')
            ax.set_xlabel('News Type')
            ax.set_ylabel('Number of Articles')
            st.pyplot(fig)
        else:
            st.error("CSV must contain both 'title' and 'text' columns.")