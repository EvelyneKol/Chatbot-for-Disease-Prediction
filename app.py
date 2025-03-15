%%writefile app.py
import streamlit as st
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from langchain_ollama import OllamaLLM

# Load the fine-tuned Intent Classifier
classifier_model = AutoModelForSequenceClassification.from_pretrained("evelynkol/distilbert-classifier")
classifier_tokenizer = AutoTokenizer.from_pretrained("evelynkol/distilbert-classifier")

# Set working directory to your Google Drive path
os.chdir('/content/drive/My Drive/thesis/')

# Load the dataset and store the symptoms
data = pd.read_csv('Training.csv')
symptoms = data.drop(columns=['prognosis']).columns.tolist()

def preprocess_symptoms(symptoms):
    return [symptom.replace("_", " ").lower() for symptom in symptoms]

normalized_symptoms = preprocess_symptoms(symptoms)

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load precomputed symptom embeddings
symptom_embeddings = np.load('/content/drive/My Drive/thesis/symptom_embeddings.npy')

# Load the pre-trained MLP ML model
ml_model = joblib.load("/content/drive/My Drive/thesis/knn.joblib")

# Load the LLaMA model using OllamaLLM
biomistral = OllamaLLM(model="cniongolo/biomistral:latest")

# Function to classify intent using the intent classifier
def classify_intent(user_input):
    inputs = classifier_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    outputs = classifier_model(**inputs)
    intent_id = torch.argmax(outputs.logits, dim=1).item()
    return intent_id  # 0 for symptoms, 1 for medical questions, 2 for irrelevant

# Function to match symptoms
def get_top_matched_symptoms(user_input, symptom_embeddings, symptoms, threshold=0.45):
    user_embedding = embedding_model.encode([user_input])
    similarities = cosine_similarity(user_embedding, symptom_embeddings).flatten()
    matches = [symptoms[i] for i, score in enumerate(similarities) if score >= threshold]
    sorted_matches = sorted(matches, key=lambda x: similarities[symptoms.index(x)], reverse=True)
    return sorted_matches

# Function to predict disease
def predict_disease(matched_symptoms):
    input_vector = [1 if symptom in matched_symptoms else 0 for symptom in symptoms]
    input_df = pd.DataFrame([input_vector], columns=symptoms)
    return ml_model.predict(input_df)[0]

# Function to generate disease description
def generate_disease_description(disease_name):
    prompt = f"The patient probably has {disease_name}. Explain briefly what this disease is."
    return biomistral.invoke(prompt)

st.set_page_config(
    page_title="Medical Assistant",  # Title of the browser tab
    page_icon="🩺",  # emoji favicon
)

st.title("Medical Chatbot 💬")
st.write("Enter your symptoms or medical question below.")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display chat input at the bottom
user_input = st.chat_input("Type your message here...")

if user_input:
    if "bye" in user_input.lower():
        bot_response = "Goodbye! Take care."
    else:
        intent = classify_intent(user_input)
        if intent == 0:
            matched_symptoms = get_top_matched_symptoms(user_input, symptom_embeddings, symptoms)
            if matched_symptoms:
                predicted_disease = predict_disease(matched_symptoms)
                description = generate_disease_description(predicted_disease)
                bot_response = f"You may have {predicted_disease}. {description}"
            else:
                bot_response = "No matching symptoms found. Please provide more details."
        elif intent == 1:
            bot_response = biomistral.invoke(user_input)
        else:
            bot_response = "I'm sorry, I can't assist with that. Can you try rephrasing?"

    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    # Add bot response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

    # Display the latest user message
    with st.chat_message("user"):
        st.markdown(user_input)
    # Display the latest bot response
    with st.chat_message("assistant"):
        st.markdown(bot_response)
