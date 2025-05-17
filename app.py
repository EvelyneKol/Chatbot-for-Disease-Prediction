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

# Load the dataset and store the symptoms
data = pd.read_csv('Training.csv')
symptoms = data.drop(columns=['prognosis']).columns.tolist()

def preprocess_symptoms(symptoms):
    return [symptom.replace("_", " ").lower() for symptom in symptoms]

normalized_symptoms = preprocess_symptoms(symptoms)

# Initialize the embedding model
embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# Load precomputed symptom embeddings
symptom_embeddings = np.load('symptom_embeddings_qa.npy')

# Load the pre-trained MLP ML model
ml_model = joblib.load("knn.joblib")

# Load the LLaMA model using OllamaLLM
biomistral = OllamaLLM(model="llama3.2:1b")

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
def predict_top_diseases(matched_symptoms, top_n=3):
    """
    Returns a list of (disease, probability) pairs.
    Any disease whose probability is exactly 0.0 is skipped.
    """
    #  Build one-hot vector for the symptoms the user mentioned
    x = [1 if s in matched_symptoms else 0 for s in symptoms]
    x_df = pd.DataFrame([x], columns=symptoms)

    #  Get probabilities from the trained KNN
    probs = ml_model.predict_proba(x_df)[0]          # shape = (n_classes,)

    #  Sort class indices by descending probability
    sorted_idx = np.argsort(probs)[::-1]

    #  Keep only non-zero probabilities, then take the first top_n
    top_pairs = [
        (ml_model.classes_[i], probs[i])
        for i in sorted_idx
        if probs[i] > 0
    ][:top_n]

    return top_pairs



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
    # Add user input to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    if "bye" in user_input.lower():
        bot_response = "Goodbye! Take care."
    else:
        intent = classify_intent(user_input)

        # labels
        intent_labels = {0: "Symptom description", 1: "Medical question", 2: "Unrelated"}

        if intent == 0:
            matched_symptoms = get_top_matched_symptoms(user_input,
                                                        symptom_embeddings, symptoms)

            if matched_symptoms:
                top3 = predict_top_diseases(matched_symptoms, top_n=3)

                # Nice chat formatting
                bullet_list = "\n".join(
                    f"• **{d}** — {p:.0%} likelihood" for d, p in top3
                )

                short_desc  = generate_disease_description(top3[0][0])

                bot_response = (
                    "Based on what you told me, the most likely conditions are:\n"
                    f"{bullet_list}\n\n{short_desc}\n\n"
                    "*Always consult a healthcare professional for a formal diagnosis.*"
                )
            else:
                bot_response = (
                    "I couldn't match enough symptoms. "
                    "Could you describe anything else you're experiencing?"
                )

        elif intent == 1:  # Medical question
            bot_response = biomistral.invoke(user_input)
        else:  # Irrelevant
            bot_response = "I'm not sure how to help with that. Can you rephrase or ask something related to health?"

    # bot response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})


    # Display the latest user message
    with st.chat_message("user"):
        st.markdown(user_input)
    # Display the latest bot response
    with st.chat_message("assistant"):
        st.markdown(bot_response)
