import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib 

# Load the fine-tuned Intent Classifier
classifier_model = AutoModelForSequenceClassification.from_pretrained("evelynkol/distilbert-classifier")
classifier_tokenizer = AutoTokenizer.from_pretrained("evelynkol/distilbert-classifier")

# Load the dataset and store the symptoms
data = pd.read_csv('Training.csv')
symptoms = data.drop(columns=['prognosis']).columns.tolist()

def preprocess_symptoms(symptoms):
    return [symptom.replace("_", " ").lower() for symptom in symptoms]

normalized_symptoms = preprocess_symptoms(symptoms)

# initializing the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

symptom_embeddings = embedding_model.encode(normalized_symptoms, convert_to_numpy=True)
#np.save('symptom_embeddings.npy', symptom_embeddings)

# Load precomputed symptom embeddings
symptom_embeddings = np.load('symptom_embeddings.npy')

# loading the pre-trained MLP ML model
ml_model = joblib.load("knn.joblib")  

# Load the LLaMA model using OllamaLLM
biomistral = OllamaLLM(model="cniongolo/biomistral:latest")

# Function to classify intent using the intent classifier
def classify_intent(user_input):
    inputs = classifier_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    outputs = classifier_model(**inputs)

    intent_id = torch.argmax(outputs.logits, dim=1).item()

    return intent_id  # 0 για συμπτωματα , 1 για med_questions, 2 για unrelevant

# Function to match symptoms
def get_top_matched_symptoms(user_input, symptom_embeddings, symptoms, threshold=0.45):
    # Compute embeddings for the user input
    user_embedding = embedding_model.encode([user_input])
    
    # Compute similarity scores
    similarities = cosine_similarity(user_embedding, symptom_embeddings).flatten()
    
    # Filter matches based on the threshold
    matches = [symptoms[i] for i, score in enumerate(similarities) if score >= threshold]
    
    # Return matches sorted by similarity score (ignoring the scores)
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

# Streamlit UI setup
st.markdown(
    """
    <style>
    .chat-container {
        max-width: 600px;
        margin: auto;
        padding: 10px;
        background-color: #e5ddd5;
        border-radius: 10px;
    }
    .chat-bubble {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
        max-width: 75%;
    }
    .user {
        background-color: #dcf8c6;
        align-self: flex-end;
    }
    .bot {
        background-color: #ffffff;
        align-self: flex-start;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Medical Chatbot 💬")
st.write("Enter your symptoms or medical question below.")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", "")

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
    
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Chatbot", bot_response))

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for role, text in st.session_state.chat_history:
    bubble_class = "user" if role == "You" else "bot"
    st.markdown(f'<div class="chat-bubble {bubble_class}"><b>{role}:</b> {text}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
