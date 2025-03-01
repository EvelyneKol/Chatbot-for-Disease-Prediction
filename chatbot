from langchain_ollama import OllamaLLM
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import re
from embeddings import get_top_matched_symptoms
import pandas as pd
import numpy as np
from langchain_ollama import OllamaLLM
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
biomistral = OllamaLLM(model="cniongolo/biomistral:latest")  #llama3.2:1b

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

# Function to predict disease based on matched symptoms
def predict_disease(matched_symptoms):
    # Initialize a zero vector with the correct feature size
    input_vector = [0] * len(symptoms)  # 132 features

    # Mark symptoms that exist in the input
    for symptom in matched_symptoms:
        if symptom in symptoms:
            index = symptoms.index(symptom)
            input_vector[index] = 1

    if len(input_vector) > 132:
            input_vector = input_vector[:132]  # truncate if too many features
    elif len(input_vector) < 132:
        raise ValueError(f"Input vector has {len(input_vector)} features but the model expects 132.")

    # Convert input vector to DataFrame with feature names
    input_df = pd.DataFrame([input_vector], columns=symptoms)

    # Make prediction
    predicted_disease = ml_model.predict(input_df)[0]
    
    return predicted_disease

# Function to query LLaMA 3.2 for a descriptive output
def generate_disease_description(disease_name):
    prompt = (
        f"You are a helpful assistant. Provide a friendly response: "
        f"The patient probably has {disease_name}. Explain briefly what this disease is."
    )
    # Generate response from LLaMA 3.2
    response = biomistral.invoke(prompt)
    return response

# Create the chat prompt template
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a knowledgeable medical assistant chatbot. Provide accurate, concise, and helpful answers to medical questions. Avoid unnecessary details. Conversation so far:\n{history}"),
        ("human", "{input}"),
    ]
)

# Initialize conversation memory
memory = ConversationBufferMemory()

# Create the conversation chain
med_question_chain = ConversationChain(
    llm=biomistral,
    prompt=chat_prompt,
    memory=memory
)

# Function to classify intent using the intent classifier
def classify_intent(user_input):
    inputs = classifier_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    outputs = classifier_model(**inputs)

    intent_id = torch.argmax(outputs.logits, dim=1).item()

    return intent_id  # 0 για συμπτωματα , 1 για med_questions, 2 για unrelevant

# Function to handle symptoms input
def handle_symptoms(user_input):
    return f"Processing your symptoms: {user_input}"

# Function to handle general medical questions or unrelated inputs
def handle_llm_response(user_input):
    response = med_question_chain.run(input=user_input)
    return response


def chatbot():
    print("Chatbot: Hello! I'm here to assist you. Type 'bye' to end the chat.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "bye":
            print("Chatbot: Goodbye! Take care.")
            break
    
        intent = classify_intent(user_input)
        if intent == 0:  # Symptoms
            matched_symptoms = get_top_matched_symptoms(user_input, symptom_embeddings, symptoms)
            if matched_symptoms:
                #print(f"Matched Symptoms: {matched_symptoms}")
                try:
                    predicted_disease = predict_disease(matched_symptoms)
                    description = generate_disease_description(predicted_disease)
                    print(f"You may have {predicted_disease}\n{description}")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("No matching symptoms found. Please provide more details.")
        elif intent == 1:  # Medical questions
            response = biomistral.invoke(user_input)
            print(f"Chatbot: {response}")
        else:
            print("Chatbot: I'm sorry, I can't assist with that. Can you try rephrasing?")

# Run the chatbot
chatbot()
