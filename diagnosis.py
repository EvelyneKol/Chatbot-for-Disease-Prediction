from embeddings import get_top_matched_symptoms
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib  # For loading the ML model

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

# Function to get the top matched symptoms
def get_top_matched_symptoms(user_input, symptom_embeddings, symptoms, threshold=0.45):
    # Compute embeddings for the user input
    user_embedding = embedding_model.encode([user_input])
    
    # Compute similarity scores
    similarities = cosine_similarity(user_embedding, symptom_embeddings).flatten()
    
    # keep the matches based on the threshold
    matches = [symptoms[i] for i, score in enumerate(similarities) if score >= threshold]
    
    # keep the matches sorted by similarity score kai return them
    sorted_matches = sorted(matches, key=lambda x: similarities[symptoms.index(x)], reverse=True)
    return sorted_matches

import pandas as pd

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
