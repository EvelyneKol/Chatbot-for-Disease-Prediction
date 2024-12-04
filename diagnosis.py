from embeddings import get_top_matched_symptoms
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib  # For loading the ML model

# Load the dataset and store the symptoms
data = pd.read_csv('Dataset.csv')
symptoms = data.columns.tolist()

# initializing the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load precomputed symptom embeddings
symptom_embeddings = np.load('symptom_embeddings.npy')

# loading the pre-trained MLP ML model
ml_model = joblib.load("ml_model.joblib")  

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

# function to predict disease based on matched symptoms
def predict_disease(matched_symptoms):
    
    # initialize a one-hot encoded vector with zeros for all symptoms
    input_vector = [0] * len(symptoms)
    
    # set 1 where the symptom exists in matched symptoms
    for symptom in matched_symptoms:
        if symptom in symptoms:
            index = symptoms.index(symptom)
            input_vector[index] = 1
    
    # ensure the input matches the model's expected feature size (which is 527)
    if len(input_vector) > 527:
        input_vector = input_vector[:527]  # truncate if too many features
    elif len(input_vector) < 527:
        raise ValueError(f"Input vector has {len(input_vector)} features but the model expects 527.")
    
    # give the vector as an input to the ML model
    predicted_disease = ml_model.predict([input_vector])[0]  # get the prediction
    
    return predicted_disease

def main():
    # Ask the user for input
    user_input = input("Please enter your symptoms: ")

    # Get top matched symptoms
    top_matches = get_top_matched_symptoms(user_input, symptom_embeddings, symptoms)

    # Print the matched symptoms
    if top_matches:
        print("Top Matched Symptoms:")
        for symptom in top_matches:
            print(symptom)
        
        # Predict disease using matched symptoms
        try:
            predicted_disease = predict_disease(top_matches)
            print(f"Predicted Disease: {predicted_disease}")
        except ValueError as e:
            print(f"Error: {e}")
    else:
        print("No matching symptoms found.")

# Run the main function
if __name__ == "__main__":
    main()
