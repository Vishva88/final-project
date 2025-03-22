import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

def create_training_data():
    data = {
        'symptoms': [
            # COVID-19 samples (5)
            ['fever', 'cough', 'fatigue', 'loss_of_taste'],
            ['cough', 'fever', 'body_aches', 'loss_of_smell'],
            ['fatigue', 'fever', 'difficulty_breathing', 'loss_of_taste'],
            ['fever', 'headache', 'body_aches', 'loss_of_smell'],
            ['cough', 'fatigue', 'difficulty_breathing', 'loss_of_taste'],
            
            # Common Cold samples (5)
            ['runny_nose', 'sore_throat', 'cough', 'congestion'],
            ['sneezing', 'cough', 'headache', 'fatigue'],
            ['sore_throat', 'congestion', 'mild_fever', 'runny_nose'],
            ['cough', 'congestion', 'sneezing', 'headache'],
            ['runny_nose', 'sore_throat', 'fatigue', 'mild_fever'],
            
            # Flu samples (5)
            ['high_fever', 'body_aches', 'fatigue', 'chills'],
            ['cough', 'sore_throat', 'muscle_pain', 'fever'],
            ['headache', 'fatigue', 'body_aches', 'congestion'],
            ['fever', 'chills', 'muscle_pain', 'headache'],
            ['body_aches', 'fatigue', 'sore_throat', 'cough'],
            
            # Migraine samples (5)
            ['headache', 'nausea', 'sensitivity_to_light', 'dizziness'],
            ['vision_changes', 'headache', 'sensitivity_to_sound', 'nausea'],
            ['throbbing_headache', 'vomiting', 'sensitivity_to_light', 'fatigue'],
            ['headache', 'sensitivity_to_light', 'vision_changes', 'nausea'],
            ['dizziness', 'sensitivity_to_sound', 'headache', 'fatigue'],
            
            # Allergies samples (5)
            ['sneezing', 'itchy_eyes', 'runny_nose', 'congestion'],
            ['watery_eyes', 'itchy_throat', 'cough', 'sneezing'],
            ['congestion', 'itchy_eyes', 'wheezing', 'headache'],
            ['itchy_throat', 'runny_nose', 'watery_eyes', 'sneezing'],
            ['congestion', 'itchy_eyes', 'cough', 'wheezing']
        ],
        'disease': [
            'COVID-19', 'COVID-19', 'COVID-19', 'COVID-19', 'COVID-19',
            'Common Cold', 'Common Cold', 'Common Cold', 'Common Cold', 'Common Cold',
            'Flu', 'Flu', 'Flu', 'Flu', 'Flu',
            'Migraine', 'Migraine', 'Migraine', 'Migraine', 'Migraine',
            'Allergies', 'Allergies', 'Allergies', 'Allergies', 'Allergies'
        ]
    }
    return data

def prepare_data(data):
    # Get all unique symptoms
    all_symptoms = set()
    for symptom_list in data['symptoms']:
        all_symptoms.update(symptom_list)
    all_symptoms = sorted(list(all_symptoms))
    
    # Create feature matrix
    X = np.zeros((len(data['symptoms']), len(all_symptoms)))
    for i, symptom_list in enumerate(data['symptoms']):
        for symptom in symptom_list:
            j = all_symptoms.index(symptom)
            X[i, j] = 1
    
    # Encode disease labels
    le = LabelEncoder()
    y = le.fit_transform(data['disease'])
    
    return X, y, all_symptoms, le

def train_model():
    print("Creating and preparing training data...")
    data = create_training_data()
    X, y, symptoms, label_encoder = prepare_data(data)
    
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'symptom': symptoms,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\nTop 10 Most Important Symptoms:")
    print(feature_importance.head(10))
    
    print("\nSaving model and related files...")
    
    # Save model and symptoms
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'symptoms': symptoms}, f)
    
    # Save label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save symptoms list
    with open('symptoms.pkl', 'wb') as f:
        pickle.dump(symptoms, f)
    
    print("All files saved successfully!")
    return model, symptoms, label_encoder

if __name__ == '__main__':
    train_model()
