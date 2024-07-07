import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess_data(input_path, output_path, sample_size=10000):
    # Load the dataset
    data = pd.read_csv(input_path)
    
    # Select a sample from the dataset
    data = data.sample(n=sample_size, random_state=42)
    
    # Drop missing values
    data = data.dropna()
    
    # Encode categorical features
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        if column != 'Flight Status':  # Avoid encoding the target variable here
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le
    
    # Encode the target column
    target_encoder = LabelEncoder()
    data['Flight Status'] = target_encoder.fit_transform(data['Flight Status'])
    
    # Save the preprocessed data
    data.to_csv(output_path, index=False)
    
    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save the label encoders and target encoder
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(target_encoder, 'models/target_encoder.pkl')
    
    # Save 100 random samples for prediction
    prediction_data = data.sample(n=100, random_state=42)
    prediction_data.to_csv('new_data.csv', index=False)

if __name__ == '__main__':
    preprocess_data('Airline-Dataset-Updated-v2.csv', 'preprocessed_data.csv')
