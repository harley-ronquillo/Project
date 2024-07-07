import pandas as pd
import numpy as np
import joblib

def predict(data_path, model_path, label_encoders_path, features_path, target_encoder_path):
    # Load the new data
    new_data = pd.read_csv(data_path)
    
    # Drop the 'Flight Status' column if it exists
    if 'Flight Status' in new_data.columns:
        new_data = new_data.drop(columns=['Flight Status'])
    
    # Load the label encoders
    label_encoders = joblib.load(label_encoders_path)
    
    # Encode categorical features in the new data
    for column, le in label_encoders.items():
        if column in new_data.columns:
            # Handle unseen labels by mapping them to -1
            new_data[column] = new_data[column].apply(lambda x: x if x in le.classes_ else -1)
            # Add -1 to the classes_ if not already present
            if -1 not in le.classes_:
                le.classes_ = np.append(le.classes_, -1)
            new_data[column] = le.transform(new_data[column])
    
    # Load the trained model and feature names
    nb = joblib.load(model_path)
    trained_features = joblib.load(features_path)
    
    # Ensure all columns present during training are in the new data
    for column in trained_features:
        if column not in new_data.columns:
            new_data[column] = 0  # Or some appropriate default value
    
    # Retain only the columns that were used in the original model
    new_data = new_data[trained_features]
    
    # Make predictions
    predictions = nb.predict(new_data)
    
    # Load the target encoder to decode the predictions
    target_encoder = joblib.load(target_encoder_path)
    decoded_predictions = target_encoder.inverse_transform(predictions)
    
    return decoded_predictions

if __name__ == '__main__':
    predictions = predict('new_data.csv', 'models/naive_bayes_model.pkl', 'models/label_encoders.pkl', 'models/features.pkl', 'models/target_encoder.pkl')
    print(predictions)
