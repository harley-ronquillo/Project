import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib
import os

def train_model(data_path, model_path, features_path):
    # Load the preprocessed dataset
    data = pd.read_csv(data_path)
    
    # Split the dataset into features and target
    X = data.drop('Flight Status', axis=1)  # Using 'Flight Status' as the target variable
    y = data['Flight Status']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Naive Bayes classifier
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(nb, model_path)
    
    # Save the feature names
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    joblib.dump(X.columns.tolist(), features_path)

if __name__ == '__main__':
    train_model('preprocessed_data.csv', 'models/naive_bayes_model.pkl', 'models/features.pkl')
