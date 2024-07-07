import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def evaluate_model(data_path, model_path):
    # Load the preprocessed dataset
    data = pd.read_csv(data_path)
    
    # Split the dataset into features and target
    X = data.drop('Flight Status', axis=1)  # Using 'Flight Status' as the target variable
    y = data['Flight Status']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load the trained model
    nb = joblib.load(model_path)
    
    # Make predictions on the testing set
    y_pred = nb.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)

if __name__ == '__main__':
    evaluate_model('preprocessed_data.csv', 'models/naive_bayes_model.pkl')
