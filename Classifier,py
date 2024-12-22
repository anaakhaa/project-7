import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from utils import extract_features  # Import feature extraction from utils.py

def train_classifier(data_path):
    # Load the dataset
    data = pd.read_csv("C:/Users/acer/PycharmProjects/resumemyself/newdata.csv")

    # Extract features (text from resume and job)
    features = data.apply(lambda row: extract_features(row['resume_text'], row['job_text']), axis=1)

    # Vectorize the features (convert text into numerical data)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(features)
    y = data['label']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the model
    model = RandomForestClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    print(classification_report(y_test, y_pred, zero_division=0))

    # Visualize the classification report and confusion matrix
    visualize_classification_report(report)
    plot_confusion_matrix(y_test, y_pred, model.classes_)

    # Save the trained model
    joblib.dump(model, 'models/resume_classifier.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')

def load_trained_model():
    # Load the trained model and vectorizer
    model = joblib.load('models/resume_classifier.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    return model, vectorizer

def predict_eligibility(model, vectorizer, resume_text, job_text):
    # Extract features from input resume and job description
    features = extract_features(resume_text, job_text)

    # Vectorize the features
    features_vectorized = vectorizer.transform([features])

    # Predict eligibility (0 or 1)
    prediction = model.predict(features_vectorized)
    return prediction[0]

def visualize_classification_report(report):
    # Prepare data for visualization
    classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', and 'weighted avg'

    metrics = ['precision', 'recall', 'f1-score']

    for metric in metrics:
        values = [report[cls][metric] for cls in classes]

        plt.figure(figsize=(6, 6))
        plt.pie(values, labels=classes, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
        plt.title(f'{metric.capitalize()} Distribution by Class')
        plt.axis('equal')
        plt.show()

def plot_confusion_matrix(y_test, y_pred, classes):
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    # Path to your dataset (update the path accordingly)
    dataset_path = "C:/Users/acer/PycharmProjects/resumemyself/newdata.csv"

    # Train the classifier
    train_classifier(dataset_path)
