import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_contract_data(base_path):
    """
    Load contract data from folders. Each folder is treated as a label/class, and
    the text files in the folder are used as training examples.

    :param base_path: Directory path containing folders of contract types.
    :return: texts (list of contract texts), labels (corresponding contract types).
    """
    texts, labels = [], []
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_name.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                        texts.append(text)
                        labels.append(folder)  # Folder name as label
    return texts, labels


def train_and_save_model(base_path='data/CONTRACT_TYPES'):
    """
    Train a document classification model using Logistic Regression and save the model,
    vectorizer, and vocabulary for later use.

    :param base_path: Directory path containing folders of contract types.
    """
    logging.info("Loading contract data...")
    texts, labels = load_contract_data(base_path)

    if not texts:
        logging.error("No data found for training.")
        return

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # TF-IDF Vectorizer for text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    logging.info("Training the Logistic Regression model...")
    model.fit(X_train_tfidf, y_train)

    # Create model directory
    model_dir = 'backend/model'
    os.makedirs(model_dir, exist_ok=True)

    # Save model and vectorizer
    model_path = os.path.join(model_dir, 'document_classifier.joblib')
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')
    dump(model, model_path)
    dump(vectorizer, vectorizer_path)
    logging.info(f"Model saved to {model_path} and vectorizer saved to {vectorizer_path}")

    # **Save vocabulary (feature names) for later interpretation**
    vocab_path = os.path.join(model_dir, 'tfidf_vocab.joblib')
    dump(vectorizer.get_feature_names_out(), vocab_path)
    logging.info(f"Vocabulary saved to {vocab_path}")

    # Evaluate the model on the test set
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    # Calculate accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    logging.info(f"Model accuracy: {accuracy}")
    logging.info(f"Classification report:\n{report}")

    # Print results to console
    print(f"Model training complete.\nAccuracy: {accuracy}\n")
    print(f"Classification Report:\n{report}\n")


if __name__ == "__main__":
    # Call the function to train and save the model
    train_and_save_model()
