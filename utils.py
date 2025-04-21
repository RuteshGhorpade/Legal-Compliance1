from transformers import pipeline
import os
import logging
from pdf2image import convert_from_path
import pytesseract
from joblib import load
import numpy as np
from multiprocessing import Pool

# Initialize the summarization model
try:
    logging.info("Loading summarization model...")
    summarizer = pipeline("summarization", model="t5-small")  # Smaller and faster model
    logging.info("Summarization model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading summarization model: {str(e)}")

# Load models globally for faster processing
model_dir = os.path.join(os.getcwd(), 'backend', 'model')
classifier = load(os.path.join(model_dir, 'document_classifier.joblib'))
vectorizer = load(os.path.join(model_dir, 'tfidf_vectorizer.joblib'))


def ocr_page(image):
    return pytesseract.image_to_string(image)


def extract_text_from_pdf(pdf_path):
    """Extracts text from each page of a PDF file using OCR."""
    try:
        logging.info(f"Starting OCR for PDF: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=100)  # Reduce DPI for faster processing
        with Pool() as pool:
            text_pages = pool.map(ocr_page, images)
        return '\n'.join(text_pages)
    except Exception as e:
        logging.error(f"Error during PDF to image conversion or OCR for PDF '{pdf_path}': {str(e)}")
        return ""


def split_text(text, max_words=500):
    """Splits a long text into chunks of `max_words`."""
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def generate_summary(text):
    """Generates a summary for the given text."""
    try:
        logging.info("Generating summary for the extracted text.")

        # Check if the text is too short for summarization
        if len(text.split()) < 50:
            logging.info("Text too short for summarization.")
            return "The extracted text is too short for a meaningful summary."

        # Split the text if it's too long for the summarizer
        if len(text.split()) > 500:
            chunks = split_text(text, max_words=500)
            logging.info(f"Text split into {len(chunks)} chunks for summarization.")
        else:
            chunks = [text]

        # Generate summary for each chunk and combine them
        summary = ''
        for chunk in chunks:
            summary_chunk = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
            summary += summary_chunk + " "

        logging.info(f"Generated summary: {summary}")
        return summary.strip()

    except Exception as e:
        logging.error(f"Error generating summary: {str(e)}")
        return "Could not generate summary."


def classify_document(pdf_path):
    """Classifies a document and generates its summary."""
    try:
        logging.info(f"Classifying document: {pdf_path}")

        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            logging.warning(f"No text extracted from PDF: {pdf_path}")
            return "Insufficient text extracted", 0, [], "No summary available."

        # Transform text into features using TF-IDF
        X_new = vectorizer.transform([text])

        # Predict the document type
        predicted_label = classifier.predict(X_new)[0]
        match_percentage = classifier.predict_proba(X_new)[0].max() * 100

        # Get more keywords influencing the classification (increased to 20)
        top_n = 20
        feature_array = np.array(vectorizer.get_feature_names_out())
        tfidf_scores = X_new.toarray().flatten()
        top_indices = np.argsort(tfidf_scores)[::-1][:top_n]
        top_keywords = feature_array[top_indices].tolist()

        # Generate AI-based summary of the text
        summary = generate_summary(text)

        logging.info(f"Prediction: {predicted_label}, Match: {match_percentage:.2f}%")
        logging.info(f"Top keywords: {top_keywords}")
        logging.info(f"Summary: {summary}")

        return predicted_label, match_percentage, top_keywords, summary

    except Exception as e:
        logging.error(f"Error classifying document '{pdf_path}': {str(e)}")
        return "Unknown", 0, [], "Could not generate summary."
