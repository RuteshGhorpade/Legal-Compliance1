from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
from utils import classify_document
from train import train_and_save_model
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, template_folder='templates')
# os.environ["PATH"] += os.pathsep + r'C:\poppler\Library\bin'
os.environ["PATH"] += os.pathsep + os.getenv('POPPLER_PATH')
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Routes for static files
@app.route('/static/css/<path:filename>')
def send_css(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'static', 'css'), filename)

@app.route('/static/js/<path:filename>')
def send_js(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'static', 'js'), filename)

@app.route('/static/img/<path:filename>')
def send_img(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'static', 'img'), filename)

# Index route
@app.route('/')
def index():
    return render_template('index.html')

# Upload file route
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'pdf' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['pdf']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            predicted_label, match_percentage, keywords, summary = classify_document(file_path)

            return jsonify({
                "document_type": predicted_label,
                "match_percentage": match_percentage,
                "keywords": keywords,
                "summary": summary  # Include the summary in the output
            })

        return jsonify({"error": "Invalid file type"}), 400

    except Exception as e:
        logging.error(f"Error occurred while processing the document: {str(e)}")
        return jsonify({"error": f"Error occurred while processing the document: {str(e)}"}), 500

# Route to trigger model training
@app.route('/train', methods=['POST'])
def train_model_route():
    base_path = 'data/CONTRACT_TYPES'
    try:
        train_and_save_model(base_path)
        return jsonify({"message": "Model trained successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # PORT from environment or fallback 10000 locally
    app.run(host="0.0.0.0", port=port, debug=True)

