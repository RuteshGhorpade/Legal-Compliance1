from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
from utils import classify_document
from train import train_and_save_model
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup Flask app
app = Flask(__name__, template_folder='templates')

# Poppler path for pdf2image
os.environ["PATH"] += os.pathsep + os.getenv('POPPLER_PATH')

# Upload folder setup
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Serve static files manually
@app.route('/static/css/<path:filename>')
def send_css(filename):
    return send_from_directory(os.path.join('static', 'css'), filename)

@app.route('/static/js/<path:filename>')
def send_js(filename):
    return send_from_directory(os.path.join('static', 'js'), filename)

@app.route('/static/img/<path:filename>')
def send_img(filename):
    return send_from_directory(os.path.join('static', 'img'), filename)

# Health check route (important for Render.com to detect app is live)
@app.route('/health')
def health_check():
    return "OK", 200

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Upload and classify PDF
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'pdf' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['pdf']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            predicted_label, match_percentage, keywords, summary = classify_document(file_path)

            return jsonify({
                "document_type": predicted_label,
                "match_percentage": match_percentage,
                "keywords": keywords,
                "summary": summary
            })

        return jsonify({"error": "Invalid file type. Only PDF allowed."}), 400

    except Exception as e:
        logging.exception("Error occurred while processing upload:")
        return jsonify({"error": str(e)}), 500

# Route to retrain model
@app.route('/train', methods=['POST'])
def train_model_route():
    base_path = 'data/CONTRACT_TYPES'
    try:
        train_and_save_model(base_path)
        return jsonify({"message": "Model trained successfully!"}), 200
    except Exception as e:
        logging.exception("Error occurred during model training:")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # use PORT from environment (Render sets it)
    app.run(host="0.0.0.0", port=port, debug=False)  # debug=False for production
