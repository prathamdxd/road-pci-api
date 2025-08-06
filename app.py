import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import Counter
import timm
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    NUM_CLASSES = 5
    CLASS_NAMES = {
        0: "1 - Very Poor",
        1: "2 - Poor",
        2: "3 - Fair",
        3: "4 - Good",
        4: "5 - Very Good"
    }
    MODEL_PATH = 'model/best_model.pth'

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Load model (with error handling)
def load_model():
    try:
        model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=Config.NUM_CLASSES)
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        app.logger.error(f"Model loading failed: {str(e)}")
        raise

try:
    model = load_model()
except Exception as e:
    app.logger.error(f"Failed to initialize model: {str(e)}")
    model = None

@app.route('/')
def home():
    return jsonify({
        "message": "Road PCI Prediction API",
        "status": "running" if model else "model_not_loaded",
        "endpoints": {
            "health": "/health (GET)",
            "predict": "/predict (POST)"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 503
        
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400
    
    files = request.files.getlist('images')
    if len(files) == 0:
        return jsonify({"error": "No files uploaded"}), 400
    
    try:
        # Process first image (for demo)
        file = files[0]
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Simulate processing (replace with actual model prediction)
        result = {
            "status": "success",
            "prediction": "4 - Good",
            "confidence": 0.85,
            "analysis": {
                "class_counts": {3: 1},
                "final_prediction": "4 - Good"
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy" if model else "unhealthy",
        "model_loaded": bool(model)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
