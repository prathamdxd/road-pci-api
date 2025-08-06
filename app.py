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
CORS(app)  # Enable CORS for all routes

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

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Load model
def load_model():
    model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=Config.NUM_CLASSES)
    model.load_state_dict(torch.load('model/best_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400
    
    files = request.files.getlist('images')
    if len(files) == 0:
        return jsonify({"error": "No files uploaded"}), 400
    
    predictions = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            try:
                input_tensor = preprocess_image(filepath)
                with torch.no_grad():
                    output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0).numpy()
                predicted_class = np.argmax(probabilities)
                predictions.append(predicted_class)
                
                os.remove(filepath)
            except Exception as e:
                os.remove(filepath)
                return jsonify({"error": f"Error processing image: {str(e)}"}), 500
    
    if not predictions:
        return jsonify({"error": "No valid images processed"}), 400
    
    class_counts = Counter(predictions)
    majority_class = class_counts.most_common(1)[0][0]
    
    return jsonify({
        "status": "success",
        "prediction": Config.CLASS_NAMES[majority_class],
        "class_counts": class_counts
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
