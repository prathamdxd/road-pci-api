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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
class Config:
    NUM_CLASSES = 5
    IMG_SIZE = 384
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "tf_efficientnetv2_s"
    CLASS_NAMES = {
        0: "1 - Very Poor",
        1: "2 - Poor",
        2: "3 - Fair",
        3: "4 - Good",
        4: "5 - Very Good"
    }
    GOOD_UPPER_THRESHOLD = 0.995
    GOOD_LOWER_THRESHOLD = 0.75
    MIN_PROB = 0.0005
    MIN_DIFFERENCE = 5
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Load model
def load_model():
    model = timm.create_model(
        Config.MODEL_NAME,
        pretrained=False,
        num_classes=Config.NUM_CLASSES
    )
    model.load_state_dict(torch.load(
        'model/best_model.pth',
        map_location=Config.DEVICE
    ))
    model.eval()
    return model.to(Config.DEVICE)

model = load_model()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(Config.DEVICE)

def get_alternative_class(probabilities, main_class):
    eligible = [(i, p) for i, p in enumerate(probabilities)
                if i != main_class and p > Config.MIN_PROB]

    if not eligible:
        return main_class

    return max(eligible, key=lambda x: x[1])[0]

def classify_image(model, image_path):
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()

    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]

    if predicted_class == 3:
        if Config.GOOD_LOWER_THRESHOLD <= confidence < Config.GOOD_UPPER_THRESHOLD:
            alternative_class = get_alternative_class(probabilities, predicted_class)
            if probabilities[alternative_class] > Config.MIN_PROB:
                predicted_class = alternative_class
                confidence = probabilities[predicted_class]

    return predicted_class, confidence, probabilities

def analyze_predictions(predictions):
    class_numbers = [int(pred) for pred in predictions]
    class_counts = Counter(class_numbers)
    most_common = class_counts.most_common(2)

    if len(most_common) == 1:
        majority_class_num = most_common[0][0]
        majority_class_name = Config.CLASS_NAMES[majority_class_num]
        second_majority_class_name = "None"
        count_diff = 0
    else:
        majority_class_num, majority_count = most_common[0]
        second_majority_class_num, second_majority_count = most_common[1]
        majority_class_name = Config.CLASS_NAMES[majority_class_num]
        second_majority_class_name = Config.CLASS_NAMES[second_majority_class_num]
        count_diff = majority_count - second_majority_count

    if count_diff >= Config.MIN_DIFFERENCE:
        final_prediction = second_majority_class_name
    else:
        final_prediction = majority_class_name

    return {
        "final_prediction": final_prediction,
        "majority_class": majority_class_name,
        "second_majority": second_majority_class_name,
        "class_counts": class_counts,
        "count_difference": count_diff
    }

@app.route('/predict', methods=['POST'])
def predict():
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400
    
    files = request.files.getlist('images')
    if len(files) < 1:
        return jsonify({"error": "At least one image is required"}), 400
    
    predictions = []
    all_probabilities = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            try:
                class_idx, confidence, probabilities = classify_image(model, filepath)
                predictions.append(class_idx)
                all_probabilities.append(probabilities.tolist())
                os.remove(filepath)
            except Exception as e:
                os.remove(filepath)
                return jsonify({"error": f"Error processing image: {str(e)}"}), 500
    
    if not predictions:
        return jsonify({"error": "No valid images processed"}), 400
    
    analysis = analyze_predictions(predictions)
    
    return jsonify({
        "status": "success",
        "predictions": predictions,
        "probabilities": all_probabilities,
        "analysis": analysis
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
