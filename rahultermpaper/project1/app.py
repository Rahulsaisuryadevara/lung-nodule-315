from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store prediction history in memory (in production, use a database)
prediction_history = []

# Define the same CNN model architecture as in training
class LungCNN(nn.Module):
    def __init__(self):
        super(LungCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 18 * 18, 128)
        self.fc2 = nn.Linear(128, 3)  # 3 classes
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 18 * 18)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model
MODEL_PATH = 'lung_cnn_model.pth'
CLASS_LABELS_PATH = 'class_labels.txt'

# Initialize model
model = LungCNN().to(device)

# Load class labels
CLASS_NAMES = []
try:
    if os.path.exists(CLASS_LABELS_PATH):
        with open(CLASS_LABELS_PATH, 'r') as f:
            for line in f:
                idx, class_name = line.strip().split(':')
                CLASS_NAMES.append(class_name)
        print(f"Loaded class names: {CLASS_NAMES}")
    else:
        # Default class names if file doesn't exist
        CLASS_NAMES = ['Benign', 'Malignant', 'Normal']
        print(f"Using default class names: {CLASS_NAMES}")
except Exception as e:
    print(f"Error loading class labels: {e}")
    CLASS_NAMES = ['Benign', 'Malignant', 'Normal']

# Load model weights
try:
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()  # Set to evaluation mode
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"WARNING: Model file '{MODEL_PATH}' not found!")
        print("Please run train_model.py first to create the model.")
        print("Using untrained model for demonstration.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Using untrained model for demonstration.")

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    """
    Preprocess the image for model prediction
    Uses the same transformations as training
    """
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(image_path):
    """
    Make prediction on the image using the trained PyTorch model
    """
    try:
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            return None
        processed_image = processed_image.to(device)
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = F.softmax(outputs, dim=1)
            probabilities = probabilities.cpu().numpy()[0]
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidences = probabilities.tolist()
        return {
            'prediction': predicted_class,
            'labels': CLASS_NAMES,
            'confidences': confidences
        }
    except Exception as e:
        print(f"Error making prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', history=prediction_history[-10:], now=datetime.now())

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        print("=" * 50)
        print("PREDICTION REQUEST RECEIVED")
        print("=" * 50)
        if 'file' not in request.files:
            print("ERROR: No file in request")
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        print(f"File received: {file.filename}")
        if file.filename == '':
            print("ERROR: Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            print(f"ERROR: Invalid file type - {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload JPEG or PNG'}), 400
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        print(f"Secure filename: {filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"File saved to: {filepath}")
        print("Making prediction...")
        result = predict_image(filepath)
        if result is None:
            print("ERROR: Prediction returned None")
            return jsonify({'error': 'Failed to process image'}), 500
        print(f"Prediction result: {result}")
        result['img_url'] = f"/static/uploads/{filename}"
        prediction_history.append({
            'prediction': result['prediction'],
            'confidence': f"{max(result['confidences']) * 100:.2f}",
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'filename': filename
        })
        print("Returning result to client")
        print(f"Result: {result}")
        print("=" * 50)
        return jsonify(result)
    except Exception as e:
        print(f"ERROR in predict endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/history')
def history():
    """Return prediction history"""
    return jsonify(prediction_history[-20:])

@app.route('/test')
def test():
    """Test endpoint to verify server is working"""
    model_exists = os.path.exists(MODEL_PATH)
    return jsonify({
        'status': 'ok',
        'message': 'Server is running!',
        'device': str(device),
        'model_loaded': model_exists,
        'model_path': MODEL_PATH,
        'class_names': CLASS_NAMES,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER'])
    })

@app.route('/model-info')
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        'model_architecture': str(model),
        'device': str(device),
        'class_names': CLASS_NAMES,
        'num_classes': len(CLASS_NAMES),
        'model_file_exists': os.path.exists(MODEL_PATH)
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 10MB'}), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print("\n" + "!" * 60)
        print("WARNING: Model file 'lung_cnn_model.pth' not found!")
        print("Please run 'python train_model.py' first to train the model.")
        print("!" * 60 + "\n")
    print("\nStarting Flask server...")
    print(f"Model: {MODEL_PATH}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Device: {device}")
    print("\nServer running at: http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
