# Import required libraries
from flask import Flask, request, jsonify, render_template
import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations
from PIL import Image  # Pillow for image handling
import io  # For handling file I/O
import tensorflow as tf  # TensorFlow for deep learning
from mtcnn import MTCNN  # MTCNN for face detection
import os  # For file operations

# Initialize Flask application
app = Flask(__name__)

# Initialize MTCNN face detector
detector = MTCNN()

# Load the pre-trained deepfake detection model
model = tf.keras.models.load_model('model/model.h5')

def preprocess_image(image):
    """
    Preprocess the input image for deepfake detection.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        
    Returns:
        numpy.ndarray: Preprocessed face image or None if no face is detected
    """
    # Convert to RGB if needed
    if len(image.shape) == 2:  # If grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # If RGBA
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    
    # Detect face using MTCNN
    faces = detector.detect_faces(image)
    if not faces:
        return None
    
    # Get the largest face (assuming it's the main subject)
    face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = face['box']
    
    # Extract face region
    face_img = image[y:y+h, x:x+w]
    
    # Resize to model input size (224x224)
    face_img = cv2.resize(face_img, (224, 224))
    
    # Normalize pixel values to [0, 1]
    face_img = face_img.astype('float32') / 255.0
    
    return face_img

@app.route('/')
def home():
    """Render the main page with the upload form."""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    """
    Handle image upload and deepfake detection.
    
    Returns:
        JSON response with detection results or error message
    """
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and decode the uploaded image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return jsonify({'error': 'No face detected'}), 400
        
        # Make prediction using the model
        prediction = model.predict(np.expand_dims(processed_image, axis=0))[0][0]
        
        # Return results
        result = {
            'is_deepfake': bool(prediction > 0.5),  # Threshold at 0.5
            'confidence': float(prediction)  # Raw prediction score
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask application in debug mode
    app.run(debug=True) 