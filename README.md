# Deepfake Detection System

A machine learning-based system for detecting deepfake images using a pre-trained model.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create the necessary directories:
```bash
mkdir -p model static/css static/js templates uploads
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Click the "Choose File" button to select an image
2. Click "Analyze Image" to process the image
3. View the results showing whether the image is a deepfake and the confidence level

## Project Structure

```
deepfake-detector/
├── app.py                 # Main Flask application
├── model/                 # Pre-trained model directory
├── static/                # Static files
│   ├── css/style.css     # Stylesheet
│   └── js/script.js      # Frontend JavaScript
├── templates/            # HTML templates
│   └── index.html       # Main page
├── uploads/             # Temporary storage for uploaded files
└── requirements.txt     # Python dependencies
```

## Note

This application requires a pre-trained model file (`model.h5`) in the `model/` directory. Make sure to place your trained model file there before running the application. 