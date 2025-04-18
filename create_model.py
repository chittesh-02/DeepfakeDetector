# Import required libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import os

def create_test_model():
    """
    Create a simple CNN model for deepfake detection.
    
    The model architecture consists of:
    - Three convolutional layers with ReLU activation
    - Max pooling layers for downsampling
    - Fully connected layers
    - Sigmoid activation for binary classification
    
    Returns:
        tf.keras.Model: Compiled CNN model
    """
    # Create a sequential model
    model = models.Sequential([
        # First convolutional layer with 32 filters
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        # Max pooling to reduce spatial dimensions
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional layer with 64 filters
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional layer with 64 filters
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten the 3D output to 1D for dense layers
        layers.Flatten(),
        
        # Dense layer with 64 neurons
        layers.Dense(64, activation='relu'),
        
        # Output layer with sigmoid activation for binary classification
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model with appropriate loss and metrics
    model.compile(
        optimizer='adam',  # Adam optimizer for training
        loss='binary_crossentropy',  # Binary cross-entropy loss for classification
        metrics=['accuracy']  # Track accuracy during training
    )
    
    return model

# Create and save the model
if __name__ == "__main__":
    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    
    # Create and save the model
    model = create_test_model()
    model.save('model/model.h5')
    print("Test model created and saved as 'model/model.h5'") 