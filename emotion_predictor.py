import tensorflow as tf
import cv2
import numpy as np
import json

def load_model_and_labels():
    """Load model và labels"""
    # Load model
    try:
        model = tf.keras.models.load_model('emotion_recognition_final.keras')
    except:
        model = tf.keras.models.load_model('emotion_recognition_final.h5')
    
    # Load labels
    with open('emotion_labels.json', 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    return model, labels_data

def preprocess_image(image):
    """Tiền xử lý ảnh cho model"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Resize to 48x48
    resized = cv2.resize(gray, (48, 48))
    
    # Normalize
    normalized = resized / 255.0
    
    # Reshape for model
    input_image = normalized.reshape(1, 48, 48, 1)
    
    return input_image

def predict_emotion(model, image, labels_data):
    """Predict cảm xúc từ ảnh"""
    # Preprocess
    processed_image = preprocess_image(image)
    
    # Predict
    predictions = model.predict(processed_image, verbose=0)[0]
    
    # Get results
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]
    
    emotion_en = labels_data['emotion_labels_en'][str(predicted_class)]
    emotion_vn = labels_data['emotion_labels_vn'][str(predicted_class)]
    
    return emotion_en, emotion_vn, confidence, predictions

if __name__ == "__main__":
    # Test function
    model, labels_data = load_model_and_labels()
    print("✅ Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Available emotions: {list(labels_data['emotion_labels_vn'].values())}")
