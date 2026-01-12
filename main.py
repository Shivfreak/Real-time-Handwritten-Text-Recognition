import numpy as np
import tensorflow as tf
import cv2
import os
import sys
import pickle

# Ensure UTF-8 encoding for console output
sys.stdout.reconfigure(encoding='utf-8')

# Load the trained model
MODEL_PATH = 'models/emnist_cnn_model.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Ensure the model is trained and saved correctly.")

model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded successfully from {MODEL_PATH}")

# Load label mapping from pickle
LABEL_MAP_PATH = 'label_map.pkl'
if os.path.exists(LABEL_MAP_PATH):
    with open(LABEL_MAP_PATH, 'rb') as f:
        label_to_char = pickle.load(f)
    print(f"Label map loaded: {label_to_char}")
else:
    # Create default A-Z mapping if file doesn't exist
    print("Label map not found. Creating default A-Z mapping...")
    label_to_char = {i: chr(65 + i) for i in range(26)}  # 0->A, 1->B, ..., 25->Z
    print(f"Using default mapping: {label_to_char}")

# Create character list for easy access
characters = [label_to_char[i] for i in sorted(label_to_char.keys())]
print(f"Total classes loaded: {len(characters)}")
print(f"Characters: {characters}")

# Function to preprocess a single character image
def preprocess_image(image):
    """
    Preprocess image for model prediction
    - Resize to 32x32
    - Normalize to [0, 1]
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    img_resized = cv2.resize(image, (32, 32))  # Resize to match input size
    img_normalized = img_resized / 255.0       # Normalize to [0, 1]
    return img_normalized

# Function to segment the word image into character images
def segment_word_image(word_img, debug=True):
    """
    Segment a word image into individual character images
    Returns list of character images in left-to-right order
    """
    if debug:
        os.makedirs("debug_chars", exist_ok=True)
    
    # Convert to grayscale
    if len(word_img.shape) == 3:
        gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = word_img.copy()
    
    # Apply threshold to binarize the image
    # For dark text on light background, use THRESH_BINARY_INV
    # For light text on dark background, use THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if debug:
        cv2.imwrite("debug_thresh.png", binary)
        print(f"Thresholded image saved to debug_thresh.png")
    
    # Apply morphological operations to clean up the image
    # Dilate slightly to connect broken parts of characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    if debug:
        cv2.imwrite("debug_dilated.png", dilated)
        print(f"Dilated image saved to debug_dilated.png")
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small contours (noise) and get bounding boxes
    min_area = 50  # Minimum contour area to consider
    bounding_boxes = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(c)
            # Filter out very thin or very short boxes (likely noise)
            if w > 5 and h > 10:
                bounding_boxes.append((x, y, w, h))
    
    # Sort bounding boxes from left to right
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    
    print(f"Total bounding boxes detected: {len(bounding_boxes)}")
    
    char_images = []
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        # Extract character
        char = binary[y:y+h, x:x+w]
        
        # Create square image with padding
        size = max(w, h) + 10  # Add some padding
        square = 255 * np.ones((size, size), dtype=np.uint8)
        
        # Center the character in the square
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = 255 - char
        
        if debug:
            cv2.imwrite(f"debug_chars/char_{i}.png", square)
        
        char_images.append(square)
    
    print(f"Total characters segmented: {len(char_images)}")
    return char_images

# Predict a single character
def predict_character(char_img, show_confidence=True):
    """
    Predict a single character from an image
    Returns the predicted character and confidence
    """
    # Preprocess
    char_preprocessed = preprocess_image(char_img)
    char_preprocessed = char_preprocessed.reshape(1, 32, 32, 1)
    
    # Predict
    prediction = model.predict(char_preprocessed, verbose=0)
    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index]
    
    if predicted_index < len(characters):
        predicted_char = characters[predicted_index]
    else:
        predicted_char = '?'
        print(f"Warning: Invalid index {predicted_index}")
    
    if show_confidence:
        return predicted_char, confidence
    return predicted_char

# Predict the English word from an image
def predict_english_word(word_img, debug=True):
    """
    Predict an English word from an image containing text
    Returns the predicted word as a string
    """
    # Segment the word into characters
    char_images = segment_word_image(word_img, debug=debug)
    
    if len(char_images) == 0:
        print("Warning: No characters detected in the image")
        return ""
    
    predicted_chars = []
    print("\nCharacter predictions:")
    print("-" * 50)
    
    for i, char_img in enumerate(char_images):
        predicted_char, confidence = predict_character(char_img, show_confidence=True)
        predicted_chars.append(predicted_char)
        print(f"Character {i+1}: {predicted_char} (confidence: {confidence*100:.2f}%)")
    
    predicted_word = ''.join(predicted_chars)
    return predicted_word

# Run the prediction with a test image
if __name__ == "__main__":
    # Path to test image
    img_path = 'test_images/test_word.jpg'  # Change this to your test image
    
    # You can also test with individual character images
    # img_path = 'test_images/letter_A.jpg'
    
    print(f"\nAttempting to load image from: {img_path}")
    print("=" * 60)
    
    try:
        # Load the image
        input_img = cv2.imread(img_path)
        
        if input_img is None:
            raise ValueError(f"Failed to load image from path: {img_path}")
        
        print(f"Image loaded successfully. Shape: {input_img.shape}")
        
        # Predict the word
        predicted_word = predict_english_word(input_img, debug=True)
        
        print("=" * 60)
        print(f"\n✓ PREDICTED ENGLISH WORD: {predicted_word}")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print("1. The image file exists at the specified path")
        print("2. The model file exists at 'models/emnist_cnn_model.h5'")
        print("3. You have trained the model using the training script")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

# Example usage for single character prediction:
def predict_single_character(img_path):
    """
    Helper function to predict a single character from an image
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image from: {img_path}")
    
    predicted_char, confidence = predict_character(img, show_confidence=True)
    print(f"Predicted character: {predicted_char} (confidence: {confidence*100:.2f}%)")
    return predicted_char

# Uncomment to test single character prediction:
# if __name__ == "__main__":
#     predict_single_character('test_images/letter_A.jpg')