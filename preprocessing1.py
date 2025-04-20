import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model


# Load the trained model
model = load_model("emnist_model.keras")


# Define EMNIST class labels (modify as per your dataset mapping)
# Correct EMNIST label mapping for 'ByClass' dataset
emnist_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z'
]

class_labels = emnist_labels

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    expanded_img = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
   
    thresh = cv2.adaptiveThreshold(expanded_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh


def segment_image(image, output_dir="segments"):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])


    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
       
        if w < 10 or h < 10:
            continue


        segment = image[y:y+h, x:x+w]


        target_size = 28
        aspect_ratio = w / h


        if aspect_ratio > 1:
            new_w = target_size
            new_h = int(target_size / aspect_ratio)
        else:
            new_h = target_size
            new_w = int(target_size * aspect_ratio)


        expanded_segment = cv2.resize(segment, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


        final_segment = np.full((target_size, target_size), 255, dtype=np.uint8)
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        final_segment[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = expanded_segment


        segment_path = os.path.join(output_dir, f"segment_{i}.png")
        cv2.imwrite(segment_path, final_segment)
        segments.append(final_segment)
   
    return segments

def predict_character(segment):
    segment = cv2.resize(segment, (28, 28)) 
    segment = segment.astype("float32") / 255.0  # Normalize
    segment = np.expand_dims(segment, axis=-1)  # Add channel dimension (28, 28, 1)
    segment = np.expand_dims(segment, axis=0)   # Add batch dimension (1, 28, 28, 1)
   
    prediction = model.predict(segment)
    predicted_label = class_labels[np.argmax(prediction)]
    return predicted_label

if __name__ == "__main__":
    image_path = "captured_image_0.jpg"  # Get image path from user
    processed_image = preprocess_image(image_path)
   
    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)
   
    segments = segment_image(processed_image)
   
    recognized_text = "".join([predict_character(seg) for seg in segments])
   
    print(f"Recognized Text: {recognized_text}")
    with open("recognized_text.txt", "w") as file:
        file.write(recognized_text)
   
    cv2.destroyAllWindows()
