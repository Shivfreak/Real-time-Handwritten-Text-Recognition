import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Two options: Load from custom dataset or download EMNIST

# OPTION 1: If you have a custom English character dataset
# Path to the train directory (structured like: dataset/train/A/, dataset/train/B/, etc.)
use_custom_dataset = False  # Set to True if you have custom English character images
train_dir = 'dataset/train/'

# OPTION 2: Download and use EMNIST dataset (recommended)
use_emnist = True  # Set to True to use EMNIST

# Image dimensions
img_size = 32

# Function to preprocess images
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))  # Resize to expected input size
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

if use_emnist:
    print("Downloading and preprocessing EMNIST dataset...")
    
    try:
        # Try using extra_keras_datasets (recommended)
        # Install: pip install extra-keras-datasets
        from extra_keras_datasets import emnist
        
        # Load EMNIST letters (A-Z)
        print("Loading EMNIST letters dataset...")
        (X_train, y_train), (X_test, y_test) = emnist.load_data(type='letters')
        
        # EMNIST labels are 1-26 for A-Z, convert to 0-25
        y_train = y_train - 1
        y_test = y_test - 1
        
    except ImportError:
        print("extra_keras_datasets not found. Trying tensorflow_datasets...")
        
        try:
            # Alternative: Use tensorflow_datasets
            # Install: pip install tensorflow-datasets
            import tensorflow_datasets as tfds
            
            # Load EMNIST/letters dataset
            ds_train = tfds.load('emnist/letters', split='train', as_supervised=True)
            ds_test = tfds.load('emnist/letters', split='test', as_supervised=True)
            
            # Convert to numpy arrays
            X_train = np.array([img.numpy() for img, _ in ds_train])
            y_train = np.array([label.numpy() for _, label in ds_train])
            X_test = np.array([img.numpy() for img, _ in ds_test])
            y_test = np.array([label.numpy() for _, label in ds_test])
            
        except ImportError:
            print("Error: Please install extra-keras-datasets or tensorflow-datasets")
            print("Run: pip install extra-keras-datasets")
            exit(1)
    
    # Preprocess EMNIST data
    print(f"Original training data shape: {X_train.shape}")
    print(f"Original test data shape: {X_test.shape}")
    
    # Reshape and normalize
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Resize to 32x32 if needed
    if img_size != 28:
        print(f"Resizing images from 28x28 to {img_size}x{img_size}...")
        X_train_resized = []
        X_test_resized = []
        
        for img in X_train:
            resized = cv2.resize(img, (img_size, img_size))
            X_train_resized.append(resized)
        
        for img in X_test:
            resized = cv2.resize(img, (img_size, img_size))
            X_test_resized.append(resized)
        
        X_train = np.array(X_train_resized).reshape(-1, img_size, img_size, 1)
        X_test = np.array(X_test_resized).reshape(-1, img_size, img_size, 1)
    
    print(f"Processed training data shape: {X_train.shape}")
    print(f"Processed test data shape: {X_test.shape}")
    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of test samples: {len(X_test)}")
    print(f"Number of classes: {len(np.unique(y_train))}")

elif use_custom_dataset:
    print("Loading custom English character dataset...")
    
    # Load images and labels from the train directory
    X = []
    y = []
    
    # Label mapping for English characters (A-Z)
    # If your folders are named like: A, B, C... or character_A, character_B, etc.
    folder_to_char = {
        'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F',
        'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L',
        'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R',
        'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X',
        'Y': 'Y', 'Z': 'Z'
    }
    
    # Dynamically generate the list of characters (labels) from the folder names
    class_names = os.listdir(train_dir)
    class_names.sort()  # To ensure characters are sorted
    
    print(f"Found {len(class_names)} classes: {class_names}")
    
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(train_dir, class_name)
        
        # Ensure it's a directory
        if os.path.isdir(class_folder):
            img_count = 0
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                
                # Skip if not an image file
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                
                # Preprocess the image and append it to the data list
                try:
                    img = preprocess_image(img_path)
                    X.append(img)
                    y.append(label)  # Assign label based on the folder name
                    img_count += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            print(f"Loaded {img_count} images for class {class_name} (label {label})")
    
    # Convert to numpy arrays
    X = np.array(X)
    X = X.reshape(-1, img_size, img_size, 1)  # Reshape to include the channel dimension
    y = np.array(y)
    
    print(f"\nTotal images loaded: {len(X)}")
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Split the data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

else:
    print("Error: Please set either use_emnist=True or use_custom_dataset=True")
    exit(1)

# Save the preprocessed data for later use
print("\nSaving preprocessed data...")
np.save('data/X_train_emnist.npy', X_train)
np.save('data/y_train_emnist.npy', y_train)
np.save('data/X_test_emnist.npy', X_test)
np.save('data/y_test_emnist.npy', y_test)

print("Preprocessing completed!")
print(f"Saved files:")
print(f"  - data/X_train_emnist.npy: {X_train.shape}")
print(f"  - data/y_train_emnist.npy: {y_train.shape}")
print(f"  - data/X_test_emnist.npy: {X_test.shape}")
print(f"  - data/y_test_emnist.npy: {y_test.shape}")

# Print class distribution
unique, counts = np.unique(y_train, return_counts=True)
print(f"\nTraining set class distribution:")
for label, count in zip(unique, counts):
    char = chr(65 + label)  # Convert 0-25 to A-Z
    print(f"  {char} (label {label}): {count} samples")
