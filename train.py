import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import pickle

# Load EMNIST data (letters variant - 26 classes A-Z)
# You can download EMNIST from: https://www.nist.gov/itl/products-and-services/emnist-dataset
# Or use tensorflow_datasets or other sources

# Option 1: If you have preprocessed EMNIST data
try:
    X_train = np.load('data/X_train_emnist.npy')
    y_train = np.load('data/y_train_emnist.npy')
    X_test = np.load('data/X_test_emnist.npy')
    y_test = np.load('data/y_test_emnist.npy')
except FileNotFoundError:
    print("EMNIST data not found. Downloading using Keras...")
    # Option 2: Download EMNIST using extra_keras_datasets
    # Install: pip install extra-keras-datasets
    from extra_keras_datasets import emnist
    
    # Load EMNIST letters (A-Z)
    (X_train, y_train), (X_test, y_test) = emnist.load_data(type='letters')
    
    # Save for future use
    os.makedirs('data', exist_ok=True)
    np.save('data/X_train_emnist.npy', X_train)
    np.save('data/y_train_emnist.npy', y_train)
    np.save('data/X_test_emnist.npy', X_test)
    np.save('data/y_test_emnist.npy', y_test)

# Preprocess the data
# EMNIST images are 28x28, resize to 32x32 if needed for consistency
from tensorflow.image import resize

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Resize to 32x32 (optional, or change input_shape to (28, 28, 1))
X_train = resize(X_train, [32, 32]).numpy()
X_test = resize(X_test, [32, 32]).numpy()

# EMNIST labels are 1-26 for A-Z, convert to 0-25
y_train = y_train - 1
y_test = y_test - 1

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Number of classes: {len(np.unique(y_train))}")

# Ensure correct input shape
input_shape = (32, 32, 1)  # 32x32 grayscale images

# Build the CNN model
model = models.Sequential([
    layers.Input(shape=input_shape),  # (32, 32, 1)
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),  # Flatten the feature map
    layers.Dropout(0.5),  # Added dropout for regularization
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(26, activation='softmax')  # 26 classes for A-Z
])

# Print model summary
model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model with early stopping and model checkpoint
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=0.00001,
    verbose=1
)

# Ensure model directory exists before saving
os.makedirs("models", exist_ok=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'models/emnist_cnn_model_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print("\nTraining the model...")
history = model.fit(
    X_train, y_train, 
    epochs=20, 
    batch_size=128,
    validation_data=(X_test, y_test), 
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_accuracy*100:.2f}%")
print(f"Test loss: {test_loss:.4f}")

# Save the final model
model.save('models/emnist_cnn_model.h5')
print("Model saved to 'models/emnist_cnn_model.h5'")

# Create label mapping for English characters (A-Z)
label_to_char = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
    12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z'
}

# Also create reverse mapping (char to label)
char_to_label = {v: k for k, v in label_to_char.items()}

# Save the label mappings
with open('label_map.pkl', 'wb') as f:
    pickle.dump(label_to_char, f)

with open('char_to_label.pkl', 'wb') as f:
    pickle.dump(char_to_label, f)

print("Label map saved to 'label_map.pkl'")
print("Character to label map saved to 'char_to_label.pkl'")

# Save training history
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

print("\nTraining completed successfully!")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
