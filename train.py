import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Reshape, TimeDistributed
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow_datasets as tfds

# EMNIST Dataset Details
num_classes = 62  # 10 digits + 26 uppercase + 26 lowercase
class_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z'
]

def load_emnist_data():
    (ds_train, ds_test), ds_info = tfds.load(
        "emnist/byclass",
        split=["train", "test"],
        as_supervised=True,
        with_info=True
    )
   
    def preprocess(image, label):
        image = tf.image.resize(image, (28, 28)) / 255.0
        return image, label
   
    ds_train = ds_train.map(preprocess).shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess).batch(64).prefetch(tf.data.AUTOTUNE)
   
    return ds_train, ds_test, ds_info

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Reshape((-1, 128)),  # Reshape for LSTM input
       // LSTM(64, return_sequences=True),
        //LSTM(32),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
   
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model():
    ds_train, ds_test, ds_info = load_emnist_data()
    model = build_model()
    model.summary()
   
    # Learning rate reduction callback
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
   
    model.fit(ds_train,
              validation_data=ds_test,
              epochs=10,
              verbose=1,
              callbacks=[lr_reduction])
   
    model.save("emnist_model.keras")
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()


