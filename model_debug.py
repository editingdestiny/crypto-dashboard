import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Preprocess an MNIST sample
x_sample = x_test[0].astype('float32') / 255.0  # Normalize
x_sample = x_sample.reshape(1, 28, 28, 1)  # Reshape for model

# Load the handwriting model
model = load_model("handwriting_model.h5")

# Make a prediction
pred = model.predict(x_sample)
pred_digit = np.argmax(pred)

print(f"Model Prediction: {pred_digit}, Actual Label: {y_test[0]}")
print(f"Confidence Scores: {pred[0]}")
