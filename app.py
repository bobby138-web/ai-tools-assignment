import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🧠 MNIST Digit Classifier")

# Load model
model = tf.keras.models.load_model("mnist_model.h5")

uploaded = st.file_uploader("Upload a handwritten digit (28x28 image)", type=["png", "jpg"])
if uploaded:
    img = Image.open(uploaded).convert('L').resize((28,28))
    img_array = np.array(img)/255.0
    pred = model.predict(img_array.reshape(1,28,28,1))
    st.image(img, caption=f"Predicted Digit: {np.argmax(pred)}")
