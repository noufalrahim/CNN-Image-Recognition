import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore
import os

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/TrainedModel/trained_fashion_mnist_model.h5"

model = tf.keras.models.load_model(model_path)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28, 28))
    img = img.convert('L') # convert to grayscale
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

st.title('Fashion MNIST Image Classification')

uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_image = image.resize((100, 100))
        st.image(resized_image)

    with col2:
        if st.button('Classify'):
            img_array = preprocess_image(uploaded_image)
            result = model.predict(img_array)

            predicted_class = np.argmax(result[0])
            prediction = class_names[predicted_class]

            st.success(f'Prediction: {prediction}')