import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image


# Load the saved model

model = load_model('dog_breed_model.keras')

#Requirement:
# Must have same name exactly as the one used during training

CLASS_NAMES = ['scottish_deerhound', 'maltese_dog', 'bernese_mountain_dog']

#App title and instructions
st.title('Dog Breed Prediction')
st.markdown('Upload an image of a dog, and the model will predict its breed.')

#File uploader

uploaded_file = st.file_uploader('Choose a dog image.. ', type=['jpg', 'jpeg', 'png'])


if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)
        breed = CLASS_NAMES[np.argmax(prediction)]
        confidence = round(float(np.max(prediction)) * 100, 2)

        st.success(f'Predicted Breed: **{breed}**')
        st.info(f'Confidence: {confidence}%')