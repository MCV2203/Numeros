import streamlit as st
from tensorflow.keras.models import model_from_json
from PIL import Image, UnidentifiedImageError
import numpy as np
import os

st.set_page_config(page_title="Classificador de Dígits", layout="centered")
st.title("🔢 Classificador de Dígits (1 al 9)")
st.markdown("Puja una imatge (28x28 píxels, fons negre) i la IA et dirà quin dígit veu. 🧠")

uploaded_file = st.file_uploader("📤 Pujar imatge (jpg, png)", type=["jpg", "jpeg", "png"])

# Comprovar que existeixen els fitxers
if not os.path.exists("model_digits_1to9.json") or not os.path.exists("model_digits_1to9.weights.h5"):
    st.error("❌ El model no s'ha trobat. Assegura't que els fitxers JSON i WEIGHTS estiguin pujats correctament al repositori.")
else:
    # Carregar el model
    with open("model_digits_1to9.json", "r") as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json)
    model.load_weights("model_digits_1to9.weights.h5")

    # Processar la imatge pujada
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("L").resize((28, 28))  # Escala de grisos
            st.image(image, caption='📷 Imatge pujada', use_container_width=True)

            img_array = np.array(image).astype("float32") / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction[0]) + 1  # Sumar 1 perquè vam restar 1 abans
            confidence = np.max(prediction[0]) * 100

            st.success(f"És un **{predicted_class}** amb una confiança del **{confidence:.2f}%** ✨")

        except UnidentifiedImageError:
            st.error("❌ No s'ha pogut llegir la imatge. Si us plau, puja un arxiu .jpg o .png vàlid.")
