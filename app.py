import streamlit as st
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import os

st.set_page_config(page_title="Classificador de Dígits 1-9", layout="centered")
st.title("🔢 Classificador de Dígits 1️⃣ a 9️⃣")
st.markdown("Puja una imatge de dígit manuscrit i la IA et dirà quin número és! 🧠")

uploaded_file = st.file_uploader("📤 Pujar imatge (28x28 en blanc i negre)", type=["jpg", "jpeg", "png"])

if not os.path.exists("model_digits_1to9.json") or not os.path.exists("model_digits_1to9.weights.h5"):
    st.error("❌ El model no s'ha trobat. Assegura't que els fitxers JSON i WEIGHTS estan al directori.")
else:
    try:
        with open("model_digits_1to9.json", "r") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights("model_digits_1to9.weights.h5")
    except Exception as e:
        st.error(f"❌ Error carregant el model: {e}")
        st.stop()

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("L")
            image = ImageOps.invert(image)
            image = image.resize((28, 28))
            st.image(image, caption="📷 Imatge pujada", use_container_width=True)

            img_array = np.array(image).astype("float32") / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            prediction = model.predict(img_array)
            predicted_class = int(np.argmax(prediction)) + 1
            confidence = float(np.max(prediction)) * 100

            st.success(f"Predicció: **{predicted_class}** amb una confiança de **{confidence:.2f}%**")

        except UnidentifiedImageError:
            st.error("❌ No s'ha pogut llegir la imatge. Si us plau, puja un arxiu .jpg o .png vàlid.")
