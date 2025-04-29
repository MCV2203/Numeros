import streamlit as st
from tensorflow.keras.models import model_from_json
from PIL import Image, UnidentifiedImageError
import numpy as np
import os

# Configurar la pàgina
st.set_page_config(page_title="Classificador de Dígits 1-9", layout="centered")
st.title("🔢 Classificador de Dígits (1 al 9)")
st.markdown("Puja una imatge de dígit escrit a mà i la IA et dirà quin número és. ✍️🧠")

# Pujar imatge
uploaded_file = st.file_uploader("📤 Pujar imatge (jpg, png)", type=["jpg", "jpeg", "png"])

# Verificació dels fitxers del model
if not os.path.exists("model_digits_1to9.json") or not os.path.exists("model_digits_1to9.weights.h5"):
    st.error("❌ Fitxers del model no trobats. Assegura't que els arxius .json i .weights.h5 són al mateix directori.")
    st.stop()

try:
    # Carregar estructura del model
    with open("model_digits_1to9.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)

    # Carregar pesos
    model.load_weights("model_digits_1to9.weights.h5")

except Exception as e:
    st.error(f"❌ Error carregant el model: {e}")
    st.stop()

# Si hi ha una imatge pujada, processar-la
if uploaded_file:
    try:
        # Obrir i processar la imatge
        image = Image.open(uploaded_file).convert("L").resize((28, 28))
        st.image(image, caption='📷 Imatge pujada', use_container_width=True)

        img_array = np.array(image).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Fer predicció
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))

        # Com que els digits anaven de 1 a 9 i es van restar 1, afegim 1 per mostrar el valor original
        st.success(f"El model prediu que és un **{predicted_class + 1}** amb una confiança del **{confidence * 100:.2f}%**.")

    except UnidentifiedImageError:
        st.error("❌ No s'ha pogut llegir la imatge. Posa una imatge vàlida (jpg, png).")
