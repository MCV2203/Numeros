import streamlit as st
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import os
import h5py

# Configuració de la pàgina
st.set_page_config(page_title="Classificador de Dígits 1-9", layout="centered")
st.title("🔢 Classificador de Dígits 1️⃣ a 9️⃣")
st.markdown("Puja una imatge de dígit manuscrit i la IA et dirà quin número és! 🧠")

# Rutes dels fitxers del model
json_path = "model_digits_1to9.json"
weights_path = "model_digits_1to9.weights.h5"

# Comprova si els fitxers existeixen
if not os.path.exists(json_path) or not os.path.exists(weights_path):
    st.error("❌ El model no s'ha trobat. Assegura't que els fitxers JSON i H5 estan al directori.")
    st.stop()

# Comprova si el fitxer .h5 és vàlid
try:
    with h5py.File(weights_path, "r") as f:
        pass
except Exception as e:
    st.error(f"❌ El fitxer de pesos no és vàlid: {e}")
    st.stop()

# Carrega el model
try:
    with open(json_path, "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
except Exception as e:
    st.error(f"❌ Error carregant el model: {e}")
    st.stop()

# Carrega la imatge
uploaded_file = st.file_uploader("📤 Pujar imatge (28x28 en blanc i negre)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("L")  # Blanc i negre
        image = ImageOps.invert(image)                  # Invertir colors (fons negre, dígit blanc)
        image = image.resize((28, 28))                   # Redimensionar a 28x28
        st.image(image, caption="📷 Imatge pujada", use_container_width=True)

        # Preprocessament
        img_array = np.array(image).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predicció
        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction)) + 1  # Dígits de 1 a 9
        confidence = float(np.max(prediction)) * 100

        # Resultat
        st.success(f"🔢 Predicció: **{predicted_class}** amb una confiança de **{confidence:.2f}%**")

    except UnidentifiedImageError:
        st.error("❌ No s'ha pogut llegir la imatge. Si us plau, puja un arxiu .jpg o .png vàlid.")
    except Exception as e:
        st.error(f"❌ Error processant la imatge: {e}")
