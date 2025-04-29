import streamlit as st
from tensorflow.keras.models import model_from_json
from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Classificador Dígits 1-9", layout="centered")
st.title("🔢 Classificador de Dígits (1-9)")
st.markdown("Puja una imatge (28x28 píxels, en escala de grisos) i la IA et dirà quin número veu! 🧠")

uploaded_file = st.file_uploader("📤 Pujar imatge (png, jpg)", type=["jpg", "jpeg", "png"])

# Verificació d'existència de fitxers
try:
    with open("model_digits_1to9.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("model_digits_1to9.weights.h5")
except FileNotFoundError:
    st.error("❌ El model no s'ha trobat. Assegura't que els fitxers JSON i WEIGHTS estiguin disponibles.")
    st.stop()

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("L").resize((28, 28))
        st.image(image, caption="📷 Imatge pujada", use_container_width=False)

        img_array = np.array(image).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction[0])) + 1  # Tornem a 1-9

        st.success(f"✏️ El model creu que és un **{predicted_class}** amb {np.max(prediction[0])*100:.2f}% de confiança.")

        # Mostrar gràfic de barres
        st.subheader("📊 Probabilitats per cada dígit:")
        fig, ax = plt.subplots()
        bars = ax.bar([str(i+1) for i in range(9)], prediction[0], color="skyblue")
        ax.set_xlabel("Dígits")
        ax.set_ylabel("Probabilitat")
        ax.set_ylim([0, 1])
        ax.set_title("Distribució de confiança")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')
        st.pyplot(fig)

    except UnidentifiedImageError:
        st.error("❌ No s'ha pogut llegir la imatge. Si us plau, puja un arxiu vàlid.")

