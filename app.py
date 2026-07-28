import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --- Configuracion ---
IMG_SIZE = (224, 224)
MODEL_PATH ="skin_disease_mobilenet.h5"
CLASS_NAMES_PATH ="class_names.json"

# Traduccion al espanol de las clases detectadas por el notebook.
# Completa/ajusta estos valores con los nombres reales que te imprimio
# el notebook al entrenar (clave = nombre original de la carpeta,
# valor = texto que se muestra en la app).
LABELS_ES = {}

st.set_page_config(page_title="Detector de enfermedades de la piel", page_icon="🩺")


@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        clases = json.load(f)
    return modelo, clases


def preprocesar_imagen(imagen_pil):
    img = imagen_pil.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def nombre_legible(clase):
    return LABELS_ES.get(clase, clase)


def main():
    st.title("🩺 Detector de enfermedades de la piel")
    st.write(
        "Sube una foto de una lesión o afección de piel y el modelo "
        "(MobileNetV2 con transfer learning) mostrará las enfermedades más probables."
    )

    modelo, class_names = cargar_modelo()

    top_k = st.slider("Número de resultados a mostrar", min_value=1, max_value=min(10, len(class_names)), value=3)

    archivo = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

    if archivo is not None:
        imagen = Image.open(archivo)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(imagen, caption="Imagen cargada", use_container_width=True)

        with st.spinner("Analizando imagen..."):
            entrada = preprocesar_imagen(imagen)
            predicciones = modelo.predict(entrada, verbose=0)[0]

        top_idx = np.argsort(predicciones)[-top_k:][::-1]

        with col2:
            st.subheader("Resultados")
            for idx in top_idx:
                clase = class_names[idx]
                prob = float(predicciones[idx]) * 100
                st.write(f"**{nombre_legible(clase)}**")
                st.progress(min(int(prob), 100))
                st.caption(f"{prob:.2f}%")

        st.warning(
            "⚠️ Este resultado es generado por un modelo de inteligencia artificial "
            "con fines educativos/demostrativos y **no reemplaza un diagnóstico médico**. "
            "Consulta a un dermatólogo ante cualquier duda."
        )


if __name__ == "__main__":
    main()
