import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --- Configuración ---
IMG_SIZE = (224, 224)
MODEL_PATH = "skin_disease_mobilenet.h5"
CLASS_NAMES_PATH = "class_names.json"

# ----------------------------------------------------------------------
# TRADUCCIÓN AL ESPAÑOL – Ajusta con los nombres exactos de tus carpetas.
# Si el modelo se entrenó con otro conjunto de datos, cambia las claves.
# ----------------------------------------------------------------------
LABELS_ES = {
    "akiec": "Queratosis actínica / Carcinoma intraepidérmico (Bowen)",
    "bcc": "Carcinoma basocelular",
    "bkl": "Queratosis seborreica",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Nevo melanocítico",
    "vasc": "Lesión vascular (angioma, hemangioma...)",
    # Añade aquí cualquier otra clase que aparezca en tu class_names.json
}

# ----------------------------------------------------------------------
# ESTILO PROFESIONAL
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="DermIA – Detector de Enfermedades de la Piel",
    page_icon="🩺",
    layout="wide",
)

# CSS personalizado para tarjetas, barras de progreso y tipografía
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5A6E8C;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 6px solid #2E86AB;
    }
    .disease-name {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1E3A5F;
    }
    .confidence {
        font-size: 0.9rem;
        color: #5A6E8C;
        margin-top: 0.2rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #A23B72 0%, #F18F01 50%, #2E86AB 100%);
        border-radius: 10px;
    }
    .disclaimer {
        background: #FFF3CD;
        border-left: 5px solid #FFC107;
        border-radius: 8px;
        padding: 1rem;
        color: #856404;
        margin-top: 2rem;
    }
    .upload-area {
        border: 2px dashed #2E86AB;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        background: #F4F8FB;
    }
</style>
""", unsafe_allow_html=True)


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


def nombre_legible(clase_original: str) -> str:
    """Devuelve el nombre en español; si no existe, muestra la clase original formateada."""
    if clase_original in LABELS_ES:
        return LABELS_ES[clase_original]
    # Fallback: reemplaza guiones bajos, capitaliza palabras
    return clase_original.replace("_", " ").title()


def barra_color(probabilidad: float) -> str:
    """Retorna un color en HSL para la barra según la confianza (verde > amarillo > rojo)."""
    # Verde (alta) -> Rojo (baja)
    hue = int(120 * probabilidad)  # 120° = verde, 0° = rojo
    return f"hsl({hue}, 70%, 50%)"


def main():
    # Encabezado profesional
    st.markdown('<div class="main-header">🩺 DermIA</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detección de enfermedades de la piel con inteligencia artificial</div>',
                unsafe_allow_html=True)

    # Descripción en tarjeta expandible
    with st.expander("ℹ️ ¿Cómo usar esta herramienta?", expanded=False):
        st.markdown("""
        1. **Sube una imagen** nítida de la lesión cutánea (formatos JPG, JPEG o PNG).
        2. Ajusta el número de resultados que deseas ver (Top-K).
        3. El modelo analizará la imagen y te mostrará las **enfermedades más probables** junto con su nivel de confianza.
        4. Recuerda que este sistema es **solo con fines educativos** y **no reemplaza la consulta con un dermatólogo**.
        """)

    modelo, class_names = cargar_modelo()

    # Control de número de resultados
    col_control, _ = st.columns([1, 3])
    with col_control:
        top_k = st.slider(
            "🔍 Resultados a mostrar",
            min_value=1,
            max_value=min(10, len(class_names)),
            value=3,
            help="Elige cuántas de las predicciones principales quieres ver."
        )

    # Subida de imagen con diseño mejorado
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    archivo = st.file_uploader(
        "📤 Arrastra o selecciona una imagen",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if archivo is not None:
        imagen = Image.open(archivo)

        col1, col2 = st.columns([1, 1.2], gap="medium")

        with col1:
            st.image(imagen, caption="Imagen cargada", use_container_width=True)

        # Predicción
        with st.spinner("🧠 Analizando imagen con MobileNetV2..."):
            entrada = preprocesar_imagen(imagen)
            predicciones = modelo.predict(entrada, verbose=0)[0]

        # Índices de mayor a menor probabilidad
        top_idx = np.argsort(predicciones)[-top_k:][::-1]

        with col2:
            st.subheader("📊 Predicciones principales")
            for i, idx in enumerate(top_idx):
                clase = class_names[idx]
                prob = float(predicciones[idx])
                porcentaje = prob * 100

                # Tarjeta de resultado
                with st.container():
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="disease-name">{i+1}. {nombre_legible(clase)}</div>
                        <div class="confidence">Confianza: {porcentaje:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Barra de progreso personalizada con color dinámico
                    color_barra = barra_color(prob)
                    st.markdown(f"""
                    <style>
                    .progress-{i} .stProgress > div > div > div > div {{
                        background: {color_barra};
                    }}
                    </style>
                    """, unsafe_allow_html=True)
                    st.progress(min(prob, 1.0), text=f"")  # la barra sola

        # Aviso legal destacado
        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Importante:</strong> Este resultado es generado por un modelo de inteligencia artificial
            con fines educativos y demostrativos. <strong>No constituye un diagnóstico médico.</strong>
            Ante cualquier duda, consulta a un dermatólogo.
        </div>
        """, unsafe_allow_html=True)

    else:
        # Placeholder cuando no hay imagen
        st.info("👆 Sube una imagen para comenzar el análisis.")

    # Footer sutil
    st.markdown("---")
    st.caption("Desarrollado con Streamlit y TensorFlow · Modelo MobileNetV2 con transfer learning")


if __name__ == "__main__":
    main()
