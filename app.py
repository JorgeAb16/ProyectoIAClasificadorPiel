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
# TRADUCCIÓN AL ESPAÑOL
# ----------------------------------------------------------------------
# Estas son las 10 clases reales del dataset "Skin Diseases Image Dataset"
# (ismailpromus/skin-diseases-image-dataset) con el que se entrenó el modelo.
# La búsqueda es por palabras clave (no por texto exacto) para que funcione
# aunque el nombre de la carpeta tenga alguna variación menor (mayúsculas,
# guiones bajos, sufijos como "Photos", etc.).
CLASES_ES = [
    ("atopic dermatitis", "Dermatitis atópica"),
    ("basal cell carcinoma", "Carcinoma basocelular (BCC)"),
    ("benign keratosis", "Queratosis benigna (tipo BKL)"),
    ("eczema", "Eccema"),
    ("melanocytic nevi", "Nevos melanocíticos (lunares)"),
    ("melanoma", "Melanoma / cáncer de piel"),
    ("psoriasis", "Psoriasis y liquen plano"),
    ("seborrheic keratos", "Queratosis seborreica y otros tumores benignos"),
    ("tinea", "Tiña, candidiasis y otras infecciones por hongos"),
    ("ringworm", "Tiña, candidiasis y otras infecciones por hongos"),
    ("warts", "Verrugas, molusco contagioso y otras infecciones virales"),
    ("molluscum", "Verrugas, molusco contagioso y otras infecciones virales"),
]


def nombre_legible(clase_original: str) -> str:
    """Traduce la clase al español buscando por palabra clave. Si no
    encuentra coincidencia, formatea el nombre original como respaldo."""
    normalizado = clase_original.lower().replace("_", " ").replace("-", " ")
    for patron, espanol in CLASES_ES:
        if patron in normalizado:
            return espanol
    return clase_original.replace("_", " ").title()


# ----------------------------------------------------------------------
# ESTILO: fondo fijo + tipografía y colores de alto contraste
# (el color base ya queda fijado también en .streamlit/config.toml,
# esta hoja de estilos solo refuerza la paleta sobre esos mismos tonos)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="DermIA – Detector de Enfermedades de la Piel",
    page_icon="🩺",
    layout="wide",
)

FONDO = "#101B2D"        # azul marino oscuro fijo
PANEL = "#1B2A45"        # panel/tarjeta, un tono mas claro que el fondo
ACENTO = "#F2A93B"       # dorado calido (titulos, bordes, marca)
TEXTO_PRINCIPAL = "#F5F8FC"   # casi blanco, alto contraste sobre el fondo oscuro
TEXTO_SECUNDARIO = "#AFC0D6"  # gris azulado, para subtitulos/detalles

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500&display=swap');

    /* Fuerza el esquema de color para que el navegador no intente
       imponer su propio modo claro/oscuro por encima del nuestro */
    html {{
        color-scheme: dark;
    }}

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stMain"],
    [data-testid="stBottomBlockContainer"],
    .block-container {{
        background-color: {FONDO} !important;
        color: {TEXTO_PRINCIPAL} !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}

    [data-testid="stHeader"] {{
        background-color: transparent !important;
    }}

    h1, h2, h3, .main-header {{
        font-family: 'Poppins', sans-serif;
    }}

    .main-header {{
        font-size: 2.6rem;
        font-weight: 700;
        color: {ACENTO};
        text-align: center;
        margin-bottom: 0.3rem;
    }}
    .sub-header {{
        font-size: 1.1rem;
        color: {TEXTO_SECUNDARIO};
        text-align: center;
        margin-bottom: 2rem;
    }}

    .result-card {{
        background: {PANEL};
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        border-left: 6px solid {ACENTO};
    }}
    .disease-name {{
        font-family: 'Poppins', sans-serif;
        font-size: 1.15rem;
        font-weight: 600;
        color: {TEXTO_PRINCIPAL};
    }}
    .confidence {{
        font-size: 0.9rem;
        color: {TEXTO_SECUNDARIO};
        margin-top: 0.2rem;
    }}

    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, #2FBF9F 0%, {ACENTO} 60%, #E85D75 100%);
        border-radius: 10px;
    }}

    .disclaimer {{
        background: #3B2F12;
        border-left: 5px solid {ACENTO};
        border-radius: 8px;
        padding: 1rem;
        color: #FFD98E;
        margin-top: 2rem;
    }}

    .upload-area {{
        border: 2px dashed {ACENTO};
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        background: {PANEL};
    }}

    /* Refuerza contraste en componentes nativos de Streamlit sobre el fondo fijo */
    .stMarkdown, .stCaption, label, p, span, .stAlert {{
        color: {TEXTO_PRINCIPAL};
    }}
    .streamlit-expanderHeader {{
        color: {TEXTO_PRINCIPAL} !important;
        background: {PANEL} !important;
    }}
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


def main():
    st.markdown('<div class="main-header">🩺 DermIA</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Detección de enfermedades de la piel con inteligencia artificial</div>',
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ ¿Cómo usar esta herramienta?", expanded=False):
        st.markdown("""
        1. **Sube una imagen** nítida de la lesión cutánea (formatos JPG, JPEG o PNG).
        2. Ajusta el número de resultados que deseas ver (Top-K).
        3. El modelo analizará la imagen y te mostrará las **enfermedades más probables** junto con su nivel de confianza.
        4. Recuerda que este sistema es **solo con fines educativos** y **no reemplaza la consulta con un dermatólogo**.
        """)

    if not Path(MODEL_PATH).exists() or not Path(CLASS_NAMES_PATH).exists():
        st.error(
            f"No se encontró el modelo ('{MODEL_PATH}') o las clases ('{CLASS_NAMES_PATH}'). "
            "Copia esos archivos, generados por el notebook de entrenamiento, en la misma carpeta que app.py."
        )
        st.stop()

    modelo, class_names = cargar_modelo()

    col_control, _ = st.columns([1, 3])
    with col_control:
        top_k = st.slider(
            "🔍 Resultados a mostrar",
            min_value=1,
            max_value=min(10, len(class_names)),
            value=3,
            help="Elige cuántas de las predicciones principales quieres ver.",
        )

    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    archivo = st.file_uploader(
        "📤 Arrastra o selecciona una imagen",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if archivo is not None:
        imagen = Image.open(archivo)

        col1, col2 = st.columns([1, 1.2], gap="medium")

        with col1:
            st.image(imagen, caption="Imagen cargada", use_container_width=True)

        with st.spinner("🧠 Analizando imagen con MobileNetV2..."):
            entrada = preprocesar_imagen(imagen)
            predicciones = modelo.predict(entrada, verbose=0)[0]

        top_idx = np.argsort(predicciones)[-top_k:][::-1]

        with col2:
            st.subheader("📊 Predicciones principales")
            for i, idx in enumerate(top_idx):
                clase = class_names[idx]
                prob = float(predicciones[idx])
                porcentaje = prob * 100

                st.markdown(f"""
                <div class="result-card">
                    <div class="disease-name">{i + 1}. {nombre_legible(clase)}</div>
                    <div class="confidence">Confianza: {porcentaje:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(min(prob, 1.0))

        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Importante:</strong> Este resultado es generado por un modelo de inteligencia artificial
            con fines educativos y demostrativos. <strong>No constituye un diagnóstico médico.</strong>
            Ante cualquier duda, consulta a un dermatólogo.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👆 Sube una imagen para comenzar el análisis.")

    st.markdown("---")
    st.caption("Desarrollado con Streamlit y TensorFlow · Modelo MobileNetV2 con transfer learning")


if __name__ == "__main__":
    main()
