"""
app.py
──────
Dashboard web interactivo construido con Streamlit para:
  1. Clasificar imágenes de mazorcas de cacao (Sano vs Moniliasis)
     usando el modelo YOLOv11 entrenado por 'entrenar.py'.
  2. Visualizar las métricas de entrenamiento (matrices de confusión,
     curvas de pérdida y accuracy).

Uso:
    streamlit run app.py
"""

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import time

# ── Configuración de Página ──────────────────────────────────────────
st.set_page_config(
    page_title="CacaoVision | Clasificador IA",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Estilos CSS Personalizados ───────────────────────────────────────
st.markdown("""
<style>
    /* Ocultar menús por defecto de Streamlit */
    #MainMenu, header, footer {visibility: hidden;}

    /* Tipografía profesional (Google Fonts: Inter) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Fondo con degradado cálido */
    .stApp {
        background: linear-gradient(160deg, #fdfcfb 0%, #f5f0eb 100%);
    }

    /* ── Encabezado principal ── */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 0.2rem 0;
    }
    .app-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        color: #3e2723;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .app-header p {
        font-size: 1rem;
        color: #8d6e63;
        font-weight: 400;
        margin-top: 0;
    }

    /* ── Fila de tarjetas métricas ── */
    .metric-row {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 1rem auto 1.5rem auto;
        flex-wrap: wrap;
        max-width: 800px;
    }
    .metric-card {
        background: white;
        border: 1px solid #e8e0da;
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        text-align: center;
        min-width: 150px;
        flex: 1;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }
    .metric-card .label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #a1887f;
        margin-bottom: 0.2rem;
    }
    .metric-card .value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #3e2723;
    }

    /* ── Tarjetas de resultado ── */
    .result-card {
        background: white;
        border-radius: 14px;
        padding: 0.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #ece7e2;
        transition: box-shadow 0.3s ease, transform 0.2s ease;
    }
    .result-card:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        transform: translateY(-3px);
    }

    /* ── Badges de clasificación ── */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
        margin-top: 0.4rem;
    }
    .badge-sano {
        background: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #c8e6c9;
    }
    .badge-enfermo {
        background: #fce4ec;
        color: #c62828;
        border: 1px solid #f8bbd0;
    }

    /* ── Pestañas ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        font-size: 0.9rem;
    }

    /* ── Estilo del file uploader ── */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 12px;
        padding: 0.5rem;
    }

    /* ── Pie de página ── */
    .app-footer {
        text-align: center;
        padding: 1.5rem 0 0.8rem 0;
        color: #bcaaa4;
        font-size: 0.75rem;
        letter-spacing: 0.5px;
    }

    /* ── Separador fino ── */
    .thin-divider {
        border: none;
        border-top: 1px solid #e8e0da;
        margin: 0.8rem auto;
        max-width: 700px;
    }
</style>
""", unsafe_allow_html=True)

# ── Encabezado ───────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>CacaoVision</h1>
    <p>Detección de Moniliasis en mazorcas de cacao mediante Inteligencia Artificial</p>
</div>
""", unsafe_allow_html=True)

# Tarjetas de métricas del modelo
st.markdown("""
<div class="metric-row">
    <div class="metric-card">
        <div class="label">Precisión</div>
        <div class="value">100%</div>
    </div>
    <div class="metric-card">
        <div class="label">Arquitectura</div>
        <div class="value">YOLOv11s</div>
    </div>
    <div class="metric-card">
        <div class="label">Inferencia</div>
        <div class="value">&lt; 3 ms</div>
    </div>
    <div class="metric-card">
        <div class="label">Clases</div>
        <div class="value">Sano · Moniliasis</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

# ── Carga del Modelo ─────────────────────────────────────────────────
MODEL_PATH = "runs/classify/resultados_cacao/modelo_refinado/weights/best.pt"


@st.cache_resource
def load_model(path):
    """Carga el modelo YOLO una sola vez y lo mantiene en caché."""
    if os.path.exists(path):
        return YOLO(path)
    return None


model = load_model(MODEL_PATH)

if model is None:
    st.error("⚠️ Modelo no encontrado. Ejecute `python3 entrenar.py` primero.")
else:
    # ── Pestañas Principales ─────────────────────────────────────────
    tab1, tab2 = st.tabs(["Clasificar Imágenes", "Métricas del Modelo"])

    # ── PESTAÑA 1: Clasificación en Vivo ─────────────────────────────
    with tab1:
        uploaded_files = st.file_uploader(
            "Arrastre o seleccione imágenes de mazorcas de cacao",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            # Indicador de carga mientras se procesan las imágenes
            with st.spinner("Analizando imágenes..."):
                time.sleep(0.3)

            st.markdown(f"**{len(uploaded_files)}** imagen(es) analizada(s)")
            st.markdown("---")

            # Distribuir resultados en cuadrícula de 3 columnas
            cols = st.columns(3)

            for i, uploaded_file in enumerate(uploaded_files):
                img = Image.open(uploaded_file)

                # Ejecutar inferencia (verbose=False para terminal limpia)
                results = model.predict(img, verbose=False)
                result = results[0]

                # Extraer clase predicha y confianza
                top_class_id = result.probs.top1
                top_class_name = result.names[top_class_id]
                confidence = float(result.probs.top1conf) * 100

                col_idx = i % 3
                with cols[col_idx]:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.image(img, use_container_width=True)

                    # Badge verde (sano) o rojo (moniliasis)
                    if top_class_name.lower() == "sano":
                        st.markdown(
                            f'<div style="text-align:center">'
                            f'<span class="badge badge-sano">✅ Sano — {confidence:.1f}%</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div style="text-align:center">'
                            f'<span class="badge badge-enfermo">⚠️ Moniliasis — {confidence:.1f}%</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown('</div>', unsafe_allow_html=True)

    # ── PESTAÑA 2: Métricas de Entrenamiento ─────────────────────────
    with tab2:
        st.markdown("#### Resultados del Entrenamiento")
        st.caption("Gráficas generadas automáticamente por YOLOv11 durante el fine-tuning.")

        # Rutas a las imágenes de métricas
        metrics = {
            "Matriz de Confusión": "runs/classify/resultados_cacao/modelo_refinado/confusion_matrix.png",
            "Matriz de Confusión (Normalizada)": "runs/classify/resultados_cacao/modelo_refinado/confusion_matrix_normalized.png",
            "Curvas de Entrenamiento": "runs/classify/resultados_cacao/modelo_refinado/results.png",
        }

        for title, img_path in metrics.items():
            if os.path.exists(img_path):
                with st.expander(title, expanded=True):
                    st.image(img_path, use_container_width=True)

# ── Pie de Página ────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    PROYECTO CAPSTONE · VISIÓN POR COMPUTADORA · MARZO 2026
</div>
""", unsafe_allow_html=True)
