"""
app.py
──────
Dashboard profesional para el clasificador de calidad de cacao.
Detecta Moniliasis en mazorcas de cacao mediante visión por computadora
con un modelo YOLOv11s fine-tuneado.

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
    page_title="CacaoVision · Clasificador de Calidad",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Sistema de Diseño CSS ────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Reset y Ocultamiento de UI de Streamlit ── */
    #MainMenu, header, footer {visibility: hidden;}

    /* ── Tipografía (Playfair Display + DM Sans) ── */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800;900&family=DM+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Fondo general ── */
    .stApp {
        background: #0f0b08;
    }

    /* ── Header Hero ── */
    .hero-section {
        position: relative;
        text-align: center;
        padding: 2.8rem 1.5rem 1.8rem;
        background: linear-gradient(135deg, #1a120b 0%, #2c1a0e 40%, #3d2417 100%);
        border-bottom: 1px solid rgba(193, 154, 107, 0.15);
        margin: -1rem -1rem 0 -1rem;
        overflow: hidden;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse at 30% 20%, rgba(193, 154, 107, 0.08) 0%, transparent 60%),
                    radial-gradient(ellipse at 70% 80%, rgba(139, 90, 43, 0.06) 0%, transparent 50%);
        pointer-events: none;
    }
    .hero-brand {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #d4a574 0%, #c19a6b 30%, #e8c89e 60%, #c19a6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
    }
    .hero-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.95rem;
        color: #8b7355;
        font-weight: 400;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-top: 0;
    }
    .hero-line {
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, transparent, #c19a6b, transparent);
        margin: 1rem auto 0;
    }

    /* ── Indicadores de rendimiento ── */
    .stats-bar {
        display: flex;
        justify-content: center;
        gap: 0;
        margin: 0 auto;
        max-width: 720px;
        background: rgba(26, 18, 11, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(193, 154, 107, 0.1);
        border-radius: 14px;
        overflow: hidden;
        margin-top: -0.5rem;
        margin-bottom: 1.8rem;
    }
    .stat-item {
        flex: 1;
        text-align: center;
        padding: 1rem 0.8rem;
        border-right: 1px solid rgba(193, 154, 107, 0.08);
        transition: background 0.3s ease;
    }
    .stat-item:last-child { border-right: none; }
    .stat-item:hover { background: rgba(193, 154, 107, 0.05); }
    .stat-label {
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        color: #7a6650;
        margin-bottom: 0.25rem;
        font-weight: 500;
    }
    .stat-value {
        font-family: 'Playfair Display', serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #d4a574;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        justify-content: center;
        background: transparent;
        border-bottom: 1px solid rgba(193, 154, 107, 0.1);
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 0.85rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #7a6650 !important;
        padding: 0.8rem 2rem;
        border: none !important;
        background: transparent !important;
        transition: color 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #d4a574 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #d4a574 !important;
        border-bottom: 2px solid #c19a6b !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #c19a6b !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* ── Upload Area ── */
    [data-testid="stFileUploader"] {
        background: rgba(26, 18, 11, 0.4);
        border: 1px dashed rgba(193, 154, 107, 0.2);
        border-radius: 16px;
        padding: 1rem;
        transition: border-color 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(193, 154, 107, 0.4);
    }
    [data-testid="stFileUploader"] label {
        color: #a08a70 !important;
    }
    [data-testid="stFileUploader"] small {
        color: #6b5a48 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(44, 26, 14, 0.3) !important;
        border-color: rgba(193, 154, 107, 0.15) !important;
    }

    /* ── Result Cards ── */
    .result-card {
        background: linear-gradient(145deg, #1a120b 0%, #231710 100%);
        border: 1px solid rgba(193, 154, 107, 0.1);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
        overflow: hidden;
    }
    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        border-color: rgba(193, 154, 107, 0.2);
    }

    /* ── Classification Badges ── */
    .diagnosis {
        text-align: center;
        padding: 0.6rem 0;
    }
    .diagnosis-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1.2rem;
        border-radius: 30px;
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }
    .diagnosis-sano {
        background: rgba(46, 125, 50, 0.12);
        color: #66bb6a;
        border: 1px solid rgba(46, 125, 50, 0.25);
    }
    .diagnosis-enfermo {
        background: rgba(198, 40, 40, 0.12);
        color: #ef5350;
        border: 1px solid rgba(198, 40, 40, 0.25);
    }
    .diagnosis-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-sano {
        background: #66bb6a;
        box-shadow: 0 0 8px rgba(102, 187, 106, 0.5);
    }
    .dot-enfermo {
        background: #ef5350;
        box-shadow: 0 0 8px rgba(239, 83, 80, 0.5);
    }
    .confidence-bar-wrap {
        margin-top: 0.4rem;
        text-align: center;
    }
    .confidence-label {
        font-size: 0.65rem;
        color: #6b5a48;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .confidence-track {
        width: 80%;
        max-width: 200px;
        height: 4px;
        background: rgba(193, 154, 107, 0.1);
        border-radius: 2px;
        margin: 0 auto;
        overflow: hidden;
    }
    .confidence-fill-sano {
        height: 100%;
        background: linear-gradient(90deg, #388e3c, #66bb6a);
        border-radius: 2px;
        transition: width 0.6s ease;
    }
    .confidence-fill-enfermo {
        height: 100%;
        background: linear-gradient(90deg, #c62828, #ef5350);
        border-radius: 2px;
        transition: width 0.6s ease;
    }

    /* ── Section Headers ── */
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #d4a574;
        margin-bottom: 0.3rem;
    }
    .section-desc {
        font-size: 0.82rem;
        color: #6b5a48;
        margin-bottom: 1.2rem;
    }

    /* ── Upload prompt ── */
    .upload-prompt {
        text-align: center;
        padding: 3rem 1rem;
        color: #5a4a3a;
    }
    .upload-prompt-icon {
        font-size: 2.5rem;
        margin-bottom: 0.8rem;
        opacity: 0.4;
    }
    .upload-prompt p {
        font-size: 0.85rem;
        color: #6b5a48;
    }

    /* ── Results count ── */
    .results-count {
        text-align: center;
        font-size: 0.8rem;
        color: #7a6650;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(193, 154, 107, 0.08);
    }

    /* ── Expanders (Metrics tab) ── */
    .streamlit-expanderHeader {
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        color: #a08a70 !important;
        background: rgba(26, 18, 11, 0.4) !important;
        border: 1px solid rgba(193, 154, 107, 0.1) !important;
        border-radius: 12px !important;
    }

    /* ── Markdown general text color ── */
    .stMarkdown, .stCaption, p, span, label {
        color: #a08a70;
    }
    h1, h2, h3, h4 {
        color: #d4a574 !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #c19a6b !important;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 2.5rem 0 1rem;
        border-top: 1px solid rgba(193, 154, 107, 0.06);
        margin-top: 2rem;
    }
    .footer-line {
        width: 40px;
        height: 1px;
        background: rgba(193, 154, 107, 0.2);
        margin: 0 auto 1rem;
    }
    .footer-text {
        font-size: 0.65rem;
        color: #4a3d30;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .footer-org {
        font-family: 'Playfair Display', serif;
        font-size: 0.8rem;
        color: #6b5a48;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-brand">CacaoVision</div>
    <div class="hero-subtitle">Diagnóstico Inteligente de Mazorcas de Cacao</div>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

# ── Stats Bar ────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-label">Precisión</div>
        <div class="stat-value">98%</div>
    </div>
    <div class="stat-item">
        <div class="stat-label">Modelo</div>
        <div class="stat-value">YOLOv11s</div>
    </div>
    <div class="stat-item">
        <div class="stat-label">Inferencia</div>
        <div class="stat-value">&lt; 3 ms</div>
    </div>
    <div class="stat-item">
        <div class="stat-label">Clases</div>
        <div class="stat-value">Sano · Moniliasis</div>
    </div>
</div>
""", unsafe_allow_html=True)

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
    st.error("Modelo no encontrado. Ejecute `python3 entrenar.py` primero.")
else:
    # ── Pestañas Principales ─────────────────────────────────────────
    tab1, tab2 = st.tabs(["DIAGNÓSTICO", "MÉTRICAS"])

    # ── PESTAÑA 1: Clasificación ─────────────────────────────────────
    with tab1:
        st.markdown("")  # Spacing
        uploaded_files = st.file_uploader(
            "Arrastre o seleccione imágenes de mazorcas de cacao para analizar",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            with st.spinner("Procesando análisis de calidad..."):
                time.sleep(0.3)

            st.markdown(
                f'<div class="results-count">{len(uploaded_files)} imagen{"es" if len(uploaded_files) > 1 else ""} analizada{"s" if len(uploaded_files) > 1 else ""}</div>',
                unsafe_allow_html=True,
            )

            cols = st.columns(3)

            for i, uploaded_file in enumerate(uploaded_files):
                img = Image.open(uploaded_file)

                results = model.predict(img, verbose=False)
                result = results[0]

                top_class_id = result.probs.top1
                top_class_name = result.names[top_class_id]
                confidence = float(result.probs.top1conf) * 100

                col_idx = i % 3
                with cols[col_idx]:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.image(img, width="stretch")

                    if top_class_name.lower() == "sano":
                        st.markdown(
                            f'<div class="diagnosis">'
                            f'<span class="diagnosis-badge diagnosis-sano">'
                            f'<span class="diagnosis-dot dot-sano"></span>'
                            f'Sano</span></div>'
                            f'<div class="confidence-bar-wrap">'
                            f'<div class="confidence-label">Confianza {confidence:.1f}%</div>'
                            f'<div class="confidence-track"><div class="confidence-fill-sano" style="width:{confidence}%"></div></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="diagnosis">'
                            f'<span class="diagnosis-badge diagnosis-enfermo">'
                            f'<span class="diagnosis-dot dot-enfermo"></span>'
                            f'Moniliasis</span></div>'
                            f'<div class="confidence-bar-wrap">'
                            f'<div class="confidence-label">Confianza {confidence:.1f}%</div>'
                            f'<div class="confidence-track"><div class="confidence-fill-enfermo" style="width:{confidence}%"></div></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown('</div>', unsafe_allow_html=True)

    # ── PESTAÑA 2: Métricas de Entrenamiento ─────────────────────────
    with tab2:
        st.markdown("")  # Spacing
        st.markdown('<div class="section-title">Resultados del Entrenamiento</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-desc">Gráficas generadas por YOLOv11 durante el proceso de fine-tuning del modelo.</div>',
            unsafe_allow_html=True,
        )

        metrics = {
            "Matriz de Confusión": "runs/classify/resultados_cacao/modelo_refinado/confusion_matrix.png",
            "Matriz de Confusión Normalizada": "runs/classify/resultados_cacao/modelo_refinado/confusion_matrix_normalized.png",
            "Curvas de Entrenamiento": "runs/classify/resultados_cacao/modelo_refinado/results.png",
        }

        for title, img_path in metrics.items():
            if os.path.exists(img_path):
                with st.expander(title, expanded=True):
                    st.image(img_path, width="stretch")

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <div class="footer-line"></div>
    <div class="footer-org">Proyecto Capstone</div>
    <div class="footer-text">Visión por Computadora · Comalcalco, Tabasco · 2026</div>
</div>
""", unsafe_allow_html=True)
