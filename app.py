"""
app.py
──────
CacaoVision — Dashboard de diagnóstico de moniliasis en mazorcas de cacao.
Modelo: YOLOv11m-cls · Streamlit · Comalcalco, Tabasco · 2026
"""

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import time

st.set_page_config(
    page_title="CacaoVision · Diagnóstico de Cacao",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

UMBRAL_SANO       = 92.0
UMBRAL_MONILIASIS = 10.0
MODEL_PATH = "runs/classify/resultados_cacao/modelo_refinado/weights/best.pt"

# ─────────────────────────────────────────────────────────────────────
# SVG ICONS — todos originales, referencia visual al cacao de Tabasco
# ─────────────────────────────────────────────────────────────────────

# Mazorca de cacao vista lateral con costillas
ICO_MAZORCA = (
    '<svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<ellipse cx="16" cy="19" rx="8" ry="11" fill="url(#mg)"/>'
    '<path d="M16 8 C16 8 13 4 10 3" stroke="#c19a6b" stroke-width="1.2" stroke-linecap="round"/>'
    '<path d="M16 8 C16 8 19 5 22 4" stroke="#c19a6b" stroke-width="1.2" stroke-linecap="round"/>'
    '<line x1="16" y1="8" x2="16" y2="30" stroke="#8b5e2a" stroke-width="1.4" stroke-linecap="round" opacity="0.6"/>'
    '<path d="M9 15 C8 17 8 21 9 24" stroke="#e8c890" stroke-width="0.9" stroke-linecap="round" opacity="0.4"/>'
    '<path d="M12 11 C11 13 11 17 12 20" stroke="#e8c890" stroke-width="0.9" stroke-linecap="round" opacity="0.4"/>'
    '<path d="M20 11 C21 13 21 17 20 20" stroke="#e8c890" stroke-width="0.9" stroke-linecap="round" opacity="0.4"/>'
    '<path d="M23 15 C24 17 24 21 23 24" stroke="#e8c890" stroke-width="0.9" stroke-linecap="round" opacity="0.4"/>'
    '<defs><linearGradient id="mg" x1="8" y1="8" x2="24" y2="30" gradientUnits="userSpaceOnUse">'
    '<stop offset="0%" stop-color="#a0723a"/>'
    '<stop offset="50%" stop-color="#c8924e"/>'
    '<stop offset="100%" stop-color="#6b3e1a"/>'
    '</linearGradient></defs>'
    '</svg>'
)

# Ojo con lupa — diagnóstico / análisis
ICO_DIAG = (
    '<svg viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="9" cy="9" r="6.5" stroke="#c19a6b" stroke-width="1.4"/>'
    '<circle cx="9" cy="9" r="3" fill="none" stroke="#d4a574" stroke-width="1.1" opacity="0.7"/>'
    '<circle cx="9" cy="9" r="1.2" fill="#d4a574" opacity="0.9"/>'
    '<line x1="13.8" y1="13.8" x2="19" y2="19" stroke="#c19a6b" stroke-width="1.6" stroke-linecap="round"/>'
    '<circle cx="18.5" cy="18.5" r="1.5" fill="#8b5e2a"/>'
    '</svg>'
)

# Onda de ADN simplificada — métricas / ciencia
ICO_METRICS = (
    '<svg viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M2 11 C4 7 6 5 8 7 C10 9 12 13 14 15 C16 17 18 15 20 11" stroke="#c19a6b" stroke-width="1.5" stroke-linecap="round" fill="none"/>'
    '<path d="M2 11 C4 15 6 17 8 15 C10 13 12 9 14 7 C16 5 18 7 20 11" stroke="#8b5e2a" stroke-width="1" stroke-linecap="round" fill="none" opacity="0.5"/>'
    '<line x1="5" y1="8" x2="5" y2="14" stroke="#d4a574" stroke-width="0.8" opacity="0.4"/>'
    '<line x1="11" y1="6" x2="11" y2="16" stroke="#d4a574" stroke-width="0.8" opacity="0.4"/>'
    '<line x1="17" y1="8" x2="17" y2="14" stroke="#d4a574" stroke-width="0.8" opacity="0.4"/>'
    '</svg>'
)

# Brújula con rosa de los vientos — Comalcalco / origen
ICO_ABOUT = (
    '<svg viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="11" cy="11" r="9" stroke="#c19a6b" stroke-width="1.3"/>'
    '<circle cx="11" cy="11" r="1.5" fill="#d4a574"/>'
    '<polygon points="11,3 12.2,9.5 11,11 9.8,9.5" fill="#d4a574"/>'
    '<polygon points="11,19 12.2,12.5 11,11 9.8,12.5" fill="#8b5e2a" opacity="0.6"/>'
    '<polygon points="3,11 9.5,9.8 11,11 9.5,12.2" fill="#8b5e2a" opacity="0.6"/>'
    '<polygon points="19,11 12.5,9.8 11,11 12.5,12.2" fill="#c19a6b" opacity="0.4"/>'
    '<circle cx="11" cy="3" r="1" fill="#d4a574"/>'
    '</svg>'
)

# Hongo estilizado — símbolo de Moniliophthora roreri
ICO_MONO = (
    '<svg viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M3 12 C3 8 5.5 5 9 5 C12.5 5 15 8 15 12" fill="url(#fm)" opacity="0.9"/>'
    '<rect x="8.2" y="12" width="1.6" height="4" rx="0.8" fill="#c04040"/>'
    '<circle cx="6" cy="9.5" r="1" fill="rgba(255,100,100,0.3)"/>'
    '<circle cx="9" cy="8" r="0.8" fill="rgba(255,100,100,0.25)"/>'
    '<circle cx="12" cy="9.5" r="1" fill="rgba(255,100,100,0.3)"/>'
    '<defs><linearGradient id="fm" x1="3" y1="5" x2="15" y2="12" gradientUnits="userSpaceOnUse">'
    '<stop offset="0%" stop-color="#8b2020"/>'
    '<stop offset="100%" stop-color="#c04040"/>'
    '</linearGradient></defs>'
    '</svg>'
)

# Hoja tropical con venación — símbolo de planta sana
ICO_SANO = (
    '<svg viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M9 16 C9 16 2 12 2 6 C2 3 5 2 9 4 C13 2 16 3 16 6 C16 12 9 16 9 16Z" fill="url(#fs)"/>'
    '<line x1="9" y1="16" x2="9" y2="4" stroke="#1a6b2a" stroke-width="0.9" stroke-linecap="round" opacity="0.5"/>'
    '<path d="M9 8 C7 7 5 7 4 8" stroke="#2e8b3e" stroke-width="0.7" stroke-linecap="round" opacity="0.6"/>'
    '<path d="M9 11 C7 10 5 10.5 4 12" stroke="#2e8b3e" stroke-width="0.7" stroke-linecap="round" opacity="0.6"/>'
    '<path d="M9 8 C11 7 13 7 14 8" stroke="#2e8b3e" stroke-width="0.7" stroke-linecap="round" opacity="0.6"/>'
    '<defs><linearGradient id="fs" x1="2" y1="2" x2="16" y2="16" gradientUnits="userSpaceOnUse">'
    '<stop offset="0%" stop-color="#2e7d32"/>'
    '<stop offset="100%" stop-color="#1b5e20"/>'
    '</linearGradient></defs>'
    '</svg>'
)

# Interrogante orgánico — incierto
ICO_DOUBT = (
    '<svg viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M7 7 C7 5 8 4 9 4 C10.5 4 11.5 5 11.5 6.5 C11.5 8 10 8.5 9 10" stroke="#c8a83a" stroke-width="1.5" stroke-linecap="round" fill="none"/>'
    '<circle cx="9" cy="13" r="1.1" fill="#c8a83a"/>'
    '<circle cx="9" cy="9" r="8" stroke="#c8a83a" stroke-width="1" opacity="0.25"/>'
    '</svg>'
)


def svg_img(svg_str, size="1.1rem"):
    """Convierte SVG inline a data URI para usar en HTML."""
    encoded = svg_str.replace('"', "'").replace('\n', '').replace('#', '%23')
    return f'<img src="data:image/svg+xml;charset=utf-8,{encoded}" style="width:{size};height:{size};vertical-align:middle;display:inline-block"/>'


# ─────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset / Streamlit overrides ── */
#MainMenu, header, footer { visibility: hidden; }
.stApp { background: #0a0805; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none; }

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    color: #c4a882;
}

/* ── HERO ── */
.hero {
    width: 100%;
    background: #0a0805;
    padding: 3.2rem 2rem 0;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 70% 80% at 15% 20%, rgba(160,110,50,0.11) 0%, transparent 60%),
        radial-gradient(ellipse 50% 70% at 85% 15%, rgba(100,50,15,0.09) 0%, transparent 55%),
        radial-gradient(ellipse 90% 40% at 50% 100%, rgba(80,40,10,0.07) 0%, transparent 55%);
    pointer-events: none;
    animation: meshDrift 14s ease-in-out infinite alternate;
}
@keyframes meshDrift {
    from { opacity: .65; transform: scale(1) rotate(0deg); }
    to   { opacity: 1;   transform: scale(1.05) rotate(1deg); }
}
.hero-inner {
    position: relative;
    max-width: 860px;
    margin: 0 auto;
}
.hero-pod {
    width: 72px; height: 72px;
    margin: 0 auto 1.4rem;
    filter: drop-shadow(0 0 22px rgba(193,154,107,0.35));
    animation: podFloat 7s ease-in-out infinite;
    display: block;
}
@keyframes podFloat {
    0%, 100% { transform: translateY(0) rotate(-1deg); }
    50%       { transform: translateY(-7px) rotate(1deg); }
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #6a5035;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Instrument Serif', serif;
    font-style: italic;
    font-size: clamp(2.6rem, 4.5vw, 4rem);
    font-weight: 400;
    line-height: 1.1;
    background: linear-gradient(130deg, #b8874a 0%, #f0d090 30%, #c19a6b 60%, #edd898 100%);
    background-size: 220% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: titleShimmer 5s linear infinite;
    margin: 0 0 0.5rem;
}
@keyframes titleShimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 220% center; }
}
.hero-desc {
    font-size: 0.87rem;
    color: #5a4232;
    max-width: 500px;
    margin: 0 auto 2.4rem;
    line-height: 1.65;
}

/* ── BENTO STATS ── */
.bento {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    max-width: 680px;
    margin: 0 auto 0;
    border: 1px solid rgba(193,154,107,0.08);
    border-radius: 16px;
    overflow: hidden;
    background: rgba(193,154,107,0.04);
}
.bc {
    background: rgba(10,8,5,0.85);
    padding: 1.1rem 0.6rem;
    text-align: center;
    border-right: 1px solid rgba(193,154,107,0.06);
    transition: background .2s;
}
.bc:last-child { border-right: none; }
.bc:hover { background: rgba(193,154,107,0.05); }
.bc-icon { margin-bottom: 0.5rem; line-height: 1; }
.bc-val {
    font-family: 'Instrument Serif', serif;
    font-style: italic;
    font-size: 1.35rem;
    color: #d4a574;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.bc-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #4a3525;
    font-weight: 500;
}

/* ── NAV DIVIDER ── */
.nav-divider {
    width: 100%; height: 1px;
    background: rgba(193,154,107,0.07);
    margin: 0;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    justify-content: center;
    background: transparent;
    border-bottom: 1px solid rgba(193,154,107,0.07);
    padding: 0;
    margin: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #3d2e1e !important;
    padding: 1rem 2.5rem;
    border: none !important;
    background: transparent !important;
    transition: color .2s;
    display: flex; align-items: center; gap: 0.5rem;
}
.stTabs [data-baseweb="tab"]:hover { color: #a08060 !important; }
.stTabs [aria-selected="true"] {
    color: #d4a574 !important;
    border-bottom: 1.5px solid #c19a6b !important;
}
.stTabs [data-baseweb="tab-highlight"] { background-color: #c19a6b !important; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ── MAIN CONTENT WRAPPER ── */
.main-pad {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1.8rem 2.5rem 0;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: rgba(16,10,6,0.6);
    border: 1px dashed rgba(193,154,107,0.15);
    border-radius: 18px;
    padding: 0.6rem;
    transition: border-color .3s;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(193,154,107,0.3); }
[data-testid="stFileUploader"] label { color: #907050 !important; }
[data-testid="stFileUploader"] small { color: #4a3828 !important; }
[data-testid="stFileUploaderDropzone"] {
    background: rgba(25,15,8,0.45) !important;
    border-color: rgba(193,154,107,0.1) !important;
}

/* ── UPLOAD PROMPT ── */
.up-prompt {
    text-align: center;
    padding: 5rem 1rem 4rem;
}
.up-pod {
    width: 64px; height: 64px;
    margin: 0 auto 1.2rem;
    opacity: 0.2;
    filter: drop-shadow(0 0 12px rgba(193,154,107,0.2));
    display: block;
}
.up-text { font-size: 0.82rem; color: #4a3828; line-height: 1.75; }
.up-hint {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; color: #3a2a1a;
    letter-spacing: 1.5px; margin-top: 0.5rem;
}

/* ── RESULTS HEADER ── */
.res-hdr {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4a3828;
    padding: 1rem 0;
    border-bottom: 1px solid rgba(193,154,107,0.05);
    margin-bottom: 1.2rem;
}
.res-hdr span { display: inline-flex; align-items: center; gap: 0.35rem; }

/* ── RESULT CARD ── */
.rcard {
    position: relative;
    background: linear-gradient(158deg, #110d08 0%, #18100a 100%);
    border-radius: 20px;
    overflow: hidden;
    margin-bottom: 1.2rem;
    transition: transform .3s cubic-bezier(.2,.8,.2,1), box-shadow .3s;
    border: 1px solid rgba(193,154,107,0.07);
}
.rcard:hover { transform: translateY(-5px); }
.rcard.g-sano:hover  { box-shadow: 0 22px 55px rgba(40,160,60,0.13);  border-color: rgba(40,160,60,0.12); }
.rcard.g-mono:hover  { box-shadow: 0 22px 55px rgba(210,50,50,0.15);  border-color: rgba(210,50,50,0.12); }
.rcard.g-doubt:hover { box-shadow: 0 22px 55px rgba(195,155,40,0.13); border-color: rgba(195,155,40,0.10); }
.rcard-body { padding: 0.85rem 0.9rem 0.9rem; }

/* ── BADGES ── */
.diag { display: flex; justify-content: center; padding: 0.3rem 0 0.1rem; }
.badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.42rem 1rem;
    border-radius: 40px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.67rem; font-weight: 700;
    letter-spacing: 1.4px; text-transform: uppercase;
}
.b-sano  { background: rgba(35,110,45,0.12); color: #55cc65; border: 1px solid rgba(35,110,45,0.2); }
.b-mono  { background: rgba(190,35,35,0.12); color: #ee5050; border: 1px solid rgba(190,35,35,0.2); }
.b-doubt { background: rgba(190,148,30,0.12); color: #d4a535; border: 1px solid rgba(190,148,30,0.2); }

/* ── DUAL BARS ── */
.conf-wrap { padding: 0.5rem 0.2rem 0.2rem; }
.cbar {
    display: flex; align-items: center; gap: 0.55rem;
    margin-bottom: 0.32rem;
}
.cbar-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem; color: #4a3828;
    width: 32px; flex-shrink: 0; text-align: right;
    letter-spacing: 0.4px;
}
.cbar-track {
    flex: 1; height: 3px;
    background: rgba(193,154,107,0.07);
    border-radius: 2px; overflow: hidden;
}
.fill-s { height:100%; background: linear-gradient(90deg,#1b5e20,#55cc65); border-radius:2px; transition: width .8s ease; }
.fill-m { height:100%; background: linear-gradient(90deg,#b71c1c,#ee5050); border-radius:2px; transition: width .8s ease; }
.cbar-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem; color: #6a5535;
    width: 36px; flex-shrink: 0; letter-spacing: 0.4px;
}

/* ── METRICS ── */
.met-hdr { padding: 0.5rem 0 1.2rem; }
.met-title {
    font-family: 'Instrument Serif', serif;
    font-style: italic;
    font-size: 1.7rem; color: #d4a574; margin-bottom: 0.3rem;
}
.met-sub { font-size: 0.78rem; color: #4a3828; }
.streamlit-expanderHeader {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500; font-size: 0.8rem;
    color: #907050 !important;
    background: rgba(16,10,6,0.5) !important;
    border: 1px solid rgba(193,154,107,0.07) !important;
    border-radius: 12px !important;
}

/* ── ABOUT ── */
.about-outer {
    display: flex;
    justify-content: center;
    padding: 1.8rem 2.5rem 0;
}
.about {
    width: 100%;
    max-width: 660px;
}
.about-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; letter-spacing: 3.5px;
    text-transform: uppercase; color: #4a3525;
    margin-bottom: 0.9rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.about-title {
    font-family: 'Instrument Serif', serif;
    font-style: italic;
    font-size: 2.1rem; color: #d4a574; margin-bottom: 1.1rem;
}
.about-body { font-size: 0.87rem; color: #6a5535; line-height: 1.78; margin-bottom: 0.9rem; }
.about-hr { height:1px; background: rgba(193,154,107,0.07); margin: 1.4rem 0; }
.about-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
}
.aitem {
    padding: 1rem 1.1rem;
    background: rgba(16,10,6,0.6);
    border-radius: 14px;
    border: 1px solid rgba(193,154,107,0.06);
    display: flex; align-items: flex-start; gap: 0.7rem;
    min-height: 72px;
}
.aitem-icon {
    flex-shrink: 0;
    width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    margin-top: 2px;
}
.aitem-k {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem; letter-spacing: 2px;
    text-transform: uppercase; color: #4a3525;
    margin-bottom: 0.35rem;
}
.aitem-v { font-size: 0.88rem; color: #c4a882; font-weight: 500; }

/* ── SPINNER ── */
.stSpinner > div { border-top-color: #c19a6b !important; }

/* ── GLOBAL TEXT ── */
.stMarkdown p, p, label { color: #6a5535; }
h1, h2, h3, h4 {
    color: #d4a574 !important;
    font-family: 'Instrument Serif', serif !important;
    font-style: italic;
}

/* ── FOOTER ── */
.site-footer {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-top: 1px solid rgba(193,154,107,0.05);
    margin-top: 3rem;
}
.ft-org {
    font-family: 'Instrument Serif', serif;
    font-style: italic;
    font-size: 0.9rem; color: #4a3525; margin-bottom: 0.3rem;
}
.ft-copy {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem; letter-spacing: 2.5px;
    text-transform: uppercase; color: #2e2015;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────
POD_HERO = (
    '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<ellipse cx="32" cy="38" rx="15" ry="20" fill="url(#hg)"/>'
    '<path d="M32 18 C32 18 27 12 22 10" stroke="#c8974a" stroke-width="1.5" stroke-linecap="round"/>'
    '<path d="M32 18 C32 18 37 13 42 11" stroke="#c8974a" stroke-width="1.5" stroke-linecap="round"/>'
    '<path d="M18 29 C16 32 16 38 17 43" stroke="#e8c478" stroke-width="1.1" stroke-linecap="round" opacity="0.5"/>'
    '<path d="M23 21 C21 24 21 30 22 35" stroke="#e8c478" stroke-width="1.1" stroke-linecap="round" opacity="0.5"/>'
    '<path d="M41 21 C43 24 43 30 42 35" stroke="#e8c478" stroke-width="1.1" stroke-linecap="round" opacity="0.5"/>'
    '<path d="M46 29 C48 32 48 38 47 43" stroke="#e8c478" stroke-width="1.1" stroke-linecap="round" opacity="0.5"/>'
    '<line x1="32" y1="18" x2="32" y2="58" stroke="#8b5c25" stroke-width="1.8" stroke-linecap="round" opacity="0.55"/>'
    '<defs><linearGradient id="hg" x1="17" y1="18" x2="47" y2="58" gradientUnits="userSpaceOnUse">'
    '<stop offset="0%" stop-color="#96692e"/>'
    '<stop offset="45%" stop-color="#c8924a"/>'
    '<stop offset="100%" stop-color="#5e3010"/>'
    '</linearGradient></defs>'
    '</svg>'
)

pod_uri = 'data:image/svg+xml;charset=utf-8,' + POD_HERO.replace('"', "'").replace('#', '%23')

# Bento icons inline
def bento_icon(svg, size="22px"):
    uri = 'data:image/svg+xml;charset=utf-8,' + svg.replace('"', "'").replace('#', '%23')
    return f'<img src="{uri}" width="{size}" height="{size}" style="display:block;margin:0 auto"/>'

ICO_ARCH = (
    '<svg viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect x="2" y="9" width="4" height="10" rx="1" fill="%23c19a6b" opacity="0.4"/>'
    '<rect x="9" y="5" width="4" height="14" rx="1" fill="%23c19a6b" opacity="0.7"/>'
    '<rect x="16" y="2" width="4" height="17" rx="1" fill="%23c19a6b"/>'
    '<line x1="2" y1="20" x2="20" y2="20" stroke="%238b5e2a" stroke-width="1.2"/>'
    '</svg>'
)
ICO_RES = (
    '<svg viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect x="3" y="3" width="16" height="16" rx="2" stroke="%23c19a6b" stroke-width="1.3"/>'
    '<line x1="3" y1="11" x2="19" y2="11" stroke="%238b5e2a" stroke-width="0.8" opacity="0.5"/>'
    '<line x1="11" y1="3" x2="11" y2="19" stroke="%238b5e2a" stroke-width="0.8" opacity="0.5"/>'
    '<circle cx="11" cy="11" r="3.5" fill="none" stroke="%23d4a574" stroke-width="1.2"/>'
    '<circle cx="11" cy="11" r="1.2" fill="%23d4a574"/>'
    '</svg>'
)
ICO_CLS = (
    '<svg viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="7.5" cy="11" r="4.5" fill="none" stroke="%2355cc65" stroke-width="1.3"/>'
    '<circle cx="14.5" cy="11" r="4.5" fill="none" stroke="%23ee5050" stroke-width="1.3"/>'
    '<path d="M11 8.5 C11 8.5 10 11 11 13.5" stroke="%23c19a6b" stroke-width="0.9" stroke-linecap="round" opacity="0.5"/>'
    '</svg>'
)
ICO_YR = (
    '<svg viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect x="3" y="5" width="16" height="15" rx="2" stroke="%23c19a6b" stroke-width="1.3"/>'
    '<line x1="3" y1="9" x2="19" y2="9" stroke="%238b5e2a" stroke-width="1" opacity="0.5"/>'
    '<line x1="7" y1="3" x2="7" y2="7" stroke="%23c19a6b" stroke-width="1.4" stroke-linecap="round"/>'
    '<line x1="15" y1="3" x2="15" y2="7" stroke="%23c19a6b" stroke-width="1.4" stroke-linecap="round"/>'
    '<circle cx="11" cy="14" r="1.5" fill="%23d4a574"/>'
    '</svg>'
)

st.markdown(
    '<div class="hero"><div class="hero-inner">'
    f'<img class="hero-pod" src="{pod_uri}" alt=""/>'
    '<div class="hero-eyebrow">// Comalcalco, Tabasco · México</div>'
    '<div class="hero-title">CacaoVision</div>'
    '<div class="hero-desc">Detección de <em>Moniliophthora roreri</em> en mazorcas de cacao mediante inteligencia artificial.</div>'
    '<div class="bento">'
    f'<div class="bc"><div class="bc-icon">{bento_icon(ICO_ARCH)}</div><div class="bc-val">YOLOv11m</div><div class="bc-lbl">Arquitectura</div></div>'
    f'<div class="bc"><div class="bc-icon">{bento_icon(ICO_RES)}</div><div class="bc-val">512 px</div><div class="bc-lbl">Resolución</div></div>'
    f'<div class="bc"><div class="bc-icon">{bento_icon(ICO_CLS)}</div><div class="bc-val">2 clases</div><div class="bc-lbl">Sano · Moniliasis</div></div>'
    f'<div class="bc"><div class="bc-icon">{bento_icon(ICO_YR)}</div><div class="bc-val">2026</div><div class="bc-lbl">Versión</div></div>'
    '</div>'
    '</div></div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────
# MODELO
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return YOLO(path)
    return None

model = load_model(MODEL_PATH)
if model is None:
    st.error("Modelo no encontrado. Ejecuta `python3 entrenar.py` primero.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────
diag_ico = svg_img(ICO_DIAG, "0.85rem")
met_ico  = svg_img(ICO_METRICS, "0.85rem")
abo_ico  = svg_img(ICO_ABOUT, "0.85rem")

tab1, tab2, tab3 = st.tabs([
    f"  DIAGNÓSTICO  ",
    f"  MÉTRICAS  ",
    f"  ACERCA DE  ",
])


# ─────────────────────────────────────────────────────────────────────
# TAB 1 · DIAGNÓSTICO
# ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="main-pad">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Arrastra o selecciona imágenes de mazorcas de cacao",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_files:
        pod_dim_uri = 'data:image/svg+xml;charset=utf-8,' + POD_HERO.replace('"', "'").replace('#', '%23')
        st.markdown(
            f'<div class="up-prompt">'
            f'<img class="up-pod" src="{pod_dim_uri}" alt=""/>'
            f'<div class="up-text">Arrastra una o varias imágenes de mazorcas<br>para analizar su estado fitosanitario.</div>'
            f'<div class="up-hint">JPG · PNG · WEBP · múltiples archivos</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Analizando imágenes…"):
            time.sleep(0.15)

        n_sano = n_mono = n_doubt = 0
        results_data = []

        for f in uploaded_files:
            img = Image.open(f).convert("RGB")
            res = model.predict(img, verbose=False)[0]
            names = res.names
            probs = res.probs.data.tolist()

            idx_sano = next((k for k, v in names.items() if v.lower() == "sano"), 0)
            idx_mono = next((k for k, v in names.items() if v.lower() != "sano"), 1)

            conf_s = probs[idx_sano] * 100
            conf_m = probs[idx_mono] * 100

            if conf_m >= UMBRAL_MONILIASIS:
                diag = "mono";  n_mono  += 1
            elif conf_s >= UMBRAL_SANO:
                diag = "sano";  n_sano  += 1
            else:
                diag = "doubt"; n_doubt += 1

            results_data.append((img, conf_s, conf_m, diag))

        total = len(uploaded_files)
        sano_icon_s  = svg_img(ICO_SANO,  "0.75rem")
        mono_icon_s  = svg_img(ICO_MONO,  "0.75rem")
        doubt_icon_s = svg_img(ICO_DOUBT, "0.75rem")

        doubt_part = (
            f'<span>{doubt_icon_s} {n_doubt} incierta{"s" if n_doubt != 1 else ""}</span>'
            if n_doubt else ""
        )
        st.markdown(
            f'<div class="res-hdr">'
            f'<span>{total} imagen{"es" if total != 1 else ""}</span>'
            f'<span>{sano_icon_s} {n_sano} sana{"s" if n_sano != 1 else ""}</span>'
            f'<span>{mono_icon_s} {n_mono} con moniliasis</span>'
            f'{doubt_part}'
            f'</div>',
            unsafe_allow_html=True,
        )

        cols = st.columns(3)
        for i, (img, conf_s, conf_m, diag) in enumerate(results_data):
            with cols[i % 3]:
                if diag == "sano":
                    glow    = "g-sano"
                    badge   = f'<span class="badge b-sano">{svg_img(ICO_SANO,"0.8rem")} &nbsp;SANO</span>'
                elif diag == "mono":
                    glow    = "g-mono"
                    badge   = f'<span class="badge b-mono">{svg_img(ICO_MONO,"0.8rem")} &nbsp;MONILIASIS</span>'
                else:
                    glow    = "g-doubt"
                    badge   = f'<span class="badge b-doubt">{svg_img(ICO_DOUBT,"0.8rem")} &nbsp;INCIERTO</span>'

                st.markdown(f'<div class="rcard {glow}">', unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                st.markdown(
                    f'<div class="rcard-body">'
                    f'<div class="diag">{badge}</div>'
                    f'<div class="conf-wrap">'
                    f'<div class="cbar"><span class="cbar-lbl">SANO</span>'
                    f'<div class="cbar-track"><div class="fill-s" style="width:{conf_s:.1f}%"></div></div>'
                    f'<span class="cbar-pct">{conf_s:.1f}%</span></div>'
                    f'<div class="cbar"><span class="cbar-lbl">MONO</span>'
                    f'<div class="cbar-track"><div class="fill-m" style="width:{conf_m:.1f}%"></div></div>'
                    f'<span class="cbar-pct">{conf_m:.1f}%</span></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 2 · MÉTRICAS
# ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="main-pad">', unsafe_allow_html=True)
    st.markdown('<div class="met-hdr"><div class="met-title">Resultados del Entrenamiento</div><div class="met-sub">Gráficas y matrices generadas por YOLOv11 durante el fine-tuning del modelo.</div></div>', unsafe_allow_html=True)

    base = "runs/classify/resultados_cacao/modelo_refinado"
    plots = [
        ("Curvas de Entrenamiento",              f"{base}/results.png"),
        ("Matriz de Confusión",                  f"{base}/confusion_matrix.png"),
        ("Matriz de Confusión Normalizada",      f"{base}/confusion_matrix_normalized.png"),
    ]
    for title, path in plots:
        if os.path.exists(path):
            with st.expander(title, expanded=True):
                st.image(path, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 3 · ACERCA DE
# ─────────────────────────────────────────────────────────────────────
with tab3:
    compass_s = svg_img(ICO_ABOUT, "0.75rem")
    arch_i  = svg_img(ICO_ARCH.replace('%23','#'), "1.1rem")
    res_i   = svg_img(ICO_RES.replace('%23','#'),  "1.1rem")
    cls_i   = svg_img(ICO_CLS.replace('%23','#'),  "1.1rem")
    yr_i    = svg_img(ICO_YR.replace('%23','#'),   "1.1rem")

    st.markdown(
        '<div class="about-outer"><div class="about">'
        f'<div class="about-eyebrow">{compass_s} Proyecto Capstone · 2026</div>'
        '<div class="about-title">Sobre CacaoVision</div>'
        '<div class="about-body">Sistema de clasificación binaria para la detección temprana de <em>Moniliophthora roreri</em> en mazorcas de cacao. Diseñado para apoyar a productores de la región cacaotera de Comalcalco, Tabasco, reduciendo pérdidas de cosecha mediante diagnóstico visual automatizado.</div>'
        '<div class="about-body">El modelo aplica fine-tuning sobre pesos ImageNet de YOLOv11m-cls, especializándolo en reconocer patrones visuales de moniliasis: decoloración parda, textura pulverulenta blanca y necrosis superficial en la mazorca.</div>'
        '<div class="about-hr"></div>'
        '<div class="about-grid">'
        f'<div class="aitem"><div class="aitem-icon">{arch_i}</div><div><div class="aitem-k">Arquitectura</div><div class="aitem-v">YOLOv11m-cls</div></div></div>'
        f'<div class="aitem"><div class="aitem-icon">{res_i}</div><div><div class="aitem-k">Resolución entrada</div><div class="aitem-v">512 × 512 px</div></div></div>'
        f'<div class="aitem"><div class="aitem-icon">{cls_i}</div><div><div class="aitem-k">Clases</div><div class="aitem-v">Sano · Moniliasis</div></div></div>'
        f'<div class="aitem"><div class="aitem-icon">{yr_i}</div><div><div class="aitem-k">Región</div><div class="aitem-v">Comalcalco, Tabasco</div></div></div>'
        '</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="site-footer">'
    '<div class="ft-org">Proyecto Capstone</div>'
    '<div class="ft-copy">Visión por Computadora · Comalcalco, Tabasco · 2026</div>'
    '</div>',
    unsafe_allow_html=True,
)
