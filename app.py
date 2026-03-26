"""
app.py — CacaoVision 2.0
Dashboard de diagnóstico de moniliasis en mazorcas de cacao.
Modelo: YOLOv11m-cls · Streamlit · Comalcalco, Tabasco · 2026
"""

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os, time, base64
from io import BytesIO

st.set_page_config(
    page_title="CacaoVision · Diagnóstico Inteligente",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

UMBRAL_SANO       = 92.0
UMBRAL_MONILIASIS = 10.0
MODEL_PATH = "runs/classify/resultados_cacao/modelo_refinado/weights/best.pt"


def img_to_b64(pil_img, fmt="JPEG", max_size=700):
    img = pil_img.copy()
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format=fmt, quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


# ═════════════════════════════════════════════════════════════════════
# CSS
# ═════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #1b120c;
    --bg-warm: #231810;
    --bg-card: #2a1d14;
    --bg-card-hover: #33241a;
    --border: rgba(255,255,255,0.06);
    --border-hover: rgba(255,255,255,0.12);
    --caramel: #d4a574;
    --caramel-light: #e8c49a;
    --caramel-dark: #a07848;
    --white: #faf8f5;
    --white-soft: #e8e0d8;
    --white-dim: #b8a898;
    --white-muted: #7a6a5a;
    --green: #4ade80;
    --green-dim: #22c55e;
    --red: #f87171;
    --red-dim: #ef4444;
    --amber: #fbbf24;
    --radius: 16px;
    --radius-sm: 10px;
}

/* ── RESET ── */
#MainMenu, header, footer { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: var(--bg); }

html, body, [class*="css"] {
    font-family: 'Poppins', -apple-system, sans-serif;
    color: var(--white-soft);
}
.stMarkdown p, p { color: var(--white-dim); }

/* ══════════════════════════════════════
   ANIMATED BACKGROUND
   ══════════════════════════════════════ */

/* Warm ambient glow */
.bg-layer {
    position: fixed; inset: 0; z-index: 0;
    pointer-events: none; overflow: hidden;
}
.glow {
    position: absolute;
    border-radius: 50%;
    filter: blur(120px);
    opacity: 0.5;
}
.glow-1 {
    width: 50vmax; height: 50vmax;
    top: -20%; left: -10%;
    background: rgba(120,70,30,0.15);
    animation: glowDrift1 22s ease-in-out infinite alternate;
}
.glow-2 {
    width: 40vmax; height: 40vmax;
    bottom: -15%; right: -10%;
    background: rgba(90,50,20,0.12);
    animation: glowDrift2 28s ease-in-out infinite alternate;
}
@keyframes glowDrift1 {
    0%   { transform: translate(0, 0); }
    100% { transform: translate(5vw, 8vh); }
}
@keyframes glowDrift2 {
    0%   { transform: translate(0, 0); }
    100% { transform: translate(-6vw, -5vh); }
}

/* Floating cacao beans */
.beans {
    position: fixed; inset: 0; z-index: 0;
    pointer-events: none; overflow: hidden;
}
.bean {
    position: absolute;
    bottom: -30px;
    opacity: 0;
    animation: beanFloat linear infinite;
}
.bean svg { filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3)); }
.bean:nth-child(1) { left: 8%;  animation-duration: 20s; animation-delay: 0s; }
.bean:nth-child(2) { left: 22%; animation-duration: 26s; animation-delay: 4s; }
.bean:nth-child(3) { left: 40%; animation-duration: 23s; animation-delay: 2s; }
.bean:nth-child(4) { left: 58%; animation-duration: 28s; animation-delay: 6s; }
.bean:nth-child(5) { left: 75%; animation-duration: 21s; animation-delay: 1s; }
.bean:nth-child(6) { left: 90%; animation-duration: 25s; animation-delay: 8s; }

@keyframes beanFloat {
    0%   { transform: translateY(0) rotate(0deg) scale(0.7); opacity: 0; }
    5%   { opacity: 0.12; }
    50%  { opacity: 0.08; }
    95%  { opacity: 0.12; }
    100% { transform: translateY(-110vh) rotate(360deg) scale(0.7); opacity: 0; }
}

/* Mesoamerican greca pattern — top border */
.greca {
    position: relative; z-index: 1;
    width: 100%; height: 6px;
    overflow: hidden;
}
.greca-inner {
    width: 200%; height: 100%;
    background: repeating-linear-gradient(
        90deg,
        var(--caramel) 0px, var(--caramel) 6px,
        transparent 6px, transparent 10px,
        var(--caramel-dark) 10px, var(--caramel-dark) 12px,
        transparent 12px, transparent 18px,
        var(--caramel) 18px, var(--caramel) 20px,
        transparent 20px, transparent 30px
    );
    opacity: 0.15;
    animation: grecaMove 40s linear infinite;
}
@keyframes grecaMove {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

/* ══════════════════════════════════════
   HERO
   ══════════════════════════════════════ */
.hero {
    position: relative; z-index: 1;
    text-align: center;
    padding: 3.5rem 1.5rem 2.5rem;
}
.hero-location {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 1.1rem;
    border-radius: 50px;
    background: rgba(212,165,116,0.08);
    border: 1px solid rgba(212,165,116,0.15);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--caramel);
    margin-bottom: 1.8rem;
}
.hero-location .live-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px var(--green);
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: clamp(2.8rem, 7vw, 5rem);
    line-height: 1.05;
    color: var(--white);
    margin: 0 0 0.6rem;
    letter-spacing: -0.01em;
}
.hero-sub {
    font-family: 'Poppins', sans-serif;
    font-weight: 300;
    font-size: clamp(0.85rem, 2vw, 1.15rem);
    color: var(--white-dim);
    margin: 0 auto 2.5rem;
    max-width: 520px;
    line-height: 1.7;
}
.hero-sub em {
    color: var(--caramel-light);
    font-style: italic;
}

/* Stats strip */
.stats {
    display: flex;
    justify-content: center;
    max-width: 720px;
    margin: 0 auto;
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--border);
    background: var(--bg-card);
}
.stat {
    flex: 1;
    padding: 1.2rem 0.8rem;
    text-align: center;
    border-right: 1px solid var(--border);
    transition: background 0.3s;
}
.stat:last-child { border-right: none; }
.stat:hover { background: var(--bg-card-hover); }
.stat-val {
    font-family: 'Poppins', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--white);
}
.stat-lbl {
    font-size: 0.6rem;
    font-weight: 500;
    color: var(--white-muted);
    margin-top: 0.2rem;
    letter-spacing: 0.5px;
}

/* ══════════════════════════════════════
   TABS
   ══════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    justify-content: center;
    background: transparent;
    border-bottom: 1px solid var(--border);
    padding: 0; margin: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Poppins', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--white-muted) !important;
    padding: 1rem 2rem;
    border: none !important;
    background: transparent !important;
    transition: color 0.3s;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--caramel) !important; }
.stTabs [aria-selected="true"] {
    color: var(--white) !important;
    border-bottom: 2px solid var(--caramel) !important;
}
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--caramel) !important; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ══════════════════════════════════════
   CONTENT
   ══════════════════════════════════════ */
.content {
    position: relative; z-index: 1;
    max-width: 1200px;
    margin: 0 auto;
    padding: 1.5rem 2rem 0;
}

/* ── UPLOADER ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card);
    border: 1.5px dashed rgba(212,165,116,0.18);
    border-radius: var(--radius);
    padding: 0.5rem;
    transition: all 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(212,165,116,0.35);
    background: var(--bg-card-hover);
}
[data-testid="stFileUploader"] label { color: var(--white-dim) !important; font-family: 'Poppins' !important; }
[data-testid="stFileUploader"] small { color: var(--white-muted) !important; }
[data-testid="stFileUploaderDropzone"] {
    background: rgba(27,18,12,0.5) !important;
    border-color: rgba(212,165,116,0.08) !important;
}

/* ── EMPTY STATE ── */
.empty {
    text-align: center;
    padding: 4rem 1rem 3rem;
}
.empty-pod {
    width: 72px; height: 72px;
    margin: 0 auto 1.2rem;
    opacity: 0.18;
    animation: floatPod 5s ease-in-out infinite;
}
@keyframes floatPod {
    0%, 100% { transform: translateY(0) rotate(-2deg); }
    50% { transform: translateY(-8px) rotate(2deg); }
}
.empty-msg {
    font-size: 0.95rem;
    font-weight: 400;
    color: var(--white-muted);
    line-height: 1.8;
}
.empty-hint {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--white-muted);
    opacity: 0.5;
    margin-top: 0.6rem;
    letter-spacing: 1.5px;
}

/* ── SUMMARY ── */
.summary {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 1.5rem;
    padding: 0.8rem 0;
    margin-bottom: 1.2rem;
    border-bottom: 1px solid var(--border);
}
.sum-item {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--white-dim);
}
.sum-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.d-all   { background: var(--caramel); }
.d-sano  { background: var(--green);   box-shadow: 0 0 6px rgba(74,222,128,0.4); }
.d-mono  { background: var(--red);     box-shadow: 0 0 6px rgba(248,113,113,0.4); }
.d-doubt { background: var(--amber);   box-shadow: 0 0 6px rgba(251,191,36,0.4); }

/* ══════════════════════════════════════
   RESULT CARDS
   ══════════════════════════════════════ */
.rcard {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    margin-bottom: 1rem;
    transition: all 0.35s ease;
}
.rcard:hover {
    transform: translateY(-3px);
    border-color: var(--border-hover);
    box-shadow: 0 12px 35px rgba(0,0,0,0.25);
}
.rcard.rc-sano:hover  { box-shadow: 0 12px 35px rgba(74,222,128,0.07); }
.rcard.rc-mono:hover  { box-shadow: 0 12px 35px rgba(248,113,113,0.09); }
.rcard.rc-doubt:hover { box-shadow: 0 12px 35px rgba(251,191,36,0.07); }

.rcard-img {
    position: relative;
    background: #120c08;
    overflow: hidden;
}
.rcard-img img {
    width: 100%;
    height: auto;
    display: block;
    transition: transform 0.4s;
}
.rcard:hover .rcard-img img { transform: scale(1.02); }

/* Scan effect */
.scan {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--caramel-light), transparent);
    opacity: 0;
    animation: scanAnim 2s ease-in-out 0.3s forwards;
}
@keyframes scanAnim {
    0%   { top: 0; opacity: 0; }
    8%   { opacity: 0.6; }
    92%  { opacity: 0.6; }
    100% { top: 100%; opacity: 0; }
}

/* Status dot */
.status-dot {
    position: absolute;
    top: 8px; right: 8px;
    width: 10px; height: 10px;
    border-radius: 50%;
    z-index: 2;
    border: 2px solid rgba(0,0,0,0.3);
}
.sd-sano  { background: var(--green); box-shadow: 0 0 10px var(--green); }
.sd-mono  { background: var(--red);   box-shadow: 0 0 10px var(--red); }
.sd-doubt { background: var(--amber); box-shadow: 0 0 10px var(--amber); }

/* Card body */
.rcard-body {
    padding: 0.9rem 1rem;
}

/* Badge */
.dbadge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.9rem;
    border-radius: 50px;
    font-family: 'Poppins', sans-serif;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.db-sano {
    background: rgba(74,222,128,0.1);
    color: var(--green);
    border: 1px solid rgba(74,222,128,0.2);
}
.db-mono {
    background: rgba(248,113,113,0.1);
    color: var(--red);
    border: 1px solid rgba(248,113,113,0.2);
}
.db-doubt {
    background: rgba(251,191,36,0.1);
    color: var(--amber);
    border: 1px solid rgba(251,191,36,0.2);
}

/* Circular gauges */
.gauges {
    display: flex;
    justify-content: center;
    gap: 1rem;
    padding: 0.8rem 0 0.2rem;
}
.gauge {
    position: relative;
    width: 60px; height: 60px;
}
.gauge svg {
    transform: rotate(-90deg);
    width: 60px; height: 60px;
}
.g-bg {
    fill: none;
    stroke: rgba(255,255,255,0.05);
    stroke-width: 3.5;
}
.g-fill {
    fill: none;
    stroke-width: 3.5;
    stroke-linecap: round;
    transition: stroke-dashoffset 1s cubic-bezier(0.4,0,0.2,1);
}
.gf-s { stroke: var(--green); }
.gf-m { stroke: var(--red); }
.g-center {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%,-50%);
    text-align: center;
}
.g-pct {
    font-family: 'Poppins', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    line-height: 1;
}
.gp-s { color: var(--green); }
.gp-m { color: var(--red); }
.g-lbl {
    font-size: 0.42rem;
    font-weight: 500;
    color: var(--white-muted);
    margin-top: 1px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ══════════════════════════════════════
   METRICS TAB
   ══════════════════════════════════════ */
.met-hdr { padding: 0.5rem 0 1.2rem; }
.met-title {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: 1.8rem;
    color: var(--white);
    margin-bottom: 0.3rem;
}
.met-sub { font-size: 0.85rem; color: var(--white-muted); }
.streamlit-expanderHeader {
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    color: var(--white-dim) !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

/* ══════════════════════════════════════
   ABOUT TAB
   ══════════════════════════════════════ */
.about {
    max-width: 700px;
    margin: 0 auto;
    padding: 1rem 0 2rem;
}
.about-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--caramel);
    margin-bottom: 0.8rem;
}
.about-h {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: clamp(1.8rem, 4vw, 2.4rem);
    color: var(--white);
    margin-bottom: 1.3rem;
    line-height: 1.2;
}
.about-p {
    font-size: 0.9rem;
    font-weight: 400;
    color: var(--white-dim);
    line-height: 1.9;
    margin-bottom: 1rem;
}
.about-hr {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1.8rem 0;
}
.specs {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.8rem;
}
.spec {
    padding: 1.1rem;
    background: var(--bg-card);
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    transition: all 0.3s;
}
.spec:hover {
    border-color: var(--border-hover);
    background: var(--bg-card-hover);
    transform: translateY(-2px);
}
.spec-k {
    font-size: 0.55rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--white-muted);
    margin-bottom: 0.4rem;
}
.spec-v {
    font-size: 1rem;
    font-weight: 700;
    color: var(--caramel-light);
}

/* ══════════════════════════════════════
   FOOTER
   ══════════════════════════════════════ */
.foot {
    position: relative; z-index: 1;
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    margin-top: 2rem;
    border-top: 1px solid var(--border);
}
.foot-brand {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--caramel-dark);
    margin-bottom: 0.3rem;
}
.foot-copy {
    font-size: 0.55rem;
    font-weight: 500;
    color: var(--white-muted);
    opacity: 0.5;
    letter-spacing: 1.5px;
}

/* ── MISC ── */
.stSpinner > div { border-top-color: var(--caramel) !important; }
h1, h2, h3, h4 {
    color: var(--white) !important;
    font-family: 'Poppins', sans-serif !important;
}

/* ══════════════════════════════════════
   RESPONSIVE
   ══════════════════════════════════════ */
@media (max-width: 768px) {
    .hero { padding: 2.5rem 1rem 2rem; }
    .hero-title { font-size: 2.4rem; }
    .hero-sub { font-size: 0.85rem; }
    .stats { flex-wrap: wrap; }
    .stat { min-width: 45%; }
    .content { padding: 1rem 1rem 0; }
    .summary { gap: 0.8rem; }
    .sum-item { font-size: 0.65rem; }
    .gauges { gap: 0.6rem; }
    .gauge { width: 52px; height: 52px; }
    .gauge svg { width: 52px; height: 52px; }
    .g-pct { font-size: 0.6rem; }
    .specs { grid-template-columns: 1fr; }
    .about-h { font-size: 1.6rem; }
    .stTabs [data-baseweb="tab"] {
        padding: 0.8rem 1rem;
        font-size: 0.6rem;
        letter-spacing: 1.5px;
    }
}
@media (max-width: 480px) {
    .hero { padding: 2rem 0.8rem 1.5rem; }
    .hero-title { font-size: 2rem; }
    .hero-location { font-size: 0.5rem; padding: 0.3rem 0.8rem; }
    .stats { flex-direction: column; }
    .stat { border-right: none; border-bottom: 1px solid var(--border); }
    .stat:last-child { border-bottom: none; }
    .content { padding: 0.8rem 0.6rem 0; }
    .rcard-body { padding: 0.7rem 0.8rem; }
    .summary { flex-direction: column; gap: 0.5rem; }
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# BACKGROUND — glows + floating cacao beans
# ═════════════════════════════════════════════════════════════════════
BEAN_SVG = (
    '<svg width="20" height="28" viewBox="0 0 20 28" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<ellipse cx="10" cy="14" rx="8" ry="12" fill="#3a2510"/>'
    '<line x1="10" y1="2" x2="10" y2="26" stroke="#5a3a1a" stroke-width="0.8" opacity="0.5"/>'
    '<path d="M4 8 C3 10 3 14 4 18" stroke="#4a3015" stroke-width="0.6" opacity="0.4"/>'
    '<path d="M16 8 C17 10 17 14 16 18" stroke="#4a3015" stroke-width="0.6" opacity="0.4"/>'
    '</svg>'
).replace('"', "'").replace('#', '%23')

st.markdown(
    '<div class="bg-layer">'
    '<div class="glow glow-1"></div>'
    '<div class="glow glow-2"></div>'
    '</div>'
    '<div class="beans">'
    + ''.join(
        f'<div class="bean"><svg width="20" height="28" viewBox="0 0 20 28" fill="none" xmlns="http://www.w3.org/2000/svg">'
        f'<ellipse cx="10" cy="14" rx="8" ry="12" fill="%233a2510"/>'
        f'<line x1="10" y1="2" x2="10" y2="26" stroke="%235a3a1a" stroke-width="0.8" opacity="0.5"/>'
        f'<path d="M4 8 C3 10 3 14 4 18" stroke="%234a3015" stroke-width="0.6" opacity="0.4"/>'
        f'<path d="M16 8 C17 10 17 14 16 18" stroke="%234a3015" stroke-width="0.6" opacity="0.4"/>'
        f'</svg></div>'
        for _ in range(6)
    )
    + '</div>',
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════════════
# GRECA + HERO
# ═════════════════════════════════════════════════════════════════════
st.markdown('<div class="greca"><div class="greca-inner"></div></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="hero">'
    '<div class="hero-location"><span class="live-dot"></span> Comalcalco, Tabasco · México</div>'
    '<div class="hero-title">CacaoVision</div>'
    '<div class="hero-sub">Detección inteligente de <em>Moniliophthora roreri</em> en mazorcas de cacao mediante visión por computadora.</div>'
    '<div class="stats">'
    '<div class="stat"><div class="stat-val">YOLOv11m</div><div class="stat-lbl">Arquitectura</div></div>'
    '<div class="stat"><div class="stat-val">384px</div><div class="stat-lbl">Resolución</div></div>'
    '<div class="stat"><div class="stat-val">Binario</div><div class="stat-lbl">Sano · Moniliasis</div></div>'
    '<div class="stat"><div class="stat-val">2026</div><div class="stat-lbl">Versión</div></div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="greca"><div class="greca-inner"></div></div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# MODELO
# ═════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return YOLO(path)
    return None

model = load_model(MODEL_PATH)
if model is None:
    st.error("Modelo no encontrado. Ejecuta `python3 entrenar.py` primero.")
    st.stop()


# ═════════════════════════════════════════════════════════════════════
# GAUGE HELPER
# ═════════════════════════════════════════════════════════════════════
def gauge(pct, label, cls):
    r = 26
    circ = 2 * 3.14159 * r
    off = circ * (1 - pct / 100)
    return (
        f'<div class="gauge">'
        f'<svg viewBox="0 0 60 60">'
        f'<circle class="g-bg" cx="30" cy="30" r="{r}"/>'
        f'<circle class="g-fill gf-{cls}" cx="30" cy="30" r="{r}" '
        f'stroke-dasharray="{circ:.1f}" stroke-dashoffset="{off:.1f}"/>'
        f'</svg>'
        f'<div class="g-center">'
        f'<div class="g-pct gp-{cls}">{pct:.0f}%</div>'
        f'<div class="g-lbl">{label}</div>'
        f'</div></div>'
    )


# ═════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "  DIAGNÓSTICO  ",
    "  MÉTRICAS  ",
    "  ACERCA DE  ",
])


# ─────────────────────────────────────────────────────────────────────
# TAB 1 — DIAGNÓSTICO
# ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="content">', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Arrastra o selecciona imágenes de mazorcas de cacao",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded:
        pod_svg_raw = (
            '<svg viewBox="0 0 64 80" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<ellipse cx="32" cy="48" rx="18" ry="26" fill="#d4a574" opacity="0.6"/>'
            '<path d="M32 22 C32 22 25 14 18 11" stroke="#e8c49a" stroke-width="2" stroke-linecap="round"/>'
            '<path d="M32 22 C32 22 39 15 46 12" stroke="#e8c49a" stroke-width="2" stroke-linecap="round"/>'
            '<line x1="32" y1="22" x2="32" y2="74" stroke="#a07848" stroke-width="1.8" opacity="0.4"/>'
            '<path d="M16 38 C14 44 14 54 16 62" stroke="#e8c49a" stroke-width="1.2" opacity="0.25"/>'
            '<path d="M48 38 C50 44 50 54 48 62" stroke="#e8c49a" stroke-width="1.2" opacity="0.25"/>'
            '</svg>'
        )
        pod_uri = "data:image/svg+xml;charset=utf-8," + pod_svg_raw.replace('"', "'").replace('#', '%23')
        st.markdown(
            f'<div class="empty">'
            f'<img class="empty-pod" src="{pod_uri}" alt=""/>'
            f'<div class="empty-msg">Arrastra una o varias imágenes de mazorcas<br>para analizar su estado fitosanitario</div>'
            f'<div class="empty-hint">JPG · PNG · WEBP · múltiples archivos</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Analizando…"):
            time.sleep(0.1)

        n_s = n_m = n_d = 0
        results = []

        for f in uploaded:
            img = Image.open(f).convert("RGB")
            res = model.predict(img, verbose=False)[0]
            names = res.names
            probs = res.probs.data.tolist()

            idx_s = next((k for k, v in names.items() if v.lower() == "sano"), 0)
            idx_m = next((k for k, v in names.items() if v.lower() != "sano"), 1)

            cs = probs[idx_s] * 100
            cm = probs[idx_m] * 100

            if cm >= UMBRAL_MONILIASIS:
                d = "mono"; n_m += 1
            elif cs >= UMBRAL_SANO:
                d = "sano"; n_s += 1
            else:
                d = "doubt"; n_d += 1

            results.append((img, cs, cm, d, f.name))

        total = len(uploaded)
        parts = [f'<div class="sum-item"><span class="sum-dot d-all"></span>{total} imagen{"es" if total != 1 else ""}</div>']
        if n_s: parts.append(f'<div class="sum-item"><span class="sum-dot d-sano"></span>{n_s} sana{"s" if n_s != 1 else ""}</div>')
        if n_m: parts.append(f'<div class="sum-item"><span class="sum-dot d-mono"></span>{n_m} moniliasis</div>')
        if n_d: parts.append(f'<div class="sum-item"><span class="sum-dot d-doubt"></span>{n_d} incierta{"s" if n_d != 1 else ""}</div>')
        st.markdown(f'<div class="summary">{"".join(parts)}</div>', unsafe_allow_html=True)

        cols = st.columns(3)
        for i, (img, cs, cm, d, fn) in enumerate(results):
            with cols[i % 3]:
                if d == "sano":
                    rc, sd = "rc-sano", "sd-sano"
                    badge = '<span class="dbadge db-sano">&#9679; Sano</span>'
                elif d == "mono":
                    rc, sd = "rc-mono", "sd-mono"
                    badge = '<span class="dbadge db-mono">&#9679; Moniliasis</span>'
                else:
                    rc, sd = "rc-doubt", "sd-doubt"
                    badge = '<span class="dbadge db-doubt">&#9679; Incierto</span>'

                b64 = img_to_b64(img)
                gs = gauge(cs, "Sano", "s")
                gm = gauge(cm, "Mono", "m")

                st.markdown(
                    f'<div class="rcard {rc}">'
                    f'<div class="rcard-img">'
                    f'<img src="{b64}" alt="{fn}"/>'
                    f'<div class="scan"></div>'
                    f'<div class="status-dot {sd}"></div>'
                    f'</div>'
                    f'<div class="rcard-body">'
                    f'<div style="text-align:center;margin-bottom:0.5rem">{badge}</div>'
                    f'<div class="gauges">{gs}{gm}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 2 — MÉTRICAS
# ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.markdown(
        '<div class="met-hdr">'
        '<div class="met-title">Resultados del Entrenamiento</div>'
        '<div class="met-sub">Gráficas y matrices generadas durante el fine-tuning.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    base = "runs/classify/resultados_cacao/modelo_refinado"
    metrics_info = [
        (
            "Curvas de Entrenamiento",
            f"{base}/results.png",
            "Muestra la evolución del modelo época por época. "
            "<b>Train Loss</b> (pérdida de entrenamiento) debe descender de forma estable — indica que el modelo aprende a distinguir las clases. "
            "<b>Val Accuracy Top-1</b> es el porcentaje de aciertos en imágenes que el modelo nunca vio durante el entrenamiento; "
            "valores cercanos al 100% indican alta capacidad de generalización. "
            "Si la val loss sube mientras la train loss baja, el modelo estaría memorizando en lugar de aprender (<em>overfitting</em>). "
            "Lo ideal es que ambas curvas converjan y se mantengan estables."
        ),
        (
            "Matriz de Confusión",
            f"{base}/confusion_matrix.png",
            "Tabla que cruza las predicciones del modelo contra la realidad. "
            "Cada celda muestra <b>cuántas imágenes</b> fueron clasificadas en cada combinación. "
            "La diagonal principal (arriba-izq → abajo-der) representa los <b>aciertos</b>: "
            "imágenes de moniliasis correctamente detectadas y mazorcas sanas correctamente identificadas. "
            "Las celdas fuera de la diagonal son <b>errores</b>: falsos positivos (sano clasificado como enfermo) "
            "y falsos negativos (enfermo clasificado como sano). Lo ideal es que solo la diagonal tenga valores."
        ),
        (
            "Matriz de Confusión Normalizada",
            f"{base}/confusion_matrix_normalized.png",
            "Misma matriz pero expresada en <b>porcentajes por clase</b> (cada fila suma 100%). "
            "Esto permite comparar el rendimiento entre clases sin importar cuántas imágenes hay de cada una. "
            "Un valor de <b>1.00</b> en la diagonal significa que el 100% de las imágenes de esa clase fueron clasificadas correctamente. "
            "Es la métrica más confiable para evaluar si el modelo funciona igual de bien para mazorcas sanas que para mazorcas con moniliasis."
        ),
    ]

    for title, path, desc in metrics_info:
        if os.path.exists(path):
            with st.expander(title, expanded=True):
                st.image(path, use_container_width=True)
                st.markdown(
                    f'<div style="padding:0.8rem 0.2rem 0.3rem;font-size:0.82rem;'
                    f'line-height:1.75;color:var(--white-dim);">{desc}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 3 — ACERCA DE
# ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(
        '<div class="content"><div class="about">'
        '<div class="about-tag">Proyecto Capstone · 2026</div>'
        '<div class="about-h">Sobre CacaoVision</div>'
        '<div class="about-p">'
        'Sistema de clasificación binaria para la detección temprana de '
        '<em>Moniliophthora roreri</em> en mazorcas de cacao. Diseñado para apoyar '
        'a productores de la región cacaotera de Comalcalco, Tabasco — cuna del chocolate '
        'en Mesoamérica y sitio de la ciudad Maya más occidental.'
        '</div>'
        '<div class="about-p">'
        'El modelo aplica fine-tuning sobre pesos ImageNet de YOLOv11m-cls, '
        'especializándolo en reconocer patrones visuales de moniliasis: decoloración parda, '
        'textura pulverulenta blanca y necrosis superficial.'
        '</div>'
        '<div class="about-hr"></div>'
        '<div class="specs">'
        '<div class="spec"><div class="spec-k">Arquitectura</div><div class="spec-v">YOLOv11m-cls</div></div>'
        '<div class="spec"><div class="spec-k">Resolución</div><div class="spec-v">384 × 384 px</div></div>'
        '<div class="spec"><div class="spec-k">Clases</div><div class="spec-v">Sano · Moniliasis</div></div>'
        '<div class="spec"><div class="spec-k">Región</div><div class="spec-v">Comalcalco, Tabasco</div></div>'
        '<div class="spec"><div class="spec-k">Optimizador</div><div class="spec-v">AdamW · Cosine LR</div></div>'
        '<div class="spec"><div class="spec-k">Framework</div><div class="spec-v">Ultralytics · PyTorch</div></div>'
        '</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="greca"><div class="greca-inner"></div></div>'
    '<div class="foot">'
    '<div class="foot-brand">CacaoVision</div>'
    '<div class="foot-copy">Visión por Computadora · Comalcalco, Tabasco · 2026</div>'
    '</div>',
    unsafe_allow_html=True,
)
