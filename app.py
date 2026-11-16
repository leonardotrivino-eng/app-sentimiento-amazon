"""App Streamlit – Análisis de sentimiento en reseñas de Amazon (DistilBERT fine-tuned)."""
import json
import os
import re
import sqlite3

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sentimiento de reseñas Amazon", page_icon="🛒", layout="wide")

# Carpeta local con el modelo exportado desde el notebook, o un repo de Hugging Face Hub
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "modelo_sentimiento")
DB_PATH = os.getenv("DB_PATH", "reviews.db")
MAX_LEN = 128
LABELS = ["negativo", "neutral", "positivo"]
EMOJI = {"negativo": "😠", "neutral": "😐", "positivo": "😀"}


@st.cache_resource(show_spinner="Cargando modelo...")
def load_model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_SOURCE)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE).eval()
    return tok, model, torch


def predict(text: str) -> dict:
    tok, model, torch = load_model()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0].tolist()
    id2label = model.config.id2label
    return {id2label[i]: probs[i] for i in range(len(probs))}


def run_select(sql: str) -> pd.DataFrame:
    if not re.match(r"^\s*select\b", sql, re.I) or ";" in sql.strip().rstrip(";"):
        raise ValueError("Solo se permite una consulta SELECT.")
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        return pd.read_sql(sql, con)
    finally:
        con.close()


st.title("🛒 Análisis de sentimiento de reseñas de Amazon")
st.caption("ACA N° 2 · Aprendizaje Automático · CUN — DistilBERT con fine-tuning + base de datos SQL")

tab_pred, tab_data, tab_metrics, tab_about = st.tabs(
    ["🔮 Predicción", "🗄️ Datos (SQL)", "📈 Métricas", "ℹ️ Acerca del modelo"]
)

# ---------------- Predicción ----------------
with tab_pred:
    examples = {
        "😀 Positiva": "Amazing tablet, the battery lasts all day and the screen is gorgeous. Totally worth it!",
        "😐 Neutral": "It works fine, nothing special. Does what it says but I expected a bit more.",
        "😠 Negativa": "Stopped working after two weeks and customer service was useless. Waste of money.",
    }
    cols = st.columns(len(examples))
    for col, (name, txt) in zip(cols, examples.items()):
        if col.button(name, use_container_width=True):
            st.session_state["texto"] = txt

    texto = st.text_area("Escribe una reseña (en inglés):", key="texto", height=140,
                         placeholder="e.g. Great product, fast shipping...")

    if st.button("Analizar sentimiento", type="primary"):
        if not texto.strip():
            st.warning("Escribe una reseña primero.")
        elif not os.path.isdir(MODEL_SOURCE) and "/" not in MODEL_SOURCE:
            st.error(f"No encuentro el modelo en '{MODEL_SOURCE}'. Descomprime artefactos.zip junto a app.py.")
        else:
            res = predict(texto)
            top = max(res, key=res.get)
            c1, c2 = st.columns([1, 2])
            c1.metric("Predicción", f"{EMOJI.get(top, '')} {top}", f"{res[top]:.1%} de confianza")
            c2.bar_chart(pd.DataFrame({"probabilidad": res}).reindex(LABELS))

# ---------------- Datos SQL ----------------
with tab_data:
    if not os.path.exists(DB_PATH):
        st.info("Coloca `reviews.db` junto a app.py para explorar los datos.")
    else:
        presets = {
            "Distribución de sentimientos": "SELECT sentimiento, COUNT(*) AS n FROM resenas GROUP BY sentimiento",
            "Peores productos (≥100 reseñas)": (
                "SELECT p.nombre, COUNT(*) AS n_resenas, ROUND(AVG(r.rating),2) AS rating_prom "
                "FROM resenas r JOIN productos p USING(producto_id) "
                "GROUP BY p.producto_id HAVING n_resenas >= 100 ORDER BY rating_prom ASC LIMIT 10"
            ),
            "Reseñas negativas más útiles": (
                "SELECT titulo, texto, utiles FROM resenas WHERE sentimiento='negativo' "
                "ORDER BY utiles DESC LIMIT 10"
            ),
        }
        choice = st.selectbox("Consulta predefinida", list(presets))
        sql = st.text_area("SQL (solo SELECT)", presets[choice], height=110)
        if st.button("Ejecutar consulta"):
            try:
                df = run_select(sql)
                st.dataframe(df, use_container_width=True)
                if df.shape[1] == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                    st.bar_chart(df.set_index(df.columns[0]))
            except Exception as e:
                st.error(f"Error: {e}")

# ---------------- Métricas ----------------
with tab_metrics:
    if os.path.exists("metricas.json"):
        m = json.load(open("metricas.json", encoding="utf-8"))
        a, b, c = st.columns(3)
        a.metric("Accuracy (test)", f"{m['accuracy']:.3f}")
        b.metric("F1 macro (test)", f"{m['f1_macro']:.3f}")
        if "baseline_tfidf" in m:
            c.metric("F1 macro baseline TF-IDF", f"{m['baseline_tfidf']['f1_macro']:.3f}")
        rows = {k: v for k, v in m["por_clase"].items() if k in LABELS}
        st.dataframe(pd.DataFrame(rows).T[["precision", "recall", "f1-score", "support"]].round(3))
    else:
        st.info("Coloca `metricas.json` (generado por el notebook) junto a app.py.")
    for img, cap in [("curvas.png", "Curvas de entrenamiento/validación"),
                     ("confusion.png", "Matriz de confusión (test)"), ("eda.png", "EDA")]:
        if os.path.exists(img):
            st.image(img, caption=cap)

# ---------------- Acerca ----------------
with tab_about:
    st.markdown("""
**Modelo:** `distilbert-base-uncased` (Transformer, ~66 M parámetros) con fine-tuning para 3 clases.  
**Datos:** Consumer Reviews of Amazon Products (Datafiniti, 2019), unificados y deduplicados en SQLite.  
**Etiquetas:** 1–2 ⭐ negativo · 3 ⭐ neutral · 4–5 ⭐ positivo.  
**Desbalance:** ~92 % de las reseñas son positivas; se usó submuestreo + pérdida ponderada y se reporta F1 macro.  
**Limitaciones:** solo inglés; la clase *neutral* es ambigua; no sustituye revisión humana.
""")
