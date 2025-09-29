# app_chat.py
# ESO • CHAT — HIST + UPLD (TF-IDF) • Ollama Cloud
# - HIST: índices pré-gerados com make_catalog_indexes.py (Sphera, GoSee, Docs)
# - UPLD: arquivos enviados no momento (TF-IDF local por sessão)
# - Combina resultados (pesos) e injeta no prompt
# - Injeta datasets_context.md (texto) como "catálogo" opcional

import os
import io
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st

# joblib / sklearn para HIST (se faltar, HIST desativa)
try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    joblib = None
    TfidfVectorizer = None

# Parsers leves
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import docx
except Exception:
    docx = None

st.set_page_config(page_title="ESO • CHAT (HIST + UPLD)", page_icon="💬", layout="wide")

# -------------------------
# Configs / Secrets
# -------------------------
OLLAMA_HOST  = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gpt-oss:20b"))
OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY", None))

if not OLLAMA_API_KEY:
    st.error("⚠️ OLLAMA_API_KEY não encontrado. Defina em **Settings → Secrets**.")
    st.stop()

HEADERS_JSON = {"Authorization": f"Bearer {OLLAMA_API_KEY}", "Content-Type": "application/json"}

DATA_DIR = "data"
ANALYTICS_DIR = os.path.join(DATA_DIR, "analytics")
DATASETS_CONTEXT_FILE = "datasets_context.md"  # texto puro (YAML ou notas), injetado como contexto adicional

# -------------------------
# Utils
# -------------------------
def read_pdf_bytes(b: bytes) -> str:
    if pypdf is None:
        raise RuntimeError("pypdf não instalado (adicione em requirements).")
    reader = pypdf.PdfReader(io.BytesIO(b))
    out = []
    for pg in reader.pages:
        try:
            out.append(pg.extract_text() or "")
        except Exception:
            pass
    return "\n".join(out)

def read_docx_bytes(b: bytes) -> str:
    if docx is None:
        raise RuntimeError("python-docx não instalado (adicione em requirements).")
    f = io.BytesIO(b)
    d = docx.Document(f)
    return "\n".join(p.text for p in d.paragraphs)

def read_any(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith(".pdf"):
        return read_pdf_bytes(data)
    if name.endswith(".docx"):
        return read_docx_bytes(data)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        xls = pd.ExcelFile(io.BytesIO(data))
        frames = []
        for s in xls.sheet_names:
            df = xls.parse(s)
            frames.append(df.astype(str))
        if frames:
            return pd.concat(frames, axis=0, ignore_index=True).to_csv(index=False)
        return ""
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(data))
        return df.astype(str).to_csv(index=False)
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def chunk_text(text: str, max_chars=1200, overlap=200):
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts, start, L = [], 0, len(text)
    ov = max(0, min(overlap, max_chars - 1))
    while start < L:
        end = min(L, start + max_chars)
        parts.append(text[start:end].strip())
        if end >= L:
            break
        start = max(0, end - ov)
    return [p for p in parts if p]

def ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2, stream=False, timeout=120):
    payload = {"model": model, "messages": messages, "temperature": float(temperature), "stream": bool(stream)}
    r = requests.post(f"{OLLAMA_HOST}/api/chat", headers=HEADERS_JSON, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

# -------------------------
# Estado
# -------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "upload_index" not in st.session_state:
    st.session_state.upload_index = {
        "texts": [],
        "metas": [],  # {"file":..., "chunk_id":...}
        "vec": None,  # TfidfVectorizer
        "X": None     # sparse matrix
    }

# -------------------------
# Carrega HIST (pré-indexado via joblib)
# -------------------------
def _try_load_joblib(path):
    if joblib is None:
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Não consegui carregar {path}: {e}")
        return None

def _try_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.warning(f"Falha ao ler {path}: {e}")
        return None

# candidatos (nomes que usamos ao longo do projeto)
sph_job = os.path.join(ANALYTICS_DIR, "sphera_tfidf.joblib")
gos_job = os.path.join(ANALYTICS_DIR, "gosee_tfidf.joblib")
his_job = os.path.join(ANALYTICS_DIR, "history_tfidf.joblib")

sph_parq = os.path.join(ANALYTICS_DIR, "sphera.parquet")  # dataframe alinhado ao joblib
gos_parq = os.path.join(ANALYTICS_DIR, "gosee.parquet")

sphera_j = _try_load_joblib(sph_job)
gosee_j  = _try_load_joblib(gos_job)
hist_j   = _try_load_joblib(his_job)

sphera_df = _try_parquet(sph_parq) if sphera_j is not None else None
gosee_df  = _try_parquet(gos_parq) if gosee_j  is not None else None

history_rows = []
hist_jsonl = os.path.join(ANALYTICS_DIR, "history_texts.jsonl")
if hist_j is not None and os.path.exists(hist_jsonl):
    try:
        with open(hist_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                history_rows.append(json.loads(line))
    except Exception as e:
        st.warning(f"Falha ao ler history_texts.jsonl: {e}")

# datasets_context.md
catalog_ctx = ""
try:
    if os.path.exists(DATASETS_CONTEXT_FILE):
        with open(DATASETS_CONTEXT_FILE, "r", encoding="utf-8") as f:
            catalog_ctx = f.read()
except Exception:
    pass

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Configurações")
with st.sidebar.expander("Ollama Cloud", expanded=True):
    st.write("Host:", OLLAMA_HOST)
    st.write("Modelo:", OLLAMA_MODEL)

st.sidebar.divider()
st.sidebar.subheader("RAG • Pesos & Limiar")
w_upload = st.sidebar.slider("Peso do UPLOAD", 0.0, 1.0, 0.7, 0.05)
w_hist   = 1.0 - w_upload
thr_upload = st.sidebar.slider("Limiar (UPLOAD)", 0.0, 1.0, 0.25, 0.01)
thr_hist   = st.sidebar.slider("Limiar (HIST)",   0.0, 1.0, 0.35, 0.01)
topk_upload = st.sidebar.slider("Top-K Upload", 1, 15, 6, 1)
topk_hist   = st.sidebar.slider("Top-K Histórico", 1, 15, 6, 1)
chunk_size  = st.sidebar.slider("Tamanho do chunk (upload)", 500, 2000, 1200, 50)
chunk_overlap = st.sidebar.slider("Overlap do chunk (upload)", 50, 600, 200, 10)
use_catalog = st.sidebar.checkbox("Injetar datasets_context.md", True)

st.sidebar.divider()
uploaded_files = st.sidebar.file_uploader(
    "Upload (PDF, DOCX, XLSX, CSV, TXT/MD)",
    type=["pdf","docx","xlsx","xls","csv","txt","md"],
    accept_multiple_files=True
)

col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Limpar uploads", use_container_width=True):
        st.session_state.upload_index = {"texts": [], "metas": [], "vec": None, "X": None}
with col_b:
    if st.button("Limpar chat", use_container_width=True):
        st.session_state.chat = []

# -------------------------
# Indexação de UPLOADS (TF-IDF sessão)
# -------------------------
def rebuild_upload_index():
    texts = st.session_state.upload_index["texts"]
    if not texts or TfidfVectorizer is None:
        st.session_state.upload_index["vec"] = None
        st.session_state.upload_index["X"] = None
        return
    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1,2),
        max_features=50000
    )
    X = vec.fit_transform(texts)
    st.session_state.upload_index["vec"] = vec
    st.session_state.upload_index["X"] = X

if uploaded_files:
    with st.spinner("Lendo arquivos e indexando (TF-IDF, local)…"):
        new_texts = []
        new_metas = []
        for uf in uploaded_files:
            try:
                text = read_any(uf)
                parts = chunk_text(text, max_chars=chunk_size, overlap=chunk_overlap)
                for i, p in enumerate(parts):
                    new_texts.append(p)
                    new_metas.append({"file": uf.name, "chunk_id": i})
            except Exception as e:
                st.warning(f"Falha ao processar {uf.name}: {e}")

        if new_texts:
            st.session_state.upload_index["texts"].extend(new_texts)
            st.session_state.upload_index["metas"].extend(new_metas)
            rebuild_upload_index()
            st.success(f"Upload indexado: {len(new_texts)} chunks.")

# -------------------------
# Busca HIST (TF-IDF pré-indexado)
# -------------------------
def search_hist(query: str, topk_hist: int, thr_hist: float):
    """Consulta Sphera/GoSee/Docs usando os TF-IDF pré-indexados (sklearn)."""
    blocks = []

    def add_block(src, score, text):
        blocks.append((score, f"[{src}] (sim={score:.3f})\n{text}"))

    # Sphera
    if sphera_j and sphera_df is not None:
        vec, X, text_col = sphera_j["vectorizer"], sphera_j["matrix"], sphera_j["text_col"]
        q = vec.transform([query])
        sims = (q @ X.T).toarray()[0]  # similaridade de cosseno implícita (L2 norm s/ sklearn)
        idx = np.argsort(-sims)[: max(topk_hist*4, 20)]
        seen = set()
        for i in idx:
            s = float(sims[i])
            if s < thr_hist:
                continue
            row = sphera_df.iloc[i]
            txt = str(row.get(text_col, ""))
            key = txt.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            ident = row.get("Event ID", row.get("EVENT_NUMBER", row.get("ID", f"row{i}")))
            add_block(f"Sphera/{ident}", s, txt)

    # GoSee
    if gosee_j and gosee_df is not None:
        vec, X, text_col = gosee_j["vectorizer"], gosee_j["matrix"], gosee_j["text_col"]
        q = vec.transform([query])
        sims = (q @ X.T).toarray()[0]
        idx = np.argsort(-sims)[: max(topk_hist*4, 20)]
        seen = set()
        for i in idx:
            s = float(sims[i])
            if s < thr_hist:
                continue
            row = gosee_df.iloc[i]
            txt = str(row.get(text_col, ""))
            key = txt.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            ident = row.get("ID", f"row{i}")
            add_block(f"GoSee/{ident}", s, txt)

    # Docs (history)
    if hist_j and history_rows:
        vec, X = hist_j["vectorizer"], hist_j["matrix"]
        q = vec.transform([query])
        sims = (q @ X.T).toarray()[0]
        idx = np.argsort(-sims)[: max(topk_hist*4, 20)]
        seen = set()
        for i in idx:
            s = float(sims[i])
            if s < thr_hist:
                continue
            row = history_rows[i]
            txt = str(row.get("text", ""))
            key = txt.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            src = f"Docs/{row.get('source','?')}/{row.get('chunk_id',0)}"
            add_block(src, s, txt)

    blocks.sort(key=lambda x: -x[0])
    return [b for _, b in blocks[:topk_hist]]

# -------------------------
# Busca UPLOAD (TF-IDF sessão)
# -------------------------
def search_upload(query: str, topk_upload: int, thr_upload: float):
    blocks = []
    vec = st.session_state.upload_index["vec"]
    X = st.session_state.upload_index["X"]
    metas = st.session_state.upload_index["metas"]
    texts = st.session_state.upload_index["texts"]
    if vec is None or X is None or not texts:
        return blocks
    q = vec.transform([query])
    sims = (q @ X.T).toarray()[0]
    idx = np.argsort(-sims)[: max(topk_upload*4, 20)]
    seen = set()
    for i in idx:
        s = float(sims[i])
        if s < thr_upload:
            continue
        m = metas[i]
        t = texts[i]
        key = t.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        blocks.append((s, f"[UPLOAD {m['file']} / {m['chunk_id']}] (sim={s:.3f})\n{t}"))
    blocks.sort(key=lambda x: -x[0])
    return [b for _, b in blocks[:topk_upload]]

# -------------------------
# UI
# -------------------------
st.title("ESO • CHAT — HIST + UPLD (TF-IDF) • Ollama Cloud")
st.caption("RAG local (TF-IDF) com foco no último upload + histórico. Catálogo (datasets_context.md) opcional.")

# mostra histórico
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Digite sua pergunta…")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Recuperação combinada
    up_blocks = search_upload(prompt, topk_upload=topk_upload, thr_upload=thr_upload)
    hi_blocks = search_hist(prompt,   topk_hist=topk_hist,       thr_hist=thr_hist)

    # Combinação com pesos (ordenando por score ponderado)
    combined = []
    def _score_of(block_str):
        try:
            return float(block_str.split("(sim=")[1].split(")")[0])
        except Exception:
            return 0.0

    for b in up_blocks:
        combined.append((w_upload * _score_of(b), b))
    for b in hi_blocks:
        combined.append((w_hist   * _score_of(b), b))

    combined.sort(key=lambda x: -x[0])
    context_blocks = [b for _, b in combined]

    SYSTEM = (
        "Você é um assistente de segurança operacional. "
        "Use os CONTEXTOS como evidências. "
        "Se for pedir quantitativos/listas, responda SOMENTE com base nos CONTEXTOS. "
        "Evite pseudo-código; entregue respostas diretas, com trechos citados quando útil."
    )

    messages = [{"role": "system", "content": SYSTEM}]
    if use_catalog and catalog_ctx:
        messages.append({"role": "system", "content": catalog_ctx})

    if context_blocks:
        ctx = "\n\n".join(context_blocks)
        messages.append({"role": "user", "content": f"CONTEXTOS (UPLOAD + HIST):\n{ctx}"} )
        messages.append({"role": "user", "content": f"PERGUNTA: {prompt}"} )
    else:
        messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            resp = ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2, stream=False)
            content = resp.get("message", {}).get("content", "").strip() or json.dumps(resp)[:1500]
        except Exception as e:
            content = f"Falha ao consultar o modelo: {e}"
        st.markdown(content)
    st.session_state.chat.append({"role": "assistant", "content": content})

# Painel status
with st.expander("📦 Índices carregados", expanded=False):
    st.write("Sphera (TF-IDF):", "✅" if sphera_j and sphera_df is not None else "—")
    st.write("GoSee (TF-IDF):", "✅" if gosee_j and gosee_df is not None else "—")
    st.write("Docs (TF-IDF):",  "✅" if hist_j and history_rows else "—")
    st.write("Uploads indexados:", len(st.session_state.upload_index["texts"]))
