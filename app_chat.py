# app_chat.py
# ESO • CHAT — HIST + UPLD (TF-IDF) • Ollama Cloud
# - HIST: índices pré-gerados com make_catalog_indexes.py (Sphera, GoSee, Docs)
# - UPLD: arquivos enviados no momento (TF-IDF local por sessão)
# - Combina resultados (pesos) e injeta no prompt
# - Injeta datasets_context.md (YAML em texto) como "catálogo" opcional
# - SEMPRE injeta texto bruto do upload (fallback) para o modelo enxergar o arquivo,
#   mesmo se scikit-learn/joblib não estiverem disponíveis.

import os
import io
import json
import textwrap
import requests
import numpy as np
import pandas as pd
import streamlit as st

# ==========================
# Dependências opcionais
# ==========================
try:
    import joblib
except Exception:
    joblib = None

try:
    import pypdf
except Exception:
    pypdf = None

try:
    import docx
except Exception:
    docx = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None

st.set_page_config(page_title="ESO • CHAT (HIST + UPLD)", page_icon="💬", layout="wide")

# ==========================
# Config / Secrets
# ==========================
OLLAMA_HOST  = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gpt-oss:20b"))
OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY", None))
if not OLLAMA_API_KEY:
    st.error("⚠️ OLLAMA_API_KEY não encontrado. Defina em **Settings → Secrets**.")
    st.stop()

HEADERS_JSON = {"Authorization": f"Bearer {OLLAMA_API_KEY}", "Content-Type": "application/json"}

DATA_DIR = "data"
ANALYTICS_DIR = os.path.join(DATA_DIR, "analytics")
DATASETS_CONTEXT_FILE = "datasets_context.md"  # YAML em texto puro

# ==========================
# Utilitários de leitura
# ==========================
def read_pdf_bytes(b: bytes) -> str:
    if pypdf is None:
        # Não derruba a app; o usuário ainda pode subir TXT/CSV/XLSX/DOCX
        return ""
    reader = pypdf.PdfReader(io.BytesIO(b))
    txt = []
    for pg in reader.pages:
        try:
            txt.append(pg.extract_text() or "")
        except Exception:
            pass
    return "\n".join(txt)

def read_docx_bytes(b: bytes) -> str:
    if docx is None:
        return ""
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
        return pd.concat(frames, axis=0, ignore_index=True).to_csv(index=False) if frames else ""
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

# ==========================
# Estado da sessão
# ==========================
if "chat" not in st.session_state:
    st.session_state.chat = []

if "upld" not in st.session_state:
    st.session_state.upld = {
        "texts": [],          # chunks de upload (para TF-IDF e para fallback)
        "metas": [],          # {"file":..., "chunk_id":...}
        "vec": None,          # TfidfVectorizer
        "X": None             # matriz TF-IDF (sparse)
    }

# ==========================
# Carregamento HIST (pré-indexado)
# ==========================
def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def _load_joblib(path):
    if joblib is None or path is None:
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Não consegui carregar {path}: {e}")
        return None

def _read_parquet(path):
    if path is None:
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.warning(f"Falha ao ler {path}: {e}")
        return None

sphera_joblib_path  = _first_existing([os.path.join(ANALYTICS_DIR, "sphera_tfidf.joblib"),
                                       os.path.join(ANALYTICS_DIR, "spheracloud.tfidf.joblib"),
                                       os.path.join(ANALYTICS_DIR, "sphera.tfidf.joblib")])
gosee_joblib_path   = _first_existing([os.path.join(ANALYTICS_DIR, "gosee_tfidf.joblib"),
                                       os.path.join(ANALYTICS_DIR, "gosee.tfidf.joblib")])
hist_joblib_path    = _first_existing([os.path.join(ANALYTICS_DIR, "history_tfidf.joblib")])

sphera_parquet_path = _first_existing([os.path.join(ANALYTICS_DIR, "sphera.parquet"),
                                       os.path.join(ANALYTICS_DIR, "spheracloud.parquet")])
gosee_parquet_path  = _first_existing([os.path.join(ANALYTICS_DIR, "gosee.parquet")])
history_jsonl_path  = os.path.join(ANALYTICS_DIR, "history_texts.jsonl")

sphera_j = _load_joblib(sphera_joblib_path)
gosee_j  = _load_joblib(gosee_joblib_path)
hist_j   = _load_joblib(hist_joblib_path)

sphera_df = _read_parquet(sphera_parquet_path)
gosee_df  = _read_parquet(gosee_parquet_path)

history_rows = []
if os.path.exists(history_jsonl_path):
    try:
        with open(history_jsonl_path, "r", encoding="utf-8") as f:
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

# ==========================
# Sidebar
# ==========================
st.sidebar.header("Configurações")
with st.sidebar.expander("Ollama Cloud", expanded=True):
    st.write("Host:", OLLAMA_HOST)
    st.write("Modelo:", OLLAMA_MODEL)

st.sidebar.divider()
st.sidebar.subheader("RAG • Pesos & Limiares")
w_upload = st.sidebar.slider("Peso do UPLOAD", 0.0, 1.0, 0.7, 0.05)
w_hist   = 1.0 - w_upload
thr_upload = st.sidebar.slider("Limiar (UPLOAD)", 0.0, 1.0, 0.25, 0.01)
thr_hist   = st.sidebar.slider("Limiar (HIST)",   0.0, 1.0, 0.35, 0.01)
topk_upload = st.sidebar.slider("Top-K Upload", 1, 15, 6, 1)
topk_hist   = st.sidebar.slider("Top-K Histórico", 1, 15, 6, 1)
chunk_size  = st.sidebar.slider("Tamanho do chunk (upload)", 500, 2000, 1200, 50)
chunk_overlap = st.sidebar.slider("Overlap do chunk (upload)", 50, 600, 200, 10)
use_catalog = st.sidebar.checkbox("Injetar datasets_context.md", True)
include_upload_raw = st.sidebar.checkbox("Sempre incluir texto bruto do upload no prompt", True)
max_upload_raw_chars = st.sidebar.number_input("Tamanho máx. do bloco UPLOAD_RAW", min_value=1000, max_value=20000, value=6000, step=500)

st.sidebar.divider()
uploaded_files = st.sidebar.file_uploader(
    "Upload (PDF, DOCX, XLSX, CSV, TXT/MD)",
    type=["pdf","docx","xlsx","xls","csv","txt","md"],
    accept_multiple_files=True
)

col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Limpar uploads", use_container_width=True):
        st.session_state.upld = {"texts": [], "metas": [], "vec": None, "X": None}
with col_b:
    if st.button("Limpar chat", use_container_width=True):
        st.session_state.chat = []

# ==========================
# Indexação de UPLOADS
# ==========================
def rebuild_upload_index():
    # Mesmo se não houver sklearn, mantemos os textos para fallback
    if TfidfVectorizer is None:
        st.session_state.upld["vec"] = None
        st.session_state.upld["X"] = None
        return
    texts = st.session_state.upld["texts"]
    if not texts:
        st.session_state.upld["vec"] = None
        st.session_state.upld["X"] = None
        return
    vec = TfidfVectorizer(lowercase=True, strip_accents="unicode",
                          analyzer="word", ngram_range=(1,2), max_features=50000)
    X = vec.fit_transform(texts)
    st.session_state.upld["vec"] = vec
    st.session_state.upld["X"] = X

if uploaded_files:
    with st.spinner("Lendo arquivos e indexando (TF-IDF, local)…"):
        new_texts, new_metas = [], []
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
            st.session_state.upld["texts"].extend(new_texts)
            st.session_state.upld["metas"].extend(new_metas)
            rebuild_upload_index()
            st.success(f"Upload indexado: {len(new_texts)} chunks.")

# ==========================
# Busca — HIST e UPLOAD
# ==========================
def _dedup_by_text(blocks_with_scores):
    seen = set()
    out = []
    for s, txt in blocks_with_scores:
        key = txt.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((s, txt))
    return out

def search_hist(query: str, topk_hist: int, thr_hist: float):
    """Consulta Sphera, GoSee, Docs (pré-indexados com TF-IDF via joblib)."""
    blocks = []

    def add_block(src, score, text):
        blocks.append((score, f"[{src}] (sim={score:.3f})\n{text}"))

    if TfidfVectorizer is None:
        return []

    # Sphera
    if sphera_j and sphera_df is not None:
        vec, X, text_col = sphera_j["vectorizer"], sphera_j["matrix"], sphera_j["text_col"]
        q = vec.transform([query])
        sims = (q @ X.T).toarray()[0]
        idx = np.argsort(-sims)[: topk_hist * 6]
        tmp = []
        for i in idx:
            s = float(sims[i])
            if s < thr_hist:
                continue
            row = sphera_df.iloc[i]
            txt = str(row.get(text_col, ""))
            ident = row.get("EVENT_NUMBER", row.get("Event ID", row.get("ID", f"row{i}")))
            tmp.append((s, f"[SpheraCloud/{ident}] (sim={s:.3f})\n{txt}"))
        blocks.extend(_dedup_by_text(tmp))

    # GoSee
    if gosee_j and gosee_df is not None:
        vec, X, text_col = gosee_j["vectorizer"], gosee_j["matrix"], gosee_j["text_col"]
        q = vec.transform([query])
        sims = (q @ X.T).toarray()[0]
        idx = np.argsort(-sims)[: topk_hist * 6]
        tmp = []
        for i in idx:
            s = float(sims[i])
            if s < thr_hist:
                continue
            row = gosee_df.iloc[i]
            txt = str(row.get(text_col, ""))
            ident = row.get("ID", f"row{i}")
            tmp.append((s, f"[GoSee/{ident}] (sim={s:.3f})\n{txt}"))
        blocks.extend(_dedup_by_text(tmp))

    # Docs
    if hist_j and history_rows:
        vec, X = hist_j["vectorizer"], hist_j["matrix"]
        q = vec.transform([query])
        sims = (q @ X.T).toarray()[0]
        idx = np.argsort(-sims)[: topk_hist * 6]
        tmp = []
        for i in idx:
            s = float(sims[i])
            if s < thr_hist:
                continue
            row = history_rows[i]
            txt = str(row.get("text", ""))
            src = f"Docs/{row.get('source','?')}/{row.get('chunk_id',0)}"
            tmp.append((s, f"[{src}] (sim={s:.3f})\n{txt}"))
        blocks.extend(_dedup_by_text(tmp))

    blocks.sort(key=lambda x: -x[0])
    return [b for _, b in blocks[:topk_hist]]

def search_upload(query: str, topk_upload: int, thr_upload: float):
    """Busca nos uploads indexados nesta sessão (TF-IDF)."""
    if TfidfVectorizer is None:
        return []
    vec = st.session_state.upld["vec"]
    X   = st.session_state.upld["X"]
    metas = st.session_state.upld["metas"]
    texts = st.session_state.upld["texts"]
    if vec is None or X is None or not texts:
        return []
    q = vec.transform([query])
    sims = (q @ X.T).toarray()[0]
    idx = np.argsort(-sims)[: topk_upload * 6]
    tmp = []
    for i in idx:
        s = float(sims[i])
        if s < thr_upload:
            continue
        m = metas[i]; t = texts[i]
        tmp.append((s, f"[UPLOAD {m['file']} / {m['chunk_id']}] (sim={s:.3f})\n{t}"))
    tmp = _dedup_by_text(tmp)
    tmp.sort(key=lambda x: -x[0])
    return [b for _, b in tmp[:topk_upload]]

def build_hist_query_from_upload(max_chars=3000, take_top=3):
    """Gera uma 'query' para HIST a partir dos top chunks do upload (pelo último prompt)."""
    texts = st.session_state.upld["texts"]
    vec   = st.session_state.upld["vec"]
    X     = st.session_state.upld["X"]
    if not texts:
        return ""
    if TfidfVectorizer is None or vec is None or X is None:
        # Sem TF-IDF: usa os primeiros chunks como resumo
        buf, total = [], 0
        for t in texts[:take_top]:
            frag = t.strip()
            if not frag:
                continue
            if total + len(frag) > max_chars:
                frag = frag[: max(0, max_chars-total)]
            buf.append(frag)
            total += len(frag)
            if total >= max_chars:
                break
        return "\n\n".join(buf)

    # Com TF-IDF: ranqueia pelos chunks mais relevantes para a última pergunta do user
    seed = ""
    for m in reversed(st.session_state.chat):
        if m["role"] == "user":
            seed = m["content"]
            break
    if not seed:
        seed = "resumo do conteúdo do upload"
    q = vec.transform([seed])
    sims = (q @ X.T).toarray()[0]
    order = np.argsort(-sims)
    buf, total = [], 0
    for i in order[:take_top]:
        t = texts[i]
        if total + len(t) > max_chars:
            t = t[: max(0, max_chars-total)]
        buf.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return "\n\n".join(buf).strip()

def build_upload_raw_block(max_chars=6000, take_top=3):
    """Monta um bloco 'UPLOAD_RAW' (sempre injetado) com texto bruto dos melhores chunks."""
    texts = st.session_state.upld["texts"]
    if not texts:
        return ""
    # Se TF-IDF existe, usa o mesmo ranking da build_hist_query_from_upload; senão, pega os primeiros
    if TfidfVectorizer is None or st.session_state.upld["vec"] is None or st.session_state.upld["X"] is None:
        chosen = texts[:take_top]
    else:
        vec = st.session_state.upld["vec"]
        X   = st.session_state.upld["X"]
        seed = ""
        for m in reversed(st.session_state.chat):
            if m["role"] == "user":
                seed = m["content"]
                break
        if not seed:
            seed = "resumo do conteúdo do upload"
        q = vec.transform([seed])
        sims = (q @ X.T).toarray()[0]
        order = np.argsort(-sims)
        chosen = [texts[i] for i in order[:take_top]]

    buf, total = [], 0
    for i, t in enumerate(chosen):
        frag = t.strip()
        if not frag:
            continue
        # protege tamanho
        if total + len(frag) > max_chars:
            frag = frag[: max(0, max_upload_raw_chars-total)]
        buf.append(f"--- UPLOAD_RAW_CHUNK_{i+1} ---\n{frag}")
        total += len(frag)
        if total >= max_chars:
            break
    return "\n\n".join(buf)

# ==========================
# UI
# ==========================
st.title("ESO • CHAT — HIST + UPLD (TF-IDF) • Ollama Cloud")
st.caption("RAG local (TF-IDF) com foco no último upload + histórico. Catálogo opcional injetado como contexto (datasets_context.md).")

# Histórico de chat
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Digite sua pergunta…")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1) Busca no UPLOAD (se possível)
    up_blocks = search_upload(prompt, topk_upload=topk_upload, thr_upload=thr_upload)

    # 2) Query do HIST baseada no próprio upload (fallback para texto bruto)
    hist_query = build_hist_query_from_upload(max_chars=3000, take_top=3) or prompt

    # 3) Busca no HIST
    hi_blocks = search_hist(hist_query, topk_hist=topk_hist, thr_hist=thr_hist)

    # 4) Combinação com pesos
    combined = []
    def _score_of(block_text):
        try:
            return float(block_text.split("(sim=")[1].split(")")[0])
        except Exception:
            return 0.0

    for b in up_blocks:
        combined.append((w_upload * _score_of(b), b))
    for b in hi_blocks:
        combined.append((w_hist * _score_of(b), b))
    combined.sort(key=lambda x: -x[0])
    context_blocks = [b for _, b in combined]

    # 5) Sempre injetar um bloco com texto bruto do upload (garante que o modelo
    #    "veja" o arquivo, mesmo sem TF-IDF ou em empates de similaridade)
    raw_block = build_upload_raw_block(max_chars=max_upload_raw_chars, take_top=3) if include_upload_raw else ""

    SYSTEM = (
        "Você é um assistente de segurança operacional. "
        "Use os CONTEXTOS a seguir como evidências. "
        "Se a pergunta solicitar quantitativos ou listas do histórico, responda SOMENTE com base nos CONTEXTOS. "
        "Evite pseudo-código; entregue respostas objetivas e explicativas com trechos citados."
    )

    messages = [{"role": "system", "content": SYSTEM}]
    if use_catalog and catalog_ctx:
        messages.append({"role": "system", "content": catalog_ctx})

    if context_blocks or raw_block:
        ctx_parts = []
        if raw_block:
            ctx_parts.append(f"[UPLOAD_RAW]\n{raw_block}")
        if context_blocks:
            ctx_parts.append("\n\n".join(context_blocks))
        ctx = "\n\n".join(ctx_parts)
        messages.append({"role": "user", "content": f"CONTEXTOS (UPLOAD + HIST):\n{ctx}"})
        messages.append({"role": "user", "content": f"PERGUNTA: {prompt}"})
    else:
        messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Consultando o modelo…"):
            try:
                resp = ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2, stream=False)
                content = resp.get("message", {}).get("content", "").strip() or json.dumps(resp)[:1000]
            except Exception as e:
                content = f"Falha ao consultar o modelo: {e}"
            st.markdown(content)
    st.session_state.chat.append({"role": "assistant", "content": content})

# ==========================
# Painel status
# ==========================
with st.expander("📦 Índices carregados", expanded=False):
    st.write("Spheracloud (HIST):", "✅" if sphera_j and sphera_df is not None else "—")
    st.write("GoSee (HIST):", "✅" if gosee_j and gosee_df is not None else "—")
    st.write("Docs (HIST):", "✅" if hist_j and history_rows else "—")
    st.write("Uploads indexados:", len(st.session_state.upld["texts"]))
    if joblib is None:
        st.info("Observação: `joblib` não instalado — se os índices HIST não carregarem, instale `joblib` e `scikit-learn`.")
    if TfidfVectorizer is None:
        st.info("Observação: `scikit-learn` não instalado — RAG por TF-IDF fica limitado; porém o texto bruto do upload será injetado.")
