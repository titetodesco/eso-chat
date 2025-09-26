# app_chat.py
# ESO • CHAT (RAG com fallback TF-IDF)
# - Chat via Ollama Cloud (/api/chat) usando OLLAMA_API_KEY
# - RAG sempre ativo:
#     • Histórico (data/analytics/history_texts.jsonl): TF-IDF local
#     • Uploads: TF-IDF local
# - Não depende de /api/embed (se estiver indisponível/401)

import os
import io
import json
import re
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ========== Config UI ==========
st.set_page_config(page_title="ESO • CHAT (Cloud + RAG)", page_icon="💬", layout="wide")
st.title("ESO • CHAT — Ollama Cloud (RAG sempre ativo)")
st.caption("Sem GPU local • Chat em nuvem • RAG com TF-IDF local (histórico + uploads)")

# ========== Secrets / ENV ==========
OLLAMA_HOST        = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
OLLAMA_MODEL       = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gpt-oss:20b"))
OLLAMA_API_KEY     = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY", None))

if not OLLAMA_API_KEY:
    st.error("⚠️ OLLAMA_API_KEY não encontrado. Defina em **Settings → Secrets**.")
    st.stop()

HEADERS_JSON = {
    "Authorization": f"Bearer {OLLAMA_API_KEY}",
    "Content-Type": "application/json",
}

# ========== Parsers simples ==========
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import docx
except Exception:
    docx = None

def read_pdf(file: bytes) -> str:
    if pypdf is None:
        raise RuntimeError("pypdf não está instalado. Adicione `pypdf` ao requirements.txt.")
    reader = pypdf.PdfReader(io.BytesIO(file))
    txt = []
    for page in reader.pages:
        try:
            txt.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(txt)

def read_docx(file: bytes) -> str:
    if docx is None:
        raise RuntimeError("python-docx não está instalado. Adicione `python-docx` ao requirements.txt.")
    f = io.BytesIO(file)
    document = docx.Document(f)
    return "\n".join(p.text for p in document.paragraphs)

def read_xlsx(file: bytes) -> str:
    f = io.BytesIO(file)
    xls = pd.ExcelFile(f)
    frames = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        frames.append(df.astype(str))
    if frames:
        df_all = pd.concat(frames, axis=0, ignore_index=True)
        return df_all.to_csv(index=False)
    return ""

def read_csv(file: bytes) -> str:
    f = io.BytesIO(file)
    df = pd.read_csv(f)
    return df.astype(str).to_csv(index=False)

def load_file_to_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        return read_pdf(data)
    if name.endswith(".docx"):
        return read_docx(data)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return read_xlsx(data)
    if name.endswith(".csv"):
        return read_csv(data)
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

# ========== TF-IDF minimalista (sem sklearn obrigatório) ==========
def _tokenize(text: str):
    return re.findall(r"[a-z0-9áéíóúâêîôûàèìòùãõç]+", text.lower())

def build_tfidf(texts, max_features=40000, min_df=1):
    """
    Constrói TF-IDF simples.
    Retorna (X, vocab, idf), onde:
      - X é (n_docs, dim) float32
      - vocab: dict token -> col
      - idf: np.ndarray (dim,)
    """
    # DF (document frequency)
    df = {}
    docs_tokens = []
    for t in texts:
        toks = set(_tokenize(t))
        docs_tokens.append(toks)
        for tok in toks:
            df[tok] = df.get(tok, 0) + 1

    # filtra por min_df e corta no max_features
    items = [(tok, c) for tok, c in df.items() if c >= min_df]
    # ordena por DF desc e corta
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:max_features]

    vocab = {tok: i for i, (tok, _) in enumerate(items)}
    dim = len(vocab)
    if dim == 0:
        return np.zeros((len(texts), 0), dtype=np.float32), {}, np.zeros((0,), dtype=np.float32)

    N = len(texts)
    idf = np.zeros((dim,), dtype=np.float32)
    for tok, c in items:
        idf[vocab[tok]] = np.log((N + 1) / (c + 1)) + 1.0

    # monta TF
    X = np.zeros((N, dim), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = _tokenize(t)
        if not toks:
            continue
        counts = {}
        for tok in toks:
            if tok in vocab:
                counts[tok] = counts.get(tok, 0) + 1
        if not counts:
            continue
        max_tf = max(counts.values())
        for tok, c in counts.items():
            j = vocab[tok]
            tf = c / max_tf
            X[i, j] = tf * idf[j]

    # normaliza L2
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    X = X / norms
    return X, vocab, idf

def tfidf_transform(texts, vocab, idf):
    dim = len(idf)
    if dim == 0:
        return np.zeros((len(texts), 0), dtype=np.float32)
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = _tokenize(t)
        if not toks:
            continue
        counts = {}
        for tok in toks:
            if tok in vocab:
                counts[tok] = counts.get(tok, 0) + 1
        if not counts:
            continue
        max_tf = max(counts.values())
        for tok, c in counts.items():
            j = vocab[tok]
            tf = c / max_tf
            X[i, j] = tf * idf[j]
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    X = X / norms
    return X

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n,d) x (m,d) -> (n,m). Assume vetores já normalizados na L2."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return a @ b.T

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200):
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts, start, L = [], 0, len(text)
    overlap = max(0, min(overlap, max_chars - 1))
    while start < L:
        end = min(L, start + max_chars)
        parts.append(text[start:end].strip())
        if end >= L:
            break
        start = max(0, end - overlap)
    return [p for p in parts if p]

# ========== Ollama /api/chat ==========
def ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2, stream=False, timeout=120):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "stream": bool(stream),
    }
    r = requests.post(f"{OLLAMA_HOST}/api/chat", headers=HEADERS_JSON, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ========== Estado ==========
if "chat" not in st.session_state:
    st.session_state.chat = []

if "rag" not in st.session_state:
    st.session_state.rag = {
        # Histórico
        "hist_texts": [],        # lista de textos
        "hist_meta": [],         # [{id, source, ...}]
        "hist_X": None,          # TF-IDF (n_hist, dim)
        "hist_vocab": {},
        "hist_idf": None,

        # Uploads
        "upl_chunks": [],        # textos dos chunks
        "upl_meta": [],          # [{file, chunk_id}]
        "upl_X": None,           # TF-IDF (n_upl, dim_up)
        "upl_vocab": {},
        "upl_idf": None,
    }

# ========== Sidebar ==========
st.sidebar.header("Configurações")
topk_hist = st.sidebar.slider("Top-K do HISTÓRICO", 1, 10, 4, 1)
topk_upl  = st.sidebar.slider("Top-K dos UPLOADS", 1, 10, 4, 1)
chunk_size = st.sidebar.slider("Tamanho do chunk (uploads)", 500, 2000, 1200, 50)
chunk_overlap = st.sidebar.slider("Overlap do chunk (uploads)", 50, 600, 200, 10)
st.sidebar.divider()

uploaded_files = st.sidebar.file_uploader(
    "Upload (PDF, DOCX, XLSX, CSV, TXT/MD)",
    type=["pdf","docx","xlsx","xls","csv","txt","md"],
    accept_multiple_files=True
)

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("Limpar uploads", use_container_width=True):
        st.session_state.rag["upl_chunks"] = []
        st.session_state.rag["upl_meta"] = []
        st.session_state.rag["upl_X"] = None
        st.session_state.rag["upl_vocab"] = {}
        st.session_state.rag["upl_idf"] = None
with col2:
    if st.button("Limpar chat", use_container_width=True):
        st.session_state.chat = []
with col3:
    if st.button("Recarregar histórico", use_container_width=True):
        st.session_state.rag["hist_texts"] = []
        st.session_state.rag["hist_meta"] = []
        st.session_state.rag["hist_X"] = None
        st.session_state.rag["hist_vocab"] = {}
        st.session_state.rag["hist_idf"] = None

with st.sidebar.expander("Diagnóstico", expanded=False):
    st.markdown(f"**Host:** `{OLLAMA_HOST}`  \n**Modelo:** `{OLLAMA_MODEL}`")
    if st.button("Teste /api/chat", use_container_width=True):
        try:
            r = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                headers=HEADERS_JSON,
                json={"model": OLLAMA_MODEL, "messages":[{"role":"user","content":"diga OK"}], "stream": False},
                timeout=60
            )
            st.write("Status:", r.status_code)
            try:
                st.json(r.json())
            except Exception:
                st.write(r.text[:1000])
        except Exception as e:
            st.error(f"Falhou: {e}")
    st.info("RAG atual usa **TF-IDF local** para histórico e uploads (independente de /api/embed).")

# ========== Carregar histórico (TF-IDF) ==========
def ensure_history_loaded():
    rag = st.session_state.rag
    if rag["hist_X"] is not None:
        return
    try:
        # Espera: data/analytics/history_texts.jsonl com linhas {"text": "...", "meta": {...}}
        path = os.path.join("data", "analytics", "history_texts.jsonl")
        texts, metas = [], []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        t = str(obj.get("text","")).strip()
                        if not t:
                            continue
                        texts.append(t)
                        meta = obj.get("meta", {})
                        metas.append(meta)
                    except Exception:
                        continue
        rag["hist_texts"] = texts
        rag["hist_meta"] = metas
        if texts:
            X, vocab, idf = build_tfidf(texts, max_features=40000, min_df=1)
            rag["hist_X"] = X
            rag["hist_vocab"] = vocab
            rag["hist_idf"] = idf
        else:
            rag["hist_X"] = np.zeros((0,0), dtype=np.float32)
            rag["hist_vocab"] = {}
            rag["hist_idf"] = np.zeros((0,), dtype=np.float32)
    except Exception as e:
        st.warning(f"Não foi possível carregar histórico (TF-IDF): {e}")
        rag["hist_texts"] = []
        rag["hist_meta"] = []
        rag["hist_X"] = np.zeros((0,0), dtype=np.float32)
        rag["hist_vocab"] = {}
        rag["hist_idf"] = np.zeros((0,), dtype=np.float32)

ensure_history_loaded()

# ========== Indexar uploads (TF-IDF) ==========
if uploaded_files:
    with st.spinner("Lendo arquivos e indexando (TF-IDF local)…"):
        new_chunks, new_meta = [], []
        for uf in uploaded_files:
            try:
                text = load_file_to_text(uf)
                parts = chunk_text(text, max_chars=chunk_size, overlap=chunk_overlap)
                for i, p in enumerate(parts):
                    new_chunks.append(p)
                    new_meta.append({"file": uf.name, "chunk_id": i})
            except Exception as e:
                st.warning(f"Falha ao processar {uf.name}: {e}")

        if new_chunks:
            # (Re)constrói TF-IDF só dos uploads (independente do histórico)
            X, vocab, idf = build_tfidf(new_chunks, max_features=40000, min_df=1)
            st.session_state.rag["upl_chunks"] = new_chunks
            st.session_state.rag["upl_meta"] = new_meta
            st.session_state.rag["upl_X"] = X
            st.session_state.rag["upl_vocab"] = vocab
            st.session_state.rag["upl_idf"] = idf
            st.success(f"Indexados {len(new_chunks)} chunks dos uploads.")

# ========== Histórico de mensagens ==========
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ========== Chat ==========
prompt = st.chat_input("Digite sua pergunta…")
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Monta contexto a partir do TF-IDF do histórico + uploads
    ctx_blocks = []

    # 1) Histórico
    rag = st.session_state.rag
    if rag["hist_X"] is not None and rag["hist_X"].shape[0] > 0:
        q_hist = tfidf_transform([prompt], rag["hist_vocab"], rag["hist_idf"])  # (1,dim)
        sims_hist = cosine_sim(q_hist, rag["hist_X"])[0]  # (n_hist,)
        order = np.argsort(-sims_hist)
        hits = order[:topk_hist]
        for idx in hits:
            meta = rag["hist_meta"][idx] if idx < len(rag["hist_meta"]) else {}
            txt = rag["hist_texts"][idx]
            s = float(sims_hist[idx])
            src = meta.get("source", meta.get("file", "history"))
            ctx_blocks.append(f"[HISTÓRICO: {src}] (sim={s:.3f})\n{txt}")

    # 2) Uploads
    if rag["upl_X"] is not None and rag["upl_X"].shape[0] > 0:
        q_upl = tfidf_transform([prompt], rag["upl_vocab"], rag["upl_idf"])  # (1,dim)
        sims_upl = cosine_sim(q_upl, rag["upl_X"])[0]  # (n_upl,)
        order = np.argsort(-sims_upl)
        hits = order[:topk_upl]
        for idx in hits:
            meta = rag["upl_meta"][idx] if idx < len(rag["upl_meta"]) else {}
            txt = rag["upl_chunks"][idx]
            s = float(sims_upl[idx])
            ctx_blocks.append(f"[UPLOAD: {meta.get('file','upload')} / {meta.get('chunk_id',0)}] (sim={s:.3f})\n{txt}")

    # Sistema + contexto + pergunta
    SYSTEM_PROMPT = (
        "Você é um assistente para gestão de segurança operacional. "
        "Use os CONTEXTOS RELEVANTES para fundamentar a resposta. "
        "Se a informação não estiver no contexto, seja transparente e peça mais detalhes.\n"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if ctx_blocks:
        ctx = "\n\n".join(ctx_blocks)
        messages.append({"role": "user", "content": f"CONTEXTOS RELEVANTES:\n{ctx}"} )
        messages.append({"role": "user", "content": f"PERGUNTA: {prompt}"} )
    else:
        messages.append({"role": "user", "content": prompt})

    # Chamada ao modelo
    with st.chat_message("assistant"):
        with st.spinner("Consultando o modelo na nuvem…"):
            try:
                resp = ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2, stream=False)
                content = resp.get("message", {}).get("content", "").strip() or json.dumps(resp)[:1000]
            except Exception as e:
                content = f"Falha ao consultar o modelo: {e}"
            st.markdown(content)
    st.session_state.chat.append({"role": "assistant", "content": content})

# ========== Painel de status ==========
with st.expander("📚 Status do RAG", expanded=False):
    rag = st.session_state.rag
    st.write(f"Histórico indexado (docs): **{len(rag['hist_texts'])}**")
    st.write(f"Uploads indexados (chunks): **{len(rag['upl_chunks'])}**")
    if rag["upl_chunks"]:
        df = pd.DataFrame(rag["upl_meta"])
        st.dataframe(df.head(50), use_container_width=True)
