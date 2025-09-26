# app_chat.py
# Chat RAG com Ollama Cloud (chat) + TF-IDF local (RAG de uploads) + Contexto de catálogo
# Usa secrets: OLLAMA_API_KEY (obrigatório), OLLAMA_HOST/OLLAMA_MODEL (opcionais)

import os
import io
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st

# Parsers leves
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import docx
except Exception:
    docx = None

# -------------------------
# Configurações básicas
# -------------------------
st.set_page_config(page_title="ESO • CHAT (Ollama Cloud)", page_icon="💬", layout="wide")

# Segredos / env
OLLAMA_HOST        = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
OLLAMA_MODEL       = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gpt-oss:20b"))
OLLAMA_API_KEY     = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY", None))

if not OLLAMA_API_KEY:
    st.error("⚠️ OLLAMA_API_KEY não encontrado. Defina em **Settings → Secrets** no Streamlit Cloud.")
    st.stop()

HEADERS_JSON = {
    "Authorization": f"Bearer {OLLAMA_API_KEY}",
    "Content-Type": "application/json",
}

# Caminhos padrão
DATA_DIR = "data"
ANALYTICS_DIR = os.path.join(DATA_DIR, "analytics")
DATASETS_CONTEXT_FILE = "datasets_context.md"  # novo

# -------------------------
# Utilitários
# -------------------------
def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200):
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = []
    start = 0
    L = len(text)
    overlap = max(0, min(overlap, max_chars - 1))
    while start < L:
        end = min(L, start + max_chars)
        parts.append(text[start:end].strip())
        if end >= L:
            break
        start = max(0, end - overlap)
    return [p for p in parts if p]

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

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1: a = a[None, :]
    if b.ndim == 1: b = b[None, :]
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    return (a @ b.T) / (a_norm * b_norm)

# TF-IDF leve (local)
def tfidf_fit_transform(texts):
    # vocabulário simples
    from collections import Counter
    toks_list = []
    vocab = {}
    for t in texts:
        toks = [w for w in t.lower().split()]
        toks_list.append(toks)
        for w in toks:
            if w not in vocab:
                vocab[w] = len(vocab)
    N = len(texts)
    D = len(vocab)
    X = np.zeros((N, D), dtype=np.float32)
    # TF
    for i, toks in enumerate(toks_list):
        c = Counter(toks)
        if not c:
            continue
        max_tf = max(c.values())
        for w, f in c.items():
            j = vocab.get(w)
            if j is not None:
                X[i, j] = f / max_tf
    # IDF
    df = np.zeros(D, dtype=np.int32)
    for j, w in enumerate(vocab):
        # presença
        cnt = 0
        for toks in toks_list:
            if w in toks:
                cnt += 1
        df[j] = max(1, cnt)
    idf = np.log((N + 1) / (df + 1)) + 1.0
    X *= idf[None, :]
    return X, vocab

def tfidf_transform(texts, vocab):
    N = len(texts)
    D = len(vocab)
    X = np.zeros((N, D), dtype=np.float32)
    from collections import Counter
    # TF
    for i, t in enumerate(texts):
        toks = [w for w in t.lower().split()]
        c = Counter(toks)
        if not c:
            continue
        max_tf = max(c.values())
        for w, f in c.items():
            j = vocab.get(w)
            if j is not None:
                X[i, j] = f / max_tf
    # IDF (reuso: como não guardamos, usamos idf=1 — suficiente para ranking simples no mesmo vocabulário)
    return X

# -------------------------
# Ollama Cloud (chat)
# -------------------------
def ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2, stream=False, timeout=120):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "stream": bool(stream),
    }
    r = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        headers=HEADERS_JSON,
        json=payload,
        timeout=timeout
    )
    r.raise_for_status()
    return r.json()

# -------------------------
# Estado
# -------------------------
if "index" not in st.session_state:
    st.session_state.index = {
        "chunks": [],          # textos de uploads
        "metas": [],           # {"file":..., "chunk_id":...}
        "tfidf_vocab": None,
        "tfidf_matrix": None,  # (n_chunks, |V|)
    }

if "chat" not in st.session_state:
    st.session_state.chat = []

if "dataset_context" not in st.session_state:
    # Carrega datasets_context.md (se existir)
    ctx = ""
    try:
        if os.path.exists(DATASETS_CONTEXT_FILE):
            with open(DATASETS_CONTEXT_FILE, "r", encoding="utf-8") as f:
                ctx = f.read()
    except Exception as e:
        ctx = ""
    st.session_state.dataset_context = ctx

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Configurações")
with st.sidebar.expander("Ollama Cloud", expanded=True):
    st.write("Host:", OLLAMA_HOST)
    st.write("Modelo (chat):", OLLAMA_MODEL)

topk = st.sidebar.slider("Top-K contexto (RAG)", 1, 10, 4, 1)
chunk_size = st.sidebar.slider("Tamanho do chunk (caracteres)", 500, 2000, 1200, 50)
chunk_overlap = st.sidebar.slider("Overlap do chunk", 50, 600, 200, 10)
sim_threshold = st.sidebar.slider("Limiar de similaridade (cos)", 0.10, 0.95, 0.35, 0.01)
use_catalog_ctx = st.sidebar.checkbox("Injetar contexto do catálogo (datasets_context.md)", value=True)
st.sidebar.divider()

uploaded_files = st.sidebar.file_uploader(
    "Upload (PDF, DOCX, XLSX, CSV, TXT/MD)",
    type=["pdf","docx","xlsx","xls","csv","txt","md"],
    accept_multiple_files=True
)

col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Limpar índice", use_container_width=True):
        st.session_state.index = {"chunks": [], "metas": [], "tfidf_vocab": None, "tfidf_matrix": None}
with col_b:
    if st.button("Limpar chat", use_container_width=True):
        st.session_state.chat = []

with st.sidebar.expander("Catálogo carregado", expanded=False):
    ok_any = False
    for fname in ["history_texts.jsonl", "ws_labels.jsonl", "precursors.csv", "ws_precursors_edges.csv", "cp_labels.jsonl"]:
        path = os.path.join(ANALYTICS_DIR, fname)
        if os.path.exists(path):
            st.write("✅", fname)
            ok_any = True
    if not ok_any:
        st.info("Suba os arquivos em data/analytics no repositório.")

# -------------------------
# Indexação de uploads (TF-IDF)
# -------------------------
def rebuild_tfidf_index():
    # Indexa SOMENTE os chunks de upload (o histórico fica para consulta na resposta do modelo)
    chunks = st.session_state.index["chunks"]
    if not chunks:
        st.session_state.index["tfidf_vocab"] = None
        st.session_state.index["tfidf_matrix"] = None
        return
    X, vocab = tfidf_fit_transform(chunks)
    st.session_state.index["tfidf_vocab"] = vocab
    st.session_state.index["tfidf_matrix"] = X

if uploaded_files:
    with st.spinner("Lendo arquivos e indexando (TF-IDF)…"):
        new_chunks, new_metas = [], []
        for uf in uploaded_files:
            try:
                text = load_file_to_text(uf)
                parts = chunk_text(text, max_chars=chunk_size, overlap=chunk_overlap)
                for i, p in enumerate(parts):
                    new_chunks.append(p)
                    new_metas.append({"file": uf.name, "chunk_id": i})
            except Exception as e:
                st.warning(f"Falha ao processar {uf.name}: {e}")

        if new_chunks:
            st.session_state.index["chunks"].extend(new_chunks)
            st.session_state.index["metas"].extend(new_metas)
            rebuild_tfidf_index()
            st.success(f"Indexados {len(new_chunks)} chunks (TF-IDF).")

# -------------------------
# UI principal
# -------------------------
st.title("ESO • CHAT — Ollama Cloud (RAG de uploads por TF-IDF)")
st.caption("Chat na nuvem; RAG local por TF-IDF; catálogo de dados injetado como sistema (opcional)")

# histórico
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Digite sua pergunta…")
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Monta contexto RAG de uploads (se houver índice)
    context_blocks = []
    if st.session_state.index["tfidf_matrix"] is not None and len(st.session_state.index["chunks"]) > 0:
        try:
            vocab = st.session_state.index["tfidf_vocab"] or {}
            # Representa a pergunta no mesmo vocabulário
            qX = tfidf_transform([prompt], vocab)
            sims = cosine_sim(qX, st.session_state.index["tfidf_matrix"])[0]  # (n_chunks,)
            order = np.argsort(-sims)
            hits = []
            for idx in order[: max(50, topk)]:
                if sims[idx] >= sim_threshold:
                    meta = st.session_state.index["metas"][idx]
                    txt = st.session_state.index["chunks"][idx]
                    hits.append((float(sims[idx]), meta, txt))
            hits = hits[:topk]
            for s, meta, txt in hits:
                context_blocks.append(f"[{meta['file']} / {meta['chunk_id']}] (sim={s:.3f})\n{txt}")
        except Exception as e:
            st.warning(f"RAG desativado nesta mensagem (TF-IDF): {e}")

    # Mensagens
    SYSTEM_PROMPT = (
        "Você é um assistente para gestão de segurança operacional. "
        "Responda de forma objetiva, cite o contexto quando relevante e "
        "seja transparente quando não houver informação suficiente."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Injetar catálogo (se existir e estiver habilitado)
    if use_catalog_ctx and st.session_state.dataset_context:
        messages.append({"role": "system", "content": st.session_state.dataset_context})

    # Enviar contextos RAG (se houver)
    if context_blocks:
        ctx = "\n\n".join(context_blocks)
        messages.append({"role": "user", "content": f"CONTEXTOS DO ARQUIVO UPLOADADO:\n{ctx}"})
        messages.append({"role": "user", "content": f"PERGUNTA: {prompt}"})
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

# -------------------------
# Painel: status do índice
# -------------------------
with st.expander("📚 Status do índice (uploads)", expanded=False):
    idx = st.session_state.index
    n_chunks = len(idx["chunks"])
    st.write(f"Chunks indexados: **{n_chunks}**")
    if n_chunks > 0:
        st.dataframe(pd.DataFrame(idx["metas"]).head(50), use_container_width=True)
        if st.button("Baixar índice (CSV de chunks)", use_container_width=True):
            df = pd.DataFrame({
                "file": [m["file"] for m in idx["metas"]],
                "chunk_id": [m["chunk_id"] for m in idx["metas"]],
                "text": idx["chunks"],
            })
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="rag_chunks.csv", mime="text/csv", use_container_width=True)
