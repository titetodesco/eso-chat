# app_chat.py
# ESO • CHAT com Ollama Cloud (preview) + fallback TF-IDF local
# - Chat:  POST https://ollama.com/api/chat
# - Embed: POST https://ollama.com/api/embed  (se falhar -> TF-IDF local)
# - Sem torch / sentence-transformers

import os
import io
import re
import json
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ESO • CHAT (Ollama Cloud)", page_icon="💬", layout="wide")

# -------------------------
# Secrets / ENV
# -------------------------
OLLAMA_HOST        = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
OLLAMA_API_KEY     = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY", None))
OLLAMA_MODEL       = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gpt-oss:20b"))
OLLAMA_EMBED_MODEL = st.secrets.get("OLLAMA_EMBED_MODEL", os.getenv("OLLAMA_EMBED_MODEL", "all-minilm"))

if not OLLAMA_API_KEY:
    st.error("⚠️ OLLAMA_API_KEY não encontrado em Settings → Secrets.")
    st.stop()

HEADERS_JSON     = {"Authorization": f"Bearer {OLLAMA_API_KEY}", "Content-Type": "application/json"}
HEADERS_JSON_RAW = {"Authorization": f"{OLLAMA_API_KEY}", "Content-Type": "application/json"}  # fallback se o provedor não exigir 'Bearer'

# -------------------------
# Parsers simples
# -------------------------
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import docx
except Exception:
    docx = None

def read_pdf(file_bytes: bytes) -> str:
    if pypdf is None:
        raise RuntimeError("pypdf não instalado (adicione 'pypdf' no requirements.txt).")
    r = pypdf.PdfReader(io.BytesIO(file_bytes))
    out = []
    for pg in r.pages:
        try:
            out.append(pg.extract_text() or "")
        except Exception:
            pass
    return "\n".join(out)

def read_docx(file_bytes: bytes) -> str:
    if docx is None:
        raise RuntimeError("python-docx não instalado (adicione 'python-docx' no requirements.txt).")
    d = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in d.paragraphs)

def read_xlsx(file_bytes: bytes) -> str:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    frames = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        frames.append(df.astype(str))
    return pd.concat(frames, axis=0, ignore_index=True).to_csv(index=False) if frames else ""

def read_csv(file_bytes: bytes) -> str:
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df.astype(str).to_csv(index=False)

def load_file_to_text(uf) -> str:
    name = uf.name.lower()
    data = uf.read()
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

# -------------------------
# RAG helpers
# -------------------------
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

# -------- TF-IDF (local, leve) --------
WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9_’-]+")

def tokenize(s: str):
    return [t.lower() for t in WORD_RE.findall(s or "")]

def build_tfidf(chunks: list[str]):
    # vocabulário
    N = len(chunks)
    df = {}  # doc freq
    docs_tokens = []
    for txt in chunks:
        toks = tokenize(txt)
        docs_tokens.append(toks)
        for w in set(toks):
            df[w] = df.get(w, 0) + 1
    # idf
    idf = {w: math.log((N + 1) / (df[w] + 1)) + 1.0 for w in df}
    vocab = {w:i for i, w in enumerate(sorted(idf))}
    D = len(vocab)

    # tf-idf matrizes esparsas (como dict word_id -> val), mas guardamos densas por simplicidade
    X = np.zeros((N, D), dtype=np.float32)
    for i, toks in enumerate(docs_tokens):
        if not toks: continue
        tf = {}
        for w in toks:
            if w in vocab:
                tf[w] = tf.get(w, 0) + 1
        L = len(toks)
        for w, cnt in tf.items():
            j = vocab[w]
            X[i, j] = (cnt / L) * idf[w]
    # normaliza linhas
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    X = X / norms
    return {"vocab": vocab, "idf": idf, "X": X}

def tfidf_query_vector(q: str, model: dict):
    vocab, idf = model["vocab"], model["idf"]
    toks = tokenize(q)
    if not toks:
        return np.zeros((1, len(vocab)), dtype=np.float32)
    tf = {}
    for w in toks:
        if w in vocab:
            tf[w] = tf.get(w, 0) + 1
    L = max(1, len(toks))
    v = np.zeros((1, len(vocab)), dtype=np.float32)
    for w, cnt in tf.items():
        j = vocab[w]
        v[0, j] = (cnt / L) * idf.get(w, 1.0)
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / n

# -------------------------
# Cloud API wrappers
# -------------------------
def _post_json(url: str, payload: dict, timeout: int = 120):
    r = requests.post(url, headers=HEADERS_JSON, json=payload, timeout=timeout)
    if r.status_code in (401, 403):  # tenta sem 'Bearer'
        r2 = requests.post(url, headers=HEADERS_JSON_RAW, json=payload, timeout=timeout)
        r2.raise_for_status()
        return r2
    r.raise_for_status()
    return r

def ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2, stream=False, timeout=120):
    payload = {"model": model, "messages": messages, "temperature": float(temperature), "stream": bool(stream)}
    r = _post_json(f"{OLLAMA_HOST}/api/chat", payload, timeout=timeout)
    return r.json() if not stream else r.iter_lines()

def ollama_embed(texts, model=OLLAMA_EMBED_MODEL, timeout=120):
    """Usa /api/embed; se 4xx ou 5xx, relança para ser capturado e ativar TF-IDF."""
    if isinstance(texts, str):
        texts = [texts]
    payload = {"model": model, "input": texts}
    r = _post_json(f"{OLLAMA_HOST}/api/embed", payload, timeout=timeout)
    data = r.json()
    if "embeddings" in data and isinstance(data["embeddings"], list):
        return data["embeddings"]
    if "embedding" in data and isinstance(data["embedding"], list):
        return [data["embedding"]]
    raise RuntimeError("Resposta inesperada de /api/embed: " + json.dumps(data)[:300])

# -------------------------
# Estado
# -------------------------
if "index" not in st.session_state:
    st.session_state.index = {
        "chunks": [],
        "embeddings": None,   # np.ndarray (n, d) quando usar Embeddings
        "metas": [],
        "tfidf": None,        # dict quando usar TF-IDF
        "backend": "auto",    # "embed" | "tfidf" | "auto"
        "active_backend": None
    }

if "chat" not in st.session_state:
    st.session_state.chat = []

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Configurações")
with st.sidebar.expander("Ollama Cloud", expanded=True):
    st.write("Host:", OLLAMA_HOST)
    st.write("Modelo (chat):", OLLAMA_MODEL)
    st.write("Modelo (embeddings):", OLLAMA_EMBED_MODEL)

topk = st.sidebar.slider("Top-K contexto (RAG)", 1, 10, 4, 1)
chunk_size = st.sidebar.slider("Tamanho do chunk", 500, 2000, 1200, 50)
chunk_overlap = st.sidebar.slider("Overlap", 50, 600, 200, 10)
sim_threshold = st.sidebar.slider("Limiar de similaridade (cos)", 0.10, 0.95, 0.35, 0.01)
backend_choice = st.sidebar.selectbox("Motor de similaridade", ["Auto", "Embeddings (Cloud)", "TF-IDF (local)"], index=0)

st.sidebar.divider()
uploaded_files = st.sidebar.file_uploader(
    "Upload base (PDF, DOCX, XLSX, CSV, TXT/MD)",
    type=["pdf","docx","xlsx","xls","csv","txt","md"],
    accept_multiple_files=True
)

col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Limpar índice", use_container_width=True):
        st.session_state.index = {"chunks": [], "embeddings": None, "metas": [], "tfidf": None, "backend": "auto", "active_backend": None}
with col_b:
    if st.button("Limpar chat", use_container_width=True):
        st.session_state.chat = []

with st.sidebar.expander("Diagnóstico de conexão", expanded=False):
    if st.button("Listar modelos (/api/tags)", use_container_width=True):
        try:
            r = requests.get(f"{OLLAMA_HOST}/api/tags", headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}, timeout=30)
            st.write("Status:", r.status_code)
            if r.headers.get("content-type","").startswith("application/json"):
                st.json(r.json())
            else:
                st.write(r.text[:1000])
        except Exception as e:
            st.error(f"Falhou: {e}")
    if st.button("Teste /api/chat", use_container_width=True):
        try:
            payload = {"model": OLLAMA_MODEL, "messages":[{"role":"user","content":"diga OK"}], "stream": False}
            r = _post_json(f"{OLLAMA_HOST}/api/chat", payload, timeout=60)
            st.write("Status:", r.status_code)
            st.json(r.json())
        except Exception as e:
            st.error(f"Falhou: {e}")
    if st.button("Teste /api/embed", use_container_width=True):
        try:
            payload = {"model": OLLAMA_EMBED_MODEL, "input": ["teste de embedding"]}
            r = _post_json(f"{OLLAMA_HOST}/api/embed", payload, timeout=60)
            st.write("Status:", r.status_code)
            st.json(r.json())
        except Exception as e:
            st.error(f"Falhou: {e}")

# -------------------------
# Indexação (RAG)
# -------------------------
if uploaded_files:
    with st.spinner("Lendo arquivos e preparando índice…"):
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

            # Decide backend
            st.session_state.index["backend"] = {"Auto":"auto","Embeddings (Cloud)":"embed","TF-IDF (local)":"tfidf"}[backend_choice]

            # Tenta embeddings se pedido (ou auto)
            use_embed = st.session_state.index["backend"] in ("embed","auto")
            used_embed = False
            embs = None

            if use_embed:
                try:
                    # gera embeddings em lotes
                    BATCH, all_vecs = 32, []
                    for i in range(0, len(new_chunks), BATCH):
                        batch = new_chunks[i:i+BATCH]
                        vecs = ollama_embed(batch, model=OLLAMA_EMBED_MODEL, timeout=120)
                        all_vecs.extend(vecs)
                    embs = np.array(all_vecs, dtype=np.float32)
                    used_embed = True
                except Exception as e:
                    st.warning(f"Embeddings indisponíveis (usarei TF-IDF): {e}")

            if used_embed and embs is not None:
                if st.session_state.index.get("embeddings") is None:
                    st.session_state.index["embeddings"] = embs
                else:
                    st.session_state.index["embeddings"] = np.vstack([st.session_state.index["embeddings"], embs])
                st.session_state.index["tfidf"] = None
                st.session_state.index["active_backend"] = "Embeddings (Cloud)"
                st.success(f"Indexados {len(new_chunks)} chunks com Embeddings (Cloud).")
            else:
                # TF-IDF
                tfidf_model = build_tfidf(st.session_state.index["chunks"])
                st.session_state.index["tfidf"] = tfidf_model
                st.session_state.index["embeddings"] = None
                st.session_state.index["active_backend"] = "TF-IDF (local)"
                st.success(f"Indexados {len(new_chunks)} chunks com TF-IDF (local).")

# -------------------------
# UI principal
# -------------------------
st.title("ESO • CHAT — Ollama Cloud (RAG opcional)")
st.caption("Cloud preview • Chat via /api/chat • RAG via Embeddings ou TF-IDF (fallback)")

if st.session_state.index["active_backend"]:
    st.info(f"Motor de similaridade em uso: **{st.session_state.index['active_backend']}**")

# histórico
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Digite sua pergunta…")
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG
    context_blocks = []
    idx = st.session_state.index
    chunks = idx["chunks"]

    if chunks:
        try:
            hits = []
            if idx["embeddings"] is not None:
                # Embeddings (Cloud)
                q_vec = np.array(ollama_embed(prompt, model=OLLAMA_EMBED_MODEL)[0], dtype=np.float32)
                V = idx["embeddings"]
                sims = cosine_sim(q_vec[None, :], V)[0]
                order = np.argsort(-sims)
                for j in order[: max(50, topk)]:
                    if sims[j] >= sim_threshold:
                        hits.append((float(sims[j]), idx["metas"][j], chunks[j]))
            elif idx["tfidf"] is not None:
                # TF-IDF (local)
                model = idx["tfidf"]
                qv = tfidf_query_vector(prompt, model)  # (1,D)
                sims = (qv @ model["X"].T).flatten()    # (N,)
                order = np.argsort(-sims)
                for j in order[: max(50, topk)]:
                    if sims[j] >= sim_threshold:
                        hits.append((float(sims[j]), idx["metas"][j], chunks[j]))

            hits = hits[:topk]
            for s, meta, txt in hits:
                context_blocks.append(f"[{meta['file']} / {meta['chunk_id']}] (sim={s:.3f})\n{txt}")
        except Exception as e:
            st.warning(f"RAG desativado nesta mensagem: {e}")

    SYSTEM_PROMPT = (
        "Você é um assistente para gestão de segurança operacional. "
        "Responda de forma objetiva; quando usar contexto, cite o trecho [arquivo/chunk]. "
        "Se a base não tiver a resposta, seja transparente."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context_blocks:
        ctx = "\n\n".join(context_blocks)
        messages.append({"role": "user", "content": f"CONTEXTOS RELEVANTES:\n{ctx}"} )
        messages.append({"role": "user", "content": f"PERGUNTA: {prompt}"} )
    else:
        messages.append({"role": "user", "content": prompt})

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
# Status do índice
# -------------------------
with st.expander("📚 Status do índice (RAG)", expanded=False):
    idx = st.session_state.index
    n_chunks = len(idx["chunks"])
    st.write(f"Chunks indexados: **{n_chunks}**")
    st.write(f"Backend ativo: **{idx['active_backend'] or '—'}**")
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
