# app_chat.py
# Chat RAG com Ollama Cloud (chat + embeddings) — pronto para Streamlit Cloud
# - Sem PyTorch e sem sentence-transformers
# - Usa secrets: OLLAMA_API_KEY (obrigatório), OLLAMA_HOST/OLLAMA_MODEL/OLLAMA_EMBED_MODEL (opcionais)

import os
import io
import json
import time
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st

# Parsers leves para documentos
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

# Segredos / env (com defaults seguros)
OLLAMA_HOST        = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
OLLAMA_MODEL       = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1"))
OLLAMA_API_KEY     = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY", None))
OLLAMA_EMBED_MODEL = st.secrets.get("OLLAMA_EMBED_MODEL", os.getenv("OLLAMA_EMBED_MODEL", "all-minilm"))

if not OLLAMA_API_KEY:
    st.error("⚠️ OLLAMA_API_KEY não encontrado. Defina em **Settings → Secrets** no Streamlit Cloud.")
    st.stop()

HEADERS_JSON = {
    "Authorization": f"Bearer {OLLAMA_API_KEY}",
    "Content-Type": "application/json",
}

# -------------------------
# Funções utilitárias
# -------------------------
def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200):
    """Divide texto em pedaços com sobreposição para RAG."""
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
    # Concatena todas as planilhas e colunas em um textão simples
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
    # txt / md ou outros
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (n,d), b:(m,d) -> (n,m)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]], dtype=np.float32) if a.size else np.zeros((0,0), dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    return (a @ b.T) / (a_norm * b_norm)

# -------------------------
# Ollama Cloud API calls
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
    if stream:
        return r.iter_lines()
    return r.json()

def ollama_embed(texts, model=OLLAMA_EMBED_MODEL, timeout=120):
    """
    Retorna uma lista de vetores (float[]) para cada texto.
    Endpoint correto na Cloud: POST /api/embed
    """
    if isinstance(texts, str):
        texts = [texts]
    payload = {"model": model, "input": texts}
    r = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        headers=HEADERS_JSON,
        json=payload,
        timeout=timeout
    )
    r.raise_for_status()
    data = r.json()
    # Resposta padrão: {"embeddings":[[...],[...],...]}
    if "embeddings" in data and isinstance(data["embeddings"], list):
        return data["embeddings"]
    # fallback muito defensivo (diferenças futuras)
    if "embedding" in data and isinstance(data["embedding"], list):
        return [data["embedding"]]
    raise RuntimeError("Resposta inesperada do /api/embed: " + json.dumps(data)[:300])

# -------------------------
# Estado da aplicação
# -------------------------
if "index" not in st.session_state:
    st.session_state.index = {
        "chunks": [],          # lista de textos
        "embeddings": None,    # np.ndarray (n, d)
        "metas": [],           # infos básicas (arquivo, posição, etc.)
    }

if "chat" not in st.session_state:
    st.session_state.chat = []  # [{"role":"user/assistant", "content": "..."}]

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Configurações")
with st.sidebar.expander("Ollama Cloud", expanded=True):
    st.write("Host:", OLLAMA_HOST)
    st.write("Modelo (chat):", OLLAMA_MODEL)
    st.write("Modelo (embeddings):", OLLAMA_EMBED_MODEL)

topk = st.sidebar.slider("Top-K contexto (RAG)", 1, 10, 4, 1)
chunk_size = st.sidebar.slider("Tamanho do chunk (caracteres)", 500, 2000, 1200, 50)
chunk_overlap = st.sidebar.slider("Overlap do chunk", 50, 600, 200, 10)
sim_threshold = st.sidebar.slider("Limiar de similaridade (cos)", 0.10, 0.95, 0.35, 0.01)
st.sidebar.divider()

uploaded_files = st.sidebar.file_uploader(
    "Upload de base (PDF, DOCX, XLSX, CSV, TXT/MD)",
    type=["pdf","docx","xlsx","xls","csv","txt","md"],
    accept_multiple_files=True
)

col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Limpar índice", use_container_width=True):
        st.session_state.index = {"chunks": [], "embeddings": None, "metas": []}
with col_b:
    if st.button("Limpar chat", use_container_width=True):
        st.session_state.chat = []

with st.sidebar.expander("Diagnóstico de conexão", expanded=False):
    if st.button("Ping /api/tags", use_container_width=True):
        try:
            r = requests.get(f"{OLLAMA_HOST}/api/tags", headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}, timeout=20)
            st.write("Status:", r.status_code)
            if r.headers.get("content-type","").startswith("application/json"):
                st.json(r.json())
            else:
                st.write(r.text[:1000])
        except Exception as e:
            st.error(f"Falhou: {e}")
    if st.button("Teste /api/chat", use_container_width=True):
        try:
            r = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                headers=HEADERS_JSON,
                json={"model": OLLAMA_MODEL, "messages":[{"role":"user","content":"diga OK"}], "stream": False},
                timeout=60)
            st.write("Status:", r.status_code)
            st.json(r.json())
        except Exception as e:
            st.error(f"Falhou: {e}")

# -------------------------
# Indexação (RAG)
# -------------------------
if uploaded_files:
    with st.spinner("Lendo arquivos e gerando embeddings no Ollama Cloud…"):
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
            embs = []
            BATCH = 32  # limite defensivo
            for i in range(0, len(new_chunks), BATCH):
                batch = new_chunks[i : i + BATCH]
                try:
                    vecs = ollama_embed(batch, model=OLLAMA_EMBED_MODEL)
                    embs.extend(vecs)
                except Exception as e:
                    st.error(f"Erro ao gerar embeddings: {e}")
                    st.stop()

            embs = np.array(embs, dtype=np.float32)
            if st.session_state.index["embeddings"] is None:
                st.session_state.index["chunks"] = new_chunks
                st.session_state.index["embeddings"] = embs
                st.session_state.index["metas"] = new_metas
            else:
                st.session_state.index["chunks"].extend(new_chunks)
                st.session_state.index["metas"].extend(new_metas)
                st.session_state.index["embeddings"] = np.vstack([st.session_state.index["embeddings"], embs])

            st.success(f"Indexados {len(new_chunks)} chunks no RAG.")

# -------------------------
# UI principal
# -------------------------
st.title("ESO • CHAT — Ollama Cloud (RAG opcional)")
st.caption("Sem GPU local • Sem torch • Chat + embeddings pela API do Ollama Cloud")

# histórico
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Digite sua pergunta…")
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Monta contexto RAG (se houver índice)
    context_blocks = []
    if st.session_state.index["embeddings"] is not None and len(st.session_state.index["chunks"]) > 0:
        try:
            q_vec = np.array(ollama_embed(prompt, model=OLLAMA_EMBED_MODEL)[0], dtype=np.float32)
            V = st.session_state.index["embeddings"]
            sims = cosine_sim(np.expand_dims(q_vec, 0), V)[0]  # (n,)
            order = np.argsort(-sims)  # desc
            hits = []
            for idx in order[: max(50, topk)]:  # pega um conjunto maior e filtra por limiar
                if sims[idx] >= sim_threshold:
                    meta = st.session_state.index["metas"][idx]
                    txt = st.session_state.index["chunks"][idx]
                    hits.append((float(sims[idx]), meta, txt))
            hits = hits[:topk]
            for s, meta, txt in hits:
                context_blocks.append(f"[{meta['file']} / {meta['chunk_id']}] (sim={s:.3f})\n{txt}")
        except Exception as e:
            st.warning(f"RAG desativado nesta mensagem (erro de embeddings): {e}")

    SYSTEM_PROMPT = (
        "Você é um assistente para gestão de segurança operacional. "
        "Responda de forma objetiva, cite o contexto quando relevante e "
        "seja transparente quando não houver informação suficiente.\n"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context_blocks:
        ctx = "\n\n".join(context_blocks)
        messages.append({"role": "user", "content": f"CONTEXTOS RELEVANTES:\n{ctx}"})
        messages.append({"role": "user", "content": f"PERGUNTA: {prompt}"})
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
# Painel à direita: status do índice
# -------------------------
with st.expander("📚 Status do índice (RAG)", expanded=False):
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
