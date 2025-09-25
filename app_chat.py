# app_chat.py
import os, io, json, time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

try:
    from ollama import Client
except Exception:
    Client = None  # evita crash em import no ambiente sem ollama lib

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="ESO • Chat (RAG)", layout="wide")

DATA_DIR = Path("data")
AN_DIR = DATA_DIR / "analytics"
EMB_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # MESMA do offline

OLLAMA_HOST  = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b-instruct"))
OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY", None))

# ----------------------------
# Utils
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_st_model(name: str):
    return SentenceTransformer(name)

def norm_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / n

def cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (m,d), b: (n,d)
    return norm_rows(a) @ norm_rows(b).T

def read_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader
    rd = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for p in rd.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts)

def read_docx(file_bytes: bytes) -> str:
    import docx
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)

def chunk_text(txt: str, max_chars=900, overlap=120) -> List[str]:
    txt = " ".join(txt.split())
    if len(txt) <= max_chars:
        return [txt]
    chunks, i = [], 0
    while i < len(txt):
        j = min(i + max_chars, len(txt))
        chunks.append(txt[i:j])
        if j == len(txt): break
        i = max(0, j - overlap)
    return chunks

# ----------------------------
# Carregar caches (se existirem)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_ws_pr_cache():
    npz_path = AN_DIR / "embeddings_ws_prec.npz"
    meta_path = AN_DIR / "embeddings_ws_prec.meta.json"
    if not npz_path.exists() or not meta_path.exists():
        return None
    npz = np.load(npz_path, allow_pickle=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    ws_vecs = npz.get("ws_vecs") or npz.get("ws") or npz.get("arr_0")
    pr_vecs = npz.get("prec_vecs") or npz.get("prec") or npz.get("arr_1")
    if ws_vecs is None or pr_vecs is None:
        return None

    ws_labels = meta.get("ws_labels") or meta.get("ws") or []
    pr_labels = meta.get("prec_labels") or meta.get("prec") or []
    return (
        ws_labels, pr_labels,
        ws_vecs.astype(np.float32), pr_vecs.astype(np.float32)
    )

@st.cache_resource(show_spinner=False)
def load_cp_cache():
    npz_path = AN_DIR / "embeddings_cp.npz"
    meta_path = AN_DIR / "embeddings_cp.meta.json"
    if not npz_path.exists() or not meta_path.exists():
        return None
    npz = np.load(npz_path, allow_pickle=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    cp_vecs   = npz.get("cp_vecs") or npz.get("arr_0")
    cp_labels = meta.get("cp_labels") or []
    if cp_vecs is None or not len(cp_labels):
        return None
    return cp_labels, cp_vecs.astype(np.float32)

# ----------------------------
# Ollama client (Cloud/local)
# ----------------------------
def get_ollama_client() -> Client | None:
    if Client is None:
        return None
    headers = {}
    # Ollama Cloud costuma exigir Authorization
    if OLLAMA_HOST.startswith("https://") and OLLAMA_API_KEY:
        headers["Authorization"] = OLLAMA_API_KEY
    try:
        return Client(host=OLLAMA_HOST, headers=headers)
    except Exception:
        return None

def call_llm(prompt: str, temperature: float = 0.1, timeout: int = 45) -> str:
    cli = get_ollama_client()
    if cli is None:
        return "(LLM indisponível neste ambiente)"
    out = []
    start = time.time()
    try:
        for part in cli.chat(
            OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": temperature}
        ):
            if "message" in part and "content" in part["message"]:
                out.append(part["message"]["content"])
            if (time.time() - start) > timeout:
                break
    except Exception as e:
        return f"(Falha ao consultar LLM: {e})"
    return "".join(out).strip()

# ----------------------------
# RAG helpers
# ----------------------------
def embed_chunks(chunks: List[str], model: SentenceTransformer) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 384), dtype=np.float32)
    V = model.encode(chunks, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return V.astype(np.float32)

def topk_indices(scores: np.ndarray, k: int) -> List[int]:
    if scores.size == 0:
        return []
    k = min(k, scores.shape[0])
    # argpartition evita custo do sort completo
    return np.argpartition(-scores, range(k))[:k].tolist()

def build_context(query: str,
                  up_chunks: List[str], V_up: np.ndarray,
                  ws_labels: List[str] | None, ws_vecs: np.ndarray | None,
                  pr_labels: List[str] | None, pr_vecs: np.ndarray | None,
                  cp_labels: List[str] | None, cp_vecs: np.ndarray | None,
                  k_chunks=3, k_ws=3, k_pr=3, k_cp=2) -> Tuple[str, List[str]]:
    st_model = load_st_model(EMB_NAME)
    vq = st_model.encode([query], normalize_embeddings=True)[0].astype(np.float32)

    ctx_parts, refs = [], []

    # Trechos do upload
    if len(up_chunks) and V_up.size:
        s = V_up @ vq
        for idx in topk_indices(s, k_chunks):
            ctx_parts.append(f"[UPLOAD_TRECHO {idx}]\n{up_chunks[idx]}")
            refs.append(f"upload:chunk:{idx}")

    # WS / PR / CP (se caches existirem)
    if ws_vecs is not None and ws_vecs.size:
        sws = ws_vecs @ vq
        for idx in topk_indices(sws, k_ws):
            ctx_parts.append(f"[WS] {ws_labels[idx]}")
    if pr_vecs is not None and pr_vecs.size:
        spr = pr_vecs @ vq
        for idx in topk_indices(spr, k_pr):
            ctx_parts.append(f"[PRECURSOR] {pr_labels[idx]}")
    if cp_vecs is not None and cp_vecs.size:
        scp = cp_vecs @ vq
        for idx in topk_indices(scp, k_cp):
            ctx_parts.append(f"[CP] {cp_labels[idx]}")

    # Limita o tamanho do contexto
    context = "\n\n".join(ctx_parts[:12])
    return context, refs

def build_prompt(query: str, context: str) -> str:
    return (
        "Você é um assistente para Segurança Operacional (ESO) com acesso a um contexto RAG.\n"
        "Regras:\n"
        "- Use APENAS o contexto e os trechos fornecidos; cite trechos entre colchetes [UPLOAD_TRECHO X] quando relevante.\n"
        "- Não invente referências; seja conciso e objetivo.\n\n"
        f"CONTEÚDO:\n{context}\n\n"
        f"PERGUNTA:\n{query}\n\n"
        "RESPOSTA:"
    )

# ----------------------------
# UI
# ----------------------------
st.title("ESO • Chat (RAG) – Ollama Cloud")

with st.sidebar:
    st.subheader("Upload (opcional)")
    up = st.file_uploader("PDF/DOCX/TXT", type=["pdf", "docx", "txt"])
    st.caption("Se nenhum arquivo for enviado, o chat usará apenas os caches (WS/PR/CP) se existirem.")

    st.markdown("---")
    st.subheader("RAG • Top-K")
    k_chunks = st.slider("Trechos do upload", 1, 6, 3)
    k_ws     = st.slider("WS", 0, 8, 3)
    k_pr     = st.slider("Precursores", 0, 8, 3)
    k_cp     = st.slider("Taxonomia CP", 0, 6, 2)

    st.markdown("---")
    st.subheader("LLM")
    st.write(f"**Modelo**: `{OLLAMA_MODEL}`")
    st.write(f"**Host**: `{OLLAMA_HOST}`")
    temp = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    max_wait = st.slider("Timeout (s)", 10, 90, 45, 5)

# Carrega caches (se existirem)
ws_cache = load_ws_pr_cache()
cp_cache = load_cp_cache()

if ws_cache is None:
    st.warning("Caches de WS/PR não encontrados (opcional). Coloque arquivos em `data/analytics/` para enriquecer o RAG.")
else:
    st.success("Caches de WS/PR carregados.")

if cp_cache is None:
    st.info("Cache da Taxonomia CP não encontrado (opcional).")
else:
    st.success("Cache da Taxonomia CP carregado.")

# Extrai caches
if ws_cache:
    ws_labels, pr_labels, ws_vecs, pr_vecs = ws_cache
else:
    ws_labels = pr_labels = []
    ws_vecs = pr_vecs = np.zeros((0, 384), dtype=np.float32)

if cp_cache:
    cp_labels, cp_vecs = cp_cache
else:
    cp_labels = []
    cp_vecs = np.zeros((0, 384), dtype=np.float32)

# Trata upload
uploaded_text = ""
if up is not None:
    if up.type == "application/pdf":
        uploaded_text = read_pdf(up.getvalue())
    elif up.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        uploaded_text = read_docx(up.getvalue())
    else:
        uploaded_text = up.getvalue().decode("utf-8", errors="ignore")

chunks = chunk_text(uploaded_text, max_chars=900, overlap=120) if uploaded_text else []
V_up = np.zeros((0, 384), dtype=np.float32)
if chunks:
    st_model = load_st_model(EMB_NAME)
    V_up = embed_chunks(chunks, st_model)

st.markdown("### Pergunta")
query = st.text_input("Faça sua pergunta sobre o upload (se houver) e/ou sobre os dicionários (WS/PR/CP)…", "")
go = st.button("Consultar")

if go and query.strip():
    with st.spinner("Consultando…"):
        ctx, refs = build_context(query, chunks, V_up, ws_labels, ws_vecs, pr_labels, pr_vecs, cp_labels, cp_vecs,
                                  k_chunks=k_chunks, k_ws=k_ws, k_pr=k_pr, k_cp=k_cp)
        prompt = build_prompt(query, ctx)
        answer = call_llm(prompt, temperature=temp, timeout=max_wait)
    st.subheader("Resposta")
    st.write(answer or "_(sem resposta)_")

    if refs:
        st.caption("Contexto usado: " + ", ".join(refs))
else:
    st.info("Dica: faça upload de 1 arquivo (opcional) e/ou pergunte diretamente. O RAG usa seus dicionários (se presentes).")
