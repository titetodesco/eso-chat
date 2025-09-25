# app_chat.py
# Chat RAG + Detecções (WS / Precursores / CP) com FastEmbed (sem torch)
# - Chat: Ollama Cloud (https://ollama.com/api/chat) com API key
# - Embeddings: FastEmbed local (consistente com arquivos gerados pelo make_embeddings_fast.py)

import os
import io
import json
import time
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st

# --------- Leitores leves ----------
try:
    import pypdf
except Exception:
    pypdf = None
try:
    import docx
except Exception:
    docx = None

# --------- FastEmbed (ONNX, sem torch) ----------
from fastembed import TextEmbedding

# -------------------------
# Configurações básicas
# -------------------------
st.set_page_config(page_title="ESO • CHAT (FastEmbed + Ollama Cloud)", page_icon="💬", layout="wide")

# Segredos (chat) — embeddings são locais com FastEmbed
OLLAMA_HOST    = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
OLLAMA_MODEL   = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gpt-oss:120b"))
OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY", None))

HEADERS_JSON = {"Authorization": f"Bearer {OLLAMA_API_KEY}", "Content-Type": "application/json"} if OLLAMA_API_KEY else None

# Nome do modelo de embedding (deve bater com o make_embeddings_fast.py)
EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-small-en-v1.5")

# -------------------------
# Utils
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

def read_pdf(b: bytes) -> str:
    if pypdf is None:
        raise RuntimeError("pypdf não está instalado. Adicione `pypdf` ao requirements.txt.")
    reader = pypdf.PdfReader(io.BytesIO(b))
    txt = []
    for page in reader.pages:
        try:
            txt.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(txt)

def read_docx(b: bytes) -> str:
    if docx is None:
        raise RuntimeError("python-docx não está instalado. Adicione `python-docx` ao requirements.txt.")
    d = docx.Document(io.BytesIO(b))
    return "\n".join(p.text for p in d.paragraphs)

def read_xlsx(b: bytes) -> str:
    xls = pd.ExcelFile(io.BytesIO(b))
    frames = []
    for sh in xls.sheet_names:
        df = xls.parse(sh)
        frames.append(df.astype(str))
    if frames:
        return pd.concat(frames, axis=0, ignore_index=True).to_csv(index=False)
    return ""

def read_csv(b: bytes) -> str:
    df = pd.read_csv(io.BytesIO(b))
    return df.astype(str).to_csv(index=False)

def load_file_to_text(uf) -> str:
    name = uf.name.lower()
    data = uf.read()
    if name.endswith(".pdf"):
        return read_pdf(data)
    if name.endswith(".docx"):
        return read_docx(data)
    if name.endswith((".xlsx", ".xls")):
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

def ensure_embedder(model_name: str):
    if "embedder" not in st.session_state:
        st.session_state.embedder = TextEmbedding(model_name=model_name)
    return st.session_state.embedder

def embed_texts(texts, model_name: str):
    """Embeddings com FastEmbed (sem torch)."""
    embedder = ensure_embedder(model_name)
    # FastEmbed retorna um iterável de vetores
    vecs = list(embedder.embed(texts))
    return np.array(vecs, dtype=np.float32)

# -------------------------
# Carregamento de analytics (gerados pelo make)
# -------------------------
def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def load_vectors(path_npz: str) -> np.ndarray:
    with np.load(path_npz, allow_pickle=False) as npz:
        if "vectors" in npz.files:
            return np.array(npz["vectors"], dtype=np.float32)
        # fallback defensivo
        key = npz.files[0]
        return np.array(npz[key], dtype=np.float32)

@st.cache_data(show_spinner=False)
def load_analytics():
    base = "data/analytics"
    data = {}

    # WS
    ws_labels_path = os.path.join(base, "ws_labels.jsonl")
    ws_vecs_path   = os.path.join(base, "ws_vectors.npz")
    if os.path.exists(ws_labels_path) and os.path.exists(ws_vecs_path):
        data["ws_labels"] = load_jsonl(ws_labels_path)  # [{label, ...}]
        data["ws_vecs"]   = load_vectors(ws_vecs_path)
    else:
        data["ws_labels"], data["ws_vecs"] = [], np.zeros((0, 384), dtype=np.float32)

    # Precursores
    prec_labels_path = os.path.join(base, "prec_labels.jsonl")
    prec_vecs_path   = os.path.join(base, "prec_vectors.npz")
    if os.path.exists(prec_labels_path) and os.path.exists(prec_vecs_path):
        data["prec_labels"] = load_jsonl(prec_labels_path)
        data["prec_vecs"]   = load_vectors(prec_vecs_path)
    else:
        data["prec_labels"], data["prec_vecs"] = [], np.zeros((0, 384), dtype=np.float32)

    # Taxonomia CP
    cp_labels_path = os.path.join(base, "cp_labels.jsonl")
    cp_vecs_path   = os.path.join(base, "cp_vectors.npz")
    if os.path.exists(cp_labels_path) and os.path.exists(cp_vecs_path):
        data["cp_labels"] = load_jsonl(cp_labels_path)
        data["cp_vecs"]   = load_vectors(cp_vecs_path)
    else:
        data["cp_labels"], data["cp_vecs"] = [], np.zeros((0, 384), dtype=np.float32)

    # Histórico (opcional)
    hist_txt_path = os.path.join(base, "history_texts.jsonl")
    hist_vec_path = os.path.join(base, "history_vectors.npz")
    if os.path.exists(hist_txt_path) and os.path.exists(hist_vec_path):
        data["hist_texts"] = load_jsonl(hist_txt_path)  # [{id, source, text, meta}]
        data["hist_vecs"]  = load_vectors(hist_vec_path)
    else:
        data["hist_texts"], data["hist_vecs"] = [], np.zeros((0, 384), dtype=np.float32)

    # Arestas WS-Precursor (opcional)
    edges_path = os.path.join(base, "edges_ws_prec.csv")
    if os.path.exists(edges_path):
        try:
            data["edges"] = pd.read_csv(edges_path)
        except Exception:
            data["edges"] = pd.DataFrame(columns=["ws","precursor","hto"])
    else:
        data["edges"] = pd.DataFrame(columns=["ws","precursor","hto"])

    return data

analytics = load_analytics()

# -------------------------
# Estado do índice de uploads
# -------------------------
if "index" not in st.session_state:
    st.session_state.index = {"chunks": [], "embeddings": None, "metas": []}

if "chat" not in st.session_state:
    st.session_state.chat = []

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Configurações")
st.sidebar.write(f"**Embeddings (local):** `{EMB_MODEL}`")
with st.sidebar.expander("Ollama Cloud (chat)", expanded=False):
    st.write("Host:", OLLAMA_HOST)
    st.write("Modelo:", OLLAMA_MODEL)
    st.write("API key definida:", bool(OLLAMA_API_KEY))

topk = st.sidebar.slider("Top-K contexto (RAG)", 1, 10, 4, 1)
chunk_size = st.sidebar.slider("Tamanho do chunk (caracteres)", 500, 2200, 1200, 50)
chunk_overlap = st.sidebar.slider("Overlap do chunk", 50, 600, 200, 10)
sim_threshold = st.sidebar.slider("Limiar de similaridade (cos)", 0.10, 0.95, 0.35, 0.01)
st.sidebar.divider()

uploaded_files = st.sidebar.file_uploader(
    "Upload (PDF, DOCX, XLSX, CSV, TXT/MD)",
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

# Diagnóstico simples de chat
with st.sidebar.expander("Diagnóstico /api/chat", expanded=False):
    if st.button("Teste /api/chat", use_container_width=True):
        if not OLLAMA_API_KEY:
            st.error("Defina OLLAMA_API_KEY em Secrets.")
        else:
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
# Indexação de uploads
# -------------------------
if uploaded_files:
    with st.spinner("Lendo arquivos e gerando embeddings (FastEmbed)…"):
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
            try:
                embs = embed_texts(new_chunks, EMB_MODEL)
            except Exception as e:
                st.error(f"Erro ao gerar embeddings (FastEmbed): {e}")
                st.stop()

            if st.session_state.index["embeddings"] is None:
                st.session_state.index["chunks"] = new_chunks
                st.session_state.index["embeddings"] = embs
                st.session_state.index["metas"] = new_metas
            else:
                st.session_state.index["chunks"].extend(new_chunks)
                st.session_state.index["metas"].extend(new_metas)
                st.session_state.index["embeddings"] = np.vstack([st.session_state.index["embeddings"], embs])

            st.success(f"Indexados {len(new_chunks)} chunks do upload.")

# -------------------------
# UI: título + status
# -------------------------
st.title("ESO • CHAT — FastEmbed + Ollama Cloud (RAG)")
st.caption("Embeddings locais (FastEmbed) • Chat remoto (Ollama Cloud) • Detecção WS/Precursores/CP")

with st.expander("📚 Status do índice (uploads + histórico)", expanded=False):
    n_chunks = len(st.session_state.index["chunks"])
    n_hist   = analytics["hist_vecs"].shape[0] if isinstance(analytics.get("hist_vecs"), np.ndarray) else 0
    st.write(f"Chunks (uploads): **{n_chunks}** | Chunks (histórico): **{n_hist}**")
    if n_chunks > 0:
        st.dataframe(pd.DataFrame(st.session_state.index["metas"]).head(50), use_container_width=True)

# -------------------------
# Chat
# -------------------------
# histórico visual
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def ollama_chat(messages, temperature=0.2, stream=False, timeout=120):
    if not OLLAMA_API_KEY:
        raise RuntimeError("Defina OLLAMA_API_KEY em Secrets.")
    payload = {"model": OLLAMA_MODEL, "messages": messages, "temperature": float(temperature), "stream": bool(stream)}
    r = requests.post(f"{OLLAMA_HOST}/api/chat", headers=HEADERS_JSON, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

prompt = st.chat_input("Digite sua pergunta…")
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context_blocks = []

    # 1) RAG de uploads
    if st.session_state.index["embeddings"] is not None and len(st.session_state.index["chunks"]) > 0:
        try:
            q_vec = embed_texts([prompt], EMB_MODEL)[0]
            V = st.session_state.index["embeddings"]
            sims = cosine_sim(np.expand_dims(q_vec, 0), V)[0]
            order = np.argsort(-sims)
            hits = []
            for idx in order[: max(50, topk)]:
                if sims[idx] >= sim_threshold:
                    meta = st.session_state.index["metas"][idx]
                    txt = st.session_state.index["chunks"][idx]
                    hits.append((float(sims[idx]), meta, txt))
            hits = hits[:topk]
            for s, meta, txt in hits:
                context_blocks.append(f"[UPLOAD:{meta['file']} #{meta['chunk_id']}] (sim={s:.3f})\n{txt}")
        except Exception as e:
            st.warning(f"RAG de uploads desabilitado: {e}")

    # 2) RAG de histórico (opcional)
    if analytics["hist_vecs"].size > 0 and len(analytics.get("hist_texts", [])) > 0:
        try:
            q_vec = embed_texts([prompt], EMB_MODEL)[0]
            Vh = analytics["hist_vecs"]
            sims_h = cosine_sim(np.expand_dims(q_vec, 0), Vh)[0]
            order_h = np.argsort(-sims_h)
            hits_h = []
            for idx in order_h[: max(50, topk)]:
                if sims_h[idx] >= sim_threshold:
                    meta = analytics["hist_texts"][idx]
                    txt = meta.get("text", "")
                    src = meta.get("source","HIST")
                    hits_h.append((float(sims_h[idx]), src, txt))
            hits_h = hits_h[:topk]
            for s, src, txt in hits_h:
                context_blocks.append(f"[HIST:{src}] (sim={s:.3f})\n{txt}")
        except Exception as e:
            st.warning(f"RAG de histórico desabilitado: {e}")

    SYSTEM_PROMPT = (
        "Você é um assistente para gestão de segurança operacional. "
        "Quando houver CONTEXTOS RELEVANTES, use-os para responder de forma objetiva e cite a origem. "
        "Se não houver contexto suficiente, seja transparente."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context_blocks:
        ctx = "\n\n".join(context_blocks)
        messages.append({"role": "user", "content": f"CONTEXTOS RELEVANTES:\n{ctx}"} )
        messages.append({"role": "user", "content": f"PERGUNTA: {prompt}"} )
    else:
        messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Consultando o modelo…"):
            try:
                resp = ollama_chat(messages, temperature=0.2, stream=False)
                content = resp.get("message", {}).get("content", "").strip() or json.dumps(resp)[:1000]
            except Exception as e:
                content = f"Falha ao consultar o modelo: {e}"
            st.markdown(content)
    st.session_state.chat.append({"role": "assistant", "content": content})

# -------------------------
# 🔎 Detecções no upload (WS / Precursores / CP)
# -------------------------
st.divider()
st.subheader("🔎 Detecções no upload (WS / Precursores / Taxonomia CP)")

col1, col2, col3 = st.columns(3)
with col1:
    thr_ws   = st.slider("Limiar WS",   0.05, 0.95, 0.35, 0.01)
with col2:
    thr_prec = st.slider("Limiar Precursores", 0.05, 0.95, 0.35, 0.01)
with col3:
    thr_cp   = st.slider("Limiar CP",   0.05, 0.95, 0.35, 0.01)

if st.session_state.index["embeddings"] is None or len(st.session_state.index["chunks"]) == 0:
    st.info("Envie um arquivo na barra lateral para analisar.")
else:
    V_chunks = st.session_state.index["embeddings"]
    chunks   = st.session_state.index["chunks"]

    def top_hits(label_rows, label_vecs, thr, label_key="label", extra_cols=()):
        """Retorna dataframe com (label, similaridade, trecho) agregando por maior sim vs chunks."""
        if label_vecs.size == 0:
            return pd.DataFrame(columns=[label_key,"similaridade","trecho"] + list(extra_cols))
        S = cosine_sim(V_chunks, label_vecs)  # (num_chunks, num_labels)
        best = S.max(axis=0)                  # (num_labels,)
        argc = S.argmax(axis=0)               # chunk id de maior sim para cada label
        rows = []
        for j, simv in enumerate(best):
            if simv >= thr:
                lbl = label_rows[j]
                ch  = chunks[int(argc[j])]
                base = {label_key: lbl[label_key] if label_key in lbl else lbl.get("label",""),
                        "similaridade": float(simv),
                        "trecho": ch}
                for k in extra_cols:
                    base[k] = lbl.get(k,"")
                rows.append(base)
        if not rows:
            return pd.DataFrame(columns=[label_key,"similaridade","trecho"] + list(extra_cols))
        return pd.DataFrame(rows).sort_values("similaridade", ascending=False).reset_index(drop=True)

    # WS
    ws_rows = [{"label": d.get("label","")} for d in analytics.get("ws_labels", [])]
    df_ws = top_hits(ws_rows, analytics.get("ws_vecs", np.zeros((0,1),dtype=np.float32)), thr_ws, label_key="label")
    st.markdown("**Weak Signals detectados (acima do limiar)**")
    st.dataframe(df_ws.head(100), use_container_width=True)

    # Precursores
    prec_rows = [{"label": d.get("label",""), "hto": d.get("hto","") } for d in analytics.get("prec_labels", [])]
    df_prec = top_hits(prec_rows, analytics.get("prec_vecs", np.zeros((0,1),dtype=np.float32)), thr_prec, label_key="label", extra_cols=("hto",))
    st.markdown("**Precursores detectados (acima do limiar)**")
    st.dataframe(df_prec.head(100), use_container_width=True)

    # Taxonomia CP
    cp_rows = [{"label": d.get("label",""),
                "dimensao": d.get("dimensao",""),
                "fator": d.get("fator",""),
                "sub1": d.get("sub1",""),
                "sub2": d.get("sub2","")} for d in analytics.get("cp_labels", [])]
    df_cp = top_hits(cp_rows, analytics.get("cp_vecs", np.zeros((0,1),dtype=np.float32)), thr_cp,
                     label_key="label", extra_cols=("dimensao","fator","sub1","sub2"))
    st.markdown("**Taxonomia CP (acima do limiar)**")
    st.dataframe(df_cp.head(200), use_container_width=True)

    # Exportar Excel (2 abas)
    if st.button("Baixar Excel (WS & Precursores)"):
        with pd.ExcelWriter("detecções.xlsx", engine="xlsxwriter") as writer:
            (df_ws if not df_ws.empty else pd.DataFrame(columns=["label","similaridade","trecho"])).to_excel(writer, index=False, sheet_name="WS")
            (df_prec if not df_prec.empty else pd.DataFrame(columns=["label","hto","similaridade","trecho"])).to_excel(writer, index=False, sheet_name="Precursores")
        with open("detecções.xlsx","rb") as f:
            st.download_button("Download detecções.xlsx", f.read(), file_name="detecções.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
