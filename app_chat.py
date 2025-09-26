# app_chat.py
# Chat RAG com Ollama Cloud (chat + embeddings) — carrega histórico (analytics) + uploads
# Correções:
# - cosine_sim: denom = a_norm @ b_norm.T (broadcast correto)
# - leitura flexível de edges: aceita "weak signal, precursor, hto" (csv/tsv)
# - mantém UI e fluxo já usados

import os
import io
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st

# Parsers leves (opcionais)
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import docx
except Exception:
    docx = None

st.set_page_config(page_title="ESO • CHAT (Ollama Cloud + Histórico)", page_icon="💬", layout="wide")

# -------------------------
# Secrets / Env
# -------------------------
OLLAMA_HOST        = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
OLLAMA_MODEL       = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gpt-oss:120b"))   # exemplo de cloud
OLLAMA_API_KEY     = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY", None))
OLLAMA_EMBED_MODEL = st.secrets.get("OLLAMA_EMBED_MODEL", os.getenv("OLLAMA_EMBED_MODEL", "all-minilm"))

if not OLLAMA_API_KEY:
    st.error("⚠️ OLLAMA_API_KEY não encontrado. Defina em Settings → Secrets.")
    st.stop()

HEADERS_JSON = {
    "Authorization": f"Bearer {OLLAMA_API_KEY}",
    "Content-Type": "application/json",
}

# -------------------------
# Utilitários
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

def read_pdf(file_bytes: bytes) -> str:
    if pypdf is None:
        raise RuntimeError("pypdf não instalado (adicione pypdf ao requirements).")
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    out = []
    for pg in reader.pages:
        try:
            out.append(pg.extract_text() or "")
        except Exception:
            pass
    return "\n".join(out)

def read_docx(file_bytes: bytes) -> str:
    if docx is None:
        raise RuntimeError("python-docx não instalado (adicione python-docx ao requirements).")
    document = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in document.paragraphs)

def read_xlsx(file_bytes: bytes) -> str:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    frames = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        frames.append(df.astype(str))
    if frames:
        return pd.concat(frames, axis=0, ignore_index=True).to_csv(index=False)
    return ""

def read_csv(file_bytes: bytes) -> str:
    df = pd.read_csv(io.BytesIO(file_bytes))
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
    """(n,d) x (m,d) -> (n,m) com broadcast correto no denominador."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1: a = a[None, :]
    if b.ndim == 1: b = b[None, :]
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-9   # (n,1)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-9   # (m,1)
    # denom correto: (n,1) @ (1,m) => (n,m)
    denom = a_norm @ b_norm.T
    return (a @ b.T) / denom

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def load_vectors_npz(path):
    with np.load(path) as z:
        if "vectors" in z:
            return z["vectors"].astype(np.float32)
        # compatilidade com arr_0
        if "arr_0" in z:
            return z["arr_0"].astype(np.float32)
        raise RuntimeError(f"{path} não contém chave 'vectors'.")

def try_read_edges(path_csv: str):
    """Lê edges aceitando cabeçalhos 'weak signal', 'precursor', 'hto' (csv/tsv)."""
    if not os.path.exists(path_csv):
        return pd.DataFrame(columns=["ws", "precursor", "hto"])
    # auto-detecção de separador
    df = pd.read_csv(path_csv, sep=None, engine="python")
    # normaliza colunas
    norm = {c.lower().strip(): c for c in df.columns}
    # mapeia possíveis nomes -> padronizado
    def pick(*cands):
        for c in cands:
            if c in norm:
                return norm[c]
        return None

    col_ws  = pick("ws", "weak signal", "weaksignal", "weak_signal")
    col_pr  = pick("precursor")
    col_hto = pick("hto")

    if not col_ws or not col_pr:
        # fallback: alguns dumps vinham como 'WeakSignal'/'Precursor'
        for c in df.columns:
            if c.strip().lower() == "weaksignal":
                col_ws = c
            if c.strip().lower() == "precursor":
                col_pr = c

    out = pd.DataFrame(columns=["ws", "precursor", "hto"])
    if col_ws and col_pr:
        out["ws"] = df[col_ws].astype(str)
        out["precursor"] = df[col_pr].astype(str)
        out["hto"] = df[col_hto].astype(str) if col_hto else ""
    return out

# -------------------------
# Ollama Cloud API
# -------------------------
def ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2, stream=False, timeout=120):
    payload = {"model": model, "messages": messages, "temperature": float(temperature), "stream": bool(stream)}
    r = requests.post(f"{OLLAMA_HOST}/api/chat", headers=HEADERS_JSON, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def ollama_embed(texts, model=OLLAMA_EMBED_MODEL, timeout=120):
    if isinstance(texts, str):
        texts = [texts]
    payload = {"model": model, "input": texts}
    # Cloud: /api/embed  (se falhar, revise o modelo de embed ou faça embed local)
    r = requests.post(f"{OLLAMA_HOST}/api/embed", headers=HEADERS_JSON, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if "embeddings" in data and isinstance(data["embeddings"], list):
        return data["embeddings"]
    if "embedding" in data and isinstance(data["embedding"], list):
        return [data["embedding"]]
    raise RuntimeError("Resposta inesperada do /api/embed: " + json.dumps(data)[:300])

# -------------------------
# Estado
# -------------------------
if "index" not in st.session_state:
    st.session_state.index = {
        "chunks": [],
        "embeddings": None,
        "metas": [],
    }
if "chat" not in st.session_state:
    st.session_state.chat = []
if "analytics" not in st.session_state:
    st.session_state.analytics = None  # carregado abaixo

# -------------------------
# Carrega analytics (histórico + dicionários)
# -------------------------
def load_analytics_once():
    base = "data/analytics"
    try:
        # Histórico
        hist_vec = load_vectors_npz(os.path.join(base, "history_vectors.npz"))
        hist_txt = read_jsonl(os.path.join(base, "history_texts.jsonl"))  # precisa campo 'text'
        history_texts = [row.get("text", "") for row in hist_txt]
        # WS
        ws_vecs = load_vectors_npz(os.path.join(base, "ws_vectors.npz"))
        ws_rows = read_jsonl(os.path.join(base, "ws_labels.jsonl"))  # precisa campo 'label'
        ws_labels = [r.get("label", "") for r in ws_rows]
        # PRE
        pr_vecs = load_vectors_npz(os.path.join(base, "prec_vectors.npz"))
        pr_rows = read_jsonl(os.path.join(base, "prec_labels.jsonl"))
        pr_labels = [r.get("label", "") for r in pr_rows]
        # CP
        cp_vecs = load_vectors_npz(os.path.join(base, "cp_vectors.npz"))
        cp_rows = read_jsonl(os.path.join(base, "cp_labels.jsonl"))
        cp_labels = [r.get("label", "") for r in cp_rows]
        # Edges
        edges = None
        for candidate in ["edges_ws_prev.csv", "ws_precursors_edges.csv"]:
            p = os.path.join(base, candidate)
            if os.path.exists(p):
                edges = try_read_edges(p)
                break
        if edges is None:
            edges = pd.DataFrame(columns=["ws", "precursor", "hto"])

        st.session_state.analytics = {
            "history_vecs": hist_vec,
            "history_texts": history_texts,
            "ws_vecs": ws_vecs,
            "ws_labels": ws_labels,
            "prec_vecs": pr_vecs,
            "prec_labels": pr_labels,
            "cp_vecs": cp_vecs,
            "cp_labels": cp_labels,
            "edges": edges,
        }
    except Exception as e:
        st.warning(f"Não foi possível carregar analytics: {e}")
        st.session_state.analytics = {
            "history_vecs": np.zeros((0, 1), dtype=np.float32),
            "history_texts": [],
            "ws_vecs": np.zeros((0, 1), dtype=np.float32),
            "ws_labels": [],
            "prec_vecs": np.zeros((0, 1), dtype=np.float32),
            "prec_labels": [],
            "cp_vecs": np.zeros((0, 1), dtype=np.float32),
            "cp_labels": [],
            "edges": pd.DataFrame(columns=["ws", "precursor", "hto"]),
        }

if st.session_state.analytics is None:
    load_analytics_once()

analytics = st.session_state.analytics

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
source_mode = st.sidebar.selectbox("Fontes para RAG", ["Ambos", "Upload", "Histórico"], index=0)

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

with st.sidebar.expander("Diagnóstico", expanded=False):
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
    if st.button("Teste /api/embed", use_container_width=True):
        try:
            r = requests.post(
                f"{OLLAMA_HOST}/api/embed",
                headers=HEADERS_JSON,
                json={"model": OLLAMA_EMBED_MODEL, "input":["teste de embed"]},
                timeout=60)
            st.write("Status:", r.status_code)
            st.json(r.json())
        except Exception as e:
            st.error(f"Falhou: {e}")

# -------------------------
# Indexação (uploads)
# -------------------------
if uploaded_files:
    with st.spinner("Lendo arquivos e gerando embeddings…"):
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
            BATCH = 32
            for i in range(0, len(new_chunks), BATCH):
                batch = new_chunks[i:i+BATCH]
                try:
                    vecs = ollama_embed(batch, model=OLLAMA_EMBED_MODEL)
                    embs.extend(vecs)
                except Exception as e:
                    st.error(f"Erro ao gerar embeddings: {e}")
                    st.stop()
            embs = np.array(embs, dtype=np.float32)
            if st.session_state.index["embeddings"] is None:
                st.session_state.index["chunks"] = new_chunks
                st.session_state.index["metas"] = new_metas
                st.session_state.index["embeddings"] = embs
            else:
                st.session_state.index["chunks"].extend(new_chunks)
                st.session_state.index["metas"].extend(new_metas)
                st.session_state.index["embeddings"] = np.vstack([st.session_state.index["embeddings"], embs])
            st.success(f"Indexados {len(new_chunks)} chunks do upload.")

# -------------------------
# Chat UI
# -------------------------
st.title("ESO • CHAT — Cloud (histórico + upload)")
st.caption("RAG combinando uploads desta sessão e histórico pré-embarcado em data/analytics.")

# histórico de mensagens
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def build_context_blocks(prompt_text: str, topk: int, sim_thr: float):
    blocks = []
    # upload
    if source_mode in ("Ambos", "Upload"):
        V_up = st.session_state.index["embeddings"]
        if V_up is not None and len(st.session_state.index["chunks"]) > 0:
            qv = np.array(ollama_embed(prompt_text, model=OLLAMA_EMBED_MODEL)[0], dtype=np.float32)
            sims = cosine_sim(np.expand_dims(qv, 0), V_up)[0]
            order = np.argsort(-sims)
            hit_count = 0
            for idx in order:
                if sims[idx] < sim_thr:
                    break
                meta = st.session_state.index["metas"][idx]
                txt = st.session_state.index["chunks"][idx]
                blocks.append(f"[UPLOAD {meta['file']} / {meta['chunk_id']}] (sim={sims[idx]:.3f})\n{txt}")
                hit_count += 1
                if hit_count >= topk:
                    break
    # histórico
    if source_mode in ("Ambos", "Histórico"):
        V_hist = analytics.get("history_vecs", np.zeros((0,1), dtype=np.float32))
        H_txt  = analytics.get("history_texts", [])
        if V_hist.size and len(H_txt) == V_hist.shape[0]:
            qv = np.array(ollama_embed(prompt_text, model=OLLAMA_EMBED_MODEL)[0], dtype=np.float32)
            sims = cosine_sim(np.expand_dims(qv, 0), V_hist)[0]
            order = np.argsort(-sims)
            hit_count = 0
            for idx in order:
                if sims[idx] < sim_thr:
                    break
                txt = H_txt[idx]
                blocks.append(f"[HIST {idx}] (sim={sims[idx]:.3f})\n{txt}")
                hit_count += 1
                if hit_count >= topk:
                    break
    return blocks

prompt = st.chat_input("Digite sua pergunta…")
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context_blocks = []
    try:
        context_blocks = build_context_blocks(prompt, topk=topk, sim_thr=sim_threshold)
    except Exception as e:
        st.warning(f"RAG desativado nesta mensagem: {e}")

    SYSTEM_PROMPT = (
        "Você é um assistente para gestão de segurança operacional (ESO). "
        "Use o contexto fornecido, quando houver, para responder com precisão. "
        "Se a informação não estiver no contexto, seja transparente.\n"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context_blocks:
        ctx = "\n\n".join(context_blocks)
        messages.append({"role": "user", "content": f"CONTEXTOS:\n{ctx}"} )
        messages.append({"role": "user", "content": f"PERGUNTA: {prompt}"} )
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

# -------------------------
# Painel: status
# -------------------------
with st.expander("📚 Status do índice e analytics", expanded=False):
    idx = st.session_state.index
    st.write(f"Chunks (upload) indexados: **{len(idx['chunks'])}**")
    st.write(f"History items: **{len(analytics.get('history_texts', []))}**")
    st.write(f"WS labels: **{len(analytics.get('ws_labels', []))}**")
    st.write(f"Precursores labels: **{len(analytics.get('prec_labels', []))}**")
    st.write(f"Taxonomia CP itens: **{len(analytics.get('cp_labels', []))}**")
    st.write("Edges (ws→prec):", len(analytics.get("edges", pd.DataFrame()).index))
    if len(idx["chunks"]) > 0:
        st.dataframe(pd.DataFrame(idx["metas"]).head(50), use_container_width=True)
