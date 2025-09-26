# app_chat.py
# Chat RAG com Ollama Cloud — usa histórico (analytics) + uploads do usuário
# - Sem PyTorch / sentence-transformers
# - Usa secrets: OLLAMA_API_KEY (obrigatório), OLLAMA_HOST/OLLAMA_MODEL/OLLAMA_EMBED_MODEL (opcionais)

import os
import io
import json
from pathlib import Path
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

# Base dirs
BASE_DIR = Path(__file__).resolve().parent
ANALYTICS_DIR = BASE_DIR / "data" / "analytics"

# Segredos / env (com defaults seguros)
OLLAMA_HOST        = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
OLLAMA_MODEL       = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gpt-oss:120b"))
OLLAMA_API_KEY     = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY", None))
OLLAMA_EMBED_MODEL = st.secrets.get("OLLAMA_EMBED_MODEL", os.getenv("OLLAMA_EMBED_MODEL", "all-minilm"))

if not OLLAMA_API_KEY:
    st.error("⚠️ OLLAMA_API_KEY não encontrado. Defina em **Settings → Secrets** no Streamlit Cloud.")
    st.stop()

# Importante: na Cloud o header NÃO usa 'Bearer '
HEADERS_JSON = {
    "Authorization": OLLAMA_API_KEY,
    "Content-Type": "application/json",
}

# -------------------------
# Utilitários
# -------------------------
def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200):
    """Divide texto em pedaços com sobreposição."""
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
        # Representa o conteúdo de forma simples
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
    """Retorna (n, m) de similaridade coseno entre linhas de a e b, com defesas."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    if a.shape[1] != b.shape[1]:
        # dimensões incompatíveis
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    return (a @ b.T) / (a_norm * b_norm)

def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def load_npz_any(*candidates: str, key_preference=("vectors", "arr_0")):
    """
    Tenta carregar um .npz de uma lista de nomes candidatos.
    Retorna np.ndarray; se houver várias chaves, usa 'vectors' ou 'arr_0' por padrão.
    """
    for name in candidates:
        fp = ANALYTICS_DIR / name
        if fp.exists():
            try:
                npz = np.load(fp)
                for key in key_preference:
                    if key in npz.files:
                        return np.array(npz[key], dtype=np.float32)
                # fallback: primeira chave
                return np.array(npz[npz.files[0]], dtype=np.float32)
            except Exception:
                # tenta com allow_pickle como fallback duro (evitar quando possível)
                try:
                    npz = np.load(fp, allow_pickle=True)
                    for key in key_preference:
                        if key in npz.files:
                            return np.array(npz[key], dtype=np.float32)
                    return np.array(npz[npz.files[0]], dtype=np.float32)
                except Exception:
                    continue
    return np.zeros((0, 1), dtype=np.float32)

def ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x[None, :]
    return x

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
    Retorna lista de vetores (float[]) para cada texto.
    Cloud API: POST /api/embed  (Authorization: <api_key>)
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
    if "embeddings" in data and isinstance(data["embeddings"], list):
        return data["embeddings"]
    if "embedding" in data and isinstance(data["embedding"], list):
        return [data["embedding"]]
    raise RuntimeError("Resposta inesperada do /api/embed: " + json.dumps(data)[:300])

# -------------------------
# Estado da app
# -------------------------
if "index" not in st.session_state:
    st.session_state.index = {
        "chunks": [],          # textos de uploads
        "embeddings": None,    # np.ndarray (n, d)
        "metas": [],           # {"file":..., "chunk_id":...}
    }

if "chat" not in st.session_state:
    st.session_state.chat = []  # [{"role":"user/assistant", "content": "..."}]

# -------------------------
# Carregar analytics (histórico + dicionários)
# -------------------------
def load_analytics():
    analytics = {
        "history_vecs": np.zeros((0, 1), dtype=np.float32),
        "history_texts": [],

        "ws_vecs": np.zeros((0, 1), dtype=np.float32),
        "ws_labels": [],

        "prec_vecs": np.zeros((0, 1), dtype=np.float32),
        "prec_labels": [],

        "cp_vecs": np.zeros((0, 1), dtype=np.float32),
        "cp_labels": [],

        "edges": pd.DataFrame(columns=["weak signal", "precursor", "hto"]),
    }
    try:
        # Histórico
        hist_vecs = load_npz_any("history_vectors.npz")
        hist_texts = read_jsonl(ANALYTICS_DIR / "history_texts.jsonl")
        analytics["history_vecs"] = ensure_2d(hist_vecs)
        analytics["history_texts"] = hist_texts

        # WS
        ws_vecs = load_npz_any("ws_vectors.npz", "embeddings_ws_prec.npz")  # fallback
        ws_labels_jsonl = ANALYTICS_DIR / "ws_labels.jsonl"
        ws_labels = [r.get("label", "") for r in read_jsonl(ws_labels_jsonl)] if ws_labels_jsonl.exists() else []
        # Caso legacy: embeddings_ws_prec.meta.json com labels
        if not ws_labels and (ANALYTICS_DIR / "embeddings_ws_prec.meta.json").exists():
            try:
                meta = json.load(open(ANALYTICS_DIR / "embeddings_ws_prec.meta.json", "r", encoding="utf-8"))
                # Espera chaves labels_ws (opcional)
                ws_labels = meta.get("labels_ws", meta.get("labels", []))
            except Exception:
                pass
        analytics["ws_vecs"] = ensure_2d(ws_vecs)
        analytics["ws_labels"] = ws_labels

        # Prec
        prec_vecs = load_npz_any("prec_vectors.npz", "embeddings_ws_prec.npz")
        prec_labels_jsonl = ANALYTICS_DIR / "prec_labels.jsonl"
        prec_labels = [r.get("label", "") for r in read_jsonl(prec_labels_jsonl)] if prec_labels_jsonl.exists() else []
        if not prec_labels and (ANALYTICS_DIR / "precursors.csv").exists():
            try:
                dfp = pd.read_csv(ANALYTICS_DIR / "precursors.csv")
                # tenta colunas comuns
                if "Precursor" in dfp.columns:
                    prec_labels = dfp["Precursor"].astype(str).tolist()
                elif "precursor" in dfp.columns:
                    prec_labels = dfp["precursor"].astype(str).tolist()
            except Exception:
                pass
        analytics["prec_vecs"] = ensure_2d(prec_vecs)
        analytics["prec_labels"] = prec_labels

        # CP
        cp_vecs = load_npz_any("cp_vectors.npz", "embeddings_cp.npz")
        cp_labels_jsonl = ANALYTICS_DIR / "cp_labels.jsonl"
        cp_labels = [r.get("label", "") for r in read_jsonl(cp_labels_jsonl)] if cp_labels_jsonl.exists() else []
        if not cp_labels and (ANALYTICS_DIR / "embeddings_cp.meta.json").exists():
            try:
                meta = json.load(open(ANALYTICS_DIR / "embeddings_cp.meta.json", "r", encoding="utf-8"))
                cp_labels = meta.get("labels", [])
            except Exception:
                pass
        analytics["cp_vecs"] = ensure_2d(cp_vecs)
        analytics["cp_labels"] = cp_labels

        # Edges WS -> Prec (HTO)
        edges_path_candidates = [
            ANALYTICS_DIR / "ws_precursors_edges.csv",
            ANALYTICS_DIR / "edges_ws_prec.csv",
        ]
        for p in edges_path_candidates:
            if p.exists():
                try:
                    df_e = pd.read_csv(p)
                    # Normaliza nomes de colunas
                    cols = {c.lower().strip(): c for c in df_e.columns}
                    # queremos "weak signal" / "precursor" / "hto"
                    ws_col = cols.get("weak signal") or cols.get("weak_signal") or cols.get("ws")
                    pr_col = cols.get("precursor") or cols.get("prec")
                    hto_col = cols.get("hto") or cols.get("categoria") or cols.get("human")
                    if ws_col and pr_col:
                        if not hto_col:
                            df_e["hto"] = ""
                            hto_col = "hto"
                        analytics["edges"] = df_e.rename(columns={ws_col: "weak signal", pr_col: "precursor", hto_col: "hto"})
                        break
                except Exception:
                    continue

    except Exception as e:
        st.warning(f"Não foi possível carregar analytics: {e}")
    return analytics

analytics = load_analytics()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Configurações")
with st.sidebar.expander("Ollama Cloud", expanded=True):
    st.write("Host:", OLLAMA_HOST)
    st.write("Modelo (chat):", OLLAMA_MODEL)
    st.write("Modelo (embeddings):", OLLAMA_EMBED_MODEL)

topk = st.sidebar.slider("Top-K contexto (RAG total)", 1, 10, 4, 1)
thr_upload = st.sidebar.slider("Limiar sim. uploads", 0.10, 0.95, 0.35, 0.01)
thr_hist   = st.sidebar.slider("Limiar sim. histórico", 0.10, 0.95, 0.35, 0.01)
chunk_size = st.sidebar.slider("Tamanho do chunk (caracteres)", 500, 2000, 1200, 50)
chunk_overlap = st.sidebar.slider("Overlap do chunk", 50, 600, 200, 10)
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

with st.sidebar.expander("Diagnóstico de conexão", expanded=False):
    if st.button("Ping /api/tags", use_container_width=True):
        try:
            r = requests.get(f"{OLLAMA_HOST}/api/tags", headers={"Authorization": OLLAMA_API_KEY}, timeout=20)
            st.write("Status:", r.status_code)
            if r.headers.get("content-type","").startswith("application/json"):
                st.json(r.json())
            else:
                st.write(r.text[:1000])
        except Exception as e:
            st.error(f"Falhou: {e}")
    c1, c2 = st.columns(2)
    with c1:
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
    with c2:
        if st.button("Teste /api/embed", use_container_width=True):
            try:
                r = requests.post(
                    f"{OLLAMA_HOST}/api/embed",
                    headers=HEADERS_JSON,
                    json={"model": OLLAMA_EMBED_MODEL, "input":["teste de embedding"]},
                    timeout=60)
                st.write("Status:", r.status_code)
                st.json(r.json())
            except Exception as e:
                st.error(f"Falhou: {e}")

with st.sidebar.expander("Arquivos em analytics", expanded=False):
    st.caption(f"Analytics: {ANALYTICS_DIR}")
    try:
        files = sorted(os.listdir(ANALYTICS_DIR)) if ANALYTICS_DIR.exists() else []
        if files:
            st.code("\n".join(files), language="bash")
        else:
            st.info("Diretório não existe ou vazio.")
    except Exception as e:
        st.warning(f"Erro ao listar analytics: {e}")

# -------------------------
# Indexação (uploads) para RAG
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
                batch = new_chunks[i: i+BATCH]
                try:
                    vecs = ollama_embed(batch, model=OLLAMA_EMBED_MODEL)
                    embs.extend(vecs)
                except Exception as e:
                    st.error(f"Erro ao gerar embeddings: {e}")
                    st.stop()
            embs = np.asarray(embs, dtype=np.float32)

            if st.session_state.index["embeddings"] is None:
                st.session_state.index["chunks"] = new_chunks
                st.session_state.index["embeddings"] = embs
                st.session_state.index["metas"] = new_metas
            else:
                st.session_state.index["chunks"].extend(new_chunks)
                st.session_state.index["metas"].extend(new_metas)
                st.session_state.index["embeddings"] = np.vstack([st.session_state.index["embeddings"], embs])

            st.success(f"Indexados {len(new_chunks)} chunks de uploads.")

# -------------------------
# UI principal
# -------------------------
st.title("ESO • CHAT — Ollama Cloud (RAG com histórico + uploads)")
st.caption("Histórico (analytics) sempre no RAG; uploads entram no RAG conforme enviados.")

# histórico visual
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Digite sua pergunta…")

def build_context_blocks(question: str, topk_total: int, thr_hist: float, thr_upload: float):
    blocks = []

    # 1) RAG do histórico (sempre que houver)
    V_hist = analytics.get("history_vecs", np.zeros((0,1), dtype=np.float32))
    hist_texts = analytics.get("history_texts", [])
    if V_hist.size > 0 and hist_texts:
        try:
            q_vec = np.asarray(ollama_embed(question, model=OLLAMA_EMBED_MODEL)[0], dtype=np.float32)
            q_vec = ensure_2d(q_vec)
            sims = cosine_sim(q_vec, V_hist)[0]  # (n_hist,)
            order = np.argsort(-sims)
            hits = []
            for idx in order[: max(50, topk_total)]:
                if sims[idx] >= thr_hist and idx < len(hist_texts):
                    row = hist_texts[idx]
                    txt = row.get("text", "")
                    src = row.get("source", "history")
                    hits.append((float(sims[idx]), f"[HIST:{src}#{idx}]", txt))
            # pega até topk_total/2 do histórico
            keep = max(1, topk_total // 2)
            for s, tag, txt in hits[:keep]:
                blocks.append(f"{tag} (sim={s:.3f})\n{txt}")
        except Exception as e:
            st.warning(f"RAG histórico desativado nesta mensagem (erro embeds): {e}")

    # 2) RAG dos uploads (se houver)
    idx = st.session_state.index
    if idx["embeddings"] is not None and len(idx["chunks"]) > 0:
        try:
            q_vec = np.asarray(ollama_embed(question, model=OLLAMA_EMBED_MODEL)[0], dtype=np.float32)
            q_vec = ensure_2d(q_vec)
            V = idx["embeddings"]
            sims = cosine_sim(q_vec, V)[0]  # (n_uploads,)
            order = np.argsort(-sims)
            hits = []
            for j in order[: max(50, topk_total)]:
                if sims[j] >= thr_upload:
                    meta = idx["metas"][j]
                    txt = idx["chunks"][j]
                    hits.append((float(sims[j]), f"[UPLOAD:{meta['file']}#{meta['chunk_id']}]", txt))
            # os restantes até completar topk_total
            remaining = max(0, topk_total - len(blocks))
            for s, tag, txt in hits[:remaining]:
                blocks.append(f"{tag} (sim={s:.3f})\n{txt}")
        except Exception as e:
            st.warning(f"RAG uploads desativado nesta mensagem (erro embeds): {e}")

    return blocks

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context_blocks = build_context_blocks(prompt, topk_total=topk, thr_hist=thr_hist, thr_upload=thr_upload)

    SYSTEM_PROMPT = (
        "Você é um assistente para gestão de segurança operacional. "
        "Responda de forma objetiva, cite trechos do contexto quando relevante e "
        "seja transparente quando não houver informação suficiente.\n"
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
# Painel: status do índice
# -------------------------
with st.expander("📚 Status do índice (RAG)", expanded=False):
    idx = st.session_state.index
    n_chunks = len(idx["chunks"])
    st.write(f"Chunks indexados (uploads): **{n_chunks}**")
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

with st.expander("🧠 Status do histórico (analytics)", expanded=False):
    st.write("Vetores do histórico:", analytics["history_vecs"].shape)
    st.write("Textos do histórico:", len(analytics["history_texts"]))
    st.write("WS (labels/vetores):", len(analytics["ws_labels"]), analytics["ws_vecs"].shape)
    st.write("Precursores (labels/vetores):", len(analytics["prec_labels"]), analytics["prec_vecs"].shape)
    st.write("Taxonomia CP (labels/vetores):", len(analytics["cp_labels"]), analytics["cp_vecs"].shape)
    st.write("Edges WS→Prec:", analytics["edges"].shape)
