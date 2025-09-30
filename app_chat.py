# app_chat.py
# ESO • CHAT — HIST + UPLD (APENAS EMBEDDINGS)
# - Carrega embeddings pré-gerados (Sphera/GoSee/Docs/WS/Prec/CP) de data/analytics/*.npz
# - Gera embeddings para UPLOADs em sessão (Sentence-Transformers)
# - Busca por similaridade (cosine) e injeta trechos como contexto para o LLM
# - NENHUM TF-IDF. Sem joblib/sklearn.
# - Requer: sentence-transformers instalado. (Usa o mesmo modelo dos seus .npz)

import os
import io
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st

# ---- Leitores leves (upload) ----
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import docx
except Exception:
    docx = None

# ---- Encoder (ST) ----
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError(
        "sentence-transformers não está instalado. "
        "Instale-o no ambiente do app para usar apenas embeddings."
    )

st.set_page_config(page_title="ESO • CHAT — (Embeddings Only)", page_icon="💬", layout="wide")

# -------------------------
# Config / Paths
# -------------------------
DATA_DIR = "data"
AN_DIR   = os.path.join(DATA_DIR, "analytics")

EMB_SPH = os.path.join(AN_DIR, "sphera_embeddings.npz")
EMB_GOS = os.path.join(AN_DIR, "gosee_embeddings.npz")
EMB_HIS = os.path.join(AN_DIR, "history_embeddings.npz")
EMB_WS  = os.path.join(AN_DIR, "ws_embeddings.npz")
EMB_PRE = os.path.join(AN_DIR, "prec_embeddings.npz")
EMB_CP  = os.path.join(AN_DIR, "cp_embeddings.npz")

PQ_SPH  = os.path.join(AN_DIR, "sphera.parquet")
PQ_GOS  = os.path.join(AN_DIR, "gosee.parquet")
HIS_JSON= os.path.join(AN_DIR, "history_texts.jsonl")

CATALOG_FILE = "datasets_context.md"  # opcional (texto)

# LLM (Ollama/OpenAI-compat)
OLLAMA_HOST  = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gpt-oss:20b"))
OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY", None))
HEADERS_JSON = {"Authorization": f"Bearer {OLLAMA_API_KEY}", "Content-Type": "application/json"} if OLLAMA_API_KEY else {"Content-Type":"application/json"}

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

def l2n(M: np.ndarray) -> np.ndarray:
    M = M.astype(np.float32, copy=False)
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return M / n

def cos_topk(E_db: np.ndarray, q: np.ndarray, k: int) -> list[tuple[int,float]]:
    if E_db is None or q is None or E_db.size == 0:
        return []
    q = q.astype(np.float32, copy=False)
    q = q / (np.linalg.norm(q) + 1e-9)
    sims = E_db @ q  # (n,)
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]

def load_embeddings_npz(path):
    """
    Espera arquivos produzidos pelo make_catalog_embeddings.py:
      keys: 'embeddings' (float32 L2), 'ids' (object), 'texts' (object)
    Carrega defensivamente se alguma chave faltar.
    """
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=True)
        keys = set(z.keys())
        if "embeddings" not in keys:
            # tenta achar alguma 2D
            best_k, best_n = None, -1
            for k in z.keys():
                arr = z[k]
                if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] > best_n:
                    best_k, best_n = k, arr.shape[0]
            E = z[best_k].astype(np.float32)
        else:
            E = z["embeddings"].astype(np.float32)
        # normaliza por segurança
        E = l2n(E)
        ids   = z["ids"]   if "ids"   in keys else np.arange(E.shape[0], dtype=object)
        texts = z["texts"] if "texts" in keys else np.array([""]*E.shape[0], dtype=object)
        return {"E": E, "ids": ids, "texts": texts}
    except Exception as e:
        st.warning(f"Falha ao ler {path}: {e}")
        return None

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

# Upload em sessão (APENAS embeddings)
if "upld_texts" not in st.session_state:
    st.session_state.upld_texts = []     # [str]
if "upld_metas" not in st.session_state:
    st.session_state.upld_metas = []     # [{"file":..., "chunk_id":...}]
if "upld_emb" not in st.session_state:
    st.session_state.upld_emb = None     # np.ndarray (n, d)

# Encoder em sessão
if "st_model_name" not in st.session_state:
    st.session_state.st_model_name = os.getenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
if "st_encoder" not in st.session_state:
    st.session_state.st_encoder = None

def ensure_encoder():
    if st.session_state.st_encoder is None:
        st.session_state.st_encoder = SentenceTransformer(st.session_state.st_model_name)

def encode_texts(texts: list[str]) -> np.ndarray:
    ensure_encoder()
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    M = st.session_state.st_encoder.encode(
        texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
    ).astype(np.float32)
    return M

def encode_query(text: str) -> np.ndarray:
    ensure_encoder()
    v = st.session_state.st_encoder.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-9)
    return v

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Configurações")
with st.sidebar.expander("Modelo de embeddings", expanded=False):
    st.write("Sentence-Transformers:", st.session_state.st_model_name)

with st.sidebar.expander("LLM (Ollama-compat)", expanded=False):
    st.write("Host:", OLLAMA_HOST)
    st.write("Model:", OLLAMA_MODEL)
    if not OLLAMA_API_KEY:
        st.info("Sem OLLAMA_API_KEY — ok se seu endpoint não exigir auth.")

st.sidebar.divider()
st.sidebar.subheader("RAG • Parâmetros")
topk_upload = st.sidebar.slider("Top-K (UPLOAD)", 1, 10, 4, 1)
topk_hist   = st.sidebar.slider("Top-K (Sphera/GoSee/Docs)", 1, 10, 5, 1)
include_ws_prec_cp = st.sidebar.checkbox("Inferir WS/Precursores/CP do upload (semântica)", True)
upload_chunk_size  = st.sidebar.slider("Chunk (upload) — caracteres", 500, 2000, 1200, 50)
upload_overlap     = st.sidebar.slider("Overlap (upload)", 50, 600, 200, 10)
upload_raw_max     = st.sidebar.slider("Tamanho máx. do bloco UPLOAD_RAW (chars)", 500, 6000, 2500, 100)
use_catalog        = st.sidebar.checkbox("Injetar datasets_context.md", True)

st.sidebar.divider()
uploaded_files = st.sidebar.file_uploader("Upload (PDF, DOCX, XLSX, CSV, TXT/MD)",
                                          type=["pdf","docx","xlsx","xls","csv","txt","md"],
                                          accept_multiple_files=True)
col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Limpar uploads", use_container_width=True):
        st.session_state.upld_texts = []
        st.session_state.upld_metas = []
        st.session_state.upld_emb   = None
with col_b:
    if st.button("Limpar chat", use_container_width=True):
        st.session_state.chat = []

# -------------------------
# Ingest de upload (APENAS embeddings)
# -------------------------
def add_uploads(files):
    new_texts, new_metas = [], []
    for uf in files:
        try:
            txt = read_any(uf)
            parts = chunk_text(txt, max_chars=upload_chunk_size, overlap=upload_overlap)
            for i, p in enumerate(parts):
                new_texts.append(p)
                new_metas.append({"file": uf.name, "chunk_id": i})
        except Exception as e:
            st.warning(f"Falha ao processar {uf.name}: {e}")
    if new_texts:
        # agrega
        st.session_state.upld_texts.extend(new_texts)
        st.session_state.upld_metas.extend(new_metas)
        # (re)embeda
        st.session_state.upld_emb = encode_texts(st.session_state.upld_texts)
        st.success(f"Upload indexado: {len(new_texts)} chunks.")

if uploaded_files:
    with st.spinner("Lendo e embedando uploads…"):
        add_uploads(uploaded_files)

def get_upload_raw_text(max_chars=2500) -> str:
    texts = st.session_state.upld_texts or []
    if not texts:
        return ""
    out, total = [], 0
    for t in texts[:3]:  # até 3 chunks
        if total + len(t) > max_chars:
            out.append(t[:max(0, max_chars-total)])
            break
        out.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return "\n".join(out).strip()

# -------------------------
# Carrega bases (embeddings + textos)
# -------------------------
sph_emb = load_embeddings_npz(EMB_SPH)
gos_emb = load_embeddings_npz(EMB_GOS)
his_emb = load_embeddings_npz(EMB_HIS)
ws_emb  = load_embeddings_npz(EMB_WS)
pre_emb = load_embeddings_npz(EMB_PRE)
cp_emb  = load_embeddings_npz(EMB_CP)

sph_df, gos_df, history_rows = None, None, []

if os.path.exists(PQ_SPH):
    try:
        sph_df = pd.read_parquet(PQ_SPH)
    except Exception as e:
        st.warning(f"Falha ao ler {PQ_SPH}: {e}")

if os.path.exists(PQ_GOS):
    try:
        gos_df = pd.read_parquet(PQ_GOS)
    except Exception as e:
        st.warning(f"Falha ao ler {PQ_GOS}: {e}")

if os.path.exists(HIS_JSON):
    try:
        with open(HIS_JSON, "r", encoding="utf-8") as f:
            for line in f:
                history_rows.append(json.loads(line))
    except Exception as e:
        st.warning(f"Falha ao ler {HIS_JSON}: {e}")

catalog_ctx = ""
if os.path.exists(CATALOG_FILE):
    try:
        with open(CATALOG_FILE, "r", encoding="utf-8") as f:
            catalog_ctx = f.read()
    except Exception:
        pass

# -------------------------
# Busca semântica (bases)
# -------------------------
def search_sphera(qv: np.ndarray, k: int) -> list[str]:
    if sph_emb is None or sph_df is None or qv is None:
        return []
    hits = cos_topk(sph_emb["E"], qv, k)
    col_desc = "Description" if "Description" in sph_df.columns else sph_df.columns[0]
    out = []
    for i, s in hits:
        row = sph_df.iloc[i]
        evid = row.get("Event ID", row.get("EVENT_NUMBER", f"row{i}"))
        snippet = str(row.get(col_desc, ""))[:700]
        out.append(f"[Sphera/{evid}] (sim={s:.3f})\n{snippet}")
    return out

def search_gosee(qv: np.ndarray, k: int) -> list[str]:
    if gos_emb is None or gos_df is None or qv is None:
        return []
    hits = cos_topk(gos_emb["E"], qv, k)
    col_obs = "Observation" if "Observation" in gos_df.columns else gos_df.columns[0]
    out = []
    for i, s in hits:
        row = gos_df.iloc[i]
        gid = row.get("ID", f"row{i}")
        snippet = str(row.get(col_obs, ""))[:700]
        out.append(f"[GoSee/{gid}] (sim={s:.3f})\n{snippet}")
    return out

def search_docs(qv: np.ndarray, k: int) -> list[str]:
    if his_emb is None or not history_rows or qv is None:
        return []
    hits = cos_topk(his_emb["E"], qv, k)
    out = []
    for i, s in hits:
        r = history_rows[i]
        tag = f"Docs/{r.get('source','?')}/{r.get('chunk_id',0)}"
        snippet = str(r.get("text", ""))[:700]
        out.append(f"[{tag}] (sim={s:.3f})\n{snippet}")
    return out

def search_upload_chunks(qv: np.ndarray, k: int) -> list[str]:
    E = st.session_state.upld_emb
    texts = st.session_state.upld_texts
    metas = st.session_state.upld_metas
    if E is None or E.size == 0 or qv is None:
        return []
    hits = cos_topk(E, qv, k)
    out = []
    for i, s in hits:
        m = metas[i]
        t = texts[i][:700]
        out.append(f"[UPLOAD {m['file']} / {m['chunk_id']}] (sim={s:.3f})\n{t}")
    return out

def infer_labels_from_upload(k_each=5) -> list[str]:
    """Inferir WS / Precursores / CP a partir do centróide dos embeddings do upload (se houver)."""
    E = st.session_state.upld_emb
    if E is None or E.size == 0:
        return []
    centroid = E.mean(axis=0).astype(np.float32)
    centroid /= (np.linalg.norm(centroid) + 1e-9)

    out = []
    def pick(db, tag):
        hits = cos_topk(db["E"], centroid, k_each)
        labels = []
        texts  = db["texts"]
        for i, s in hits:
            # Mostra o próprio texto/label curto (primeiros 120 chars)
            lbl = str(texts[i]) if i < len(texts) else f"{tag}-{i}"
            labels.append(f"{lbl} (sim={s:.2f})")
        if labels:
            out.append(f"[{tag}] Prováveis: " + "; ".join(labels))

    if ws_emb is not None:
        pick(ws_emb, "WeakSignals")
    if pre_emb is not None:
        pick(pre_emb, "Precursores")
    if cp_emb is not None:
        pick(cp_emb, "CP-Taxonomia")

    return out

# -------------------------
# UI
# -------------------------
st.title("ESO • CHAT — HIST + UPLD (Embeddings Only)")
st.caption("RAG semântico com embeddings pré-gerados (Sphera/GoSee/Docs) e upload embeddado em sessão. Sem TF-IDF.")

# Histórico
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
prompt = st.chat_input("Digite sua pergunta…")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Embedding da query
    try:
        qv = encode_query(prompt)
    except Exception as e:
        st.error(f"Falha ao gerar embedding da consulta: {e}")
        qv = None

    # Recuperações (sempre em embeddings)
    up_blocks  = search_upload_chunks(qv, k=topk_upload) if qv is not None else []
    sph_blocks = search_sphera(qv, k=topk_hist) if qv is not None else []
    gos_blocks = search_gosee(qv, k=topk_hist) if qv is not None else []
    his_blocks = search_docs(qv, k=topk_hist)  if qv is not None else []

    # Opcional: rótulos semânticos inferidos do upload
    labels_blocks = infer_labels_from_upload(k_each=5) if include_ws_prec_cp else []

    # Upload RAW: garantir que o conteúdo do arquivo entre no prompt
    upload_raw = get_upload_raw_text(upload_raw_max)
    raw_block = [f"[UPLOAD_RAW]\n{upload_raw}"] if upload_raw else []

    # Monta contexto (ordem: upload raw, labels inferidos, upload knn, sphera, gosee, docs)
    context_blocks = raw_block + labels_blocks + up_blocks + sph_blocks + gos_blocks + his_blocks

    SYSTEM = (
        "Você é um assistente de segurança operacional. "
        "Use os CONTEXTOS a seguir como evidências e cite trechos quando fizer afirmações específicas. "
        "Se pedirem contagem/listas, responda SOMENTE com base nos CONTEXTOS. "
        "Evite extrapolações sem amparo textual."
    )

    messages = [{"role": "system", "content": SYSTEM}]
    if use_catalog and os.path.exists(CATALOG_FILE):
        messages.append({"role": "system", "content": catalog_ctx})

    if context_blocks:
        ctx = "\n\n".join(context_blocks)
        messages.append({"role": "user", "content": f"CONTEXTOS (UPLOAD + HIST):\n{ctx}"} )
        messages.append({"role": "user", "content": f"PERGUNTA: {prompt}"} )
    else:
        messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Consultando o modelo…"):
            try:
                resp = ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2, stream=False)
                content = resp.get("message", {}).get("content", "").strip() or json.dumps(resp)[:1200]
            except Exception as e:
                content = f"Falha ao consultar o modelo: {e}"
            st.markdown(content)

    st.session_state.chat.append({"role": "assistant", "content": content})

# -------------------------
# Painel de Status
# -------------------------
with st.expander("📦 Status dos índices", expanded=False):
    st.write("Embeddings Sphera:", "✅" if sph_emb is not None and sph_df is not None else "—")
    st.write("Embeddings GoSee :", "✅" if gos_emb is not None and gos_df is not None else "—")
    st.write("Embeddings Docs  :", "✅" if his_emb is not None and history_rows else "—")
    st.write("Embeddings WS    :", "✅" if ws_emb is not None else "—")
    st.write("Embeddings Prec  :", "✅" if pre_emb is not None else "—")
    st.write("Embeddings CP    :", "✅" if cp_emb is not None else "—")
    st.write("Encoder (ST)     :", "✅" if st.session_state.st_encoder is not None else "—")
    st.write("Uploads indexados:", len(st.session_state.upld_texts))
