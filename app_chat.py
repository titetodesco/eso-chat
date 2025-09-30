# app_chat.py
# ESO • CHAT — HIST + UPLD (Embeddings preferencial; TF‑IDF como fallback)
# - HIST: carrega embeddings pré‑gerados (data/analytics/*_embeddings.npz)
#         + tabelas de texto (sphera.parquet, gosee.parquet, history_texts.jsonl)
# - UPLD: arquivos enviados no momento (TF‑IDF por padrão; opcional: embeddings se houver ST)
# - Combina resultados (pesos) e injeta no prompt
# - Injeta datasets_context.md (texto) como "catálogo" opcional
#
# Requisitos mínimos: streamlit, requests, numpy, pandas, pyarrow
# Opcionais: pypdf, python-docx, joblib, scikit-learn, sentence-transformers

import os
import io
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st

# --- Opcionais ---
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import docx
except Exception:
    docx = None

try:
    import joblib
except Exception:
    joblib = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# -------------------------
st.set_page_config(page_title="ESO • CHAT (Embeddings | TF‑IDF)", page_icon="💬", layout="wide")

DATA_DIR = "data"
ANALYTICS_DIR = os.path.join(DATA_DIR, "analytics")
DATASETS_CONTEXT_FILE = "datasets_context.md"

# -------------------------
# Segredos / Modelo de chat
# -------------------------
OLLAMA_HOST  = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gpt-oss:20b"))
OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY", None))

HEADERS_JSON = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}

# -------------------------
# Utilitários de leitura de arquivos de upload
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
    d = docx.Document(io.BytesIO(b))
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


# -------------------------
# Carrega embeddings pré‑gerados e tabelas de texto
# -------------------------

def _npz_load_any(path):
    if not os.path.exists(path):
        return None
    try:
        npz = np.load(path, allow_pickle=True)
        emb = (npz.get("embeddings") or npz.get("X") or npz.get("E") or npz.get("arr_0"))
        ids = (npz.get("ids") or npz.get("labels") or npz.get("arr_1"))
        if emb is None:
            return None
        emb = np.asarray(emb, dtype=np.float32)
        if ids is None:
            ids = np.arange(emb.shape[0])
        return {"emb": emb, "ids": ids}
    except Exception as e:
        st.warning(f"Falha ao ler {path}: {e}")
        return None


def _l2norm(M):
    M = np.asarray(M, dtype=np.float32)
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return M / n


def _cosine_topk(M_norm, q_norm, topk=5, thr=0.0):
    sims = (M_norm @ q_norm.reshape(-1, 1)).ravel()
    order = np.argsort(-sims)
    out = []
    for i in order[: max(topk * 4, topk)]:
        s = float(sims[i])
        if s < thr:
            continue
        out.append((i, s))
        if len(out) >= topk:
            break
    return out


@st.cache_resource(show_spinner=False)
def _load_parquet(path: str):
    return pd.read_parquet(path)


@st.cache_resource(show_spinner=False)
def _load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# Embeddings + textos
SPH_EMB_PATH = os.path.join(ANALYTICS_DIR, "sphera_embeddings.npz")
GOS_EMB_PATH = os.path.join(ANALYTICS_DIR, "gosee_embeddings.npz")
HIS_EMB_PATH = os.path.join(ANALYTICS_DIR, "history_embeddings.npz")

SPH_TBL_PATH = os.path.join(ANALYTICS_DIR, "sphera.parquet")
GOS_TBL_PATH = os.path.join(ANALYTICS_DIR, "gosee.parquet")
HIS_TXT_PATH = os.path.join(ANALYTICS_DIR, "history_texts.jsonl")

sph_emb = _npz_load_any(SPH_EMB_PATH)
gos_emb = _npz_load_any(GOS_EMB_PATH)
his_emb = _npz_load_any(HIS_EMB_PATH)

sph_df = _load_parquet(SPH_TBL_PATH) if os.path.exists(SPH_TBL_PATH) else None
gos_df = _load_parquet(GOS_TBL_PATH) if os.path.exists(GOS_TBL_PATH) else None
his_rows = _load_jsonl(HIS_TXT_PATH) if os.path.exists(HIS_TXT_PATH) else []

sph_M = _l2norm(sph_emb["emb"]) if sph_emb else None
gos_M = _l2norm(gos_emb["emb"]) if gos_emb else None
his_M = _l2norm(his_emb["emb"]) if his_emb else None

# coluna de texto padrão
sph_text_col = "Description" if (sph_df is not None and "Description" in sph_df.columns) else (sph_df.columns[0] if sph_df is not None else None)
gos_text_col = "Observation" if (gos_df is not None and "Observation" in gos_df.columns) else (gos_df.columns[0] if gos_df is not None else None)

# -------------------------
# Embedding de consulta (Sentence-Transformers opcional)
# -------------------------
@st.cache_resource(show_spinner=False)
def _load_st_model(name="sentence-transformers/all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(name)
    except Exception as e:
        st.warning(f"Não consegui carregar SentenceTransformer: {e}")
        return None


# -------------------------
# Estado de sessão
# -------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "upload_index" not in st.session_state:
    st.session_state.upload_index = {
        "texts": [],
        "metas": [],  # {file, chunk_id}
        "vec": None,  # TF‑IDF vectorizer
        "X": None,    # TF‑IDF matrix
        "embU": None, # Upload embeddings (opcional)
    }

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Configurações")
st.sidebar.write("Host:", OLLAMA_HOST)
st.sidebar.write("Modelo:", OLLAMA_MODEL)

st.sidebar.divider()
st.sidebar.subheader("RAG • Pesos & Limiar")
w_upload = st.sidebar.slider("Peso do UPLOAD", 0.0, 1.0, 0.7, 0.05)
w_hist   = 1.0 - w_upload
thr_upload = st.sidebar.slider("Limiar (UPLOAD)", 0.0, 1.0, 0.20, 0.01)
thr_hist   = st.sidebar.slider("Limiar (HIST)",   0.0, 1.0, 0.25, 0.01)
topk_upload = st.sidebar.slider("Top‑K Upload", 1, 15, 5, 1)
topk_hist   = st.sidebar.slider("Top‑K Histórico", 1, 15, 5, 1)
chunk_size  = st.sidebar.slider("Tamanho do chunk (upload)", 500, 2000, 1200, 50)
chunk_overlap = st.sidebar.slider("Overlap do chunk (upload)", 50, 600, 200, 10)
use_catalog = st.sidebar.checkbox("Injetar datasets_context.md", True)

st.sidebar.divider()
force_embeddings = st.sidebar.checkbox("Forçar EMBEDDINGS (desliga TF‑IDF)", True)
model = _load_st_model() if force_embeddings else None

uploaded_files = st.sidebar.file_uploader(
    "Upload (PDF, DOCX, XLSX, CSV, TXT/MD)",
    type=["pdf","docx","xlsx","xls","csv","txt","md"],
    accept_multiple_files=True
)

col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Limpar uploads", use_container_width=True):
        st.session_state.upload_index = {"texts": [], "metas": [], "vec": None, "X": None, "embU": None}
with col_b:
    if st.button("Limpar chat", use_container_width=True):
        st.session_state.chat = []

# -------------------------
# Indexação de UPLOADS (TF‑IDF por sessão; embeddings opcionais)
# -------------------------

def rebuild_upload_index_tfidf():
    texts = st.session_state.upload_index["texts"]
    if TfidfVectorizer is None or not texts:
        st.session_state.upload_index["vec"] = None
        st.session_state.upload_index["X"] = None
        return
    vec = TfidfVectorizer(lowercase=True, strip_accents="unicode", analyzer="word", ngram_range=(1,2), max_features=50000)
    X = vec.fit_transform(texts)
    st.session_state.upload_index["vec"] = vec
    st.session_state.upload_index["X"] = X


def embed_upload_texts_if_possible():
    if model is None:
        return
    texts = st.session_state.upload_index["texts"]
    if not texts:
        return
    try:
        U = model.encode(texts, normalize_embeddings=True).astype(np.float32)
        st.session_state.upload_index["embU"] = U
    except Exception as e:
        st.warning(f"Falha ao embutir uploads: {e}")
        st.session_state.upload_index["embU"] = None


if uploaded_files:
    with st.spinner("Lendo arquivos e indexando…"):
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
            st.session_state.upload_index["texts"].extend(new_texts)
            st.session_state.upload_index["metas"].extend(new_metas)
            rebuild_upload_index_tfidf()
            embed_upload_texts_if_possible()
            st.success(f"Upload indexado: {len(new_texts)} chunks.")

# -------------------------
# Busca HIST por embeddings (preferencial)
# -------------------------

def _embed_query(text: str):
    if model is None:
        return None
    try:
        v = model.encode([text], normalize_embeddings=True)
        return v.astype(np.float32)[0]
    except Exception as e:
        st.warning(f"Falha ao embutir consulta (ST): {e}")
        return None


def search_hist_embeddings(query: str, topk_hist: int, thr_hist: float):
    blocks = []
    q_vec = _embed_query(query)

    # Sphera
    if sph_M is not None and sph_df is not None and sph_text_col:
        if q_vec is not None:
            hits = _cosine_topk(sph_M, q_vec, topk=topk_hist, thr=thr_hist)
            for idx, s in hits:
                row = sph_df.iloc[idx]
                ev_id = row.get("Event ID", row.get("EVENT_NUMBER", f"row{idx}"))
                txt   = str(row.get(sph_text_col, ""))[:800]
                blocks.append((s, f"[Sphera/{ev_id}] (sim={s:.3f})\n{txt}"))
    # GoSee
    if gos_M is not None and gos_df is not None and gos_text_col:
        if q_vec is not None:
            hits = _cosine_topk(gos_M, q_vec, topk=topk_hist, thr=thr_hist)
            for idx, s in hits:
                row = gos_df.iloc[idx]
                gid = row.get("ID", f"row{idx}")
                txt = str(row.get(gos_text_col, ""))[:800]
                blocks.append((s, f"[GoSee/{gid}] (sim={s:.3f})\n{txt}"))
    # History (docs)
    if his_M is not None and his_rows:
        if q_vec is not None:
            hits = _cosine_topk(his_M, q_vec, topk=topk_hist, thr=thr_hist)
            for idx, s in hits:
                row = his_rows[idx]
                src = f"{row.get('source','?')}/{row.get('chunk_id',idx)}"
                txt = str(row.get("text", ""))[:800]
                blocks.append((s, f"[Docs/{src}] (sim={s:.3f})\n{txt}"))

    blocks.sort(key=lambda x: -x[0])
    return [b for _, b in blocks[:topk_hist]]


# -------------------------
# Busca UPLOAD (embeddings se possível; senão TF‑IDF)
# -------------------------

def search_upload(query: str, topk_upload: int, thr_upload: float):
    texts = st.session_state.upload_index["texts"]
    metas = st.session_state.upload_index["metas"]
    U = st.session_state.upload_index.get("embU")

    blocks = []
    if texts:
        if model is not None and U is not None:
            q_vec = _embed_query(query)
            if q_vec is not None:
                hits = _cosine_topk(U, q_vec, topk=topk_upload, thr=thr_upload)
                for idx, s in hits:
                    m = metas[idx]; t = texts[idx][:800]
                    blocks.append((s, f"[UPLOAD {m['file']} / {m['chunk_id']}] (sim={s:.3f})\n{t}"))
        else:
            # TF‑IDF fallback
            vec = st.session_state.upload_index["vec"]
            X = st.session_state.upload_index["X"]
            if vec is not None and X is not None and TfidfVectorizer is not None:
                q = vec.transform([query])
                sims = (q @ X.T).toarray()[0]
                idxs = np.argsort(-sims)[: topk_upload * 4]
                for i in idxs:
                    s = float(sims[i])
                    if s < thr_upload:
                        continue
                    m = metas[i]; t = texts[i]
                    blocks.append((s, f"[UPLOAD {m['file']} / {m['chunk_id']}] (sim={s:.3f})\n{t}"))
    blocks.sort(key=lambda x: -x[0])
    return [b for _, b in blocks[:topk_upload]]


# -------------------------
# Catálogo (texto)
# -------------------------
try:
    catalog_ctx = open(DATASETS_CONTEXT_FILE, "r", encoding="utf-8").read() if os.path.exists(DATASETS_CONTEXT_FILE) else ""
except Exception:
    catalog_ctx = ""

# -------------------------
# UI principal
# -------------------------
st.title("ESO • CHAT — HIST + UPLD (Embeddings preferencial)")
st.caption("RAG local com embeddings (Sphera/GoSee/Docs) + upload; TF‑IDF usado apenas como fallback quando permitido.")

# Histórico de chat
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Digite sua pergunta…")

# Função de chat remoto (Ollama Cloud compatível com /api/chat)

def ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2, stream=False, timeout=120):
    payload = {"model": model, "messages": messages, "temperature": float(temperature), "stream": bool(stream)}
    url = f"{OLLAMA_HOST}/api/chat"
    r = requests.post(url, headers={**HEADERS_JSON, "Content-Type": "application/json"}, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Recuperação
    up_blocks = search_upload(prompt, topk_upload=topk_upload, thr_upload=thr_upload)
    hi_blocks = search_hist_embeddings(prompt, topk_hist=topk_hist, thr_hist=thr_hist)

    # Combinação com pesos (para ordenação final)
    combined = []
    for b in up_blocks:
        try:
            s = float(b.split("(sim=")[1].split(")")[0])
        except Exception:
            s = 0.0
        combined.append((w_upload * s, b))
    for b in hi_blocks:
        try:
            s = float(b.split("(sim=")[1].split(")")[0])
        except Exception:
            s = 0.0
        combined.append((w_hist * s, b))

    combined.sort(key=lambda x: -x[0])
    context_blocks = [b for _, b in combined]

    SYSTEM = (
        "Você é um assistente de segurança operacional. "
        "Use os CONTEXTOS a seguir como evidências. "
        "Se a pergunta solicitar quantitativos ou listas do histórico, responda SOMENTE com base nos CONTEXTOS. "
        "Entregue respostas objetivas e explicativas com trechos citados."
    )

    messages = [{"role": "system", "content": SYSTEM}]
    if use_catalog and catalog_ctx:
        messages.append({"role": "system", "content": catalog_ctx})

    if context_blocks:
        ctx = "\n\n".join(context_blocks)
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

# -------------------------
# Painel de status
# -------------------------
with st.expander("📦 Status dos índices", expanded=False):
    st.write("Sphera (embeddings):", "✅" if sph_M is not None and sph_df is not None else "—")
    st.write("GoSee  (embeddings):", "✅" if gos_M is not None and gos_df is not None else "—")
    st.write("Docs   (embeddings):", "✅" if his_M is not None and his_rows else "—")
    st.write("Uploads indexados:", len(st.session_state.upload_index.get("texts", [])))
    st.caption(f"Sphera texto: {sph_text_col or '—'} | GoSee texto: {gos_text_col or '—'}")


# -------------------------
# (Opcional) requirements.txt sugerido — mantenha abaixo como comentário
# -------------------------
"""
# requirements.txt (sugerido)
streamlit==1.50.0
requests==2.32.5
pandas==2.2.2
numpy==1.26.4
pyarrow==16.1.0

# Upload parsers (opcional)
pypdf==4.3.1
python-docx==1.1.2

# Fallback TF‑IDF (opcional)
scikit-learn==1.5.1
joblib==1.4.2

# Embeddings de consulta (opcional; usa CPU)
sentence-transformers==2.2.2
# (se seu ambiente exigir, instale também torch CPU)
# torch==2.3.1
"""
