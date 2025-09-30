# app_chat.py
# ESO • CHAT — HIST + UPLD (Embeddings preferencial; TF-IDF como fallback)
# - HIST: Sphera, GoSee, Docs
#   • Preferir embeddings (*.npz) + encoder (Sentence-Transformers OU ONNX local)
#   • Se encoder indisponível: cair para TF-IDF pré-gerado (*.joblib)
# - UPLD: arquivos enviados no momento (TF-IDF local)
# - Combina resultados (pesos) e injeta no prompt
# - Injeta datasets_context.md (opcional)

import os
import io
import json
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

# === Dependências opcionais ===
try:
    import joblib
except Exception:
    joblib = None

# Encoders possíveis para embeddings
_HAS_ST = False      # sentence-transformers
_HAS_ORT = False     # onnxruntime + transformers (tokenizer)
_ST_MODEL = None
_ONNX_SESS = None
_ONNX_TOK = None

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    _HAS_ORT = True
except Exception:
    _HAS_ORT = False

# Parsers leves (apenas para uploads)
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import docx
except Exception:
    docx = None

st.set_page_config(page_title="ESO • CHAT (Embeddings + TF-IDF fallback)", page_icon="💬", layout="wide")

# -------------------------
# Configs / Secrets
# -------------------------
OLLAMA_HOST  = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gpt-oss:20b"))
OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY", None))

HEADERS_JSON = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}

DATA_DIR = "data"
ANALYTICS_DIR = os.path.join(DATA_DIR, "analytics")
DATASETS_CONTEXT_FILE = "datasets_context.md"  # texto (YAML em markdown está OK)

MODELS_ONNX_DIR = "models/all-MiniLM-L6-v2-onnx"  # se existir "model.onnx" + tokenizer.*

# -------------------------
# Utilitários
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

def cosine_sim_dense(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # A:(n,d) B:(m,d) -> (n,m)
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    a = A.astype(np.float32)
    b = B.astype(np.float32)
    a /= (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b /= (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T

def ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2, stream=False, timeout=120):
    if not OLLAMA_API_KEY:
        # Permite rodar sem chave (ambiente local) — server pode estar aberto
        pass
    payload = {"model": model, "messages": messages, "temperature": float(temperature), "stream": bool(stream)}
    r = requests.post(f"{OLLAMA_HOST}/api/chat", headers={**HEADERS_JSON, "Content-Type": "application/json"}, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ==== Loader robusto para embeddings .npz ====
def load_embeddings_npz(path):
    """
    Carrega um .npz de embeddings de forma defensiva.
    Retorna dict: {"E": np.ndarray (n,d) float32 L2-normalizada}
    """
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=True) as npz:
            keys = list(npz.keys())
            cand = None
            for k in ("embeddings", "E", "X", "vecs", "vectors"):
                if k in npz:
                    cand = k
                    break
            if cand is None:
                # fallback: pega maior matriz 2D
                best_k, best_n = None, -1
                for k in keys:
                    arr = npz[k]
                    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] > best_n:
                        best_k, best_n = k, arr.shape[0]
                if best_k is None:
                    raise RuntimeError(f"Nenhuma matriz 2D encontrada em {path}. Chaves: {keys}")
                cand = best_k
            E = np.array(npz[cand])
            E = E.astype(np.float32, copy=False)
            norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
            E = E / norms
            return {"E": E}
    except Exception as e:
        st.warning(f"Falha ao ler {path}: {e}")
        return None

def topk_by_cosine(E_db: np.ndarray, q: np.ndarray, k: int = 5):
    if E_db is None or q is None:
        return []
    q = q.astype(np.float32, copy=False)
    q = q / (np.linalg.norm(q) + 1e-9)
    sims = E_db @ q  # (n,)
    idx = np.argsort(-sims)[:k]
    return list(zip(idx, sims[idx]))

# -------------------------
# Estado
# -------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "upload_index" not in st.session_state:
    st.session_state.upload_index = {"texts": [], "metas": [], "vec": None, "X": None}

# encoder em sessão
if "encoder_mode" not in st.session_state:
    st.session_state.encoder_mode = "none"  # "st" | "onnx" | "none"
if "st_encoder" not in st.session_state:
    st.session_state.st_encoder = None
if "onnx_sess" not in st.session_state:
    st.session_state.onnx_sess = None
if "onnx_tok" not in st.session_state:
    st.session_state.onnx_tok = None

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Configurações")
with st.sidebar.expander("Ollama Cloud", expanded=False):
    st.write("Host:", OLLAMA_HOST)
    st.write("Modelo:", OLLAMA_MODEL)
    if not OLLAMA_API_KEY:
        st.info("Sem OLLAMA_API_KEY — ok para testes locais.")

st.sidebar.divider()
st.sidebar.subheader("RAG • Pesos & Limiar")
w_upload = st.sidebar.slider("Peso do UPLOAD", 0.0, 1.0, 0.7, 0.05)
w_hist   = 1.0 - w_upload
topk_upload = st.sidebar.slider("Top-K Upload", 1, 15, 5, 1)
topk_hist   = st.sidebar.slider("Top-K HIST",   1, 15, 5, 1)
use_catalog = st.sidebar.checkbox("Injetar datasets_context.md", True)
force_embeddings = st.sidebar.checkbox("Usar EMBEDDINGS (se encoder disponível)", True)
upload_chunk_size  = st.sidebar.slider("Tamanho do chunk (upload)", 500, 2000, 1200, 50)
upload_overlap = st.sidebar.slider("Overlap do chunk (upload)", 50, 600, 200, 10)
upload_raw_max = st.sidebar.slider("Tamanho máx. do bloco UPLOAD_RAW (chars)", 500, 6000, 2500, 100)

st.sidebar.divider()
uploaded_files = st.sidebar.file_uploader("Upload (PDF, DOCX, XLSX, CSV, TXT/MD)",
                                          type=["pdf","docx","xlsx","xls","csv","txt","md"],
                                          accept_multiple_files=True)

col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Limpar uploads", use_container_width=True):
        st.session_state.upload_index = {"texts": [], "metas": [], "vec": None, "X": None}
with col_b:
    if st.button("Limpar chat", use_container_width=True):
        st.session_state.chat = []

# -------------------------
# Indexação de UPLOADS (TF-IDF por sessão)
# -------------------------
def rebuild_upload_index():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as e:
        st.warning(f"scikit-learn ausente para TF-IDF de uploads: {e}")
        return
    texts = st.session_state.upload_index["texts"]
    if not texts:
        st.session_state.upload_index["vec"] = None
        st.session_state.upload_index["X"] = None
        return
    vec = TfidfVectorizer(lowercase=True, strip_accents="unicode",
                          analyzer="word", ngram_range=(1,2),
                          max_features=50000)
    X = vec.fit_transform(texts)
    st.session_state.upload_index["vec"] = vec
    st.session_state.upload_index["X"] = X

if uploaded_files:
    with st.spinner("Lendo files e indexando (TF-IDF, local)…"):
        new_texts, new_metas = [], []
        for uf in uploaded_files:
            try:
                text = read_any(uf)
                parts = chunk_text(text, max_chars=upload_chunk_size, overlap=upload_overlap)
                for i, p in enumerate(parts):
                    new_texts.append(p)
                    new_metas.append({"file": uf.name, "chunk_id": i})
            except Exception as e:
                st.warning(f"Falha ao processar {uf.name}: {e}")
        if new_texts:
            st.session_state.upload_index["texts"].extend(new_texts)
            st.session_state.upload_index["metas"].extend(new_metas)
            rebuild_upload_index()
            st.success(f"Upload indexado: {len(new_texts)} chunks.")

# -------------------------
# Carrega HIST (Embeddings preferido; TF-IDF fallback)
# -------------------------
# Embeddings (se existirem)
SPH_EMB_PATH = os.path.join(ANALYTICS_DIR, "sphera_embeddings.npz")
GOS_EMB_PATH = os.path.join(ANALYTICS_DIR, "gosee_embeddings.npz")
HIS_EMB_PATH = os.path.join(ANALYTICS_DIR, "history_embeddings.npz")

sph_emb = load_embeddings_npz(SPH_EMB_PATH)
gos_emb = load_embeddings_npz(GOS_EMB_PATH)
his_emb = load_embeddings_npz(HIS_EMB_PATH)

# Dados de texto/metadata correspondentes
SPH_PQ_PATH = os.path.join(ANALYTICS_DIR, "sphera.parquet")
GOS_PQ_PATH = os.path.join(ANALYTICS_DIR, "gosee.parquet")
HIS_JSONL   = os.path.join(ANALYTICS_DIR, "history_texts.jsonl")

sphera_df = None
gosee_df  = None
history_rows = []

if os.path.exists(SPH_PQ_PATH):
    try:
        sphera_df = pd.read_parquet(SPH_PQ_PATH)
    except Exception as e:
        st.warning(f"Falha ao ler {SPH_PQ_PATH}: {e}")

if os.path.exists(GOS_PQ_PATH):
    try:
        gosee_df = pd.read_parquet(GOS_PQ_PATH)
    except Exception as e:
        st.warning(f"Falha ao ler {GOS_PQ_PATH}: {e}")

if os.path.exists(HIS_JSONL):
    try:
        with open(HIS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                history_rows.append(json.loads(line))
    except Exception as e:
        st.warning(f"Falha ao ler {HIS_JSONL}: {e}")

# TF-IDF fallback (pré-gerado pelo make_catalog_indexes.py)
def load_joblib_safe(path):
    if joblib is None:
        return None
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Não consegui carregar {path}: {e}")
        return None

sphera_tfidf = load_joblib_safe(os.path.join(ANALYTICS_DIR, "sphera_tfidf.joblib"))
gosee_tfidf  = load_joblib_safe(os.path.join(ANALYTICS_DIR, "gosee_tfidf.joblib"))
hist_tfidf   = load_joblib_safe(os.path.join(ANALYTICS_DIR, "history_tfidf.joblib"))

# Catalogo opcional
catalog_ctx = ""
try:
    if os.path.exists(DATASETS_CONTEXT_FILE):
        with open(DATASETS_CONTEXT_FILE, "r", encoding="utf-8") as f:
            catalog_ctx = f.read()
except Exception:
    pass

# -------------------------
# Encoder (embeddings) — inicialização sob demanda
# -------------------------
def ensure_encoder(want_embeddings: bool):
    """
    Inicializa um encoder se embeddings estiverem habilitados e necessário.
    Tenta sentence-transformers; se não, ONNX local; se não, 'none'.
    """
    if not want_embeddings:
        st.session_state.encoder_mode = "none"
        st.session_state.st_encoder = None
        st.session_state.onnx_sess = None
        st.session_state.onnx_tok  = None
        return

    # Já inicializado?
    if st.session_state.encoder_mode in ("st", "onnx"):
        return

    # 1) Sentence-Transformers (se instalado)
    if _HAS_ST:
        try:
            model_name = os.getenv("ST_MODEL_NAME", "all-MiniLM-L6-v2")
            st.session_state.st_encoder = SentenceTransformer(model_name)
            st.session_state.encoder_mode = "st"
            return
        except Exception as e:
            st.info(f"ST indisponível: {e}")

    # 2) ONNX local (se existir model.onnx + tokenizer)
    if _HAS_ORT and os.path.exists(os.path.join(MODELS_ONNX_DIR, "model.onnx")):
        try:
            tok = AutoTokenizer.from_pretrained(MODELS_ONNX_DIR, local_files_only=True)
            sess = ort.InferenceSession(os.path.join(MODELS_ONNX_DIR, "model.onnx"),
                                        providers=["CPUExecutionProvider"])
            st.session_state.onnx_tok = tok
            st.session_state.onnx_sess = sess
            st.session_state.encoder_mode = "onnx"
            return
        except Exception as e:
            st.info(f"ONNX indisponível: {e}")

    # 3) Sem encoder
    st.session_state.encoder_mode = "none"

def encode_query(text: str) -> np.ndarray | None:
    """Gera embedding L2-normalizado da consulta, conforme encoder disponível."""
    mode = st.session_state.encoder_mode
    if mode == "st" and st.session_state.st_encoder is not None:
        v = st.session_state.st_encoder.encode([text])[0].astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        return v
    if mode == "onnx" and st.session_state.onnx_sess is not None and st.session_state.onnx_tok is not None:
        tok = st.session_state.onnx_tok(text, return_tensors="np", truncation=True, max_length=256)
        inputs = {k: v.astype(np.int64) for k, v in tok.items()}
        out = st.session_state.onnx_sess.run(None, inputs)
        # tenta achar a última saída como embedding
        v = out[-1].squeeze()
        v = v.astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        return v
    return None

# -------------------------
# Busca UPLOAD (TF-IDF sessão)
# -------------------------
def search_upload(query: str, topk_upload: int):
    blocks = []
    vec = st.session_state.upload_index["vec"]
    X = st.session_state.upload_index["X"]
    metas = st.session_state.upload_index["metas"]
    texts = st.session_state.upload_index["texts"]
    if vec is None or X is None or not texts:
        return blocks
    q = vec.transform([query])
    sims = (q @ X.T).toarray()[0]
    idx = np.argsort(-sims)[: topk_upload * 4]
    for i in idx:
        s = float(sims[i])
        m = metas[i]
        t = texts[i]
        blocks.append((s, f"[UPLOAD {m['file']} / {m['chunk_id']}] (sim={s:.3f})\n{t}"))
    blocks.sort(key=lambda x: -x[0])
    return [b for _, b in blocks[:topk_upload]]

def get_upload_raw_text(max_chars=2500):
    texts = st.session_state.upload_index["texts"] or []
    if not texts:
        return ""
    # concatena os 2 primeiros chunks (ou menos) — simples e rápido
    out = ""
    for t in texts[:2]:
        if len(out) + len(t) > max_chars:
            out += "\n" + t[:max(0, max_chars - len(out))]
            break
        out += ("\n" if out else "") + t
    return out

# -------------------------
# Busca HIST (Embeddings → TF-IDF)
# -------------------------
def search_hist_embeddings(query: str, topk_hist: int):
    """Busca usando embeddings, se encoder disponível e base embeddings carregada."""
    ensure_encoder(force_embeddings)
    qv = encode_query(query) if force_embeddings else None
    blocks = []

    # Sphera
    if qv is not None and sph_emb is not None and sphera_df is not None:
        sph_text_col = "Description" if "Description" in sphera_df.columns else sphera_df.columns[0]
        hits = topk_by_cosine(sph_emb["E"], qv, k=topk_hist)
        for i, s in hits:
            i = int(i)
            row = sphera_df.iloc[i]
            evid = row.get("Event ID", row.get("EVENT_NUMBER", f"row{i}"))
            snippet = str(row.get(sph_text_col, ""))[:600]
            blocks.append((float(s), f"[Sphera/{evid}] (sim={float(s):.3f})\n{snippet}"))

    # GoSee
    if qv is not None and gos_emb is not None and gosee_df is not None:
        gos_text_col = "Observation" if "Observation" in gosee_df.columns else gosee_df.columns[0]
        hits = topk_by_cosine(gos_emb["E"], qv, k=topk_hist)
        for i, s in hits:
            i = int(i)
            row = gosee_df.iloc[i]
            gid = row.get("ID", f"row{i}")
            snippet = str(row.get(gos_text_col, ""))[:600]
            blocks.append((float(s), f"[GoSee/{gid}] (sim={float(s):.3f})\n{snippet}"))

    # Docs (history)
    if qv is not None and his_emb is not None and history_rows:
        E = his_emb["E"]
        hits = topk_by_cosine(E, qv, k=topk_hist)
        for i, s in hits:
            i = int(i)
            row = history_rows[i]
            src = f"Docs/{row.get('source','?')}/{row.get('chunk_id',0)}"
            snippet = str(row.get("text", ""))[:600]
            blocks.append((float(s), f"[{src}] (sim={float(s):.3f})\n{snippet}"))

    blocks.sort(key=lambda x: -x[0])
    return [b for _, b in blocks[:topk_hist]]

def search_hist_tfidf(query: str, topk_hist: int):
    """Fallback TF-IDF pré-indexado."""
    blocks = []

    # Sphera
    if sphera_tfidf is not None and sphera_df is not None:
        vec, X, text_col = sphera_tfidf["vectorizer"], sphera_tfidf["matrix"], sphera_tfidf.get("text_col", "Description")
        q = vec.transform([query])
        sims = (q @ X.T).toarray()[0]
        idx = np.argsort(-sims)[: topk_hist]
        for i in idx:
            i = int(i)
            s = float(sims[i])
            row = sphera_df.iloc[i]
            evid = row.get("Event ID", row.get("EVENT_NUMBER", f"row{i}"))
            snippet = str(row.get(text_col, ""))[:600]
            blocks.append((s, f"[Sphera/{evid}] (sim={s:.3f})\n{snippet}"))

    # GoSee
    if gosee_tfidf is not None and gosee_df is not None:
        vec, X, text_col = gosee_tfidf["vectorizer"], gosee_tfidf["matrix"], gosee_tfidf.get("text_col", "Observation")
        q = vec.transform([query])
        sims = (q @ X.T).toarray()[0]
        idx = np.argsort(-sims)[: topk_hist]
        for i in idx:
            i = int(i)
            s = float(sims[i])
            row = gosee_df.iloc[i]
            gid = row.get("ID", f"row{i}")
            snippet = str(row.get(text_col, ""))[:600]
            blocks.append((s, f"[GoSee/{gid}] (sim={s:.3f})\n{snippet}"))

    # Docs
    if hist_tfidf is not None and history_rows:
        vec, X = hist_tfidf["vectorizer"], hist_tfidf["matrix"]
        q = vec.transform([query])
        sims = (q @ X.T).toarray()[0]
        idx = np.argsort(-sims)[: topk_hist]
        for i in idx:
            i = int(i)
            s = float(sims[i])
            row = history_rows[i]
            src = f"Docs/{row.get('source','?')}/{row.get('chunk_id',0)}"
            snippet = str(row.get("text", ""))[:600]
            blocks.append((s, f"[{src}] (sim={s:.3f})\n{snippet}"))

    blocks.sort(key=lambda x: -x[0])
    return [b for _, b in blocks[:topk_hist]]

# -------------------------
# UI
# -------------------------
st.title("ESO • CHAT — HIST + UPLD (Embeddings preferencial)")
st.caption("RAG local com embeddings (Sphera/GoSee/Docs) + upload; TF-IDF usado apenas como fallback quando permitido.")

# Histórico de chat
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Digite sua pergunta…")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1) Recuperação no UPLOAD (sempre TF-IDF sessão)
    up_blocks = search_upload(prompt, topk_upload=topk_upload)

    # 2) Recuperação no HIST (Embeddings → TF-IDF)
    hi_blocks = []
    try:
        if force_embeddings:
            hi_blocks = search_hist_embeddings(prompt, topk_hist=topk_hist)
        if not hi_blocks:
            hi_blocks = search_hist_tfidf(prompt, topk_hist=topk_hist)
    except Exception as e:
        st.warning(f"Falha na busca HIST: {e}")
        hi_blocks = []

    # 3) Combinação com pesos (ordenação por score ponderado)
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

    # 4) Opcional: inserir RAW do upload (até N chars) para garantir que o arquivo "entra" no prompt
    upload_raw = get_upload_raw_text(upload_raw_max)
    if upload_raw:
        context_blocks = [f"[UPLOAD_RAW] (recorte)\n{upload_raw}"] + context_blocks

    SYSTEM = (
        "Você é um assistente de segurança operacional. "
        "Use os CONTEXTOS a seguir como evidências. "
        "Se a pergunta solicitar quantitativos ou listas do histórico, responda SOMENTE com base nos CONTEXTOS. "
        "Cite trechos relevantes quando fizer afirmações específicas."
    )

    messages = [{"role": "system", "content": SYSTEM}]
    if use_catalog and catalog_ctx:
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
                content = resp.get("message", {}).get("content", "").strip() or json.dumps(resp)[:1000]
            except Exception as e:
                content = f"Falha ao consultar o modelo: {e}"
            st.markdown(content)
    st.session_state.chat.append({"role": "assistant", "content": content})

# -------------------------
# Painel status
# -------------------------
with st.expander("📦 Status dos índices", expanded=False):
    st.write("Embeddings Sphera:", "✅" if sph_emb is not None and sphera_df is not None else "—")
    st.write("Embeddings GoSee :", "✅" if gos_emb is not None and gosee_df  is not None else "—")
    st.write("Embeddings Docs  :", "✅" if his_emb is not None and history_rows else "—")
    st.write("TF-IDF Sphera    :", "✅" if sphera_tfidf is not None and sphera_df is not None else "—")
    st.write("TF-IDF GoSee     :", "✅" if gosee_tfidf  is not None and gosee_df  is not None else "—")
    st.write("TF-IDF Docs      :", "✅" if hist_tfidf   is not None and history_rows else "—")

    enc_label = {"st":"Sentence-Transformers", "onnx":"ONNX local", "none":"—"}[st.session_state.encoder_mode]
    st.write("Encoder ativo (embeddings):", enc_label)
    st.write("Uploads indexados:", len(st.session_state.upload_index["texts"]))
