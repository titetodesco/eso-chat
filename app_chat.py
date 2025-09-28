# app_chat.py
# ESO • CHAT — HIST + UPLD (TF-IDF) • Ollama Cloud para respostas
# - HIST: índices pré-gerados com make_catalog_indexes.py (Sphera, GoSee, Docs)
# - UPLD: arquivos enviados no momento (TF-IDF local)
# - Combina resultados (pesos) e injeta no prompt
# - Injeta datasets_context.md (YAML em texto) como "catálogo" opcional

import os
import io
import json
import time  # <-- NOVO
import requests
import numpy as np
import pandas as pd
import streamlit as st

# joblib opcional: se não estiver instalado, HIST fica desativado
try:
    import joblib  # noqa
except Exception:
    joblib = None

# Parsers leves (apenas para uploads)
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import docx
except Exception:
    docx = None


st.set_page_config(page_title="ESO • CHAT (HIST + UPLD)", page_icon="💬", layout="wide")
# --- PATCH: inicialização segura para índices de histórico em memória ---

# 1) Garante estrutura no session_state
if "hist_idx" not in st.session_state:
    # cada item: {"vectorizer":..., "matrix":..., "text_col":..., "rows":...}
    st.session_state.hist_idx = {"sphera": None, "gosee": None, "docs": None}

# 2) Função utilitária para construir TF-IDF (igual ao que usamos para uploads)
def _build_tfidf_from_texts(texts, n_features=50000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = ["" if t is None or (isinstance(t, float) and np.isnan(t)) else str(t) for t in texts]
    vec = TfidfVectorizer(lowercase=True, strip_accents="unicode",
                          analyzer="word", ngram_range=(1,2),
                          max_features=n_features)
    X = vec.fit_transform(texts)
    return vec, X

# 3) Inicializa listas vazias para evitar NameError
sph_texts, go_texts, hist_texts = [], [], []

# 4) Fallback opcional: só constrói em memória se NÃO carregou os joblibs prontos
#    (mantém prioridade para os índices pré-gerados via make_catalog_indexes.py)
try:
    # estes objetos devem existir no escopo se você carregou os .joblib/.parquet
    have_sphera_joblib = ("sphera_j" in globals()) and (sphera_j is not None)
    have_gosee_joblib  = ("gosee_j"  in globals()) and (gosee_j  is not None)
    have_hist_joblib   = ("hist_j"   in globals()) and (hist_j   is not None)

    # Sphera: se não há joblib, mas há dataframe, constrói TF-IDF em memória
    if (not have_sphera_joblib) and ("sphera_df" in globals()) and (sphera_df is not None) \
       and (st.session_state.hist_idx["sphera"] is None):
        # escolha de coluna padrão quando não há joblib dizendo qual é:
        text_col = "DESCRIPTION" if "DESCRIPTION" in sphera_df.columns else sphera_df.columns[0]
        sph_texts = sphera_df[text_col].astype(str).fillna("").tolist()
        vec, X = _build_tfidf_from_texts(sph_texts)
        st.session_state.hist_idx["sphera"] = {"vectorizer": vec, "matrix": X, "text_col": text_col, "rows": None}

    # GoSee: idem
    if (not have_gosee_joblib) and ("gosee_df" in globals()) and (gosee_df is not None) \
       and (st.session_state.hist_idx["gosee"] is None):
        text_col = "Observation" if "Observation" in gosee_df.columns else gosee_df.columns[0]
        go_texts = gosee_df[text_col].astype(str).fillna("").tolist()
        vec, X = _build_tfidf_from_texts(go_texts)
        st.session_state.hist_idx["gosee"] = {"vectorizer": vec, "matrix": X, "text_col": text_col, "rows": None}

    # Docs (history_texts.jsonl): idem
    if (not have_hist_joblib) and ("history_rows" in globals()) and history_rows \
       and (st.session_state.hist_idx["docs"] is None):
        hist_texts = [str(r.get("text","")) for r in history_rows]
        vec, X = _build_tfidf_from_texts(hist_texts)
        st.session_state.hist_idx["docs"] = {"vectorizer": vec, "matrix": X, "text_col": "text", "rows": history_rows}
except Exception as _e:
    # não bloqueia a app por causa do fallback; só registra no log server
    print("[hist-fallback] aviso:", repr(_e))



# -------------------------
# Configs / Secrets
# -------------------------
OLLAMA_HOST  = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gpt-oss:20b"))
OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY", None))

if not OLLAMA_API_KEY:
    st.error("⚠️ OLLAMA_API_KEY não encontrado. Defina em **Settings → Secrets**.")
    st.stop()

HEADERS_JSON = {"Authorization": f"Bearer {OLLAMA_API_KEY}", "Content-Type": "application/json"}

DATA_DIR = "data"
ANALYTICS_DIR = os.path.join(DATA_DIR, "analytics")
DATASETS_CONTEXT_FILE = "datasets_context.md"  # YAML em markdown (conteúdo puro YAML)

# -------------------------
# Utilitários
# -------------------------
# -------------------------
# Carrega HIST (pré-indexado)
# -------------------------

# === PATCH: índices TF-IDF leves para HIST (Sphera/GoSee/Docs) ===
# Reaproveita as funções tfidf_fit, tfidf_transform, cosine_sim já existentes no app.

if "hist_idx" not in st.session_state:
    st.session_state.hist_idx = {
        "sphera": None,  # dict: {"texts": [...], "vocab":..., "idf":..., "X":...}
        "gosee":  None,
        "hist":   None,
    }

def build_hist_index(name, texts_list):
    # texts_list: lista de strings (por ex., lidas de sphera_texts.jsonl)
    texts = [t if isinstance(t, str) else "" for t in texts_list]
    # Remover vazios
    texts = [t for t in texts if t.strip()]
    if not texts:
        return None
    vocab, idf, X = tfidf_fit(texts)
    return {"texts": texts, "vocab": vocab, "idf": idf, "X": X}

# Construção dos índices, apenas 1x por sessão
if st.session_state.hist_idx["sphera"] is None and sph_texts:
    st.session_state.hist_idx["sphera"] = build_hist_index("sphera", sph_texts)
if st.session_state.hist_idx["gosee"] is None and gos_texts:
    st.session_state.hist_idx["gosee"]  = build_hist_index("gosee", gos_texts)
if st.session_state.hist_idx["hist"] is None and hist_texts:
    st.session_state.hist_idx["hist"]   = build_hist_index("hist", hist_texts)

def search_hist_tfidf(query_text: str, topk_hist: int, thr_hist: float):
    """Busca TF-IDF (leve) nos índices HIST (sphera/gosee/hist)."""
    out_blocks = []

    def do_search(tag, idx):
        if not idx: 
            return
        qX = tfidf_transform([query_text], idx["vocab"], idx["idf"])
        sims = cosine_sim(qX, idx["X"])[0]
        order = np.argsort(-sims)
        seen_snippets = set()
        for i in order[: max(50, topk_hist*4)]:
            s = float(sims[i])
            if s < thr_hist:
                continue
            t = idx["texts"][i]
            # desduplicação simples por texto normalizado
            key = t.strip().lower()
            if key in seen_snippets:
                continue
            seen_snippets.add(key)
            out_blocks.append((s, f"[{tag}/{i}] (sim={s:.3f})\n{t}"))

    do_search("Sphera", st.session_state.hist_idx["sphera"])
    do_search("GoSee",  st.session_state.hist_idx["gosee"])
    do_search("Docs",   st.session_state.hist_idx["hist"])

    out_blocks.sort(key=lambda x: -x[0])
    return [b for _, b in out_blocks[:topk_hist]]

# === PATCH: extrair consulta automática a partir do upload ===
def get_upload_query_text(max_chars=3000):
    """Monta uma consulta a partir dos melhores chunks do upload para usar no HIST.
       Se não houver upload indexado, retorna string vazia."""
    texts = st.session_state.upld.get("texts") or []
    metas = st.session_state.upld.get("metas") or []
    vocab = st.session_state.upld.get("vocab")
    idf   = st.session_state.upld.get("idf")
    X     = st.session_state.upld.get("X")
    if not texts or vocab is None or idf is None or X is None:
        return ""
    # Usamos o último prompt do usuário como "semente" para ranquear os chunks e pegar os top-3
    # (se quiser, pode usar outra estratégia: últimos N chunks, ou tudo junto).
    seed = st.session_state.chat[-1]["content"] if st.session_state.chat else ""
    if not seed:
        seed = "resumo do conteúdo do upload"
    qX = tfidf_transform([seed], vocab, idf)
    sims = cosine_sim(qX, X)[0]
    order = np.argsort(-sims)
    top_idxs = order[:3]  # pega até 3 trechos mais relevantes
    buf = []
    total = 0
    for i in top_idxs:
        t = texts[i]
        if total + len(t) > max_chars:
            t = t[: max(0, max_chars-total)]
        buf.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return "\n\n".join(buf).strip()

# ---- Cache helpers ----
@st.cache_resource(show_spinner=False)
def _load_parquet_cached(path: str):
    return pd.read_parquet(path)

@st.cache_resource(show_spinner=False)
def _load_jsonl_cached(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

@st.cache_resource(show_spinner=False)
def _load_joblib_cached(path: str):
    if joblib is None:
        return None
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def _load_text_cached(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def load_joblib_any(*candidates):
    for p in candidates:
        if os.path.exists(p):
            try:
                return joblib.load(p)
            except Exception as e:
                st.warning(f"Não consegui carregar {p}: {e}")
    return None

def read_parquet_any(*candidates):
    for p in candidates:
        if os.path.exists(p):
            try:
                return pd.read_parquet(p)
            except Exception as e:
                st.warning(f"Falha ao ler {p}: {e}")
    return None

# caminhos base
sphera_parquet_candidates = [
    os.path.join(ANALYTICS_DIR, "spheracloud.parquet"),
    os.path.join(ANALYTICS_DIR, "sphera.parquet"),
]
sphera_tfidf_candidates = [
    os.path.join(ANALYTICS_DIR, "spheracloud.tfidf.joblib"),
    os.path.join(ANALYTICS_DIR, "sphera_tfidf.joblib"),
]

gosee_parquet_candidates = [
    os.path.join(ANALYTICS_DIR, "gosee.parquet"),
]
gosee_tfidf_candidates = [
    os.path.join(ANALYTICS_DIR, "gosee.tfidf.joblib"),
]

hist_texts_path = os.path.join(ANALYTICS_DIR, "history_texts.jsonl")
hist_tfidf_candidates = [
    os.path.join(ANALYTICS_DIR, "history_tfidf.joblib"),
]

# carrega joblibs
sphera_j = load_joblib_any(*sphera_tfidf_candidates)
gosee_j  = load_joblib_any(*gosee_tfidf_candidates)
hist_j   = load_joblib_any(*hist_tfidf_candidates)

# carrega dataframes
sphera_df = read_parquet_any(*sphera_parquet_candidates)
gosee_df  = read_parquet_any(*gosee_parquet_candidates)

# carrega histórico (jsonl)
history_rows = []
if os.path.exists(hist_texts_path):
    try:
        with open(hist_texts_path, "r", encoding="utf-8") as f:
            for line in f:
                history_rows.append(json.loads(line))
    except Exception as e:
        st.warning(f"Falha ao ler history_texts.jsonl: {e}")


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
    payload = {"model": model, "messages": messages, "temperature": float(temperature), "stream": bool(stream)}
    r = requests.post(f"{OLLAMA_HOST}/api/chat", headers=HEADERS_JSON, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

# -------------------------
# Estado
# -------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "upload_index" not in st.session_state:
    st.session_state.upload_index = {
        "texts": [],
        "metas": [],  # {"file":..., "chunk_id":...}
        "vec": None,  # tfidf_vectorizer (joblib do sklearn) re-fit em sessão
        "X": None     # sparse matrix dos uploads
    }

# -------------------------
# Carrega HIST (pré-indexado)
# -------------------------
def load_joblib(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Não consegui carregar {path}: {e}")
        return None

sphera_j = load_joblib(os.path.join(ANALYTICS_DIR, "sphera_tfidf.joblib"))
gosee_j  = load_joblib(os.path.join(ANALYTICS_DIR, "gosee_tfidf.joblib"))
hist_j   = load_joblib(os.path.join(ANALYTICS_DIR, "history_tfidf.joblib"))

sphera_df = None
gosee_df  = None
history_rows = []

if os.path.exists(os.path.join(ANALYTICS_DIR, "sphera.parquet")):
    try:
        sphera_df = pd.read_parquet(os.path.join(ANALYTICS_DIR, "sphera.parquet"))
    except Exception as e:
        st.warning(f"Falha ao ler sphera.parquet: {e}")

if os.path.exists(os.path.join(ANALYTICS_DIR, "gosee.parquet")):
    try:
        gosee_df = pd.read_parquet(os.path.join(ANALYTICS_DIR, "gosee.parquet"))
    except Exception as e:
        st.warning(f"Falha ao ler gosee.parquet: {e}")

if os.path.exists(os.path.join(ANALYTICS_DIR, "history_texts.jsonl")):
    try:
        with open(os.path.join(ANALYTICS_DIR, "history_texts.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                history_rows.append(json.loads(line))
    except Exception as e:
        st.warning(f"Falha ao ler history_texts.jsonl: {e}")

# datasets_context.md (texto puro)
catalog_ctx = ""
try:
    if os.path.exists(DATASETS_CONTEXT_FILE):
        with open(DATASETS_CONTEXT_FILE, "r", encoding="utf-8") as f:
            catalog_ctx = f.read()
except Exception:
    pass

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Configurações")
with st.sidebar.expander("Ollama Cloud", expanded=True):
    st.write("Host:", OLLAMA_HOST)
    st.write("Modelo:", OLLAMA_MODEL)

st.sidebar.divider()
st.sidebar.subheader("RAG • Pesos & Limiar")
w_upload = st.sidebar.slider("Peso do UPLOAD", 0.0, 1.0, 0.7, 0.05)
w_hist   = 1.0 - w_upload
thr_upload = st.sidebar.slider("Limiar (UPLOAD)", 0.0, 1.0, 0.25, 0.01)
thr_hist   = st.sidebar.slider("Limiar (HIST)",   0.0, 1.0, 0.35, 0.01)
topk_upload = st.sidebar.slider("Top-K Upload", 1, 15, 6, 1)
topk_hist   = st.sidebar.slider("Top-K Histórico", 1, 15, 6, 1)
chunk_size  = st.sidebar.slider("Tamanho do chunk (upload)", 500, 2000, 1200, 50)
chunk_overlap = st.sidebar.slider("Overlap do chunk (upload)", 50, 600, 200, 10)
use_catalog = st.sidebar.checkbox("Injetar datasets_context.md", True)

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
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = st.session_state.upload_index["texts"]
    if not texts:
        st.session_state.upload_index["vec"] = None
        st.session_state.upload_index["X"] = None
        return
    vec = TfidfVectorizer(lowercase=True, strip_accents="unicode", analyzer="word", ngram_range=(1,2), max_features=50000)
    X = vec.fit_transform(texts)
    st.session_state.upload_index["vec"] = vec
    st.session_state.upload_index["X"] = X

if uploaded_files:
    with st.spinner("Lendo files e indexando (TF-IDF, local)…"):
        new_texts = []
        new_metas = []
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
            rebuild_upload_index()
            st.success(f"Upload indexado: {len(new_texts)} chunks.")

# -------------------------
# Busca HIST
# -------------------------
def search_hist(query: str, topk_hist: int, thr_hist: float):
    blocks = []

    def add_block(src, score, meta, text):
        blocks.append((score, f"[{src}] (sim={score:.3f})\n{text}"))

    # Spheracloud
    if sphera_j and sphera_df is not None:
        vec, X, text_col = sphera_j["vectorizer"], sphera_j["matrix"], sphera_j["text_col"]
        q = vec.transform([query])
        sims = (q @ X.T).toarray()[0]  # cos-like (TFIDF l2 normalizado no sklearn)
        idx = np.argsort(-sims)[: topk_hist * 4]  # pega mais e filtra por thr
        for i in idx:
            s = float(sims[i])
            if s < thr_hist:
                continue
            row = sphera_df.iloc[i].to_dict()
            txt = row.get(text_col, "")
            ident = row.get("EVENT_NUMBER", row.get("ID", f"row{i}"))
            add_block(f"SpheraCloud/{ident}", s, row, txt)

    # GoSee
    if gosee_j and gosee_df is not None:
        vec, X, text_col = gosee_j["vectorizer"], gosee_j["matrix"], gosee_j["text_col"]
        q = vec.transform([query])
        sims = (q @ X.T).toarray()[0]
        idx = np.argsort(-sims)[: topk_hist * 4]
        for i in idx:
            s = float(sims[i])
            if s < thr_hist:
                continue
            row = gosee_df.iloc[i].to_dict()
            txt = row.get(text_col, "")
            ident = row.get("ID", f"row{i}")
            add_block(f"GoSee/{ident}", s, row, txt)

    # History (docs)
    if hist_j and history_rows:
        vec, X = hist_j["vectorizer"], hist_j["matrix"]
        q = vec.transform([query])
        sims = (q @ X.T).toarray()[0]
        idx = np.argsort(-sims)[: topk_hist * 4]
        for i in idx:
            s = float(sims[i])
            if s < thr_hist:
                continue
            row = history_rows[i]
            txt = row.get("text", "")
            src = f"Docs/{row.get('source','?')}/{row.get('chunk_id',0)}"
            add_block(src, s, row, txt)

    # Ordena e corta Top-K
    blocks.sort(key=lambda x: -x[0])
    return [b for _, b in blocks[:topk_hist]]

# -------------------------
# Busca UPLOAD
# -------------------------
def search_upload(query: str, topk_upload: int, thr_upload: float):
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
        if s < thr_upload:
            continue
        m = metas[i]
        t = texts[i]
        blocks.append((s, f"[UPLOAD {m['file']} / {m['chunk_id']}] (sim={s:.3f})\n{t}"))
    blocks.sort(key=lambda x: -x[0])
    return [b for _, b in blocks[:topk_upload]]

# -------------------------
# UI
# -------------------------
st.title("ESO • CHAT — HIST + UPLD (TF-IDF) • Ollama Cloud")
st.caption("RAG local (TF-IDF) com foco no último upload + histórico. Catálogo opcional injetado como contexto (datasets_context.md).")

# Histórico de chat
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Digite sua pergunta…")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Recuperação combinada
    up_blocks = search_upload(prompt, topk_upload=topk_upload, thr_upload=thr_upload)
    # 1) Busca em UPLOAD pela pergunta do usuário
    up_blocks = search_upload(prompt, topk_upload=topk_upload, thr_upload=thr_upload)
    
    # 2) Se houver upload, gera uma CONSULTA baseada no próprio upload para HIST
    upload_query = get_upload_query_text()  # usa top-chunks do upload
    hist_query = upload_query if upload_query else prompt
    
    # 3) Busca HIST com TF-IDF leve usando a 'hist_query'
    hi_blocks = search_hist_tfidf(hist_query, topk_hist=topk_hist, thr_hist=thr_hist)
    

    # Combinação com pesos (nota: servem para ordenação; cada bloco já traz sim do seu domínio)
    combined = []
    for i, b in enumerate(up_blocks):
        # extrai sim= de b
        try:
            s = float(b.split("(sim=")[1].split(")")[0])
        except Exception:
            s = 0.0
        combined.append((w_upload * s, b))
    for i, b in enumerate(hi_blocks):
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
        "Evite pseudo-código; entregue respostas objetivas e explicativas com trechos citados."
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

# Painel status
with st.expander("📦 Índices carregados", expanded=False):
    st.write("Spheracloud:", "✅" if sphera_j and sphera_df is not None else "—")
    st.write("GoSee:", "✅" if gosee_j and gosee_df is not None else "—")
    st.write("Histórico (docs):", "✅" if hist_j and history_rows else "—")
    st.write("Uploads indexados:", len(st.session_state.upload_index["texts"]))
