# app_chat.py
# ESO • CHAT — HIST + UPLD (Embeddings preferencial) • Ollama Cloud
# - HIST (Sphera/GoSee/Docs): usa embeddings *.npz se existirem; opcional fallback TF-IDF
# - UPLD: por padrão gera embeddings com sentence-transformers; fallback TF-IDF se não houver ST
# - Combina resultados com pesos; injeta datasets_context.md (opcional)

import os
import io
import json
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

# ==== Dependências opcionais ====
# sentence-transformers para embeddings no upload (recomendado)
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # fallback para TF-IDF no upload

# TF-IDF (fallback para upload e/ou histórico)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
except Exception:
    TfidfVectorizer = None  # sem TF-IDF

# joblib só para carregar TF-IDF pré-gerado (histórico)
try:
    import joblib  # type: ignore
except Exception:
    joblib = None

# Parsers leves (uploads)
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import docx
except Exception:
    docx = None


# ==== Config base ====
st.set_page_config(page_title="ESO • CHAT (Embeddings + TF-IDF Fallback)", page_icon="💬", layout="wide")

DATA_DIR = "data"
ANALYTICS_DIR = os.path.join(DATA_DIR, "analytics")
DATASETS_CONTEXT_FILE = "datasets_context.md"  # texto (YAML/markdown) puro

OLLAMA_HOST  = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gpt-oss:20b"))
OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY", None))
if not OLLAMA_API_KEY:
    st.error("⚠️ OLLAMA_API_KEY não encontrado. Defina em **Settings → Secrets**.")
    st.stop()
HEADERS_JSON = {"Authorization": f"Bearer {OLLAMA_API_KEY}", "Content-Type": "application/json"}


# ==== Utils ====
def read_pdf_bytes(b: bytes) -> str:
    if pypdf is None:
        raise RuntimeError("pypdf não instalado (adicione 'pypdf' ao requirements).")
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
        raise RuntimeError("python-docx não instalado (adicione 'python-docx' ao requirements).")
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

def l2_normalize(M: np.ndarray) -> np.ndarray:
    M = M.astype(np.float32, copy=False)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return M / norms

def cosine_sim_dense(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    a = l2_normalize(A)
    b = l2_normalize(B)
    return a @ b.T

def ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2, stream=False, timeout=120):
    payload = {"model": model, "messages": messages, "temperature": float(temperature), "stream": bool(stream)}
    r = requests.post(f"{OLLAMA_HOST}/api/chat", headers=HEADERS_JSON, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def load_embeddings_npz(path: str):
    """Carrega um .npz flexível (aceita chaves variadas). Retorna np.ndarray (N,D)."""
    if not os.path.exists(path):
        return None
    try:
        npz = np.load(path, allow_pickle=True)
        # tenta chaves comuns:
        for key in ("emb", "X", "vectors", "arr_0"):
            if key in npz:
                arr = np.array(npz[key])
                if arr.ndim == 2:
                    return arr.astype(np.float32, copy=False)
        # fallback: pega o primeiro 2D que achar
        for k in npz.files:
            arr = np.array(npz[k])
            if arr.ndim == 2:
                return arr.astype(np.float32, copy=False)
    except Exception as e:
        st.warning(f"Falha ao carregar embeddings {path}: {e}")
    return None

def safe_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    # fallback para 1ª coluna textual
    for c in df.columns:
        return c

def normalize_text(t: str) -> str:
    return " ".join((t or "").strip().lower().split())


# ==== Estado ====
if "chat" not in st.session_state:
    st.session_state.chat = []
if "upload" not in st.session_state:
    st.session_state.upload = {
        "texts": [],     # chunks
        "metas": [],     # {"file":..., "chunk_id":...}
        "emb": None,     # np.ndarray (N,D) se embeddings
        "tfidf_vec": None,  # TfidfVectorizer se fallback
        "tfidf_X": None,    # sparse matrix
        "backend": "none"   # "emb" | "tfidf" | "none"
    }


# ==== Carregamento HIST (dados + embeddings + tfidf) ====
# Dataframes para recuperar textos
sphera_df = None
gosee_df  = None
history_rows = []

# dataframes
try:
    if os.path.exists(os.path.join(ANALYTICS_DIR, "sphera.parquet")):
        sphera_df = pd.read_parquet(os.path.join(ANALYTICS_DIR, "sphera.parquet"))
    if os.path.exists(os.path.join(ANALYTICS_DIR, "gosee.parquet")):
        gosee_df = pd.read_parquet(os.path.join(ANALYTICS_DIR, "gosee.parquet"))
except Exception as e:
    st.warning(f"Falha ao ler parquet: {e}")

# history texts jsonl
hist_jsonl = os.path.join(ANALYTICS_DIR, "history_texts.jsonl")
if os.path.exists(hist_jsonl):
    try:
        with open(hist_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                history_rows.append(json.loads(line))
    except Exception as e:
        st.warning(f"Falha ao ler history_texts.jsonl: {e}")

# embeddings
sph_emb = load_embeddings_npz(os.path.join(ANALYTICS_DIR, "sphera_embeddings.npz"))
gos_emb = load_embeddings_npz(os.path.join(ANALYTICS_DIR, "gosee_embeddings.npz"))
his_emb = load_embeddings_npz(os.path.join(ANALYTICS_DIR, "history_embeddings.npz"))

# TF-IDF fallback do HIST (opcional)
def load_joblib_maybe(path):
    if (joblib is None) or (not os.path.exists(path)):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Não consegui carregar {path}: {e}")
        return None

sph_tfidf = load_joblib_maybe(os.path.join(ANALYTICS_DIR, "sphera_tfidf.joblib"))
gos_tfidf = load_joblib_maybe(os.path.join(ANALYTICS_DIR, "gosee_tfidf.joblib"))
his_tfidf = load_joblib_maybe(os.path.join(ANALYTICS_DIR, "history_tfidf.joblib"))

# catálogo opcional
catalog_ctx = ""
try:
    if os.path.exists(DATASETS_CONTEXT_FILE):
        with open(DATASETS_CONTEXT_FILE, "r", encoding="utf-8") as f:
            catalog_ctx = f.read()
except Exception:
    pass


# ==== Sidebar ====
st.sidebar.header("Configurações")

with st.sidebar.expander("Ollama Cloud", expanded=True):
    st.write("Host:", OLLAMA_HOST)
    st.write("Modelo:", OLLAMA_MODEL)

st.sidebar.divider()
st.sidebar.subheader("RAG • Pesos & Limiar")
w_upload = st.sidebar.slider("Peso do UPLOAD", 0.0, 1.0, 0.7, 0.05)
w_hist   = 1.0 - w_upload
thr_upload = st.sidebar.slider("Limiar (UPLOAD)", 0.0, 1.0, 0.20, 0.01)
thr_hist   = st.sidebar.slider("Limiar (HIST)",   0.0, 1.0, 0.30, 0.01)
topk_upload = st.sidebar.slider("Top-K Upload", 1, 15, 6, 1)
topk_hist   = st.sidebar.slider("Top-K Histórico", 1, 15, 6, 1)
use_catalog = st.sidebar.checkbox("Injetar datasets_context.md", True)

st.sidebar.divider()
st.sidebar.subheader("Backends")
use_embeddings_only = st.sidebar.checkbox("Forçar apenas EMBEDDINGS (sem TF-IDF)", True)
use_upload_embeddings = st.sidebar.checkbox("Embeddings para UPLOAD (recomendado)", True)
upload_model_name = st.sidebar.text_input("Modelo embeddings (upload)", "sentence-transformers/all-MiniLM-L6-v2")
max_upload_raw = st.sidebar.slider("Tamanho máx. do bloco UPLOAD_RAW (chars)", 0, 4000, 2000, 100)

st.sidebar.divider()
uploaded_files = st.sidebar.file_uploader("Upload (PDF, DOCX, XLSX, CSV, TXT/MD)",
                                          type=["pdf","docx","xlsx","xls","csv","txt","md"],
                                          accept_multiple_files=True)

col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Limpar uploads", use_container_width=True):
        st.session_state.upload = {"texts": [], "metas": [], "emb": None, "tfidf_vec": None, "tfidf_X": None, "backend": "none"}
with col_b:
    if st.button("Limpar chat", use_container_width=True):
        st.session_state.chat = []


# ==== Indexação UPLOAD ====
def rebuild_upload_embeddings():
    texts = st.session_state.upload["texts"]
    if not texts:
        st.session_state.upload.update({"emb": None, "backend": "none"})
        return
    if not use_upload_embeddings or SentenceTransformer is None:
        # fallback TF-IDF (só para upload)
        if TfidfVectorizer is None:
            st.session_state.upload.update({"emb": None, "tfidf_vec": None, "tfidf_X": None, "backend": "none"})
            return
        vec = TfidfVectorizer(lowercase=True, strip_accents="unicode", analyzer="word", ngram_range=(1,2), max_features=50000)
        X = vec.fit_transform(texts)
        st.session_state.upload.update({"emb": None, "tfidf_vec": vec, "tfidf_X": X, "backend": "tfidf"})
        return
    # embeddings
    try:
        model = SentenceTransformer(upload_model_name)
        emb = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        st.session_state.upload.update({"emb": emb.astype(np.float32, copy=False), "tfidf_vec": None, "tfidf_X": None, "backend": "emb"})
    except Exception as e:
        st.warning(f"Falha ao gerar embeddings do upload ({e}); usando TF-IDF no upload.")
        if TfidfVectorizer is None:
            st.session_state.upload.update({"emb": None, "tfidf_vec": None, "tfidf_X": None, "backend": "none"})
        else:
            vec = TfidfVectorizer(lowercase=True, strip_accents="unicode", analyzer="word", ngram_range=(1,2), max_features=50000)
            X = vec.fit_transform(texts)
            st.session_state.upload.update({"emb": None, "tfidf_vec": vec, "tfidf_X": X, "backend": "tfidf"})

def handle_uploads():
    if not uploaded_files:
        return
    new_texts = []
    new_metas = []
    for uf in uploaded_files:
        try:
            text = read_any(uf)
            # 2 trilhas: (i) blocos chunkados (para embeddings/tfidf), (ii) “raw” para injetar no prompt
            parts = chunk_text(text, max_chars=1200, overlap=200)
            for i, p in enumerate(parts):
                new_texts.append(p)
                new_metas.append({"file": uf.name, "chunk_id": i})
            # salva 1 trecho raw para prompt (limitado)
            raw = text[:max_upload_raw].strip()
            if raw:
                new_texts.append(f"(RAW {uf.name})\n{raw}")
                new_metas.append({"file": uf.name, "chunk_id": "RAW"})
        except Exception as e:
            st.warning(f"Falha ao processar {uf.name}: {e}")
    if new_texts:
        st.session_state.upload["texts"].extend(new_texts)
        st.session_state.upload["metas"].extend(new_metas)
        with st.spinner("Indexando upload…"):
            rebuild_upload_embeddings()
        st.success(f"Upload indexado: {len(new_texts)} chunks.")

handle_uploads()


# ==== Busca HIST ====
def search_hist_embeddings(query: str, topk: int, thr: float):
    blocks = []
    seen = set()

    def push(tag, score, txt, ident):
        key = (tag, normalize_text(txt))
        if key in seen:
            return
        seen.add(key)
        blocks.append((score, f"[{tag}/{ident}] (sim={score:.3f})\n{txt}"))

    # Sphera
    if sph_emb is not None and sphera_df is not None:
        # texto candidato: Description (preferível)
        text_col = "Description" if "Description" in sphera_df.columns else safe_col(sphera_df, list(sphera_df.columns))
        # embedding da query via upload backend (se houver embeddings no upload); caso contrário, TF-IDF não se aplica aqui
        q_emb = None
        if st.session_state.upload["backend"] == "emb" and st.session_state.upload["emb"] is not None:
            # gerar com mesmo modelo do upload — OK para consulta semântica ad-hoc
            try:
                model = SentenceTransformer(upload_model_name) if SentenceTransformer else None
                if model is not None:
                    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
            except Exception:
                q_emb = None
        # fallback: se não temos encoder disponível agora, tenta TF-IDF do hist (se permitido)
        if q_emb is not None:
            sims = (q_emb @ l2_normalize(sph_emb).T)[0]
            idx = np.argsort(-sims)[: max(50, topk*4)]
            for i in idx:
                s = float(sims[i])
                if s < thr:
                    continue
                row = sphera_df.iloc[i]
                ident = row.get("Event ID", row.get("EVENT_NUMBER", f"row{i}"))
                push("Sphera", s, str(row.get(text_col, "")), ident)
        elif (not use_embeddings_only) and (sph_tfidf and sphera_df is not None and TfidfVectorizer is not None):
            vec, X, text_col2 = sph_tfidf.get("vectorizer"), sph_tfidf.get("matrix"), sph_tfidf.get("text_col")
            if vec is not None and X is not None and text_col2:
                q = vec.transform([query])
                sims = (q @ X.T).toarray()[0]
                idx = np.argsort(-sims)[: max(50, topk*4)]
                for i in idx:
                    s = float(sims[i])
                    if s < thr:
                        continue
                    row = sphera_df.iloc[i]
                    ident = row.get("Event ID", row.get("EVENT_NUMBER", f"row{i}"))
                    push("Sphera", s, str(row.get(text_col2, "")), ident)

    # GoSee
    if gos_emb is not None and gosee_df is not None:
        text_col = "Observation" if "Observation" in gosee_df.columns else safe_col(gosee_df, list(gosee_df.columns))
        q_emb = None
        if st.session_state.upload["backend"] == "emb" and st.session_state.upload["emb"] is not None:
            try:
                model = SentenceTransformer(upload_model_name) if SentenceTransformer else None
                if model is not None:
                    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
            except Exception:
                q_emb = None
        if q_emb is not None:
            sims = (q_emb @ l2_normalize(gos_emb).T)[0]
            idx = np.argsort(-sims)[: max(50, topk*4)]
            for i in idx:
                s = float(sims[i])
                if s < thr:
                    continue
                row = gosee_df.iloc[i]
                ident = row.get("ID", f"row{i}")
                push("GoSee", s, str(row.get(text_col, "")), ident)
        elif (not use_embeddings_only) and (gos_tfidf and gosee_df is not None and TfidfVectorizer is not None):
            vec, X, text_col2 = gos_tfidf.get("vectorizer"), gos_tfidf.get("matrix"), gos_tfidf.get("text_col")
            if vec is not None and X is not None and text_col2:
                q = vec.transform([query])
                sims = (q @ X.T).toarray()[0]
                idx = np.argsort(-sims)[: max(50, topk*4)]
                for i in idx:
                    s = float(sims[i])
                    if s < thr:
                        continue
                    row = gosee_df.iloc[i]
                    ident = row.get("ID", f"row{i}")
                    push("GoSee", s, str(row.get(text_col2, "")), ident)

    # Docs (history)
    if his_emb is not None and history_rows:
        q_emb = None
        if st.session_state.upload["backend"] == "emb" and st.session_state.upload["emb"] is not None:
            try:
                model = SentenceTransformer(upload_model_name) if SentenceTransformer else None
                if model is not None:
                    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
            except Exception:
                q_emb = None
        if q_emb is not None:
            sims = (q_emb @ l2_normalize(his_emb).T)[0]
            idx = np.argsort(-sims)[: max(50, topk*4)]
            for i in idx:
                s = float(sims[i])
                if s < thr:
                    continue
                row = history_rows[i]
                txt = str(row.get("text", ""))
                src = f"{row.get('source','?')}/{row.get('chunk_id',i)}"
                push("Docs", s, txt, src)
        elif (not use_embeddings_only) and (his_tfidf and history_rows and TfidfVectorizer is not None):
            vec, X = his_tfidf.get("vectorizer"), his_tfidf.get("matrix")
            if vec is not None and X is not None:
                q = vec.transform([query])
                sims = (q @ X.T).toarray()[0]
                idx = np.argsort(-sims)[: max(50, topk*4)]
                for i in idx:
                    s = float(sims[i])
                    if s < thr:
                        continue
                    row = history_rows[i]
                    txt = str(row.get("text", ""))
                    src = f"{row.get('source','?')}/{row.get('chunk_id',i)}"
                    push("Docs", s, txt, src)

    blocks.sort(key=lambda x: -x[0])
    return [b for _, b in blocks[:topk]]

def search_upload(query: str, topk: int, thr: float):
    blocks = []
    texts = st.session_state.upload["texts"]
    metas = st.session_state.upload["metas"]
    backend = st.session_state.upload["backend"]
    if not texts:
        return blocks

    if backend == "emb" and st.session_state.upload["emb"] is not None:
        # gera embedding da query com o mesmo modelo do upload
        try:
            model = SentenceTransformer(upload_model_name) if SentenceTransformer else None
        except Exception:
            model = None
        if model is None:
            return blocks
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        sims = (q_emb @ st.session_state.upload["emb"].T)[0]
        idx = np.argsort(-sims)[: max(50, topk*4)]
        seen = set()
        for i in idx:
            s = float(sims[i])
            if s < thr:
                continue
            m = metas[i]
            t = texts[i]
            key = normalize_text(t)
            if key in seen:
                continue
            seen.add(key)
            blocks.append((s, f"[UPLOAD {m['file']} / {m['chunk_id']}] (sim={s:.3f})\n{t}"))
        blocks.sort(key=lambda x: -x[0])
        return [b for _, b in blocks[:topk]]

    if backend == "tfidf" and st.session_state.upload["tfidf_vec"] is not None:
        vec = st.session_state.upload["tfidf_vec"]
        X = st.session_state.upload["tfidf_X"]
        if vec is None or X is None:
            return []
        q = vec.transform([query])
        sims = (q @ X.T).toarray()[0]
        idx = np.argsort(-sims)[: max(50, topk*4)]
        seen = set()
        for i in idx:
            s = float(sims[i])
            if s < thr:
                continue
            m = metas[i]
            t = texts[i]
            key = normalize_text(t)
            if key in seen:
                continue
            seen.add(key)
            blocks.append((s, f"[UPLOAD {m['file']} / {m['chunk_id']}] (sim={s:.3f})\n{t}"))
        blocks.sort(key=lambda x: -x[0])
        return [b for _, b in blocks[:topk]]

    return []


# ==== UI ====
st.title("ESO • CHAT — HIST + UPLD (Embeddings preferencial)")
st.caption("RAG local com embeddings (Sphera/GoSee/Docs) + upload; TF-IDF usado apenas como fallback quando permitido.")

# Histórico do chat
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Digite sua pergunta…")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Busca
    up_blocks = search_upload(prompt, topk=topk_upload, thr=thr_upload)
    hi_blocks = search_hist_embeddings(prompt, topk=topk_hist, thr=thr_hist)

    # Combinação ponderada (ordenação)
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
        "Evite pseudo-código; entregue respostas objetivas, com trechos citados entre aspas quando fizer afirmações."
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


# ==== Painel de status (badges do backend) ====
def _badge(ok: bool, text_ok: str = "✅", text_no: str = "—"):
    return text_ok if ok else text_no

with st.expander("📦 Status dos índices", expanded=False):
    st.write("Modo:", "Embeddings only" if use_embeddings_only else "Embeddings + TF-IDF fallback")
    st.write("Upload backend:", st.session_state.upload["backend"])
    st.write("---")

    st.write("Sphera (embeddings):", _badge(sph_emb is not None and sphera_df is not None))
    st.write("GoSee (embeddings):", _badge(gos_emb is not None and gosee_df is not None))
    st.write("Docs (embeddings):",  _badge(his_emb is not None and len(history_rows) > 0))

    if not use_embeddings_only:
        st.write("Sphera (TF-IDF):", _badge(sph_tfidf is not None and sphera_df is not None))
        st.write("GoSee (TF-IDF):", _badge(gos_tfidf is not None and gosee_df is not None))
        st.write("Docs (TF-IDF):",  _badge(his_tfidf is not None and len(history_rows) > 0))

    st.write("Uploads indexados:", len(st.session_state.upload["texts"]))

# Avisos úteis na sidebar (curtos)
with st.sidebar.expander("ℹ️ Avisos", expanded=False):
    if use_upload_embeddings and SentenceTransformer is None:
        st.info("Instale `sentence-transformers` para usar embeddings no upload. Fallback TF-IDF será usado.")
    if use_embeddings_only:
        if (sph_emb is None) and (gos_emb is None) and (his_emb is None):
            st.warning("Sem nenhum embeddings *.npz disponível no histórico — nada a recuperar. Gere com make_catalog_embeddings.py.")
