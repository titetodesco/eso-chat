# app_chat.py — ESO • CHAT (Embeddings-only)
# - Busca SEMÂNTICA usando embeddings:
#   • Sphera:   data/analytics/sphera_embeddings.npz   + sphera.parquet
#   • GoSee:    data/analytics/gosee_embeddings.npz    + gosee.parquet
#   • History:  data/analytics/history_embeddings.npz  + history_texts.jsonl
# - Uploads: faz chunk + embeddings em tempo real (Sentence-Transformers)
# - Injeta apenas TRECHOS recuperados (não envia vetores ao LLM)
# - Sem TF-IDF, sem ONNX: apenas ST + Torch CPU

import os
import io
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Config básica ----------
st.set_page_config(page_title="ESO • CHAT (Embeddings)", page_icon="💬", layout="wide")

DATA_DIR = "data"
AN_DIR = os.path.join(DATA_DIR, "analytics")
DATASETS_CONTEXT_FILE = "datasets_context.md"  # opcional (YAML em texto)
ST_MODEL_NAME = os.getenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Modelo de chat (Ollama-compatible). Se não tiver chave, tenta mesmo assim.
OLLAMA_HOST  = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gpt-oss:20b"))
OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY"))
HEADERS_JSON = {"Authorization": f"Bearer {OLLAMA_API_KEY}", "Content-Type": "application/json"} if OLLAMA_API_KEY else {"Content-Type": "application/json"}

# ---------- Dependências necessárias (sem elas o app para de forma elegante) ----------
def _fatal(msg: str):
    st.error(msg)
    st.stop()

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    _fatal(
        "❌ sentence-transformers não está disponível.\n\n"
        "Instale as dependências (incluindo torch CPU) conforme o requirements.txt recomendado."
        f"\n\nDetalhe: {e}"
    )

try:
    import pypdf
except Exception:
    pypdf = None

try:
    import docx
except Exception:
    docx = None

# ---------- Utilidades ----------
def ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2, stream=False, timeout=120):
    payload = {"model": model, "messages": messages, "temperature": float(temperature), "stream": bool(stream)}
    r = requests.post(f"{OLLAMA_HOST}/api/chat", headers=HEADERS_JSON, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def l2norm(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(np.float32, copy=False)
    n = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return mat / n

def cos_topk(E_db: np.ndarray, q: np.ndarray, k: int) -> list[tuple[int, float]]:
    # E_db: (n,d) L2; q: (d,) (L2)
    if E_db is None or E_db.size == 0:
        return []
    q = q.astype(np.float32, copy=False)
    q = q / (np.linalg.norm(q) + 1e-9)
    sims = E_db @ q  # (n,)
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]

def load_npz_embeddings(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=True) as z:
            # procurar chave provável
            for key in ("embeddings", "E", "X", "vectors", "vecs"):
                if key in z:
                    E = np.array(z[key]).astype(np.float32, copy=False)
                    return l2norm(E)
            # fallback: maior matriz 2D
            best_k, best_n = None, -1
            for k in z.files:
                arr = z[k]
                if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] > best_n:
                    best_k, best_n = k, arr.shape[0]
            if best_k is None:
                st.warning(f"{os.path.basename(path)} não contém matriz 2D de embeddings.")
                return None
            E = np.array(z[best_k]).astype(np.float32, copy=False)
            return l2norm(E)
    except Exception as e:
        st.warning(f"Falha ao ler {path}: {e}")
        return None

def read_pdf_bytes(b: bytes) -> str:
    if pypdf is None:
        return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(b))
        out = []
        for pg in reader.pages:
            try:
                out.append(pg.extract_text() or "")
            except Exception:
                pass
        return "\n".join(out)
    except Exception:
        return ""

def read_docx_bytes(b: bytes) -> str:
    if docx is None:
        return ""
    try:
        doc = docx.Document(io.BytesIO(b))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""

def read_any(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith(".pdf"):
        return read_pdf_bytes(data)
    if name.endswith(".docx"):
        return read_docx_bytes(data)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            xls = pd.ExcelFile(io.BytesIO(data))
            frames = []
            for s in xls.sheet_names:
                df = xls.parse(s)
                frames.append(df.astype(str))
            return pd.concat(frames, axis=0, ignore_index=True).to_csv(index=False) if frames else ""
        except Exception:
            return ""
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(data))
            return df.astype(str).to_csv(index=False)
        except Exception:
            return ""
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
        part = text[start:end].strip()
        if part:
            parts.append(part)
        if end >= L:
            break
        start = max(0, end - ov)
    return parts

# ---------- Estado ----------
if "chat" not in st.session_state:
    st.session_state.chat = []

# Uploads (embeddings de sessão)
if "upld_texts" not in st.session_state:
    st.session_state.upld_texts = []      # lista[str]
if "upld_meta" not in st.session_state:
    st.session_state.upld_meta = []       # lista[dict]
if "upld_emb" not in st.session_state:
    st.session_state.upld_emb = None      # np.ndarray (n,d) L2

# Encoder ST singleton
if "st_encoder" not in st.session_state:
    st.session_state.st_encoder = None

def ensure_st_encoder():
    if st.session_state.st_encoder is None:
        try:
            st.session_state.st_encoder = SentenceTransformer(ST_MODEL_NAME)
        except Exception as e:
            _fatal(
                "❌ Não foi possível carregar o encoder de embeddings (Sentence-Transformers). "
                f"Modelo: {ST_MODEL_NAME}\n\nDetalhe: {e}"
            )

def encode_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    ensure_st_encoder()
    M = st.session_state.st_encoder.encode(
        texts, batch_size=batch_size, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)
    return M

def encode_query(q: str) -> np.ndarray:
    ensure_st_encoder()
    v = st.session_state.st_encoder.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-9)
    return v

# ---------- Carregamento dos catálogos (embeddings + texto base) ----------
SPH_EMB_PATH = os.path.join(AN_DIR, "sphera_embeddings.npz")
GOS_EMB_PATH = os.path.join(AN_DIR, "gosee_embeddings.npz")
HIS_EMB_PATH = os.path.join(AN_DIR, "history_embeddings.npz")

SPH_PQ_PATH = os.path.join(AN_DIR, "sphera.parquet")
GOS_PQ_PATH = os.path.join(AN_DIR, "gosee.parquet")
HIS_JSONL   = os.path.join(AN_DIR, "history_texts.jsonl")

E_sph = load_npz_embeddings(SPH_EMB_PATH)
E_gos = load_npz_embeddings(GOS_EMB_PATH)
E_his = load_npz_embeddings(HIS_EMB_PATH)

df_sph = None
df_gos = None
rows_his = []

if os.path.exists(SPH_PQ_PATH):
    try:
        df_sph = pd.read_parquet(SPH_PQ_PATH)
    except Exception as e:
        st.warning(f"Falha ao ler {SPH_PQ_PATH}: {e}")
if os.path.exists(GOS_PQ_PATH):
    try:
        df_gos = pd.read_parquet(GOS_PQ_PATH)
    except Exception as e:
        st.warning(f"Falha ao ler {GOS_PQ_PATH}: {e}")
if os.path.exists(HIS_JSONL):
    try:
        with open(HIS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                rows_his.append(json.loads(line))
    except Exception as e:
        st.warning(f"Falha ao ler {HIS_JSONL}: {e}")

# ---------- Sidebar ----------
st.sidebar.header("Configurações")
with st.sidebar.expander("Modelo de Resposta", expanded=False):
    st.write("Host:", OLLAMA_HOST)
    st.write("Modelo:", OLLAMA_MODEL)
    if not OLLAMA_API_KEY:
        st.info("Sem OLLAMA_API_KEY — ok para ambientes locais se o host não exigir auth.")

st.sidebar.subheader("Recuperação (Embeddings)")
k_sph = st.sidebar.slider("Top-K Sphera", 0, 10, 5, 1)
k_gos = st.sidebar.slider("Top-K GoSee",  0, 10, 5, 1)
k_his = st.sidebar.slider("Top-K Docs",   0, 10, 3, 1)
k_upl = st.sidebar.slider("Top-K Upload", 0, 10, 5, 1)

st.sidebar.subheader("Upload")
chunk_size  = st.sidebar.slider("Tamanho do chunk", 500, 2000, 1200, 50)
chunk_ovlp  = st.sidebar.slider("Overlap do chunk", 50, 600, 200, 10)
upload_raw_max = st.sidebar.slider("Tamanho máx. de UPLOAD_RAW (chars)", 300, 8000, 2500, 100)

use_catalog = st.sidebar.checkbox("Injetar datasets_context.md", True)

uploaded_files = st.sidebar.file_uploader(
    "Upload (PDF, DOCX, XLSX, CSV, TXT/MD)",
    type=["pdf", "docx", "xlsx", "xls", "csv", "txt", "md"],
    accept_multiple_files=True
)

c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("Limpar uploads", use_container_width=True):
        st.session_state.upld_texts = []
        st.session_state.upld_meta = []
        st.session_state.upld_emb = None
with c2:
    if st.button("Limpar chat", use_container_width=True):
        st.session_state.chat = []

# ---------- Indexação de Uploads (embeddings em sessão) ----------
if uploaded_files:
    with st.spinner("Lendo e embutindo uploads (embeddings)…"):
        new_texts, new_meta = [], []
        for uf in uploaded_files:
            try:
                raw = read_any(uf)
                parts = chunk_text(raw, max_chars=chunk_size, overlap=chunk_ovlp)
                for i, p in enumerate(parts):
                    new_texts.append(p)
                    new_meta.append({"file": uf.name, "chunk_id": i})
            except Exception as e:
                st.warning(f"Falha ao processar {uf.name}: {e}")
        if new_texts:
            # embute apenas os novos
            M_new = encode_texts(new_texts, batch_size=64)
            if st.session_state.upld_emb is None:
                st.session_state.upld_emb = M_new
            else:
                st.session_state.upld_emb = np.vstack([st.session_state.upld_emb, M_new])
            st.session_state.upld_texts.extend(new_texts)
            st.session_state.upld_meta.extend(new_meta)
            st.success(f"Upload indexado: {len(new_texts)} chunks.")

# ---------- Busca ----------
def search_all(query: str) -> list[str]:
    """Embute a query e busca nos 4 conjuntos (Sphera/GoSee/Docs/Upload). Retorna blocos formatados."""
    qv = encode_query(query)
    blocks: list[tuple[float, str]] = []

    # Sphera
    if k_sph > 0 and E_sph is not None and df_sph is not None and len(df_sph) >= E_sph.shape[0]:
        text_col = "Description" if "Description" in df_sph.columns else df_sph.columns[0]
        id_col = "Event ID" if "Event ID" in df_sph.columns else ("EVENT_NUMBER" if "EVENT_NUMBER" in df_sph.columns else None)
        hits = cos_topk(E_sph, qv, k=k_sph)
        for i, s in hits:
            row = df_sph.iloc[i]
            evid = row.get(id_col, f"row{i}") if id_col else f"row{i}"
            snippet = str(row.get(text_col, ""))[:800]
            blocks.append((s, f"[Sphera/{evid}] (sim={s:.3f})\n{snippet}"))

    # GoSee
    if k_gos > 0 and E_gos is not None and df_gos is not None and len(df_gos) >= E_gos.shape[0]:
        text_col = "Observation" if "Observation" in df_gos.columns else df_gos.columns[0]
        id_col = "ID" if "ID" in df_gos.columns else None
        hits = cos_topk(E_gos, qv, k=k_gos)
        for i, s in hits:
            row = df_gos.iloc[i]
            gid = row.get(id_col, f"row{i}") if id_col else f"row{i}"
            snippet = str(row.get(text_col, ""))[:800]
            blocks.append((s, f"[GoSee/{gid}] (sim={s:.3f})\n{snippet}"))

    # Docs (history)
    if k_his > 0 and E_his is not None and rows_his:
        hits = cos_topk(E_his, qv, k=k_his)
        for i, s in hits:
            r = rows_his[i]
            src = f"Docs/{r.get('source','?')}/{r.get('chunk_id', 0)}"
            snippet = str(r.get("text", ""))[:800]
            blocks.append((s, f"[{src}] (sim={s:.3f})\n{snippet}"))

    # Upload
    if k_upl > 0 and st.session_state.upld_emb is not None and len(st.session_state.upld_texts) == st.session_state.upld_emb.shape[0]:
        hits = cos_topk(st.session_state.upld_emb, qv, k=k_upl)
        for i, s in hits:
            meta = st.session_state.upld_meta[i]
            snippet = st.session_state.upld_texts[i][:800]
            blocks.append((s, f"[UPLOAD {meta['file']} / {meta['chunk_id']}] (sim={s:.3f})\n{snippet}"))

    blocks.sort(key=lambda x: -x[0])
    return [b for _, b in blocks]

def get_upload_raw(max_chars: int) -> str:
    if not st.session_state.upld_texts:
        return ""
    buf, total = [], 0
    for t in st.session_state.upld_texts[:3]:  # 3 trechos é suficiente
        if total >= max_chars:
            break
        t = t[: max_chars - total]
        buf.append(t)
        total += len(t)
    return "\n\n".join(buf).strip()

# ---------- UI ----------
st.title("ESO • CHAT — HIST + UPLD (Embeddings preferencial)")
st.caption("RAG local 100% embeddings (Sphera / GoSee / Docs / Upload).")

# Mostrar histórico
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Digite sua pergunta…")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Recuperação
    blocks = search_all(prompt)

    # Opcional: injeta um recorte 'cru' do upload (máx N chars)
    up_raw = get_upload_raw(upload_raw_max)
    if up_raw:
        blocks = [f"[UPLOAD_RAW]\n{up_raw}"] + blocks

    # Monta mensagens p/ LLM
    SYSTEM = (
        "Você é um assistente de segurança operacional. "
        "Use os CONTEXTOS abaixo como evidências. "
        "Quando citar fatos específicos, inclua aspas de trechos dos contextos e a etiqueta "
        "[Fonte/ID] do bloco correspondente. Não invente dados fora dos contextos."
    )
    msgs = [{"role": "system", "content": SYSTEM}]

    if use_catalog and os.path.exists(DATASETS_CONTEXT_FILE):
        try:
            with open(DATASETS_CONTEXT_FILE, "r", encoding="utf-8") as f:
                msgs.append({"role": "system", "content": f.read()})
        except Exception:
            pass

    if blocks:
        ctx = "\n\n".join(blocks)
        msgs.append({"role": "user", "content": f"CONTEXTOS (HIST + UPLOAD):\n{ctx}"})
        msgs.append({"role": "user", "content": f"PERGUNTA: {prompt}"})
    else:
        msgs.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Consultando o modelo…"):
            try:
                resp = ollama_chat(msgs, model=OLLAMA_MODEL, temperature=0.2, stream=False)
                content = resp.get("message", {}).get("content", "").strip() or json.dumps(resp)[:1200]
            except Exception as e:
                content = f"Falha ao consultar o modelo: {e}"
            st.markdown(content)
    st.session_state.chat.append({"role": "assistant", "content": content})

# ---------- Painel / Diagnóstico ----------
with st.expander("📦 Status dos índices", expanded=False):
    def _ok(x): return "✅" if x else "—"
    st.write("Sphera embeddings:", _ok(E_sph is not None and df_sph is not None))
    if E_sph is not None and df_sph is not None:
        st.write(f" • shape: {E_sph.shape} | linhas df: {len(df_sph)}")
    st.write("GoSee embeddings :", _ok(E_gos is not None and df_gos is not None))
    if E_gos is not None and df_gos is not None:
        st.write(f" • shape: {E_gos.shape} | linhas df: {len(df_gos)}")
    st.write("Docs embeddings  :", _ok(E_his is not None and len(rows_his) > 0))
    if E_his is not None and rows_his:
        st.write(f" • shape: {E_his.shape} | chunks: {len(rows_his)}")
    st.write("Uploads indexados:", len(st.session_state.upld_texts))
    st.write("Encoder ativo    :", ST_MODEL_NAME)
