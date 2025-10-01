# app_chat.py — ESO • CHAT (Embeddings-only)
# - Busca SEMÂNTICA usando embeddings:
#   • Sphera:     data/analytics/sphera_embeddings.npz   + sphera.parquet
#   • GoSee:      data/analytics/gosee_embeddings.npz    + gosee.parquet
#   • History:    data/analytics/history_embeddings.npz  + history_texts.jsonl
#   • WeakSignals: data/analytics/ws_embeddings.npz      + ws.parquet | data/xlsx/DicionarioWeakSignals.xlsx
#   • Precursores: data/analytics/prec_embeddings.npz    + prec.parquet | data/xlsx/precursores_expandido.xlsx
#   • CP Taxonomy: data/analytics/cp_embeddings.npz      + cp.parquet   | data/xlsx/TaxonomiaCP_Por.xlsx
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
from pathlib import Path

# ---------- Config básica (primeiro comando Streamlit!) ----------
st.set_page_config(page_title="ESO • CHAT (Embeddings)", page_icon="💬", layout="wide")

# ---------- Contexto (system prompt) ----------
CONTEXT_MD_REL_PATH = Path(__file__).parent / "docs" / "contexto_eso_chat.md"
DATASETS_CONTEXT_FILE = "datasets_context.md"  # opcional (YAML/MD adicional)

@st.cache_data(show_spinner=False)
def load_file_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"[AVISO] Não consegui ler {p}: {e}\n(Prosseguindo sem esse contexto.)"

def build_system_prompt() -> str:
    preambulo = (
        "Você é o ESO-CHAT (segurança operacional).\n"
        "REGRAS RÍGIDAS:\n"
        "1) Ao listar Weak Signals, Precursores (H-T-O) e Fatores da Taxonomia CP, USE EXCLUSIVAMENTE os itens presentes nos dicionários embutidos (WS/Prec/CP), "
        "com base nos trechos recuperados como contexto ([WS/...], [Prec/...], [CP/...]).\n"
        "2) NÃO crie sinais/fatores novos a partir do upload; o upload serve apenas para evidência de correspondência semântica.\n"
        "3) Sempre cite os blocos de contexto (ex.: [WS/<id>] e [UPLOAD ...]) ao justificar.\n"
        "4) Se não houver correspondência ≥ limiar, diga explicitamente que não foi encontrado.\n"
        "Responda em PT-BR por padrão. Não invente dados fora dos contextos fornecidos.\n"
    )
    ctx_md = load_file_text(CONTEXT_MD_REL_PATH)
    return preambulo + "\n\n=== CONTEXTO ESO-CHAT (.md) ===\n" + ctx_md

# Inicializa uma vez por sessão
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = build_system_prompt()

# ---------- Parâmetros / caminhos ----------
DATA_DIR = "data"
AN_DIR = os.path.join(DATA_DIR, "analytics")
ST_MODEL_NAME = os.getenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Modelo de chat (Ollama-compatible). Se não tiver chave, tenta mesmo assim.
OLLAMA_HOST  = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gpt-oss:20b"))
OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY"))
HEADERS_JSON = {"Authorization": f"Bearer {OLLAMA_API_KEY}", "Content-Type": "application/json"} if OLLAMA_API_KEY else {"Content-Type": "application/json"}

# ---------- Dependências necessárias ----------
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

def cos_topk(E_db: np.ndarray, q: np.ndarray, k: int):
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
# Sphera / GoSee / Docs
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

# Weak Signals / Precursores / CP (embeddings + tabela)
WS_EMB_PATH = os.path.join(AN_DIR, "ws_embeddings.npz")                # ou weak_signals_embeddings.npz
WS_PQ_PATH  = os.path.join(AN_DIR, "ws.parquet")
WS_XLSX_FB  = os.path.join("data", "xlsx", "DicionarioWeakSignals.xlsx")

PREC_EMB_PATH = os.path.join(AN_DIR, "prec_embeddings.npz")
PREC_PQ_PATH  = os.path.join(AN_DIR, "precursores.parquet")
PREC_XLSX_FB  = os.path.join("data", "xlsx", "precursores_expandido.xlsx")

CP_EMB_PATH = os.path.join(AN_DIR, "cp_embeddings.npz")
CP_PQ_PATH  = os.path.join(AN_DIR, "cp.parquet")
CP_XLSX_FB  = os.path.join("data", "xlsx", "TaxonomiaCP_Por.xlsx")

E_ws   = load_npz_embeddings(WS_EMB_PATH)
E_prec = load_npz_embeddings(PREC_EMB_PATH)
E_cp   = load_npz_embeddings(CP_EMB_PATH)

def _read_df_with_fallback(parquet_path, xlsx_path):
    df = None
    if os.path.exists(parquet_path):
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            st.warning(f"Falha ao ler {parquet_path}: {e}")
    elif os.path.exists(xlsx_path):
        try:
            df = pd.read_excel(xlsx_path)
        except Exception as e:
            st.warning(f"Falha ao ler {xlsx_path}: {e}")
    return df

df_ws   = _read_df_with_fallback(WS_PQ_PATH,   WS_XLSX_FB)
df_prec = _read_df_with_fallback(PREC_PQ_PATH, PREC_XLSX_FB)
df_cp   = _read_df_with_fallback(CP_PQ_PATH,   CP_XLSX_FB)

# Descobrir colunas relevantes de cada dicionário
def _normalize_dict_cols(df: pd.DataFrame, preferred_terms=("PT","pt","Term","term","Sinal","sinal","Nome","name"), id_candidates=("ID","Id","id")):
    if df is None or df.empty:
        return None, None
    cols_map = {c.lower(): c for c in df.columns}
    term_col = None
    for cand in preferred_terms:
        if cand in df.columns:
            term_col = cand
            break
        if cand.lower() in cols_map:
            term_col = cols_map[cand.lower()]
            break
    if term_col is None:
        term_col = list(df.columns)[0]  # fallback: primeira coluna

    id_col = None
    for cand in id_candidates:
        if cand in df.columns:
            id_col = cand
            break
        if cand.lower() in cols_map:
            id_col = cols_map[cand.lower()]
            break
    if id_col is None:
        df["_ID_"] = np.arange(len(df))
        id_col = "_ID_"
    return term_col, id_col

ws_term_col, ws_id_col     = _normalize_dict_cols(df_ws)   if df_ws is not None else (None, None)
prec_term_col, prec_id_col = _normalize_dict_cols(df_prec) if df_prec is not None else (None, None)
cp_term_col, cp_id_col     = _normalize_dict_cols(df_cp)   if df_cp is not None else (None, None)

# Para CP, se existirem colunas Dimensão/Fator/Subfator, cria um campo exibível composto (sem alterar embeddings)
def _compose_cp_label(row: pd.Series):
    parts = []
    for key in ("Dimensão","Dimensao","Dimensao/Dimension","Dimension","Dimens\u00e3o"):
        if key in row:
            parts.append(str(row[key]).strip())
            break
    for key in ("Fator","Fator/Factor","Factor"):
        if key in row:
            parts.append(str(row[key]).strip())
            break
    for key in ("Subfator","Sub-fator","SubFactor","Sub-fator/SubFactor"):
        if key in row:
            parts.append(str(row[key]).strip())
            break
    if parts:
        return " / ".join([p for p in parts if p])
    # fallback: termo base
    return str(row.get(cp_term_col, "")).strip() if cp_term_col else ""

# ---------- Sidebar ----------
st.sidebar.header("Configurações")
with st.sidebar.expander("Modelo de Resposta", expanded=False):
    st.write("Host:", OLLAMA_HOST)
    st.write("Modelo:", OLLAMA_MODEL)
    if not OLLAMA_API_KEY:
        st.info("Sem OLLAMA_API_KEY — ok para ambientes locais se o host não exigir auth.")
    if st.button("Recarregar contexto (.md)", use_container_width=True):
        st.session_state.system_prompt = build_system_prompt()
        st.success("Contexto recarregado.")

st.sidebar.subheader("Recuperação (Embeddings)")
k_sph = st.sidebar.slider("Top-K Sphera", 0, 10, 5, 1)
k_gos = st.sidebar.slider("Top-K GoSee",  0, 10, 5, 1)
k_his = st.sidebar.slider("Top-K Docs",   0, 10, 3, 1)
k_upl = st.sidebar.slider("Top-K Upload", 0, 10, 5, 1)

st.sidebar.subheader("Dicionários (WS / Precursores / CP)")
k_ws_query   = st.sidebar.slider("Top-K WS (por query)",   0, 20, 10, 1)
k_prec_query = st.sidebar.slider("Top-K Precursores (por query)", 0, 20, 10, 1)
k_cp_query   = st.sidebar.slider("Top-K CP (por query)",   0, 20, 10, 1)

ws_upl_thresh   = st.sidebar.number_input("Limiar Upload↔WS (cos)",   min_value=0.0, max_value=1.0, value=0.25, step=0.01)
prec_upl_thresh = st.sidebar.number_input("Limiar Upload↔Prec (cos)", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
cp_upl_thresh   = st.sidebar.number_input("Limiar Upload↔CP (cos)",   min_value=0.0, max_value=1.0, value=0.25, step=0.01)

include_ws_upload   = st.sidebar.checkbox("Cruzar Upload com Dicionário WS",   True)
include_prec_upload = st.sidebar.checkbox("Cruzar Upload com Dicionário Prec", True)
include_cp_upload   = st.sidebar.checkbox("Cruzar Upload com Dicionário CP",   True)

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
            M_new = encode_texts(new_texts, batch_size=64)
            if st.session_state.upld_emb is None:
                st.session_state.upld_emb = M_new
            else:
                st.session_state.upld_emb = np.vstack([st.session_state.upld_emb, M_new])
            st.session_state.upld_texts.extend(new_texts)
            st.session_state.upld_meta.extend(new_meta)
            st.success(f"Upload indexado: {len(new_texts)} chunks.")

# ---------- Funções de busca nos dicionários ----------
def _search_dict_by_query(E_dict, df_dict, term_col, id_col, tag: str, query: str, topk: int = 10):
    if E_dict is None or df_dict is None or term_col is None or id_col is None:
        return []
    qv = encode_query(query)
    hits = cos_topk(E_dict, qv, k=min(topk, E_dict.shape[0]))
    blocks = []
    for i, s in hits:
        row = df_dict.iloc[i]
        did = row.get(id_col, f"row{i}")
        term = str(row.get(term_col, "")).strip()
        # Para CP, tente exibir rótulo composto
        if tag == "CP":
            label = _compose_cp_label(row) or term
            blocks.append((s, f"[{tag}/{did}] (sim={s:.3f}) {label}"))
        else:
            blocks.append((s, f"[{tag}/{did}] (sim={s:.3f}) {term}"))
    return blocks

def _match_upload_to_dict(E_dict, df_dict, term_col, id_col, tag: str, threshold: float, max_upload_chunks: int = 200):
    if E_dict is None or df_dict is None or term_col is None or id_col is None:
        return []
    if st.session_state.upld_emb is None or not st.session_state.upld_texts:
        return []
    E_upl = st.session_state.upld_emb
    texts = st.session_state.upld_texts
    meta  = st.session_state.upld_meta

    n = min(len(texts), max_upload_chunks)
    E_upl = E_upl[:n]
    texts = texts[:n]
    meta  = meta[:n]

    blocks = []
    for i in range(E_dict.shape[0]):
        vec = E_dict[i]
        sims = E_upl @ vec
        j = int(np.argmax(sims))
        s = float(sims[j])
        if s >= threshold:
            row = df_dict.iloc[i]
            did = row.get(id_col, f"row{i}")
            term = str(row.get(term_col, "")).strip()
            if tag == "CP":
                term = _compose_cp_label(row) or term
            snippet = texts[j][:300].replace("\n", " ")
            label = f"[UPLOAD {meta[j]['file']} / {meta[j]['chunk_id']}]"
            blocks.append((s, f"[{tag}/{did}] (sim={s:.3f}) {term}\n  ↳ {label}  “{snippet}”"))
    blocks.sort(key=lambda x: -x[0])
    return blocks[:100]

# ---------- Busca principal ----------
def search_all(query: str) -> list[str]:
    """Embute a query e busca em Sphera/GoSee/Docs/Upload + Dicionários (WS/Prec/CP). Retorna blocos formatados."""
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

    # Upload (top-k pelos chunks mais parecidos à query)
    if k_upl > 0 and st.session_state.upld_emb is not None and len(st.session_state.upld_texts) == st.session_state.upld_emb.shape[0]:
        hits = cos_topk(st.session_state.upld_emb, qv, k=k_upl)
        for i, s in hits:
            meta = st.session_state.upld_meta[i]
            snippet = st.session_state.upld_texts[i][:800]
            blocks.append((s, f"[UPLOAD {meta['file']} / {meta['chunk_id']}] (sim={s:.3f})\n{snippet}"))

    # DICIONÁRIOS: por query
    if k_ws_query > 0:
        blocks.extend(_search_dict_by_query(E_ws, df_ws, ws_term_col, ws_id_col, "WS", query, topk=k_ws_query))
    if k_prec_query > 0:
        blocks.extend(_search_dict_by_query(E_prec, df_prec, prec_term_col, prec_id_col, "Prec", query, topk=k_prec_query))
    if k_cp_query > 0:
        blocks.extend(_search_dict_by_query(E_cp, df_cp, cp_term_col, cp_id_col, "CP", query, topk=k_cp_query))

    # DICIONÁRIOS: match Upload ↔ Dict (somente termos existentes)
    if include_ws_upload:
        blocks.extend(_match_upload_to_dict(E_ws, df_ws, ws_term_col, ws_id_col, "WS", ws_upl_thresh))
    if include_prec_upload:
        blocks.extend(_match_upload_to_dict(E_prec, df_prec, prec_term_col, prec_id_col, "Prec", prec_upl_thresh))
    if include_cp_upload:
        blocks.extend(_match_upload_to_dict(E_cp, df_cp, cp_term_col, cp_id_col, "CP", cp_upl_thresh))

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
st.title("ESO • CHAT — HIST + UPLD + WS/Prec/CP (Embeddings)")
st.caption("RAG local 100% embeddings (Sphera / GoSee / Docs / Upload / Dicionários).")

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
    msgs = [{"role": "system", "content": st.session_state.system_prompt}]

    if use_catalog and os.path.exists(DATASETS_CONTEXT_FILE):
        try:
            with open(DATASETS_CONTEXT_FILE, "r", encoding="utf-8") as f:
                msgs.append({"role": "system", "content": f.read()})
        except Exception:
            pass

    if blocks:
        ctx = "\n\n".join(blocks)
        msgs.append({"role": "user", "content": f"CONTEXTOS (HIST + UPLOAD + DICIONÁRIOS):\n{ctx}"})
        msgs.append({"role": "user", "content": f"PERGUNTA: {prompt}"})
    else:
        msgs.append({"role": "user", "content": prompt})

    # Chamada ao modelo
    with st.chat_message("assistant"):
        with st.spinner("Consultando o modelo…"):
            try:
                resp = ollama_chat(msgs, model=OLLAMA_MODEL, temperature=0.2, stream=False)
                content = resp.get("message", {}).get("content", "").strip() or json.dumps(resp)[:1200]
            except Exception as e:
                content = f"Falha ao consultar o modelo: {e}"
            st.markdown(content)
    st.session_state.chat.append({"role": "assistant", "content": content})

# ---------- Painel / Diagnóstico (opcional) ----------
debug = st.sidebar.checkbox("Mostrar painel de diagnóstico", False)

if debug:
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

        st.write("---")
        st.write("WS embeddings     :", _ok(E_ws is not None and df_ws is not None))
        if E_ws is not None and df_ws is not None:
            st.write(f" • shape: {E_ws.shape} | linhas df: {len(df_ws)} | col termo: {ws_term_col} | id: {ws_id_col}")
        st.write("Precursores emb.  :", _ok(E_prec is not None and df_prec is not None))
        if E_prec is not None and df_prec is not None:
            st.write(f" • shape: {E_prec.shape} | linhas df: {len(df_prec)} | col termo: {prec_term_col} | id: {prec_id_col}")
        st.write("CP Taxonomy emb.  :", _ok(E_cp is not None and df_cp is not None))
        if E_cp is not None and df_cp is not None:
            st.write(f" • shape: {E_cp.shape} | linhas df: {len(df_cp)} | col termo: {cp_term_col} | id: {cp_id_col}")

        st.write("Uploads indexados :", len(st.session_state.upld_texts))
        st.write("Encoder ativo     :", ST_MODEL_NAME)

    with st.expander("🔎 Versões dos pacotes", expanded=False):
        import importlib, sys
        pkgs = [
            ("torch", "torch"),
            ("transformers", "transformers"),
            ("sentence-transformers", "sentence_transformers"),
            ("pandas", "pandas"),
            ("numpy", "numpy"),
            ("pyarrow", "pyarrow"),
            ("pypdf", "pypdf"),
            ("python-docx", "docx"),
            ("scikit-learn", "sklearn"),
        ]
        st.write("Python:", sys.version)
        for disp, mod in pkgs:
            try:
                m = importlib.import_module(mod)
                ver = getattr(m, "__version__", "sem __version__")
                st.write(f"{disp}: {ver}")
            except Exception as e:
                st.write(f"{disp}: não instalado ({e})")
