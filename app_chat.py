# app_chat.py — ESO • CHAT (Embeddings-only)
# - Busca SEMÂNTICA usando embeddings:
#   • Sphera:   data/analytics/sphera_embeddings.npz   + sphera.parquet
#   • GoSee:    data/analytics/gosee_embeddings.npz    + gosee.parquet
#   • History:  data/analytics/history_embeddings.npz  + history_texts.jsonl
#   • WS/Prec/CP: dicionários (parquet/xlsx) + embeddings .npz (com ids, se possível)
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

# ---------- Contexto (system prompt) ----------
CONTEXT_MD_REL_PATH = Path(__file__).parent / "docs" / "contexto_eso_chat.md"
DATASETS_CONTEXT_FILE = "datasets_context.md"  # mantido

@st.cache_data(show_spinner=False)
def load_file_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"[AVISO] Não consegui ler {p}: {e}\n(Prosseguindo sem esse contexto.)"

def build_system_prompt() -> str:
    preambulo = (
        "Você é o ESO-CHAT (segurança operacional).\n"
        "Siga estritamente as regras do contexto abaixo. Responda em PT-BR por padrão.\n"
        "Quando usar buscas semânticas, sempre mostre IDs/Fonte e similaridade.\n"
        "Não invente dados fora dos contextos fornecidos.\n"
    )
    ctx_md = load_file_text(CONTEXT_MD_REL_PATH)
    return preambulo + "\n\n=== CONTEXTO ESO-CHAT (.md) ===\n" + ctx_md

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = build_system_prompt()

# (Opcional) botão para recarregar o .md sem reiniciar o app
st.sidebar.button(
    "Recarregar contexto (.md)",
    on_click=lambda: st.session_state.update(system_prompt=build_system_prompt()),
)

# ---------- Config básica ----------
st.set_page_config(page_title="ESO • CHAT (Embeddings)", page_icon="💬", layout="wide")

DATA_DIR = "data"
AN_DIR = os.path.join(DATA_DIR, "analytics")
XLSX_DIR = os.path.join(DATA_DIR, "xlsx")
ST_MODEL_NAME = os.getenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Modelo de chat (Ollama-compatible)
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

# ---------- Utilidades de rede / LLM ----------
def ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2, stream=False, timeout=120):
    payload = {"model": model, "messages": messages, "temperature": float(temperature), "stream": bool(stream)}
    r = requests.post(f"{OLLAMA_HOST}/api/chat", headers=HEADERS_JSON, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ---------- Utilidades de vetor ----------
def l2norm(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(np.float32, copy=False)
    n = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return mat / n

def cos_topk(E_db: np.ndarray, q: np.ndarray, k: int) -> list[tuple[int, float]]:
    if E_db is None or E_db.size == 0:
        return []
    q = q.astype(np.float32, copy=False)
    q = q / (np.linalg.norm(q) + 1e-9)
    sims = E_db @ q
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]

def load_npz_with_ids(path: str):
    """Retorna (E_l2, ids[str] | None). Aceita chaves 'embeddings' e 'ids'."""
    if not os.path.exists(path):
        return None, None
    try:
        with np.load(path, allow_pickle=True) as z:
            E = None
            for key in ("embeddings", "E", "X", "vectors", "vecs"):
                if key in z:
                    E = np.array(z[key]).astype(np.float32, copy=False)
                    break
            if E is None:
                # fallback: maior matriz 2D
                best_k, best_n = None, -1
                for k in z.files:
                    arr = z[k]
                    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] > best_n:
                        best_k, best_n = k, arr.shape[0]
                if best_k is None:
                    st.warning(f"{os.path.basename(path)} não contém matriz 2D de embeddings.")
                    return None, None
                E = np.array(z[best_k]).astype(np.float32, copy=False)
            ids = None
            for key in ("ids", "row_ids", "index", "keys"):
                if key in z:
                    ids = [str(x) for x in z[key].tolist()]
                    break
            return l2norm(E), ids
    except Exception as e:
        st.warning(f"Falha ao ler {path}: {e}")
        return None, None

def align_df_by_ids(df: pd.DataFrame, id_col: str, ids: list[str], name: str):
    """Reordena df para a ordem dos ids do .npz (inner-join)."""
    if df is None or not isinstance(df, pd.DataFrame) or not ids:
        return df, None
    if id_col not in df.columns:
        st.warning(f"[{name}] '{id_col}' não existe no DataFrame — não foi possível alinhar por ids.")
        return df, None
    df["_IDX_TMP_"] = np.arange(len(df))
    m = pd.Series(df["_IDX_TMP_"].values, index=df[id_col].astype(str), dtype="Int64")
    take_idx = [int(m[i]) for i in ids if i in m.index and pd.notna(m[i])]
    if not take_idx:
        st.warning(f"[{name}] Nenhum id do .npz foi encontrado no DataFrame ({id_col}).")
        df.drop(columns=["_IDX_TMP_"], errors="ignore", inplace=True)
        return df, None
    df2 = df.take(take_idx).reset_index(drop=True)
    df.drop(columns=["_IDX_TMP_"], errors="ignore", inplace=True)
    if len(df2) != len(ids):
        st.warning(f"[{name}] Alinhamento parcial: {len(df2)} de {len(ids)} ids casaram (inner-join).")
    return df2, take_idx

# ---------- Leitura de arquivos arbitrários ----------
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

# ---------- Carregamento dos catálogos (Sphera/GoSee/History) ----------
SPH_EMB_PATH = os.path.join(AN_DIR, "sphera_embeddings.npz")
GOS_EMB_PATH = os.path.join(AN_DIR, "gosee_embeddings.npz")
HIS_EMB_PATH = os.path.join(AN_DIR, "history_embeddings.npz")

SPH_PQ_PATH = os.path.join(AN_DIR, "sphera.parquet")
GOS_PQ_PATH = os.path.join(AN_DIR, "gosee.parquet")
HIS_JSONL   = os.path.join(AN_DIR, "history_texts.jsonl")

def load_parquet_or_none(path):
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception as e:
            st.warning(f"Falha ao ler {path}: {e}")
    return None

E_sph, _ = load_npz_with_ids(SPH_EMB_PATH)
E_gos, _ = load_npz_with_ids(GOS_EMB_PATH)
E_his, _ = load_npz_with_ids(HIS_EMB_PATH)

df_sph = load_parquet_or_none(SPH_PQ_PATH)
df_gos = load_parquet_or_none(GOS_PQ_PATH)
rows_his = []
if os.path.exists(HIS_JSONL):
    try:
        with open(HIS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                rows_his.append(json.loads(line))
    except Exception as e:
        st.warning(f"Falha ao ler {HIS_JSONL}: {e}")

# ---------- Carregamento dicionários (WS / Precursores / CP) ----------
WS_EMB_PATH   = os.path.join(AN_DIR, "ws_embeddings.npz")
PREC_EMB_PATH = os.path.join(AN_DIR, "prec_embeddings.npz")
CP_EMB_PATH   = os.path.join(AN_DIR, "cp_embeddings.npz")

# Tabelas: tenta primeiro parquet no analytics, depois xlsx no data/xlsx
WS_TABLE_CANDIDATES = [
    os.path.join(AN_DIR, "ws.parquet"),
    os.path.join(XLSX_DIR, "DicionarioWeakSignals.xlsx"),
    os.path.join(XLSX_DIR, "DicionarioWaekSignals.xlsx"),  # fallback caso de nome antigo
]
PREC_TABLE_CANDIDATES = [
    os.path.join(AN_DIR, "precursores.parquet"),
    os.path.join(XLSX_DIR, "precursores_expandido.xlsx"),
]
CP_TABLE_CANDIDATES = [
    os.path.join(AN_DIR, "cp.parquet"),
    os.path.join(XLSX_DIR, "TaxonomiaCP_Por.xlsx"),
]

def load_table(candidates):
    for p in candidates:
        if os.path.exists(p):
            try:
                if p.endswith(".parquet"):
                    return pd.read_parquet(p), p
                else:
                    # pega a primeira planilha
                    return pd.read_excel(p), p
            except Exception as e:
                st.warning(f"Falha ao ler {p}: {e}")
    return None, None

df_ws,   WS_TABLE_SRC   = load_table(WS_TABLE_CANDIDATES)
df_prec, PREC_TABLE_SRC = load_table(PREC_TABLE_CANDIDATES)
df_cp,   CP_TABLE_SRC   = load_table(CP_TABLE_CANDIDATES)

E_ws,   WS_IDS   = load_npz_with_ids(WS_EMB_PATH)
E_prec, PREC_IDS = load_npz_with_ids(PREC_EMB_PATH)
E_cp,   CP_IDS   = load_npz_with_ids(CP_EMB_PATH)

# ---------- Normalização de colunas / idioma ----------
def _guess_lang_from_text(sample_text: str) -> str:
    """Heurística simples para PT vs EN."""
    if not sample_text:
        return "pt"
    s = sample_text.lower()
    score_pt = 0
    for token in (" que ", " não ", " para ", " foi ", " guindaste ", " cabo ", " operação "):
        if token in s:
            score_pt += 1
    accented = any(ch in s for ch in "áéíóúâêôãõç")
    if accented or score_pt >= 2:
        return "pt"
    return "en"

def detect_upload_lang(default="pt") -> str:
    # Toma alguns chunks do upload para avaliar
    if st.session_state.upld_texts:
        sample = " ".join(st.session_state.upld_texts[:2])[:2000]
        return _guess_lang_from_text(sample)
    return default

def _pick_first(df: pd.DataFrame, cols: list[str]) -> str | None:
    for c in cols:
        if c in df.columns:
            return c
    return None

def _norm_cols_ws(df: pd.DataFrame, lang: str):
    id_col = _pick_first(df, ["id","ID","codigo","Código","code","ws_id"])
    if lang == "pt":
        term_col = _pick_first(df, ["PT","pt","Termo PT","Termo_PT","Weak Signal (PT)"])
    else:
        term_col = _pick_first(df, ["EN","en","Termo EN","Termo_EN","Weak Signal (EN)"])
    # fallback
    if term_col is None:
        term_col = df.select_dtypes(include=["object"]).columns[0]
    if id_col is None:
        id_col = df.columns[0]
    return term_col, id_col

def _norm_cols_prec(df: pd.DataFrame, lang: str):
    id_col = _pick_first(df, ["id","ID","prec_id","codigo","Código","code"])
    if lang == "pt":
        term_col = _pick_first(df, ["Precursor (PT)","Precursor PT","PT","precursor","precursor_pt"])
    else:
        term_col = _pick_first(df, ["Precursor EN","EN","precursor_en"])
    hto_col = _pick_first(df, ["HTO","H-T-O","Tipo","Type","class"])
    if term_col is None:
        term_col = df.select_dtypes(include=["object"]).columns[0]
    if id_col is None:
        id_col = df.columns[0]
    return term_col, id_col, hto_col

def _norm_cols_cp(df: pd.DataFrame, lang: str):
    id_col  = _pick_first(df, ["id","ID","cp_id","codigo","Código","code"])
    bag_col = _pick_first(df, ["Bag de termos","bag","Bag","Bag of terms","terms"])
    dim_col = _pick_first(df, ["Dimensão","Dimensao","Dimension","Dimensão EN","Dimension EN"])
    fat_col = _pick_first(df, ["Fator","Factor","Fator EN","Factor EN"])
    sub_col = _pick_first(df, ["Sub-fator","Subfator","Subfactor","Sub-fator EN"])
    # Preferir coluna do idioma na exibição do termo (bag)
    if lang == "en" and "Bag of terms" in df.columns:
        bag_col = "Bag of terms"
    elif lang == "pt" and "Bag de termos" in df.columns:
        bag_col = "Bag de termos"
    # fallback
    if bag_col is None:
        bag_col = df.select_dtypes(include=["object"]).columns[0]
    if id_col is None:
        id_col = df.columns[0]
    return bag_col, id_col, dim_col, fat_col, sub_col

# ---------- Alinhamento por IDs (se o .npz tiver ids) ----------
def _check_align(E, df, name, have_ids=False):
    if E is not None and df is not None and E.shape[0] != len(df) and not have_ids:
        st.warning(f"[{name}] Embeddings ({E.shape[0]}) e tabela ({len(df)}) com contagem diferente — rótulos podem não bater o índice.")

# alinhamento inicial (usar PT só para descobrir colunas-ID; exibição usa idioma real depois)
if df_ws is not None and WS_IDS:
    _, ws_id_col = _norm_cols_ws(df_ws, "pt")
    df_ws, _ = align_df_by_ids(df_ws, ws_id_col, WS_IDS, "WS")

if df_prec is not None and PREC_IDS:
    _, prec_id_col_tmp, _ = _norm_cols_prec(df_prec, "pt")
    df_prec, _ = align_df_by_ids(df_prec, prec_id_col_tmp, PREC_IDS, "Precursores")

if df_cp is not None and CP_IDS:
    _, cp_id_col_tmp, _, _, _ = _norm_cols_cp(df_cp, "pt")
    df_cp, _ = align_df_by_ids(df_cp, cp_id_col_tmp, CP_IDS, "CP")

_check_align(E_ws, df_ws, "WS", have_ids=bool(WS_IDS))
_check_align(E_prec, df_prec, "Precursores", have_ids=bool(PREC_IDS))
_check_align(E_cp, df_cp, "CP", have_ids=bool(CP_IDS))

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

# --- WS/Precursores/CP (documento → dicionários) ---
st.sidebar.subheader("Extração: WS / Precursores / CP (do UPLOAD)")
enable_dict_match = st.sidebar.checkbox("Habilitar extração semântica dos dicionários", True)
lang_auto = detect_upload_lang()
ui_lang = st.sidebar.selectbox("Idioma esperado do UPLOAD", ["auto", "pt", "en"], index=0)
resolved_lang = lang_auto if ui_lang == "auto" else ui_lang

# Limiares
ws_topk = st.sidebar.slider("Top-K Weak Signals", 0, 20, 10, 1)
ws_thr  = st.sidebar.slider("Limiar WS (cos)", 0.0, 1.0, 0.25, 0.01)

prec_topk = st.sidebar.slider("Top-K Precursores", 0, 20, 10, 1)
prec_thr  = st.sidebar.slider("Limiar Precursores (cos)", 0.0, 1.0, 0.25, 0.01)

cp_topk = st.sidebar.slider("Top-K CP", 0, 20, 10, 1)
cp_thr  = st.sidebar.slider("Limiar CP (cos)", 0.0, 1.0, 0.25, 0.01)

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

def _upload_centroid() -> np.ndarray | None:
    """Retorna o vetor centróide (L2) dos embeddings do upload (ou None)."""
    E = st.session_state.upld_emb
    if E is None or E.size == 0:
        return None
    c = E.mean(axis=0)
    c = c.astype(np.float32)
    c /= (np.linalg)
