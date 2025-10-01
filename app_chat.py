# app_chat.py — ESO • CHAT (Embeddings-only)
# - Busca SEMÂNTICA usando embeddings:
#   • Sphera:   data/analytics/sphera_embeddings.npz   + sphera.parquet
#   • GoSee:    data/analytics/gosee_embeddings.npz    + gosee.parquet
#   • History:  data/analytics/history_embeddings.npz  + history_texts.jsonl
#   • WS/Prec/CP: data/analytics/{ws,prec,cp}_embeddings.npz + tabelas (.parquet ou .xlsx fallback)
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
DATASETS_CONTEXT_FILE = "datasets_context.md"  # você já tinha isso (mantido)

@st.cache_data(show_spinner=False)
def load_file_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"[AVISO] Não consegui ler {p}: {e}\n(Prosseguindo sem esse contexto.)"

def build_system_prompt() -> str:
    # Preâmbulo curto + contexto operacional
    preambulo = (
        "Você é o ESO-CHAT (segurança operacional).\n"
        "Siga estritamente as regras e convenções do contexto abaixo.\n"
        "Responda em PT-BR por padrão.\n"
        "Quando usar buscas semânticas, sempre mostre IDs/Fonte e similaridade.\n"
        "Não invente dados fora dos contextos fornecidos.\n"
    )
    ctx_md = load_file_text(CONTEXT_MD_REL_PATH)
    # Nota: o datasets_context.md (se existir) será adicionado mais abaixo como outra mensagem de sistema.
    return preambulo + "\n\n=== CONTEXTO ESO-CHAT (.md) ===\n" + ctx_md

# Inicializa uma vez por sessão
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = build_system_prompt()

# (Opcional) botão para recarregar o .md sem reiniciar o app
if st.sidebar.button("Recarregar contexto (.md)"):
    st.session_state.system_prompt = build_system_prompt()
    st.sidebar.success("Contexto recarregado.")

# ---------- Config básica ----------
st.set_page_config(page_title="ESO • CHAT (Embeddings)", page_icon="💬", layout="wide")

DATA_DIR = "data"
AN_DIR = os.path.join(DATA_DIR, "analytics")
XLS_DIR = os.path.join(DATA_DIR, "xlsx")

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

# ---------- Dicionários WS / Precursores / CP ----------
WS_EMB_PATH   = os.path.join(AN_DIR, "ws_embeddings.npz")
PREC_EMB_PATH = os.path.join(AN_DIR, "prec_embeddings.npz")   # nome visto nos logs
CP_EMB_PATH   = os.path.join(AN_DIR, "cp_embeddings.npz")

E_ws   = load_npz_embeddings(WS_EMB_PATH)
E_prec = load_npz_embeddings(PREC_EMB_PATH)
E_cp   = load_npz_embeddings(CP_EMB_PATH)

# Tabelas: tenta .parquet padrão na pasta analytics; se não houver, procura no /data/xlsx
def _try_read_parquet(*names):
    for n in names:
        p = os.path.join(AN_DIR, n)
        if os.path.exists(p):
            try:
                return pd.read_parquet(p)
            except Exception as e:
                st.warning(f"Falha ao ler {p}: {e}")
    return None

def _try_read_xlsx(*names):
    for n in names:
        p = os.path.join(XLS_DIR, n)
        if os.path.exists(p):
            try:
                return pd.read_excel(p)
            except Exception as e:
                st.warning(f"Falha ao ler {p}: {e}")
    return None

# WS
df_ws = _try_read_parquet("ws.parquet", "weak_signals.parquet")
if df_ws is None:
    df_ws = _try_read_xlsx("DicionarioWeakSignals.xlsx", "DicionarioWaekSignals.xlsx")

# Precursores
df_prec = _try_read_parquet("prec.parquet", "precursores.parquet")
if df_prec is None:
    df_prec = _try_read_xlsx("precursores_expandido.xlsx", "precursores.xlsx")

# CP
df_cp = _try_read_parquet("cp.parquet", "taxonomy_cp.parquet", "taxonomia_cp.parquet")
if df_cp is None:
    df_cp = _try_read_xlsx("TaxonomiaCP_Por.xlsx", "TaxonomiaCP.xlsx", "taxonomy_cp.xlsx")

def _check_align(E, df, name):
    if E is not None and df is not None and E.shape[0] != len(df):
        st.warning(f"[{name}] Embeddings ({E.shape[0]}) e tabela ({len(df)}) com contagem diferente — rótulos podem não bater o índice.")

_check_align(E_ws, df_ws, "WS")
_check_align(E_prec, df_prec, "Precursores")
_check_align(E_cp, df_cp, "CP")

# ---------- Idioma (PT/EN) do upload ----------
def detect_lang_pt_en(text: str) -> str:
    """Heurística simples PT/EN: conta stopwords básicas e acentuação."""
    if not text:
        return "pt"
    t = text.lower()
    # listas mínimas (evita precisar de libs extras)
    pt_sw = {"de","do","da","que","para","com","não","na","no","os","as","um","uma","foi","ser","estar","há","houve"}
    en_sw = {"the","of","and","to","in","for","on","with","is","are","was","were","be","been","not","no","an","a"}
    pt_score = sum(t.count(f" {w} ") for w in pt_sw) + sum(1 for ch in t if ch in "áàãâéêíóôõúç")
    en_score = sum(t.count(f" {w} ") for w in en_sw)
    return "pt" if pt_score >= en_score else "en"

# ---------- Descoberta de colunas (versões especializadas) ----------
def _norm_cols_ws(df: pd.DataFrame, lang: str):
    if df is None or df.empty: return None, None
    cols = {c.lower(): c for c in df.columns}
    # Preferências por idioma
    if lang == "pt":
        for key in ("pt","termo","sinal","nome","descricao","descrição","term"):
            if key in cols: term_col = cols[key]; break
        else:
            term_col = list(df.columns)[0]
    else:  # en
        for key in ("en","term","name","signal","description","desc"):
            if key in cols: term_col = cols[key]; break
        else:
            # se não tiver EN, cai para PT
            for key in ("pt","termo","sinal","nome"):
                if key in cols: term_col = cols[key]; break
            else:
                term_col = list(df.columns)[0]
    # id
    for key in ("id","ws_id","codigo","código","code"):
        if key in cols: id_col = cols[key]; break
    else:
        df["_WS_ID_"] = np.arange(len(df)); id_col = "_WS_ID_"
    return term_col, id_col

def _norm_cols_prec(df: pd.DataFrame, lang: str):
    if df is None or df.empty: return None, None, None
    cols = {c.lower(): c for c in df.columns}
    # termo do precursor
    if lang == "pt":
        for key in ("precursor","nome","pt","descricao","descrição","term"):
            if key in cols: term_col = cols[key]; break
        else:
            term_col = list(df.columns)[0]
    else:
        for key in ("precursor_en","en","name","term","description"):
            if key in cols: term_col = cols[key]; break
        else:
            # fallback para PT
            for key in ("precursor","pt","nome"):
                if key in cols: term_col = cols[key]; break
            else:
                term_col = list(df.columns)[0]
    # HTO
    hto_col = None
    for key in ("hto","h-t-o","tipo","class","categoria","category"):
        if key in cols: hto_col = cols[key]; break
    # id
    for key in ("id","prec_id","codigo","código","code"):
        if key in cols: id_col = cols[key]; break
    else:
        df["_PREC_ID_"] = np.arange(len(df)); id_col = "_PREC_ID_"
    return term_col, id_col, hto_col

def _norm_cols_cp(df: pd.DataFrame, lang: str):
    if df is None or df.empty: return None, None, {}
    cols = {c.lower(): c for c in df.columns}
    # termo base (quando não der para compor)
    if lang == "pt":
        for key in ("pt","termo","nome","label","descricao","descrição","term"):
            if key in cols: term_col = cols[key]; break
        else:
            term_col = list(df.columns)[0]
    else:
        for key in ("en","term","name","label","description","desc"):
            if key in cols: term_col = cols[key]; break
        else:
            # fallback pt
            for key in ("pt","termo","nome"):
                if key in cols: term_col = cols[key]; break
            else:
                term_col = list(df.columns)[0]

    # id
    for key in ("id","cp_id","codigo","código","code"):
        if key in cols: id_col = cols[key]; break
    else:
        df["_CP_ID_"] = np.arange(len(df)); id_col = "_CP_ID_"

    # mapeia colunas semânticas para compor o label
    keys = {"dim": None, "fator": None, "subfator": None, "bag": None}
    # Dimensão/Fator/Subfator
    for want, candidates in {
        "dim": ("dimensão","dimensao","dimension","dimension_en","dim"),
        "fator": ("fator","factor","factor_en"),
        "subfator": ("subfator","sub-fator","subfactor","sub-factor","subfactor_en"),
    }.items():
        for c in candidates:
            if c in cols: keys[want] = cols[c]; break
    # Bag de termos (PT) / Bag of terms (EN)
    if lang == "pt":
        for c in ("bag de termos","bag","termos","keywords"):
            if c in cols: keys["bag"] = cols[c]; break
        if keys["bag"] is None:
            # fallback en bag
            for c in ("bag of terms","terms","keywords_en"):
                if c in cols: keys["bag"] = cols[c]; break
    else:
        for c in ("bag of terms","terms","keywords","bag"):
            if c in cols: keys["bag"] = cols[c]; break
        if keys["bag"] is None:
            for c in ("bag de termos","bag","termos"):
                if c in cols: keys["bag"] = cols[c]; break

    return term_col, id_col, keys

# ---------- Rotinas de exibição/match para WS/Prec/CP ----------
def _compose_cp_label(row: pd.Series, cp_keys: dict, cp_term_col: str | None):
    parts = []
    dim = cp_keys.get("dim")
    fat = cp_keys.get("fator")
    sub = cp_keys.get("subfator")
    bag = cp_keys.get("bag")

    if dim and pd.notna(row.get(dim, None)):
        parts.append(str(row[dim]).strip())
    if fat and pd.notna(row.get(fat, None)):
        parts.append(str(row[fat]).strip())
    if sub and pd.notna(row.get(sub, None)):
        parts.append(str(row[sub]).strip())

    label = " / ".join([p for p in parts if p])
    if bag and pd.notna(row.get(bag, None)):
        bt = str(row[bag]).strip()
        if bt:
            label = f"{label} — {bt}" if label else bt
    if not label and cp_term_col:
        t = str(row.get(cp_term_col,"")).strip()
        if t: label = t
    return label

def _search_dict_by_query(E_dict, df_dict, term_col, id_col, tag: str, query: str, topk: int,
                          prec_hto_col=None, cp_keys=None, cp_term_col=None):
    if E_dict is None or df_dict is None or term_col is None or id_col is None:
        return []
    qv = encode_query(query)
    hits = cos_topk(E_dict, qv, k=min(topk, E_dict.shape[0]))
    blocks = []
    for i, s in hits:
        row = df_dict.iloc[i]
        did = row.get(id_col, f"row{i}")
        term = str(row.get(term_col, "")).strip()
        if not term:
            continue

        if tag == "CP":
            label = _compose_cp_label(row, cp_keys or {}, cp_term_col) or term
            blocks.append((s, f"[{tag}/{did}] (sim={s:.3f}) {label}"))
        elif tag == "Prec":
            hto = ""
            try:
                if prec_hto_col and pd.notna(row.get(prec_hto_col, None)):
                    hto = f" [{str(row[prec_hto_col]).strip()}]"
            except Exception:
                pass
            blocks.append((s, f"[{tag}/{did}] (sim={s:.3f}) {term}{hto}"))
        else:
            blocks.append((s, f"[{tag}/{did}] (sim={s:.3f}) {term}"))
    return blocks

def _match_upload_to_dict(E_dict, df_dict, term_col, id_col, tag: str,
                          upld_vec: np.ndarray, topk: int, threshold: float,
                          prec_hto_col=None, cp_keys=None, cp_term_col=None):
    """Compara o embedding médio do upload com todos os termos do dicionário."""
    if E_dict is None or df_dict is None or term_col is None or id_col is None:
        return []
    if upld_vec is None or upld_vec.size == 0:
        return []

    sims = (E_dict @ upld_vec.astype(np.float32))
    order = np.argsort(-sims)[:min(topk, E_dict.shape[0])]
    blocks = []
    for i in order:
        s = float(sims[i])
        if s < float(threshold):
            continue
        row = df_dict.iloc[int(i)]
        did = row.get(id_col, f"row{i}")
        term = str(row.get(term_col, "")).strip()
        if not term:
            continue
        if tag == "CP":
            label = _compose_cp_label(row, cp_keys or {}, cp_term_col) or term
            blocks.append((s, f"[{tag}/{did}] (sim={s:.3f}) {label}"))
        elif tag == "Prec":
            hto = ""
            try:
                if prec_hto_col and pd.notna(row.get(prec_hto_col, None)):
                    hto = f" [{str(row[prec_hto_col]).strip()}]"
            except Exception:
                pass
            blocks.append((s, f"[{tag}/{did}] (sim={s:.3f}) {term}{hto}"))
        else:
            blocks.append((s, f"[{tag}/{did}] (sim={s:.3f}) {term}"))
    # ordenar por similaridade descendente
    blocks.sort(key=lambda x: -x[0])
    return blocks

def _mean_upload_vec() -> np.ndarray | None:
    if st.session_state.upld_emb is None or st.session_state.upld_emb.size == 0:
        return None
    v = st.session_state.upld_emb.mean(axis=0)
    v = v / (np.linalg.norm(v) + 1e-9)
    return v.astype(np.float32)

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

st.sidebar.subheader("WS / Precursores / CP (sobre UPLOAD)")
k_ws   = st.sidebar.slider("Top-K WS",   0, 15, 10, 1)
thr_ws = st.sidebar.slider("Limiar WS (cos)", 0.00, 1.00, 0.25, 0.01)

k_prec   = st.sidebar.slider("Top-K Precursores", 0, 15, 10, 1)
thr_prec = st.sidebar.slider("Limiar Precursores (cos)", 0.00, 1.00, 0.25, 0.01)

k_cp   = st.sidebar.slider("Top-K CP", 0, 15, 10, 1)
thr_cp = st.sidebar.slider("Limiar CP (cos)", 0.00, 1.00, 0.25, 0.01)

st.sidebar.subheader("Upload")
chunk_size  = st.sidebar.slider("Tamanho do chunk", 500, 2000, 1200, 50)
chunk_ovlp  = st.sidebar.slider("Overlap do chunk", 50, 600, 200, 10)
upload_raw_max = st.sidebar.slider("Tamanho máx. de UPLOAD_RAW (chars)", 300, 8000, 2500, 100)

use_catalog = st.sidebar.checkbox("Injetar datasets_context.md", True)

# Detecção de idioma do upload (auto), com override
st.sidebar.subheader("Idioma (dicionários)")
lang_override = st.sidebar.selectbox("Forçar idioma dos dicionários", ["auto", "pt", "en"], index=0)

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

# ---------- Funções de busca ----------
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
st.caption("RAG local 100% embeddings (Sphera / GoSee / Docs / Upload / WS / Precursores / CP).")

# Mostrar histórico
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Digite sua pergunta…")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Recuperação básica
    blocks = search_all(prompt)

    # Opcional: injeta um recorte 'cru' do upload (máx N chars)
    up_raw = get_upload_raw(upload_raw_max)
    if up_raw:
        blocks = [f"[UPLOAD_RAW]\n{up_raw}"] + blocks

    # ---------- WS/Precursores/CP (match vs upload) ----------
    # Detecta idioma do upload (se houver) para escolher colunas corretas nos dicionários
    upld_all_text = "\n".join(st.session_state.upld_texts) if st.session_state.upld_texts else ""
    auto_lang = detect_lang_pt_en(upld_all_text) if upld_all_text else "pt"
    lang = auto_lang if lang_override == "auto" else lang_override

    # Descobrir colunas de cada dicionário conforme idioma
    ws_term_col, ws_id_col = _norm_cols_ws(df_ws, lang) if df_ws is not None else (None, None)
    prec_term_col, prec_id_col, prec_hto_col = _norm_cols_prec(df_prec, lang) if df_prec is not None else (None, None, None)
    cp_term_col, cp_id_col, cp_keys = _norm_cols_cp(df_cp, lang) if df_cp is not None else (None, None, {})

    # Vetor médio do upload
    upld_vec = _mean_upload_vec()

    # WS
    if k_ws > 0 and upld_vec is not None and E_ws is not None and df_ws is not None:
        ws_blocks = _match_upload_to_dict(
            E_ws, df_ws, ws_term_col, ws_id_col, "WS",
            upld_vec, topk=k_ws, threshold=thr_ws
        )
        if ws_blocks:
            blocks.append("### Busca WS vs UPLOAD")
            blocks.extend([b for _, b in ws_blocks])

    # Precursores
    if k_prec > 0 and upld_vec is not None and E_prec is not None and df_prec is not None:
        prec_blocks = _match_upload_to_dict(
            E_prec, df_prec, prec_term_col, prec_id_col, "Prec",
            upld_vec, topk=k_prec, threshold=thr_prec,
            prec_hto_col=prec_hto_col
        )
        if prec_blocks:
            blocks.append("### Busca Precursores vs UPLOAD")
            blocks.extend([b for _, b in prec_blocks])

    # CP
    if k_cp > 0 and upld_vec is not None and E_cp is not None and df_cp is not None:
        cp_blocks = _match_upload_to_dict(
            E_cp, df_cp, cp_term_col, cp_id_col, "CP",
            upld_vec, topk=k_cp, threshold=thr_cp,
            cp_keys=cp_keys, cp_term_col=cp_term_col
        )
        if cp_blocks:
            blocks.append("### Busca CP vs UPLOAD")
            blocks.extend([b for _, b in cp_blocks])

    # ---------- Monta mensagens p/ LLM ----------
    # 1) Mensagem de SISTEMA principal: vem do .md
    msgs = [{"role": "system", "content": st.session_state.system_prompt}]

    # 2) (Opcional) acrescenta o datasets_context.md como outra mensagem de sistema
    if use_catalog and os.path.exists(DATASETS_CONTEXT_FILE):
        try:
            with open(DATASETS_CONTEXT_FILE, "r", encoding="utf-8") as f:
                msgs.append({"role": "system", "content": f.read()})
        except Exception:
            pass

    # 3) Injeta os CONTEXTOS recuperados (RAG) como user-message
    if blocks:
        ctx = "\n\n".join(blocks)
        msgs.append({"role": "user", "content": f"CONTEXTOS (HIST + UPLOAD + WS/Prec/CP):\n{ctx}"})
        msgs.append({"role": "user", "content": f"PERGUNTA: {prompt}"})
    else:
        msgs.append({"role": "user", "content": prompt})

    # 4) Chamada ao modelo
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

        st.write("WS embeddings    :", _ok(E_ws is not None and df_ws is not None))
        if E_ws is not None and df_ws is not None:
            st.write(f" • shape: {E_ws.shape} | linhas WS: {len(df_ws)}")

        st.write("Prec embeddings  :", _ok(E_prec is not None and df_prec is not None))
        if E_prec is not None and df_prec is not None:
            st.write(f" • shape: {E_prec.shape} | linhas Prec: {len(df_prec)}")

        st.write("CP embeddings    :", _ok(E_cp is not None and df_cp is not None))
        if E_cp is not None and df_cp is not None:
            st.write(f" • shape: {E_cp.shape} | linhas CP: {len(df_cp)}")

        st.write("Uploads indexados:", len(st.session_state.upld_texts))
        st.write("Encoder ativo    :", ST_MODEL_NAME)

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
