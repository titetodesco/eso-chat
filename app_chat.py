# app_chat.py — ESO • CHAT (Embeddings-only)  — v2 (WS/Precursores/CP patches)
# - Busca SEMÂNTICA usando embeddings:
#   • Sphera:   data/analytics/sphera_embeddings.npz   + sphera.parquet
#   • GoSee:    data/analytics/gosee_embeddings.npz    + gosee.parquet
#   • History:  data/analytics/history_embeddings.npz  + history_texts.jsonl
# - Uploads: faz chunk + embeddings em tempo real (Sentence-Transformers)
# - Injeta apenas TRECHOS recuperados (não envia vetores ao LLM)
# - Taxonomias (WS/Precursores/CP): usa SOMENTE embeddings pré-gerados dos dicionários
# - Sem TF-IDF, sem ONNX: apenas ST + Torch CPU

import os
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ========== Contexto (system prompt) ==========
CONTEXT_MD_REL_PATH = Path(__file__).parent / "docs" / "contexto_eso_chat.md"
DATASETS_CONTEXT_FILE = "datasets_context.md"  # opcional, em texto

@st.cache_data(show_spinner=False)
def load_file_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"[AVISO] Não consegui ler {p}: {e}\n(Prosseguindo sem esse contexto.)"

def build_system_prompt() -> str:
    preambulo = (
        "Você é o ESO-CHAT (segurança operacional).\n"
        "Responda em PT-BR por padrão.\n"
        "Siga as regras do contexto abaixo e NÃO invente dados fora dos blocos de contexto.\n"
        "Quando usar buscas semânticas, sempre mostre IDs/Fonte e similaridade.\n"
        "Para Weak Signals, Precursores e Taxonomia CP, utilize SOMENTE os itens listados nos blocos [WS_MATCH], [PREC_MATCH] e [CP_MATCH].\n"
    )
    ctx_md = load_file_text(CONTEXT_MD_REL_PATH)
    return preambulo + "\n\n=== CONTEXTO ESO-CHAT (.md) ===\n" + ctx_md

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = build_system_prompt()

# botão para recarregar o .md sem reiniciar o app
st.sidebar.button(
    "Recarregar contexto (.md)",
    on_click=lambda: st.session_state.update(system_prompt=build_system_prompt())
)

# ========== Config básica ==========
st.set_page_config(page_title="ESO • CHAT (Embeddings)", page_icon="💬", layout="wide")

DATA_DIR = "data"
AN_DIR = os.path.join(DATA_DIR, "analytics")
ALT_DIR = "/mnt/data"  # fallback para arquivos enviados no runtime

ST_MODEL_NAME = os.getenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Modelo de chat (Ollama-compatible). Se não tiver chave, tenta mesmo assim.
OLLAMA_HOST  = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gpt-oss:20b"))
OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY"))
HEADERS_JSON = {"Authorization": f"Bearer {OLLAMA_API_KEY}", "Content-Type": "application/json"} if OLLAMA_API_KEY else {"Content-Type": "application/json"}

# ========== Dependências (falha elegante) ==========
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

# ========== Utilidades ==========
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

def batched_cosine_max(E_dict: np.ndarray, E_chunks: np.ndarray) -> np.ndarray:
    """
    Para cada item do dicionário (linha em E_dict), obtém a similaridade máxima vs. todos os chunks do upload (E_chunks).
    Retorna vetor (n_dict,) com os máximos.
    """
    if E_dict is None or E_chunks is None or E_dict.size == 0 or E_chunks.size == 0:
        return np.zeros((0,), dtype=np.float32) if (E_dict is None or E_dict.size == 0) else np.zeros((E_dict.shape[0],), dtype=np.float32)
    # ambos já normalizados
    sims = E_dict @ E_chunks.T  # (n_dict, n_chunks)
    return sims.max(axis=1)     # (n_dict,)

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

def load_npz_text_id(path: str):
    """
    Tenta carregar 'texts' e 'ids' de um .npz (quando presentes).
    Retorna (ids_list|None, texts_list|None).
    """
    if not os.path.exists(path):
        return None, None
    ids, texts = None, None
    try:
        with np.load(path, allow_pickle=True) as z:
            for key in ("texts", "labels", "label_texts"):
                if key in z:
                    t = z[key]
                    texts = list(t.tolist()) if hasattr(t, "tolist") else list(t)
                    break
            for key in ("ids", "id", "indexes"):
                if key in z:
                    i = z[key]
                    ids = list(i.tolist()) if hasattr(i, "tolist") else list(i)
                    break
    except Exception:
        pass
    return ids, texts

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

def detect_lang_pt(text: str) -> bool:
    """Heurística simples: se muitas stop-words PT aparecerem, assume PT."""
    if not text:
        return True
    pt_sw = {"que", "não", "para", "com", "uma", "como", "foi", "estava", "de", "do", "da", "no", "na", "os", "as", "um", "uma"}
    en_sw = {"the", "and", "with", "for", "was", "were", "is", "are", "of", "to", "in", "on"}
    t = text.lower()
    pt_hits = sum(1 for w in pt_sw if f" {w} " in f" {t} ")
    en_hits = sum(1 for w in en_sw if f" {w} " in f" {t} ")
    return pt_hits >= en_hits

# ========== Estado ==========
if "chat" not in st.session_state:
    st.session_state.chat = []

# Uploads (embeddings de sessão)
if "upld_texts" not in st.session_state:
    st.session_state.upld_texts = []      # lista[str]
if "upld_meta" not in st.session_state:
    st.session_state.upld_meta = []       # lista[dict]
if "upld_emb" not in st.session_state:
    st.session_state.upld_emb = None      # np.ndarray (n,d) L2
if "upld_lang_pt" not in st.session_state:
    st.session_state.upld_lang_pt = True  # assume PT até ler upload

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

# ========== Carregamento dos catálogos (embeddings + texto base) ==========
def path_first_existing(*candidates: str) -> str | None:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

# Sphera/GoSee/History (como antes)
SPH_EMB_PATH = path_first_existing(os.path.join(AN_DIR, "sphera_embeddings.npz"))
GOS_EMB_PATH = path_first_existing(os.path.join(AN_DIR, "gosee_embeddings.npz"))
HIS_EMB_PATH = path_first_existing(os.path.join(AN_DIR, "history_embeddings.npz"))

SPH_PQ_PATH  = path_first_existing(os.path.join(AN_DIR, "sphera.parquet"))
GOS_PQ_PATH  = path_first_existing(os.path.join(AN_DIR, "gosee.parquet"))
HIS_JSONL    = path_first_existing(os.path.join(AN_DIR, "history_texts.jsonl"))

E_sph = load_npz_embeddings(SPH_EMB_PATH) if SPH_EMB_PATH else None
E_gos = load_npz_embeddings(GOS_EMB_PATH) if GOS_EMB_PATH else None
E_his = load_npz_embeddings(HIS_EMB_PATH) if HIS_EMB_PATH else None

df_sph = None
df_gos = None
rows_his = []

if SPH_PQ_PATH:
    try:
        df_sph = pd.read_parquet(SPH_PQ_PATH)
    except Exception as e:
        st.warning(f"Falha ao ler {SPH_PQ_PATH}: {e}")
if GOS_PQ_PATH:
    try:
        df_gos = pd.read_parquet(GOS_PQ_PATH)
    except Exception as e:
        st.warning(f"Falha ao ler {GOS_PQ_PATH}: {e}")
if HIS_JSONL:
    try:
        with open(HIS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                rows_his.append(json.loads(line))
    except Exception as e:
        st.warning(f"Falha ao ler {HIS_JSONL}: {e}")

# ========== WS/Precursores/CP — embeddings + labels ==========
def load_labels_any(*cands: str) -> pd.DataFrame | None:
    p = path_first_existing(*cands)
    if not p:
        return None
    try:
        if p.endswith(".parquet"):
            return pd.read_parquet(p)
        if p.endswith(".csv"):
            return pd.read_csv(p)
        if p.endswith(".jsonl") or p.endswith(".json"):
            rows = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    rows.append(json.loads(line))
            return pd.DataFrame(rows)
        if p.endswith(".xlsx") or p.endswith(".xls"):
            xls = pd.ExcelFile(p)
            # concat todas as sheets
            frames = [xls.parse(s) for s in xls.sheet_names]
            return pd.concat(frames, ignore_index=True)
    except Exception as e:
        st.warning(f"Falha ao ler labels {p}: {e}")
        return None
    return None

# caminhos principais (repo) e alternativos (/mnt/data)
WS_NPZ = path_first_existing(os.path.join(AN_DIR, "ws_embeddings.npz"),
                             os.path.join(ALT_DIR, "ws_embeddings.npz"))
PREC_NPZ = path_first_existing(os.path.join(AN_DIR, "prec_embeddings.npz"),
                               os.path.join(ALT_DIR, "prec_embeddings.npz"),
                               os.path.join(ALT_DIR, "prec_vectors.npz"))
CP_NPZ = path_first_existing(os.path.join(AN_DIR, "cp_embeddings.npz"),
                             os.path.join(ALT_DIR, "cp_embeddings.npz"))

# labels/ids possíveis
WS_LABELS = load_labels_any(
    os.path.join(AN_DIR, "ws_embeddings.parquet"),
    os.path.join(AN_DIR, "ws_labels.parquet"),
    os.path.join(ALT_DIR, "ws_labels.parquet"),
    os.path.join(ALT_DIR, "ws_labels.jsonl"),
    os.path.join(ALT_DIR, "ws_labels.csv")
)

PREC_LABELS = load_labels_any(
    os.path.join(AN_DIR, "precursors.parquet"),
    os.path.join(AN_DIR, "precursors.csv"),
    os.path.join(ALT_DIR, "precursors.parquet"),
    os.path.join(ALT_DIR, "precursors.csv"),
    os.path.join(ALT_DIR, "prec_labels.jsonl"),
    os.path.join(DATA_DIR, "xlsx", "precursores_expandido.xlsx"),
    os.path.join(ALT_DIR, "precursores_expandido.xlsx"),
)

CP_LABELS = load_labels_any(
    os.path.join(AN_DIR, "cp_labels.parquet"),
    os.path.join(ALT_DIR, "cp_labels.parquet"),
    os.path.join(DATA_DIR, "xlsx", "TaxonomiaCP_Por.xlsx"),
    os.path.join(AN_DIR, "cp_embeddings.parquet"),
)

# carregar embeddings
E_ws   = load_npz_embeddings(WS_NPZ)   if WS_NPZ   else None
E_prec = load_npz_embeddings(PREC_NPZ) if PREC_NPZ else None
E_cp   = load_npz_embeddings(CP_NPZ)   if CP_NPZ   else None

# tentar obter ids/texts dos .npz
ws_ids_npz, ws_texts_npz     = load_npz_text_id(WS_NPZ)   if WS_NPZ   else (None, None)
prec_ids_npz, prec_texts_npz = load_npz_text_id(PREC_NPZ) if PREC_NPZ else (None, None)
cp_ids_npz, cp_texts_npz     = load_npz_text_id(CP_NPZ)   if CP_NPZ   else (None, None)

def extract_label_cols(df: pd.DataFrame, pt_cols: list[str], en_cols: list[str]):
    pt = None
    en = None
    for c in pt_cols:
        if c in df.columns:
            pt = df[c].astype(str).tolist()
            break
    for c in en_cols:
        if c in df.columns:
            en = df[c].astype(str).tolist()
            break
    # fallback: se não tiver PT/EN, use uma coluna "text"/"label"/"term" se existir
    if pt is None and en is None:
        for c in ("text", "label", "term", "bag", "bag_pt", "bag_en"):
            if c in df.columns:
                pt = df[c].astype(str).tolist()
                en = pt
                break
    return pt, en

# tentar extrair PT/EN das labels
ws_pt, ws_en = None, None
if WS_LABELS is not None:
    ws_pt, ws_en = extract_label_cols(WS_LABELS, pt_cols=["PT", "pt"], en_cols=["EN", "en"])

prec_pt, prec_en = None, None
if PREC_LABELS is not None:
    prec_pt, prec_en = extract_label_cols(PREC_LABELS, pt_cols=["Precursor_PT", "PT", "prec_pt", "Precursor (PT)"],
                                          en_cols=["Precursor_EN", "EN", "prec_en", "Precursor (EN)"])

cp_pt, cp_en = None, None
if CP_LABELS is not None:
    cp_pt, cp_en = extract_label_cols(CP_LABELS,
                                      pt_cols=["Bag de termos", "Bag_de_termos", "bag_pt", "bag_terms_pt"],
                                      en_cols=["Bag of terms", "bag_en", "bag_terms_en"])

def safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0

# sanidade: alinhar labels aos embeddings por tamanho; se não bater, mostrar aviso e usar npz texts se existir
def align_labels(E: np.ndarray, labels_pt, labels_en, npz_texts):
    nE = E.shape[0] if E is not None else 0
    n_pt = safe_len(labels_pt)
    n_en = safe_len(labels_en)
    n_npz = safe_len(npz_texts)
    notes = []

    labels_final_pt, labels_final_en = None, None

    if nE == 0:
        return None, None, ["Sem embeddings."]

    # caso ideal: alguma lista bate nE
    if n_pt == nE:
        labels_final_pt = labels_pt
    if n_en == nE:
        labels_final_en = labels_en

    # se nenhuma bate, mas npz_texts bate, usa como PT e EN
    if labels_final_pt is None and labels_final_en is None and n_npz == nE:
        labels_final_pt = npz_texts
        labels_final_en = npz_texts
        notes.append("Labels ausentes/desalinhados — usando 'texts' do .npz como rótulos.")

    # se só uma bate, duplica na outra
    if labels_final_pt is None and labels_final_en is not None:
        labels_final_pt = labels_final_en
        notes.append("PT rótulos ausentes — usando EN como PT.")
    if labels_final_en is None and labels_final_pt is not None:
        labels_final_en = labels_final_pt
        notes.append("EN rótulos ausentes — usando PT como EN.")

    # nenhum bate: corta/expande de forma segura se existir alguma lista > 0
    if labels_final_pt is None and labels_final_en is None:
        any_labels = None
        for cand in (labels_pt, labels_en, npz_texts):
            if safe_len(cand) > 0:
                any_labels = cand
                break
        if any_labels is not None:
            if safe_len(any_labels) >= nE:
                labels_final_pt = any_labels[:nE]
                labels_final_en = any_labels[:nE]
            else:
                # expande com placeholders
                fill = ["—"] * (nE - safe_len(any_labels))
                tmp = list(any_labels) + fill
                labels_final_pt = tmp
                labels_final_en = tmp
            notes.append("Rótulo e embeddings com contagem diferente — ajustado por corte/preenchimento.")
        else:
            labels_final_pt = ["—"] * nE
            labels_final_en = ["—"] * nE
            notes.append("Sem rótulos — usando placeholders.")

    return labels_final_pt, labels_final_en, notes

ws_labels_pt, ws_labels_en, ws_notes     = align_labels(E_ws,   ws_pt,   ws_en,   ws_texts_npz)
prec_labels_pt, prec_labels_en, prec_notes = align_labels(E_prec, prec_pt, prec_en, prec_texts_npz)
cp_labels_pt, cp_labels_en, cp_notes     = align_labels(E_cp,   cp_pt,   cp_en,   cp_texts_npz)

# ========== Sidebar ==========
st.sidebar.header("Configurações • Modelo")
st.sidebar.write("Host:", OLLAMA_HOST)
st.sidebar.write("Modelo:", OLLAMA_MODEL)
if not OLLAMA_API_KEY:
    st.sidebar.info("Sem OLLAMA_API_KEY — ok para ambientes locais se o host não exigir auth.")

st.sidebar.header("Recuperação (Embeddings)")
k_sph = st.sidebar.slider("Top-K Sphera", 0, 10, 5, 1)
k_gos = st.sidebar.slider("Top-K GoSee",  0, 10, 5, 1)
k_his = st.sidebar.slider("Top-K Docs",   0, 10, 3, 1)
k_upl = st.sidebar.slider("Top-K Upload", 0, 10, 5, 1)

st.sidebar.header("Upload")
chunk_size     = st.sidebar.slider("Tamanho do chunk", 500, 2000, 1200, 50)
chunk_ovlp     = st.sidebar.slider("Overlap do chunk", 50, 600, 200, 10)
upload_raw_max = st.sidebar.slider("Tamanho máx. de UPLOAD_RAW (chars)", 300, 8000, 2500, 100)

use_catalog = st.sidebar.checkbox("Injetar datasets_context.md", True)

st.sidebar.header("Taxonomias (WS/Prec/CP)")
ws_threshold   = st.sidebar.slider("Limiar WS (cos)",   0.0, 1.0, 0.25, 0.01)
prec_threshold = st.sidebar.slider("Limiar Prec (cos)", 0.0, 1.0, 0.25, 0.01)
cp_threshold   = st.sidebar.slider("Limiar CP (cos)",   0.0, 1.0, 0.25, 0.01)
max_per_dict   = st.sidebar.slider("Máx. itens por grupo (WS/Prec/CP)", 1, 20, 10, 1)

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

# ========== Indexação de Uploads (embeddings em sessão) ==========
def update_upload_embeddings(files):
    new_texts, new_meta = [], []
    for uf in files:
        try:
            raw = read_any(uf)
            parts = chunk_text(raw, max_chars=chunk_size, overlap=chunk_ovlp)
            for i, p in enumerate(parts):
                new_texts.append(p)
                new_meta.append({"file": uf.name, "chunk_id": i})
        except Exception as e:
            st.warning(f"Falha ao processar {uf.name}: {e}")
    if new_texts:
        # lingua
        st.session_state.upld_lang_pt = detect_lang_pt(" ".join(new_texts[:3]))
        M_new = encode_texts(new_texts, batch_size=64)
        if st.session_state.upld_emb is None:
            st.session_state.upld_emb = M_new
        else:
            st.session_state.upld_emb = np.vstack([st.session_state.upld_emb, M_new])
        st.session_state.upld_texts.extend(new_texts)
        st.session_state.upld_meta.extend(new_meta)
        st.success(f"Upload indexado: {len(new_texts)} chunks.")

if uploaded_files:
    with st.spinner("Lendo e embutindo uploads (embeddings)…"):
        update_upload_embeddings(uploaded_files)

# ========== Recuperação padrão ==========
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

# ========== Match Upload → Dicionários (WS/Prec/CP) ==========
def match_upload_to_dict(E_dict: np.ndarray, labels_pt: list[str], labels_en: list[str], threshold: float, topk: int):
    """
    Retorna lista de dicts: {idx, sim, label_pt, label_en}
    Calcula sim máxima de cada termo do dicionário vs. TODOS os chunks do upload.
    Filtra por limiar e ordena por similaridade desc.
    """
    if E_dict is None or st.session_state.upld_emb is None:
        return []
    sims_max = batched_cosine_max(E_dict, st.session_state.upld_emb)  # (n_dict,)
    idxs = np.where(sims_max >= threshold)[0]
    items = []
    lang_pt = st.session_state.upld_lang_pt
    for i in idxs:
        lp = labels_pt[i] if labels_pt and i < len(labels_pt) else "—"
        le = labels_en[i] if labels_en and i < len(labels_en) else lp
        items.append({
            "idx": int(i),
            "sim": float(sims_max[i]),
            "label_pt": lp,
            "label_en": le,
            "label": lp if lang_pt else le
        })
    items.sort(key=lambda x: -x["sim"])
    return items[:topk]

def build_taxonomy_blocks():
    """Monta blocos [WS_MATCH], [PREC_MATCH], [CP_MATCH] a partir do upload + dicionários."""
    blocks = []
    ws_items   = match_upload_to_dict(E_ws,   ws_labels_pt,   ws_labels_en,   ws_threshold,   max_per_dict)
    prec_items = match_upload_to_dict(E_prec, prec_labels_pt, prec_labels_en, prec_threshold, max_per_dict)
    cp_items   = match_upload_to_dict(E_cp,   cp_labels_pt,   cp_labels_en,   cp_threshold,   max_per_dict)

    def fmt(group_name, items, tag):
        if not items:
            return f"[{tag}] (nenhum ≥ limiar)"
        lines = [f"[{tag}] {group_name} (top {len(items)} — limiar={ws_threshold if tag=='WS_MATCH' else prec_threshold if tag=='PREC_MATCH' else cp_threshold:.2f})"]
        for r in items:
            lines.append(f"- {tag.split('_')[0]}/{r['idx']} | sim={r['sim']:.3f} | {r['label']}")
        return "\n".join(lines)

    blocks.append(fmt("Weak Signals", ws_items, "WS_MATCH"))
    blocks.append(fmt("Precursores (HTO incluído no rótulo, se houver)", prec_items, "PREC_MATCH"))
    blocks.append(fmt("Taxonomia CP (bag of terms)", cp_items, "CP_MATCH"))
    return "\n\n".join(blocks), ws_items, prec_items, cp_items

# ========== UI ==========
st.title("ESO • CHAT — HIST + UPLD (Embeddings preferencial)")
st.caption("RAG local 100% embeddings (Sphera / GoSee / Docs / Upload) + Dicionários (WS/Precursores/CP).")

# Histórico
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Digite sua pergunta…")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Recuperação clássica
    blocks = search_all(prompt)

    # Opcional: injeta recorte do upload
    up_raw = get_upload_raw(upload_raw_max)
    if up_raw:
        blocks = [f"[UPLOAD_RAW]\n{up_raw}"] + blocks

    # Taxonomias (WS/Prec/CP) a partir do upload
    taxo_block, ws_found, prec_found, cp_found = build_taxonomy_blocks()

    # Monta mensagens para o LLM
    messages = [{"role": "system", "content": st.session_state.system_prompt}]
    if use_catalog and os.path.exists(DATASETS_CONTEXT_FILE):
        try:
            with open(DATASETS_CONTEXT_FILE, "r", encoding="utf-8") as f:
                messages.append({"role": "system", "content": f.read()})
        except Exception:
            pass

    # Blocos de contexto (RAG clássico)
    if blocks:
        ctx = "\n\n".join(blocks)
        messages.append({"role": "user", "content": f"CONTEXTOS (HIST + UPLOAD):\n{ctx}"})

    # Bloco Taxonomias (somente os itens achados nos dicionários)
    messages.append({"role": "user", "content": taxo_block})

    # Pergunta do usuário
    messages.append({"role": "user", "content": f"PERGUNTA: {prompt}\n\n"
                                                f"IMPORTANTE: Ao citar Weak Signals, Precursores e Fatores CP, "
                                                f"USE SOMENTE os itens listados nos blocos [WS_MATCH], [PREC_MATCH] e [CP_MATCH]. "
                                                f"Não invente termos fora desses blocos."})

    # Chamada ao modelo
    with st.chat_message("assistant"):
        with st.spinner("Consultando o modelo…"):
            try:
                resp = ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2, stream=False)
                content = resp.get("message", {}).get("content", "").strip() or json.dumps(resp)[:1200]
            except Exception as e:
                content = f"Falha ao consultar o modelo: {e}"
            st.markdown(content)
    st.session_state.chat.append({"role": "assistant", "content": content})

# ========== Painel / Diagnóstico ==========
debug = st.sidebar.checkbox("Mostrar painel de diagnóstico", False)

if debug:
    with st.expander("📦 Status dos índices (bases)", expanded=False):
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
        st.write("Idioma upload    :", "PT" if st.session_state.upld_lang_pt else "EN")

    with st.expander("🔎 WS/Precursores/CP — embeddings & labels", expanded=True):
        def show_group(name, E, ids_npz, texts_npz, labels_pt, labels_en, notes):
            nE = E.shape[0] if E is not None else 0
            n_ids = len(ids_npz) if ids_npz is not None else 0
            n_txt = len(texts_npz) if texts_npz is not None else 0
            n_pt  = len(labels_pt) if labels_pt is not None else 0
            n_en  = len(labels_en) if labels_en is not None else 0
            st.markdown(f"**{name}**: E={nE}, ids={n_ids}, npz_texts={n_txt}, PT={n_pt}, EN={n_en}")
            if notes:
                for nt in notes:
                    st.caption(f"ℹ️ {nt}")
            # exemplo
            if nE > 0:
                try:
                    ex = labels_pt[0] if labels_pt else (texts_npz[0] if texts_npz else "—")
                    st.write("Exemplo:", ex)
                except Exception:
                    pass

        show_group("WS",   E_ws,   ws_ids_npz,   ws_texts_npz,   ws_labels_pt,   ws_labels_en,   ws_notes)
        show_group("Prec", E_prec, prec_ids_npz, prec_texts_npz, prec_labels_pt, prec_labels_en, prec_notes)
        show_group("CP",   E_cp,   cp_ids_npz,   cp_texts_npz,   cp_labels_pt,   cp_labels_en,   cp_notes)

    with st.expander("🧪 Match (upload → dicionários)", expanded=False):
        tb, ws_found, prec_found, cp_found = build_taxonomy_blocks()
        st.text(tb)
