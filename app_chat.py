# app_chat.py — ESO • CHAT (Embeddings-only) — versão com patches PT/EN e “Somente Sphera”
# - Busca SEMÂNTICA usando embeddings:
#   • Sphera:   data/analytics/sphera_embeddings.npz + sphera.parquet
#   • GoSee:    data/analytics/gosee_embeddings.npz  + gosee.parquet
#   • History:  data/analytics/history_embeddings.npz + history_texts.jsonl
# - Dicionários (seleção automática de idioma):
#   • WS:   ws_embeddings_pt/en.(npz|parquet|jsonl)
#   • Prec: prec_embeddings_pt/en.(npz|parquet|jsonl)
#   • CP:   cp_embeddings.npz + cp_labels.(parquet|jsonl)
# - Uploads: faz chunk + embeddings em tempo real (Sentence-Transformers)
# - “Somente Sphera”: cálculo e filtro no app (threshold e últimos N anos), sem misturar outras fontes
# - Injeta apenas TRECHOS recuperados (quando não estiver em “Somente Sphera”)

import os
import io
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta

# ---------- Contexto (system prompt) ----------
CONTEXT_MD_REL_PATH = Path(__file__).parent / "docs" / "contexto_eso_chat.md"
DATASETS_CONTEXT_FILE = "datasets_context.md"  # opcional

@st.cache_data(show_spinner=False)
def load_file_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"[AVISO] Não consegui ler {p}: {e}\n(Prosseguindo sem esse contexto.)"

def build_system_prompt() -> str:
    preambulo = (
        "Você é o ESO-CHAT (segurança operacional).\n"
        "Siga estritamente as regras e convenções do contexto abaixo.\n"
        "Responda em PT-BR por padrão.\n"
        "Quando usar buscas semânticas, sempre mostre IDs/Fonte e similaridade.\n"
        "Não invente dados fora dos contextos fornecidos.\n"
    )
    ctx_md = load_file_text(CONTEXT_MD_REL_PATH)
    return preambulo + "\n\n=== CONTEXTO ESO-CHAT (.md) ===\n" + ctx_md

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = build_system_prompt()

if st.sidebar.button("Recarregar contexto (.md)"):
    st.session_state.system_prompt = build_system_prompt()
    st.sidebar.success("Contexto recarregado.")

# ---------- Config básica ----------
st.set_page_config(page_title="ESO • CHAT (Embeddings)", page_icon="💬", layout="wide")

DATA_DIR = "data"
AN_DIR = os.path.join(DATA_DIR, "analytics")
ALT_DIR = "/mnt/data"  # fallback em ambientes gerenciados
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

def cos_topk(E_db: np.ndarray, q: np.ndarray, k: int) -> list[tuple[int, float]]:
    if E_db is None or E_db.size == 0:
        return []
    q = q.astype(np.float32, copy=False)
    q = q / (np.linalg.norm(q) + 1e-9)
    sims = E_db @ q
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

# --- Heurística de idioma (PT/EN) ---
def guess_lang(text: str) -> str:
    if not text:
        return "pt"
    t = text.lower()
    pt_hits = sum(kw in t for kw in [
        " guindaste", " cabo ", " limit switch", "lança", "convés",
        "devido", "foi decidido", "observado", "pendurado", "equipamento",
        "procedimento", "manutenção", "investigação", "faina"
    ])
    en_hits = sum(kw in t for kw in [
        " crane", " wire", " limit switch", "boom", "deck",
        "due to", "decided", "observed", "hanging", "equipment",
        "procedure", "maintenance", "investigation", "sling"
    ])
    return "pt" if pt_hits >= en_hits else "en"

# ---------- Estado ----------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "upld_texts" not in st.session_state:
    st.session_state.upld_texts = []
if "upld_meta" not in st.session_state:
    st.session_state.upld_meta = []
if "upld_emb" not in st.session_state:
    st.session_state.upld_emb = None

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

# ---------- Carregamento dos catálogos ----------
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

# --- Dicionários PT/EN (caminhos) ---
WS_PT_NPZ = os.path.join(AN_DIR, "ws_embeddings_pt.npz")
WS_EN_NPZ = os.path.join(AN_DIR, "ws_embeddings_en.npz")
WS_PT_LBL_PARQ = os.path.join(AN_DIR, "ws_embeddings_pt.parquet")
WS_EN_LBL_PARQ = os.path.join(AN_DIR, "ws_embeddings_en.parquet")

PREC_PT_NPZ = os.path.join(AN_DIR, "prec_embeddings_pt.npz")
PREC_EN_NPZ = os.path.join(AN_DIR, "prec_embeddings_en.npz")
PREC_PT_LBL_PARQ = os.path.join(AN_DIR, "prec_embeddings_pt.parquet")
PREC_EN_LBL_PARQ = os.path.join(AN_DIR, "prec_embeddings_en.parquet")

CP_NPZ = os.path.join(AN_DIR, "cp_embeddings.npz")
CP_LBL_PARQ = os.path.join(AN_DIR, "cp_labels.parquet")

def load_dict_bank(npz_path: str, labels_parquet: str):
    E = load_npz_embeddings(npz_path)
    labels = None
    if os.path.exists(labels_parquet):
        try:
            labels = pd.read_parquet(labels_parquet)
        except Exception:
            labels = None
    if E is None or labels is None or len(labels) != E.shape[0]:
        return None, None
    return E, labels

def select_ws_bank(lang: str):
    if lang == "en" and os.path.exists(WS_EN_NPZ):
        return load_dict_bank(WS_EN_NPZ, WS_EN_LBL_PARQ)
    return load_dict_bank(WS_PT_NPZ, WS_PT_LBL_PARQ)

def select_prec_bank(lang: str):
    if lang == "en" and os.path.exists(PREC_EN_NPZ):
        return load_dict_bank(PREC_EN_NPZ, PREC_EN_LBL_PARQ)
    return load_dict_bank(PREC_PT_NPZ, PREC_PT_LBL_PARQ)

def select_cp_bank():
    return load_dict_bank(CP_NPZ, CP_LBL_PARQ)

# ---------- Sidebar ----------
st.sidebar.header("Configurações")
with st.sidebar.expander("Modelo de Resposta", expanded=False):
    st.write("Host:", OLLAMA_HOST)
    st.write("Modelo:", OLLAMA_MODEL)
    if not OLLAMA_API_KEY:
        st.info("Sem OLLAMA_API_KEY — ok para ambientes locais se o host não exigir auth.")

st.sidebar.subheader("Recuperação (Embeddings padrão)")
k_sph = st.sidebar.slider("Top-K Sphera", 0, 10, 5, 1)
k_gos = st.sidebar.slider("Top-K GoSee",  0, 10, 5, 1)
k_his = st.sidebar.slider("Top-K Docs",   0, 10, 3, 1)
k_upl = st.sidebar.slider("Top-K Upload", 0, 10, 5, 1)

st.sidebar.subheader("Upload")
chunk_size  = st.sidebar.slider("Tamanho do chunk", 500, 2000, 1200, 50)
chunk_ovlp  = st.sidebar.slider("Overlap do chunk", 50, 600, 200, 10)
upload_raw_max = st.sidebar.slider("Tamanho máx. de UPLOAD_RAW (chars)", 300, 8000, 2500, 100)

st.sidebar.subheader("Regras de Escopo")
only_sphera = st.sidebar.checkbox("Somente Sphera (ignorar GoSee/Docs/Upload)", True)
apply_time_filter = st.sidebar.checkbox("Sphera: filtrar últimos N anos", True)
years_back = st.sidebar.slider("N (anos)", 1, 10, 3, 1)

st.sidebar.subheader("Limiares de Similaridade (0–1)")
thr_sphera = st.sidebar.slider("Limiar Sphera (Description)", 0.0, 1.0, 0.25, 0.01)
thr_ws     = st.sidebar.slider("Limiar WS", 0.0, 1.0, 0.25, 0.01)
thr_prec   = st.sidebar.slider("Limiar Precursores", 0.0, 1.0, 0.25, 0.01)
thr_cp     = st.sidebar.slider("Limiar CP", 0.0, 1.0, 0.25, 0.01)

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
        st.session_state.pop("last_upload_digest", None)
with c2:
    if st.button("Limpar chat", use_container_width=True):
        st.session_state.chat = []

# ---------- Indexação de Uploads ----------
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

# ---------- Funções de busca ----------
def filter_sphera_by_date(df: pd.DataFrame, years: int) -> pd.DataFrame:
    if df is None or "EVENT_DATE" not in df.columns:
        return df
    try:
        d = df.copy()
        d["EVENT_DATE"] = pd.to_datetime(d["EVENT_DATE"], errors="coerce")
        cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=365*years))
        return d[d["EVENT_DATE"] >= cutoff]
    except Exception:
        return df

def sphera_similar_to_text(query_text: str, min_sim: float, years: int | None = None, topk: int = 50):
    """Retorna [(event_id, sim, row)] com sim >= min_sim, usando apenas Sphera/Description."""
    if df_sph is None or E_sph is None or E_sph.size == 0:
        return []
    base = df_sph
    if years is not None:
        base = filter_sphera_by_date(base, years)

    text_col = "Description" if "Description" in base.columns else base.columns[0]
    id_col = "Event ID" if "Event ID" in base.columns else ("EVENT_NUMBER" if "EVENT_NUMBER" in base.columns else None)

    # alinhar E_sph com o índice filtrado
    try:
        base_idx = base.index.to_list()
        E_view = E_sph[base_idx, :]
    except Exception:
        E_view = E_sph
        base = df_sph

    qv = encode_query(query_text)
    sims = E_view @ qv
    idx = np.argsort(-sims)
    out = []
    for i in idx[:max(topk, len(idx))]:
        s = float(sims[i])
        if s < min_sim:
            break
        row = base.iloc[i]
        evid = row.get(id_col, f"row{i}") if id_col else f"row{i}"
        out.append((evid, s, row))
    return out

def match_from_dicts(query_text: str, lang: str, thr_ws: float, thr_prec: float, thr_cp: float, topk: int = 20):
    out = {"ws": [], "prec": [], "cp": []}

    # WS
    E_ws, L_ws = select_ws_bank(lang)
    if E_ws is not None:
        qv = encode_query(query_text)
        sims = E_ws @ qv
        idx = np.argsort(-sims)
        for i in idx[:min(topk, len(idx))]:
            s = float(sims[i])
            if s < thr_ws:
                break
            label = str(L_ws.iloc[i].get("label", L_ws.iloc[i].get("text", f"WS_{i}")))
            out["ws"].append((label, s))

    # Prec
    E_pr, L_pr = select_prec_bank(lang)
    if E_pr is not None:
        qv = encode_query(query_text)
        sims = E_pr @ qv
        idx = np.argsort(-sims)
        for i in idx[:min(topk, len(idx))]:
            s = float(sims[i])
            if s < thr_prec:
                break
            label = str(L_pr.iloc[i].get("label", L_pr.iloc[i].get("text", f"Prec_{i}")))
            out["prec"].append((label, s))

    # CP (único banco)
    E_cp, L_cp = select_cp_bank()
    if E_cp is not None:
        qv = encode_query(query_text)
        sims = E_cp @ qv
        idx = np.argsort(-sims)
        for i in idx[:min(topk, len(idx))]:
            s = float(sims[i])
            if s < thr_cp:
                break
            label = str(L_cp.iloc[i].get("label", L_cp.iloc[i].get("text", f"CP_{i}")))
            out["cp"].append((label, s))

    return out

def get_upload_raw(max_chars: int) -> str:
    if not st.session_state.upld_texts:
        return ""
    buf, total = [], 0
    for t in st.session_state.upld_texts[:3]:
        if total >= max_chars:
            break
        t = t[: max_chars - total]
        buf.append(t)
        total += len(t)
    return "\n\n".join(buf).strip()

def search_all(query: str) -> list[str]:
    """Embute a query e busca nos 4 conjuntos (Sphera/GoSee/Docs/Upload). Retorna blocos formatados."""
    qv = encode_query(query)
    blocks: list[tuple[float, str]] = []

    # Sphera (apenas quando NÃO está em 'Somente Sphera', pois lá é calculado localmente)
    if not only_sphera:
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
    if not only_sphera:
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
    if not only_sphera:
        if k_his > 0 and E_his is not None and rows_his:
            hits = cos_topk(E_his, qv, k=k_his)
            for i, s in hits:
                r = rows_his[i]
                src = f"Docs/{r.get('source','?')}/{r.get('chunk_id', 0)}"
                snippet = str(r.get("text", ""))[:800]
                blocks.append((s, f"[{src}] (sim={s:.3f})\n{snippet}"))

    # Upload
    if not only_sphera:
        if k_upl > 0 and st.session_state.upld_emb is not None and len(st.session_state.upld_texts) == st.session_state.upld_emb.shape[0]:
            hits = cos_topk(st.session_state.upld_emb, qv, k=k_upl)
            for i, s in hits:
                meta = st.session_state.upld_meta[i]
                snippet = st.session_state.upld_texts[i][:800]
                blocks.append((s, f"[UPLOAD {meta['file']} / {meta['chunk_id']}] (sim={s:.3f})\n{snippet}"))

    blocks.sort(key=lambda x: -x[0])
    return [b for _, b in blocks]

# ---------- UI ----------
st.title("ESO • CHAT — HIST + UPLD (Embeddings preferencial) + Dicionários PT/EN")
st.caption("RAG local (Sphera / GoSee / Docs / Upload) + WS/Precursores/CP com seleção automática de idioma.")

# Mostrar histórico
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Digite sua pergunta…")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Opcional: injeta um recorte 'cru' do upload (máx N chars)
    up_raw = get_upload_raw(upload_raw_max)
    lang = guess_lang((prompt or "") + "\n" + (up_raw or ""))

    if only_sphera:
        # -------- Fluxo "Somente Sphera": cálculo no app --------
        query_text = up_raw if up_raw else prompt
        years = years_back if apply_time_filter else None

        # 1) Eventos Sphera semelhantes (threshold aplicado AQUI)
        hits = sphera_similar_to_text(query_text, thr_sphera, years=years, topk=200)
        if hits:
            md = ["**Eventos do Sphera (calculado no app, limiar aplicado)**\n",
                  "| Event Id | Similaridade | Description |",
                  "|---:|---:|---|"]
            for evid, s, row in hits:
                desc = str(row.get("Description", ""))[:4000].replace("\n", " ")
                md.append(f"| {evid} | {s:.3f} | {desc} |")
            tbl = "\n".join(md)
            with st.chat_message("assistant"):
                st.markdown(tbl)
            st.session_state.chat.append({"role": "assistant", "content": tbl})
        else:
            msg = "Nenhum evento do Sphera encontrado com similaridade ≥ " + str(thr_sphera)
            with st.chat_message("assistant"):
                st.markdown(msg)
            st.session_state.chat.append({"role": "assistant", "content": msg})

        # 2) Dicionários (WS / Precursores / CP) — estritamente dos bancos
        dict_matches = match_from_dicts(query_text, lang, thr_ws, thr_prec, thr_cp, topk=50)
        md2 = []
        if dict_matches["ws"]:
            md2.append("\n**WS (≥ limiar, calculado no app)**")
            md2.append("| Rank | Termo | Similaridade |")
            md2.append("|---:|---|---:|")
            for r, (label, s) in enumerate(dict_matches["ws"], 1):
                md2.append(f"| {r} | {label} | {s:.3f} |")
        if dict_matches["prec"]:
            md2.append("\n**Precursores (≥ limiar, calculado no app)**")
            md2.append("| Rank | Termo | Similaridade |")
            md2.append("|---:|---|---:|")
            for r, (label, s) in enumerate(dict_matches["prec"], 1):
                md2.append(f"| {r} | {label} | {s:.3f} |")
        if dict_matches["cp"]:
            md2.append("\n**CP (≥ limiar, calculado no app)**")
            md2.append("| Rank | Fator | Similaridade |")
            md2.append("|---:|---|---:|")
            for r, (label, s) in enumerate(dict_matches["cp"], 1):
                md2.append(f"| {r} | {label} | {s:.3f} |")

        if md2:
            out2 = "\n".join(md2)
            with st.chat_message("assistant"):
                st.markdown(out2)
            st.session_state.chat.append({"role": "assistant", "content": out2})

        # Mensagens para o LLM apenas para “explicar” (sem buscar fora)
        msgs = [{"role": "system", "content": st.session_state.system_prompt}]
        if use_catalog and os.path.exists(DATASETS_CONTEXT_FILE):
            try:
                with open(DATASETS_CONTEXT_FILE, "r", encoding="utf-8") as f:
                    msgs.append({"role": "system", "content": f.read()})
            except Exception:
                pass
        msgs.append({"role": "user", "content": f"Explique, sem buscar outras fontes, os resultados calculados no app. Limiar Sphera={thr_sphera}, anos={'todos' if not years else years}."})

        with st.chat_message("assistant"):
            with st.spinner("Consultando o modelo (análise explicativa)…"):
                try:
                    resp = ollama_chat(msgs, model=OLLAMA_MODEL, temperature=0.2, stream=False)
                    content = resp.get("message", {}).get("content", "").strip() or json.dumps(resp)[:1200]
                except Exception as e:
                    content = f"[Comentário do modelo indisponível] {e}"
                st.markdown(content)
        st.session_state.chat.append({"role": "assistant", "content": content})

    else:
        # -------- Fluxo RAG “clássico” (mistura Sphera/GoSee/Docs/Upload) --------
        blocks = search_all(prompt)
        up_raw = get_upload_raw(upload_raw_max)
        if up_raw:
            blocks = [f"[UPLOAD_RAW]\n{up_raw}"] + blocks

        msgs = [{"role": "system", "content": st.session_state.system_prompt}]
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
