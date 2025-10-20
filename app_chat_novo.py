# app_chat.py — ESO • CHAT (Embeddings-only)
# Versão com patches PT/EN, “Somente Sphera”, Sumário de Resultados e **Filtros avançados (Location / Description contém)**
# - Busca SEMÂNTICA usando embeddings:
#   • Sphera:   data/analytics/sphera_embeddings.npz + sphera.parquet
#   • GoSee:    data/analytics/gosee_embeddings.npz  + gosee.parquet
#   • History:  data/analytics/history_embeddings.npz + history_texts.jsonl
# - Dicionários (seleção automática de idioma): WS / Precursores / CP
# - Uploads: chunk + embeddings em tempo real (Sentence-Transformers)
# - “Somente Sphera”: cálculo local (limiar de **similaridade do cosseno** e últimos N anos)
# - Sumário ao final: (2) Estatísticas, (3) Visualizações (exemplo), (4) Interpretação + Resumo descritivo
# - NOVO: Filtros avançados: Location (multiselect) e "Description contém" (substring, case-insensitive)

import os
import io
import re
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

from pathlib import Path
import re

PROMPTS_MD_PATH = Path("data/prompts/prompts.md")

@st.cache_data(show_spinner=False)
def load_prompts_md(md_path: Path):
    """
    Lê data/prompts/prompts.md e retorna:
    {
      "Texto":  [{"title": "1) ...", "body": "..."} , ...],
      "Upload": [{"title": "1) ...", "body": "..."} , ...]
    }
    Regras:
      - Seções: '## Texto' e '## Upload'
      - Items: '### <n>) <título>' seguidos do corpo até o próximo '###' ou '##'
    """
    if not md_path.exists():
        return {"Texto": [], "Upload": []}

    raw = md_path.read_text(encoding="utf-8")

    # Quebra por grandes seções
    sections = re.split(r"(?m)^##\s+", raw)
    data = {"Texto": [], "Upload": []}
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        # primeira linha = nome da seção (Texto/Upload)
        first_line, _, rest = sec.partition("\n")
        section_name = first_line.strip()
        if section_name not in ("Texto", "Upload"):
            continue

        # Itens "###"
        parts = re.split(r"(?m)^###\s+", rest)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            title_line, _, body = p.partition("\n")
            title_line = title_line.strip()
            # limpa numeração, mas mantém no título exibido
            title = title_line
            body = body.strip()
            data[section_name].append({"title": title, "body": body})

    # Ordena por prefixo numérico se houver (1), 2), etc.)
    def _key(x):
        m = re.match(r"^(\d+)\)", x["title"])
        return int(m.group(1)) if m else 9999
    for k in data:
        data[k].sort(key=_key)
    return data


@st.cache_data(show_spinner=False)
def load_file_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"[AVISO] Não consegui ler {p}: {e} (Prosseguindo sem esse contexto.)"

def build_system_prompt() -> str:
    preambulo = (
        "Você é o ESO-CHAT (segurança operacional)."
        "Siga estritamente as regras e convenções do contexto abaixo."
        "Responda em PT-BR por padrão."
        "Quando usar buscas semânticas, sempre mostre IDs/Fonte e similaridade."
        "Não invente dados fora dos contextos fornecidos."
    )
    ctx_md = load_file_text(CONTEXT_MD_REL_PATH)
    return preambulo + " === CONTEXTO ESO-CHAT (.md) === " + ctx_md

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = build_system_prompt()

if st.sidebar.button("Recarregar contexto (.md)"):
    st.session_state.system_prompt = build_system_prompt()
    st.sidebar.success("Contexto recarregado.")
    
# ===== Assistente de Prompts =====
st.sidebar.subheader("Assistente de Prompts")
prompts_bank = load_prompts_md(PROMPTS_MD_PATH)

# Escolha do tipo
prompt_type = st.sidebar.selectbox("Tipo de análise", options=["Texto", "Upload"], index=0)

# Opções de prompt para o tipo escolhido
titles = [it["title"] for it in prompts_bank.get(prompt_type, [])]
if not titles:
    st.sidebar.info("Nenhum prompt encontrado em {} (seção {}).".format(PROMPTS_MD_PATH, prompt_type))
else:
    selected_title = st.sidebar.selectbox("Modelo de prompt", options=titles, index=0, key="prompt_title_{}".format(prompt_type))
    # Recupera corpo
    selected = next((it for it in prompts_bank[prompt_type] if it["title"] == selected_title), None)
    body = selected["body"] if selected else ""

    # Coloca o corpo do prompt no rascunho (session_state)
    if "draft_prompt" not in st.session_state:
        st.session_state.draft_prompt = ""

    if st.sidebar.button("Carregar no rascunho", use_container_width=True):
        st.session_state["draft_prompt"] = body
        st.sidebar.success("Modelo carregado no rascunho (edite antes de enviar).")
        st.rerun()

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
        "❌ sentence-transformers não está disponível."
        "Instale as dependências (incluindo torch CPU) conforme o requirements.txt recomendado."
        f"Detalhe: {e}"
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
    if E_db is None or E_db.size == 0 or k <= 0:
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
        return "".join(out)
    except Exception:
        return ""

def read_docx_bytes(b: bytes) -> str:
    if docx is None:
        return ""
    try:
        doc = docx.Document(io.BytesIO(b))
        return "".join(p.text for p in doc.paragraphs)
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
    text = text.replace("", "").replace("", "")
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

# ---------- Preferências de saída ----------
st.sidebar.subheader("Saídas (Sumário)")
show_summary = st.sidebar.checkbox("Exibir sumário da consulta", True)
summary_via_model = st.sidebar.checkbox("Resumo descritivo com modelo", True)

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

# ---------- Funções de embeddings ----------

def ensure_st_encoder():
    if st.session_state.st_encoder is None:
        try:
            st.session_state.st_encoder = SentenceTransformer(ST_MODEL_NAME)
        except Exception as e:
            _fatal("❌ Não foi possível carregar o encoder de embeddings (Sentence-Transformers). "
        f"Modelo: {ST_MODEL_NAME} Detalhe: {e}"
            )

def encode_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    ensure_st_encoder()
    M = st.session_state.st_encoder.encode(
        texts, batch_size=batch_size, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)
    return M


def aggregate_dict_matches_over_hits(
    hits, lang: str,
    thr_ws: float, thr_prec: float, thr_cp: float,
    topn_ws: int, topn_prec: int, topn_cp: int,
    agg_mode: str = "max",
    per_event_thr: float = 0.30,
    min_support: int = 2,
):
    """
    WS/Precursores/CP somente dos dicionários embutidos vs DESCRIPTIONS dos hits Sphera.
    Agrega por 'max' ou 'mean', aplica limiar por evento e suporte mínimo.
    Retorna dict com listas de tuplas (label, sim, suporte).
    """
    try:
        if not hits:
            return {"ws": [], "prec": [], "cp": []}

        descs = []
        for _, _, row in hits:
            d = str(row.get("Description", row.get("DESCRIPTION", ""))).strip()
            if d:
                descs.append(d)
        if not descs:
            return {"ws": [], "prec": [], "cp": []}

        V_desc = encode_texts(descs, batch_size=32)  # (M, D)
        V_desc_T = V_desc.T

        def _score_bank(E_bank, labels_df, thr_global, topn_target):
            if E_bank is None or labels_df is None or len(labels_df) != E_bank.shape[0]:
                return []
            S = (E_bank @ V_desc_T)  # (N_terms x M_events)
            support = (S >= per_event_thr).sum(axis=1)
            sims = S.mean(axis=1) if agg_mode == "mean" else S.max(axis=1)
            mask = (support >= min_support) & (sims >= thr_global)
            idx = np.where(mask)[0]
            if idx.size == 0:
                return []
            order = idx[np.argsort(sims[idx])[::-1]]
            out = []
            for i in order[:topn_target]:
                label = str(labels_df.iloc[i].get("label", labels_df.iloc[i].get("text", f"TERM_{i}")))
                out.append((label, float(sims[i]), int(support[i])))
            return out

        E_ws, L_ws = select_ws_bank(lang)
        E_pr, L_pr = select_prec_bank(lang)
        E_cp, L_cp = select_cp_bank()

        return {
            "ws":  _score_bank(E_ws, L_ws, thr_ws,  topn_ws),
            "prec": _score_bank(E_pr, L_pr, thr_prec, topn_prec),
            "cp":  _score_bank(E_cp, L_cp, thr_cp,  topn_cp),
        }
    except Exception as e:
        try:
            st.warning(f"[Dict/Hits] Falha ao agregar dicionários sobre hits: {e}")
        except Exception:
            pass
        return {"ws": [], "prec": [], "cp": []}


def encode_query(q: str) -> np.ndarray:
    ensure_st_encoder()
    v = st.session_state.st_encoder.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-9)
    return v

def get_sphera_location_col(df: pd.DataFrame) -> str | None:
    """
    Retorna a coluna correta para 'Location' na Sphera, por ordem de preferência:
    1) LOCATION
    2) FPSO
    3) Location
    4) FPSO/Unidade
    5) Unidade
    (Só cai para AREA/Setor se nada acima existir — e avisa no UI.)
    """
    if df is None:
        return None
    preferred = ["LOCATION", "FPSO", "Location", "FPSO/Unidade", "Unidade"]
    fallback  = ["AREA", "Area", "Setor"]
    for c in preferred:
        if c in df.columns:
            return c
    for c in fallback:
        if c in df.columns:
            st.warning(
                "⚠️ Usando '{}' como fallback de Location (colunas LOCATION/FPSO/Location ausentes)."
                .format(c)
            )
            return c
    return None


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
thr_sphera = st.sidebar.slider("Limiar Sphera (Description — cos sim)", 0.0, 1.0, 0.25, 0.01)
thr_ws     = st.sidebar.slider("Limiar WS", 0.0, 1.0, 0.25, 0.01)
thr_prec   = st.sidebar.slider("Limiar Precursores", 0.0, 1.0, 0.25, 0.01)
thr_cp     = st.sidebar.slider("Limiar CP", 0.0, 1.0, 0.25, 0.01)

use_catalog = st.sidebar.checkbox("Injetar datasets_context.md", True)

# ---------- Filtros Avançados — Sphera ----------
st.sidebar.subheader("Filtros avançados — Sphera")
_sph_loc_col = None
_sph_loc_options = []
_sph_has_desc = False
desc_candidates = ["Description", "DESCRIPTION"]
_sph_desc_col = next((c for c in desc_candidates if c in (df_sph.columns if df_sph is not None else [])), None)
if df_sph is not None:
    _sph_loc_col = get_sphera_location_col(df_sph)  # << aqui
    if _sph_loc_col:
        _sph_loc_options = sorted([str(x) for x in df_sph[_sph_loc_col].dropna().unique()])[:500]
    _sph_has_desc = "Description" in df_sph.columns or "DESCRIPTION" in df_sph.columns


sph_loc_selected = st.sidebar.multiselect(
    "Location (se disponível)", options=_sph_loc_options, default=[]
) if _sph_loc_col else []

sph_desc_contains = st.sidebar.text_input(
    "Description contém (substring)", value=""
) if _sph_has_desc else ""

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

# ---------- Funções de busca / filtros ----------

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


def apply_advanced_filters(base: pd.DataFrame) -> pd.DataFrame:
    d = base
    if _sph_loc_col and sph_loc_selected:
        d = d[d[_sph_loc_col].astype(str).isin(set(sph_loc_selected))]
    if _sph_has_desc and sph_desc_contains:
        pat = re.escape(sph_desc_contains)
        desc_col = _sph_desc_col or ("Description" if "Description" in d.columns else None)
        if desc_col:
            d = d[d[desc_col].astype(str).str.contains(pat, case=False, na=False)]
    return d

def sphera_similar_to_text(query_text: str, min_sim: float, years: int | None = None, topk: int = 50):
    """Retorna [(event_id, sim, row)] com sim >= min_sim (cosine), usando Sphera/Description e filtros avançados."""
    if df_sph is None or E_sph is None or E_sph.size == 0:
        return []
    base = df_sph
    if years is not None:
        base = filter_sphera_by_date(base, years)
    base = apply_advanced_filters(base)

    text_col = "Description" if "Description" in base.columns else base.columns[0]
    id_col = "Event ID" if "Event ID" in base.columns else ("EVENT_NUMBER" if "EVENT_NUMBER" in base.columns else None)

    # alinhar E_sph com o índice filtrado (apenas se índice for inteiro)
    try:
        base_idx = base.index.to_numpy()
        if np.issubdtype(base_idx.dtype, np.integer):
            E_view = E_sph[base_idx, :]
        else:
            raise TypeError("Índice não inteiro; usando E_sph completo.")
    except Exception:
        E_view = E_sph
        base = df_sph
        base = apply_advanced_filters(base)  # reaplicar se caiu no fallback
        if years is not None:
            base = filter_sphera_by_date(base, years)

    qv = encode_query(query_text)
    sims = E_view @ qv
    idx = np.argsort(-sims)

    out = []
    upto = min(topk, len(idx))
    for i in idx[:upto]:
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

    # Precursores
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

    # CP
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
    return "".join(buf).strip()

# (NOVO) Parser simples para blocos do RAG misto

def parse_blocks(blocks: list[str]):
    stats = {
        "Sphera": {"count": 0, "sims": []},
        "GoSee": {"count": 0, "sims": []},
        "Docs":   {"count": 0, "sims": []},
        "Upload": {"count": 0, "sims": []},
    }
    for b in blocks or []:
        if b.startswith("[UPLOAD_RAW]"):
            continue
        m = re.search(r"\(sim=([0-9.]+)\)", b)
        sim = float(m.group(1)) if m else None
        if b.startswith("[Sphera/"):
            stats["Sphera"]["count"] += 1
            if sim is not None: stats["Sphera"]["sims"].append(sim)
        elif b.startswith("[GoSee/"):
            stats["GoSee"]["count"] += 1
            if sim is not None: stats["GoSee"]["sims"].append(sim)
        elif b.startswith("[Docs/"):
            stats["Docs"]["count"] += 1
            if sim is not None: stats["Docs"]["sims"].append(sim)
        elif b.startswith("[UPLOAD "):
            stats["Upload"]["count"] += 1
            if sim is not None: stats["Upload"]["sims"].append(sim)
    return stats

# (NOVO) Funções utilitárias para sumário

def _agg_sims(v):
    if not v: return {"n": 0, "min": None, "max": None, "avg": None}
    return {"n": len(v), "min": float(np.min(v)), "max": float(np.max(v)), "avg": float(np.mean(v))}


def render_visual_layout_example():
    st.markdown(
        """
**3. Visualizações que o app oferece (exemplo de layout)**
- Heatmap: *Location × Risk Area* (contagem de incidentes)
- Série temporal mensal: número de eventos por mês (últimos N anos)
- Top termos WS/Precursores/CP: ranking por similaridade
- Tabela exportável: eventos Sphera filtrados com ID, data, descrição e similaridade
        """
    )


def render_interpretation_via_model(prompt: str, context_hint: str):
    msgs = [
        {"role": "system", "content": st.session_state.system_prompt},
        {"role": "user", "content": (
            "Você é um analista de Segurança Operacional."
            "Escreva uma interpretação breve e objetiva dos resultados, com 3–6 bullet points,"
            "indicando padrões, possíveis causas (WS/Precursores/CP) e sugestões práticas de follow-up."
            f"Contexto: {context_hint}"
            f"Consulta do usuário: {prompt}"
        )}
    ]
    try:
        msgs.append({"role":"user","content":"Importante: NÃO gere novas listas de WS, Precursores ou Fatores CP; apenas interprete as tabelas calculadas pelo app (embeddings dos dicionários sobre as DESCRIPTIONS dos eventos Sphera recuperados)."})
        resp = ollama_chat(msgs, model=OLLAMA_MODEL, temperature=0.2, stream=False)
        return resp.get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"[Interpretação automática indisponível] {e}"


def render_descriptive_summary_via_model(prompt: str, stats_text: str):
    msgs = [
        {"role": "system", "content": st.session_state.system_prompt},
        {"role": "user", "content": (
            "Produza um resumo descritivo em 4–6 linhas sobre a busca realizada,"
            "mencionando fontes com resultados, nível de similaridade observado e limitações,"
            "usando tom técnico e claro." + stats_text + f"Pergunta do usuário: {prompt}"
        )}
    ]
    try:
        resp = ollama_chat(msgs, model=OLLAMA_MODEL, temperature=0.2, stream=False)
        return resp.get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"[Resumo descritivo automático indisponível] {e}"


def render_stats_section(title: str, per_source_stats: dict, extra_lines: list[str] | None = None):
    st.markdown(f"**2. {title}**")
    lines = []
    for src in ("Sphera", "GoSee", "Docs", "Upload"):
        s = per_source_stats.get(src, {"count": 0, "sims": []})
        agg = _agg_sims(s["sims"]) if "sims" in s else _agg_sims([])
        lines.append(
            f"- **{src}**: {s['count']} itens | sim avg={agg['avg']:.3f} máx={agg['max']:.3f} mín={agg['min']:.3f}" if agg['n']>0 else f"- **{src}**: {s['count']} itens"
        )
    if extra_lines:
        lines.extend(extra_lines)
    st.markdown("".join(lines))

# ---------- Busca mista (com filtros aplicados à Sphera) ----------

def search_all(query: str) -> list[str]:
    """Embute a query e busca nos 4 conjuntos (Sphera/GoSee/Docs/Upload). Retorna blocos formatados."""
    qv = encode_query(query)
    blocks: list[tuple[float, str]] = []

    # Sphera (apenas quando NÃO está em 'Somente Sphera') com filtros avançados
    if not only_sphera:
        if k_sph > 0 and E_sph is not None and df_sph is not None and len(df_sph) >= E_sph.shape[0]:
            base = df_sph
            if apply_time_filter:
                base = filter_sphera_by_date(base, years_back)
            base = apply_advanced_filters(base)

            text_col = "Description" if "Description" in base.columns else base.columns[0]
            id_col = "Event ID" if "Event ID" in base.columns else ("EVENT_NUMBER" if "EVENT_NUMBER" in base.columns else None)

            # alinhar E com base filtrada
            try:
                base_idx = base.index.to_numpy()
                if np.issubdtype(base_idx.dtype, np.integer):
                    E_view = E_sph[base_idx, :]
                else:
                    raise TypeError
            except Exception:
                E_view = E_sph
                base = df_sph
                if apply_time_filter:
                    base = filter_sphera_by_date(base, years_back)
                base = apply_advanced_filters(base)

            sims = (E_view @ qv).astype(float)
            ord_idx = np.argsort(-sims)
            kept = 0
            for i in ord_idx:
                if kept >= k_sph: break
                s = float(sims[i])
                if s < thr_sphera:  # aplica limiar de SIMILARIDADE do cosseno
                    continue
                row = base.iloc[int(i)]
                evid = row.get(id_col, f"row{i}") if id_col else f"row{i}"
                snippet = str(row.get(text_col, ""))[:800]
                blocks.append((s, f"[Sphera/{evid}] (sim={s:.3f}){snippet}"))
                kept += 1

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
                blocks.append((s, f"[GoSee/{gid}] (sim={s:.3f}){snippet}"))

    # Docs (history)
    if not only_sphera:
        if k_his > 0 and E_his is not None and rows_his:
            hits = cos_topk(E_his, qv, k=k_his)
            for i, s in hits:
                r = rows_his[i]
                src = f"Docs/{r.get('source','?')}/{r.get('chunk_id', 0)}"
                snippet = str(r.get("text", ""))[:800]
                blocks.append((s, f"[{src}] (sim={s:.3f}){snippet}"))

    # Upload
    if not only_sphera:
        if k_upl > 0 and st.session_state.upld_emb is not None and len(st.session_state.upld_texts) == st.session_state.upld_emb.shape[0]:
            hits = cos_topk(st.session_state.upld_emb, qv, k=k_upl)
            for i, s in hits:
                meta = st.session_state.upld_meta[i]
                snippet = st.session_state.upld_texts[i][:800]
                blocks.append((s, f"[UPLOAD {meta['file']} / {meta['chunk_id']}] (sim={s:.3f}){snippet}"))

    blocks.sort(key=lambda x: -x[0])
    return [b for _, b in blocks]


def _send_prompt_to_chat():
    text_to_send = (st.session_state.get("draft_prompt") or "").strip()
    if not text_to_send:
        return
    # adiciona ao histórico como 'user'
    if "chat" not in st.session_state:
        st.session_state.chat = []
    st.session_state.chat.append({"role": "user", "content": text_to_send})
    # sinaliza para o pipeline do chat processar após o rerun
    st.session_state["pending_user_prompt"] = text_to_send
    # limpa o rascunho
    st.session_state["draft_prompt"] = ""
    st.rerun()

# ---------- UI ----------
st.title("ESO • CHAT — HIST + UPLD (Embeddings preferencial) + Dicionários PT/EN")
st.caption("RAG local (Sphera / GoSee / Docs / Upload) + WS/Precursores/CP com seleção automática de idioma.")

# Mostrar histórico
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# === Saída – Dicionários (WS/Prec/CP) ===
st.sidebar.markdown("### Saída – Dicionários (WS/Prec/CP)")
topn_ws  = st.sidebar.slider("Top-N WS", 3, 50, 10, 1)
topn_prec = st.sidebar.slider("Top-N Precursores", 3, 50, 10, 1)
topn_cp  = st.sidebar.slider("Top-N CP", 3, 50, 10, 1)
st.sidebar.markdown("**Agregação sobre eventos recuperados (Sphera)**")
agg_mode = st.sidebar.selectbox("Como agregar similaridade por termo", ["max", "mean"], index=0)
per_event_thr = st.sidebar.slider("Limiar por evento (dicionários)", 0.0, 1.0, 0.30, 0.01)
min_support = st.sidebar.slider("Suporte mínimo (nº de eventos)", 1, 10, 2, 1)
st.sidebar.markdown("### Modo de Saída")
output_mode = st.sidebar.selectbox("Layout do resultado", ["Auto", "Investigação", "Aprendizado", "Comportamento", "Métricas"], index=0)

prompt = st.chat_input("Digite sua pergunta ou cole seu texto")
if prompt and _is_freq_by_type_intent(prompt) and df_sph is not None:
    render_frequency_by_type(df_sph)
    prompt = None
if not prompt and "pending_user_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_user_prompt")
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Opcional: injeta um recorte 'cru' do upload (máx N chars)
    up_raw = get_upload_raw(upload_raw_max)
    lang = guess_lang((prompt or "") + "" + (up_raw or ""))

    if only_sphera:
        # -------- Fluxo "Somente Sphera" --------
        query_text = up_raw if up_raw else prompt
        years = years_back if apply_time_filter else None

        # 1) Eventos Sphera semelhantes (limiar de similaridade do cosseno)
        hits = sphera_similar_to_text(query_text, thr_sphera, years=years, topk=200)
        loc_col = get_sphera_location_col(df_sph)  # << escolha centralizada
        desc_col = _sph_desc_col or ("Description" if "Description" in (df_sph.columns if df_sph is not None else []) else None)
        
        if hits:
            md = [
                "**Eventos do Sphera (calculado no app, limiar de similaridade aplicado)**",
                "| Event Id | Similaridade (cos) | Location | Description |",
                "|---:|---:|---|---|",
            ]
            for evid, s, row in hits:
                loc = str(row.get(loc_col, "N/D")) if loc_col else "N/D"
                desc_val = str(row.get(desc_col, "")) if desc_col else str(row.get("Description",""))
                desc = desc_val.replace("\n", " ")[:4000]
                md.append("| {} | {:.3f} | {} | {} |".format(evid, s, loc, desc))
            tbl = "\n".join(md)
            with st.chat_message("assistant"):
                st.markdown(tbl)
            st.session_state.chat.append({"role": "assistant", "content": tbl})
        else:
            msg = "Nenhum evento do Sphera com **similaridade do cosseno** ≥ " + str(thr_sphera)
            with st.chat_message("assistant"):
                st.markdown(msg)
            st.session_state.chat.append({"role": "assistant", "content": msg})

        # 2) Dicionários (WS / Precursores / CP)
        dict_matches = aggregate_dict_matches_over_hits(hits, lang, thr_ws, thr_prec, thr_cp, topn_ws, topn_prec, topn_cp, agg_mode, per_event_thr, min_support)
        md2 = []
        # WS
        if dict_matches["ws"]:
            md2 += [
                "",  # linha em branco antes da tabela
                "**WS (≥ limiar, calculado no app)**",
                "| Rank | Termo | Similaridade |",
                "|---:|---|---:|",
            ]
            
_ws_support = any(isinstance(x, (list, tuple)) and len(x) >= 3 for x in dict_matches.get("ws", []))
if _ws_support:
    md2[-2] = "| Rank | Termo | Similaridade | Suporte |"
    md2[-1] = "|---:|---|---:|---:|"
    for r, item in enumerate(dict_matches.get("ws", []), 1):
        try:
            if isinstance(item, (list, tuple)):
                if len(item) >= 3:
                    label, s, sup = item[0], float(item[1]), int(item[2])
                    md2.append(f"| {r} | {label} | {s:.3f} | {sup} |")
                elif len(item) >= 2:
                    label, s = item[0], float(item[1])
                    md2.append(f"| {r} | {label} | {s:.3f} |")
                else:
                    md2.append(f"| {r} | {str(item)} |  |")
            else:
                md2.append(f"| {r} | {str(item)} |  |")
        except Exception:
            md2.append(f"| {r} | {str(item)} |  |")
else:
            md2 += [
                "",
                "**WS (≥ limiar, calculado no app)**",
                "Nenhum WS ≥ limiar.",
            ]
        
        # Precursores
if dict_matches["prec"]:
            md2 += [
                "",
                "**Precursores (≥ limiar, calculado no app)**",
                "| Rank | Termo | Similaridade |",
                "|---:|---|---:|",
            ]
            
_prec_support = any(isinstance(x, (list, tuple)) and len(x) >= 3 for x in dict_matches.get("prec", []))
if _prec_support:
    md2[-2] = "| Rank | Termo | Similaridade | Suporte |"
    md2[-1] = "|---:|---|---:|---:|"
for r, item in enumerate(dict_matches.get("prec", []), 1):
    try:
        if isinstance(item, (list, tuple)):
            if len(item) >= 3:
                label, s, sup = item[0], float(item[1]), int(item[2])
                md2.append(f"| {r} | {label} | {s:.3f} | {sup} |")
            elif len(item) >= 2:
                label, s = item[0], float(item[1])
                md2.append(f"| {r} | {label} | {s:.3f} |")
            else:
                md2.append(f"| {r} | {str(item)} |  |")
        else:
            md2.append(f"| {r} | {str(item)} |  |")
    except Exception:
        md2.append(f"| {r} | {str(item)} |  |")
        else:
            md2 += [
                "",
                "**Precursores (≥ limiar, calculado no app)**",
                "Nenhum Precursor ≥ limiar.",
            ]
        
        # CP
        if dict_matches["cp"]:
            md2 += [
                "",
                "**CP (≥ limiar, calculado no app)**",
                "| Rank | Fator | Similaridade |",
                "|---:|---|---:|",
            ]
            
_cp_support = any(isinstance(x, (list, tuple)) and len(x) >= 3 for x in dict_matches.get("cp", []))
if _cp_support:
    md2[-2] = "| Rank | Fator | Similaridade | Suporte |"
    md2[-1] = "|---:|---|---:|---:|"
for r, item in enumerate(dict_matches.get("cp", []), 1):
    try:
        if isinstance(item, (list, tuple)):
            if len(item) >= 3:
                label, s, sup = item[0], float(item[1]), int(item[2])
                md2.append(f"| {r} | {label} | {s:.3f} | {sup} |")
            elif len(item) >= 2:
                label, s = item[0], float(item[1])
                md2.append(f"| {r} | {label} | {s:.3f} |")
            else:
                md2.append(f"| {r} | {str(item)} |  |")
        else:
            md2.append(f"| {r} | {str(item)} |  |")
    except Exception:
        md2.append(f"| {r} | {str(item)} |  |")
        else:
            md2 += [
                "",
                "**CP (≥ limiar, calculado no app)**",
                "Nenhum CP ≥ limiar.",
            ]
        
        if md2:
            out2 = "\n".join(md2)        # ← AGORA COM QUEBRAS
            with st.chat_message("assistant"):
                st.markdown(out2)
            st.session_state.chat.append({"role": "assistant", "content": out2})


        md2_lines = []
        if dict_matches["ws"]:
            md2_lines.append("")  # linha em branco antes da tabela
            md2_lines.append("**WS (≥ limiar, calculado no app)**")
            md2_lines.append("| Rank | Termo | Similaridade |")
            md2_lines.append("|---:|---|---:|")
            for r_idx, (label, s) in enumerate(dict_matches["ws"], 1):
                md2_lines.append(f"| {r_idx} | {label} | {s:.3f} |")
        else:
            md2_lines.append("")
            md2_lines.append("**WS (≥ limiar, calculado no app)**")
            md2_lines.append("Nenhum WS ≥ limiar.")

        
        # 3) Comentário do LLM sobre os resultados (sem buscar fora)
        msgs = [{"role": "system", "content": st.session_state.system_prompt}]
        if use_catalog and os.path.exists(DATASETS_CONTEXT_FILE):
            try:
                with open(DATASETS_CONTEXT_FILE, "r", encoding="utf-8") as f:
                    msgs.append({"role": "system", "content": f.read()})
            except Exception:
                pass
        msgs.append({"role": "user", "content": f"Explique, sem buscar outras fontes, os resultados calculados no app. Limiar Sphera={thr_sphera}, anos={'todos' if not years else years}."})
        msgs.append({
          "role": "user",
          "content": (
              "Regra obrigatória (Sphera): Location deve vir da coluna LOCATION, "
              "ou do campo FPSO quando LOCATION não existir; nunca usar AREA como Location. "
              "Se a coluna não existir nos blocos, retornar 'N/D'."
          )
        })
        msgs.append({
          "role": "user",
          "content": (
            "Formate a saída em três seções separadas com tabelas Markdown, sem texto entre elas, "
            "seguindo exatamente o padrão do contexto: "
            "1) **WS (≥ limiar, calculado no app)**, 2) **Precursores (≥ limiar, calculado no app)**, "
            "3) **CP (≥ limiar, calculado no app)**. "
            "Use cabeçalho de tabela e 3 casas decimais na similaridade. "
            "Se uma categoria não tiver itens, escreva ‘Nenhum <categoria> ≥ limiar.’"
          )
        })

        with st.chat_message("assistant"):
            with st.spinner("Consultando o modelo (análise explicativa)…"):
                try:
                    resp = ollama_chat(msgs, model=OLLAMA_MODEL, temperature=0.2, stream=False)
                    content = resp.get("message", {}).get("content", "").strip() or json.dumps(resp)[:1200]
                except Exception as e:
                    content = f"[Comentário do modelo indisponível] {e}"
                st.markdown(content)
        st.session_state.chat.append({"role": "assistant", "content": content})

        # 4) SUMÁRIO
        if show_summary:
            sims = [s for _, s, _ in hits] if hits else []
            per_source = {
                "Sphera": {"count": len(sims), "sims": sims},
                "GoSee": {"count": 0, "sims": []},
                "Docs": {"count": 0, "sims": []},
                "Upload": {"count": len(st.session_state.upld_texts) if st.session_state.upld_texts else 0, "sims": []},
            }
            extra = [
                f"- Filtro temporal: {'sem filtro' if years is None else f'últimos {years} anos'}",
                f"- Limiar de similaridade aplicado: {thr_sphera:.2f}",
                f"- Idioma inferido: {lang.upper()}",
                (f"- Location: {', '.join(sph_loc_selected)}" if sph_loc_selected else "- Location: (sem filtro)"),
                (f"- Description contém: '{sph_desc_contains}'" if sph_desc_contains else "- Description contém: (vazio)"),
                f"- WS/Prec/CP retornados: {len(dict_matches['ws'])}/{len(dict_matches['prec'])}/{len(dict_matches['cp'])}",
            ]
            with st.chat_message("assistant"):
                render_stats_section("Estatísticas principais geradas", per_source, extra)
                render_visual_layout_example()
                if summary_via_model:
                    context_hint = f"Sphera hits={len(sims)}, thr={thr_sphera}, years={'all' if years is None else years}"
                    interp = render_interpretation_via_model(prompt, context_hint)
                else:
                    interp = (
                        "- Similaridades indicam proximidade textual com descrições Sphera;"
                        "- Ajuste de limiar pode aumentar precisão (↑) ou abrangência (↓);"
                        "- Verificar manualmente top eventos;"
                        "- Revisar WS/Precursores/CP com maior similaridade para ações preventivas."
                    )
                st.markdown("**4. Interpretação dos resultados (exemplo típico)**" + interp)

                stats_text = "".join(extra)
                if summary_via_model:
                    desc = render_descriptive_summary_via_model(prompt, stats_text)
                else:
                    desc = (
                        "Foram retornados eventos do Sphera acima do limiar de similaridade definido, "
                        "considerando o escopo e filtros aplicados. As correspondências em WS, "
                        "Precursores e CP reforçam a leitura contextual e subsidiam decisões de risco."
                    )
                st.markdown("**Resumo descritivo da consulta**" + desc)

    else:
        # -------- Fluxo RAG “clássico” --------
        blocks = search_all(prompt)
        up_raw = get_upload_raw(upload_raw_max)
        if up_raw:
            blocks = [f"[UPLOAD_RAW]{up_raw}"] + blocks

        msgs = [{"role": "system", "content": st.session_state.system_prompt}]
        if use_catalog and os.path.exists(DATASETS_CONTEXT_FILE):
            try:
                with open(DATASETS_CONTEXT_FILE, "r", encoding="utf-8") as f:
                    msgs.append({"role": "system", "content": f.read()})
            except Exception:
                pass

        if blocks:
            ctx = "".join(blocks)
            msgs.append({"role": "user", "content": f"CONTEXTOS (HIST + UPLOAD):{ctx}"})
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

        # SUMÁRIO
        if show_summary:
            blocks_wo_raw = [b for b in blocks if not b.startswith("[UPLOAD_RAW]")]
            per_source = parse_blocks(blocks_wo_raw)
            extra = [
                f"- Top-K: Sphera={k_sph}, GoSee={k_gos}, Docs={k_his}, Upload={k_upl}",
                f"- Limiar WS/Prec/CP: {thr_ws:.2f}/{thr_prec:.2f}/{thr_cp:.2f}",
                f"- Idioma inferido: {lang.upper()}",
                (f"- Location: {', '.join(sph_loc_selected)}" if sph_loc_selected else "- Location: (sem filtro)"),
                (f"- Description contém: '{sph_desc_contains}'" if sph_desc_contains else "- Description contém: (vazio)"),
                f"- Uploads indexados: {len(st.session_state.upld_texts)} chunks" if st.session_state.upld_texts else "- Sem uploads no contexto",
            ]
            with st.chat_message("assistant"):
                render_stats_section("Estatísticas principais geradas", per_source, extra)
                render_visual_layout_example()
                if summary_via_model:
                    context_hint = (
                        f"Sphera n={per_source['Sphera']['count']} avg={_agg_sims(per_source['Sphera']['sims'])['avg']}; "
                        f"GoSee n={per_source['GoSee']['count']}; Docs n={per_source['Docs']['count']}; "
                        f"Upload n={per_source['Upload']['count']}"
                    )
                    interp = render_interpretation_via_model(prompt, context_hint)
                else:
                    interp = (
                        "- Resultados agregam múltiplas fontes com base em similaridade;"
                        "- Priorize itens com maior similaridade do cosseno e origem Sphera;"
                        "- Use WS/Prec/CP como apoio a ações corretivas/preventivas;"
                        "- Ajuste Top-K/limiares para refinar o escopo."
                    )
                st.markdown("**4. Interpretação dos resultados (exemplo típico)**" + interp)

                stats_text = "".join(extra)
                if summary_via_model:
                    desc = render_descriptive_summary_via_model(prompt, stats_text)
                else:
                    desc = (
                        "A consulta integrou Sphera, GoSee, Docs e Uploads segundo os Top-K e filtros definidos. "
                        "As similaridades mais altas (cosseno) indicam proximidade textual e relevância operacional. "
                        "Ajustes de limiar/Top-K podem ampliar ou reduzir a abrangência."
                    )
                st.markdown("**Resumo descritivo da consulta**" + desc)

st.markdown("### 📝 Rascunho do prompt (edite antes de enviar)")
st.caption("Dica: cole o seu texto do evento onde indicado; se for usar upload, envie os arquivos na barra lateral antes de enviar.")

draft = st.text_area("Conteúdo do prompt", height=220, key="draft_prompt")

c_a, c_c = st.columns([1,3])
with c_a:
    st.button("Enviar para o chat", use_container_width=True, on_click=_send_prompt_to_chat)
# (sem botão de limpar — o _send_prompt_to_chat já limpa o rascunho)


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


def _is_freq_by_type_intent(text: str) -> bool:
    t = (text or "").lower()
    keys = ["frequência", "frequencia", "frequency", "freq", "por tipo", "event type", "observation", "near miss", "incident"]
    return any(k in t for k in keys)

def render_frequency_by_type(df_sph):
    type_cols = ["event_type", "EVENT_TYPE", "tipo", "Tipo", "TYPE"]
    col = next((c for c in type_cols if c in df_sph.columns), None)
    if not col:
        st.warning("Não encontrei coluna de tipo de evento (ex.: event_type).")
        return

    s = df_sph[col].astype(str).str.strip().str.lower()
    map_alias = {
        "observation": "Observation",
        "near miss": "Near Miss",
        "incident": "Incident",
        "incidente": "Incident",
        "quase acidente": "Near Miss",
        "observação": "Observation",
    }
    s = s.map(lambda x: map_alias.get(x, x.title()))

    freq = s.value_counts().rename_axis("Tipo").reset_index(name="Contagem")
    total = int(freq["Contagem"].sum()) if not freq.empty else 0
    if total == 0:
        st.info("Não há eventos na base para calcular frequência por tipo.")
        return
    freq["Proporção"] = (freq["Contagem"] / total).round(3)

    md = []
    md += ["**Frequência por tipo (Sphera)**", ""]
    md += ["| Tipo | Contagem | Proporção |", "|---|---:|---:|"]
    for _, r in freq.iterrows():
        md.append(f"| {{r['Tipo']}} | {{int(r['Contagem'])}} | {{r['Proporção']:.3f}} |")

    out = "\\n".join(md)
    with st.chat_message("assistant"):
        st.markdown(out)
    st.session_state.chat.append({{"role": "assistant", "content": out}})
