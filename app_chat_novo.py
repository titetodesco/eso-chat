# -*- coding: utf-8 -*-
"""
app_chat_novo.py — FINAL (ESO • CHAT, Somente Sphera + Dicionários)

Atende aos objetivos:
- Corrige contagens: tabela de **Sphera** sempre reflete nº de hits; **WS/Precursores/CP** calculados apenas sobre os hits (agregação max/mean, limiar por evento, suporte mínimo). Evita zero indevido quando há hits (defaults mais permissivos e correções de filtro/colunas).
- Remove qualquer toggle de “Injetar datasets_context.md” — o arquivo é SEMPRE injetado.
- “Limpar uploads” e “Limpar chat” apenas zeram estado e fazem rerun **sem** disparar prompts.
- Remove “Modo de Saída” e qualquer bloco fixo de resposta — a síntese é do modelo (via system + contexto).
- Upload: **apenas** “Tamanho máx. de UPLOAD_RAW (chars)” (chunk/overlap ocultos e fixos internamente quando necessário).
- Mantém **Filtros avançados – Sphera** e **Agregação sobre eventos recuperados (Sphera)**.
- “Description contém (substring)” corrigido (case-insensitive, regex escapado, coluna correta).
- Seletor de prompts: **dois combos simultâneos** (“Texto” e “Upload”) lidos de `data/prompts/prompts.md` + botão **“Carregar no rascunho”**; rascunho editável e botão **“Enviar para o chat”**.
- Usa **somente** bancos existentes `.npz/.parquet` (Sphera + dicionários PT/EN). **Não** gera novos termos; usa labels existentes.
- Location: usa **LOCATION**; se indisponível tenta **FPSO**, **Location**, **FPSO/Unidade**, **Unidade**. **Nunca** usa AREA como location; se nada existir, mostra **“N/D”**.

Arquivos esperados:
- `data/analytics/sphera_embeddings.npz` + `data/analytics/sphera.parquet`
- Dicionários: `ws_embeddings_*.npz + .parquet`, `prec_embeddings_*.npz + .parquet`, `cp_embeddings.npz` + `cp_labels.parquet`
- Contexto: `data/datasets_context.md` (sempre injetado) e, se existir, `docs/contexto_eso_chat.md` (complementar)

Requisitos: `streamlit`, `pandas`, `numpy`, `sentence-transformers`, `requests`.
Config de modelo: `OLLAMA_HOST`, `OLLAMA_MODEL` (e opcional `OLLAMA_API_KEY`).
"""

import os
import re
import io
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ========================== Config inicial ==========================
st.set_page_config(page_title="ESO • CHAT", page_icon="💬", layout="wide")

DATA_DIR = "data"
AN_DIR   = os.path.join(DATA_DIR, "analytics")
ALT_DIR  = "/mnt/data"  # fallback em ambientes gerenciados
DOCS_DIR = Path("docs")
DATASETS_CONTEXT_PATH = Path("data/datasets_context.md")
CONTEXTO_ESO_MD_PATH  = DOCS_DIR / "contexto_eso_chat.md"  # opcional complementar
PROMPTS_MD_PATH       = Path("data/prompts/prompts.md")

# Modelo (chat)
OLLAMA_HOST    = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", ""))
OLLAMA_MODEL   = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", ""))
OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY"))
HEADERS_JSON   = {"Authorization": f"Bearer {OLLAMA_API_KEY}", "Content-Type": "application/json"} if OLLAMA_API_KEY else {"Content-Type": "application/json"}

# Embeddings (Sentence-Transformers para query/upload; corpus já embutido em .npz)
ST_MODEL_NAME = os.getenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# ========================== Helpers base ==========================
def _fatal(msg: str):
    st.error(msg)
    st.stop()

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    _fatal(f"❌ sentence-transformers indisponível: {e}")

@st.cache_resource(show_spinner=False)
def ensure_st_encoder():
    try:
        return SentenceTransformer(ST_MODEL_NAME)
    except Exception as e:
        _fatal(f"❌ Não foi possível carregar o encoder: {e}")

@st.cache_data(show_spinner=False)
def load_npz_embeddings(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=True) as z:
            for key in ("embeddings", "E", "X", "vectors", "vecs"):
                if key in z:
                    E = np.array(z[key]).astype(np.float32, copy=False)
                    # l2 normalize
                    n = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
                    return (E / n).astype(np.float32)
            # fallback: maior matriz 2D
            best_k, best_n = None, -1
            for k in z.files:
                arr = z[k]
                if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] > best_n:
                    best_k, best_n = k, arr.shape[0]
            if best_k is None:
                st.warning(f"{os.path.basename(path)} não contém matriz 2D.")
                return None
            E = np.array(z[best_k]).astype(np.float32, copy=False)
            n = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
            return (E / n).astype(np.float32)
    except Exception as e:
        st.warning(f"Falha ao ler {path}: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_prompts_md(md_path: Path) -> Dict[str, List[Dict[str, str]]]:
    """Retorna {"Texto": [{title,body}], "Upload": [{title,body}]} a partir de data/prompts/prompts.md."""
    if not md_path.exists():
        return {"Texto": [], "Upload": []}
    raw = md_path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^##\s+", raw)
    data = {"Texto": [], "Upload": []}
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        first, _, rest = sec.partition("\n")
        if first.strip() not in ("Texto", "Upload"):
            continue
        parts = re.split(r"(?m)^###\s+", rest)
        for p in parts:
            p = p.strip()
            if not p: continue
            title, _, body = p.partition("\n")
            data[first.strip()].append({"title": title.strip(), "body": body.strip()})
    # Ordena por prefixo numérico "1) ..." se houver
    def _k(x):
        m = re.match(r"^(\d+)\)", x["title"])
        return int(m.group(1)) if m else 9999
    for k in data:
        data[k].sort(key=_k)
    return data

@st.cache_data(show_spinner=False)
def load_file_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"[AVISO] Não consegui ler {p}: {e} (continuando sem este contexto)"

def build_system_prompt() -> str:
    """Injeta SEMPRE datasets_context.md; contexto_eso_chat.md é complementar (se existir)."""
    pre = (
        "Você é o ESO-CHAT para segurança operacional (óleo e gás). "
        "Responda em PT-BR, cite IDs/similaridade quando usar buscas locais, "
        "e não invente dados fora dos contextos fornecidos.\n\n"
    )
    ctx = []
    if DATASETS_CONTEXT_PATH.exists():
        ctx.append("=== DATASETS_CONTEXT ===\n" + load_file_text(DATASETS_CONTEXT_PATH))
    if CONTEXTO_ESO_MD_PATH.exists():
        ctx.append("=== CONTEXTO ESO-CHAT ===\n" + load_file_text(CONTEXTO_ESO_MD_PATH))
    return pre + "\n\n".join(ctx)

# ========================== Estado ==========================
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = build_system_prompt()
if "chat" not in st.session_state:
    st.session_state.chat = []
if "draft_prompt" not in st.session_state:
    st.session_state.draft_prompt = ""
if "upld_texts" not in st.session_state:
    st.session_state.upld_texts = []
if "upld_meta" not in st.session_state:
    st.session_state.upld_meta = []
if "upld_emb" not in st.session_state:
    st.session_state.upld_emb = None
if "st_encoder" not in st.session_state:
    st.session_state.st_encoder = ensure_st_encoder()

# ========================== Encoder wrappers ==========================
@st.cache_data(show_spinner=False)
def encode_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    M = st.session_state.st_encoder.encode(
        texts, batch_size=batch_size, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)
    return M

@st.cache_data(show_spinner=False)
def encode_query(q: str) -> np.ndarray:
    v = st.session_state.st_encoder.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-9)
    return v

# ========================== Dados Sphera & Dicionários ==========================
SPH_EMB_PATH = os.path.join(AN_DIR, "sphera_embeddings.npz")
SPH_PQ_PATH  = os.path.join(AN_DIR, "sphera.parquet")

E_sph = load_npz_embeddings(SPH_EMB_PATH)
df_sph = None
if os.path.exists(SPH_PQ_PATH):
    try:
        df_sph = pd.read_parquet(SPH_PQ_PATH)
    except Exception as e:
        st.warning(f"Falha ao ler {SPH_PQ_PATH}: {e}")

# Dicionários PT/EN
WS_PT_NPZ, WS_PT_LBL_PARQ   = os.path.join(AN_DIR, "ws_embeddings_pt.npz"),   os.path.join(AN_DIR, "ws_embeddings_pt.parquet")
WS_EN_NPZ, WS_EN_LBL_PARQ   = os.path.join(AN_DIR, "ws_embeddings_en.npz"),   os.path.join(AN_DIR, "ws_embeddings_en.parquet")
PREC_PT_NPZ, PREC_PT_LBL_PARQ = os.path.join(AN_DIR, "prec_embeddings_pt.npz"), os.path.join(AN_DIR, "prec_embeddings_pt.parquet")
PREC_EN_NPZ, PREC_EN_LBL_PARQ = os.path.join(AN_DIR, "prec_embeddings_en.npz"), os.path.join(AN_DIR, "prec_embeddings_en.parquet")
CP_NPZ, CP_LBL_PARQ         = os.path.join(AN_DIR, "cp_embeddings.npz"),      os.path.join(AN_DIR, "cp_labels.parquet")

@st.cache_data(show_spinner=False)
def load_dict_bank(npz_path: str, labels_parquet: str):
    E = load_npz_embeddings(npz_path)
    labels = None
    if os.path.exists(labels_parquet):
        try: labels = pd.read_parquet(labels_parquet)
        except Exception: labels = None
    # fallback ALT_DIR
    if (E is None or labels is None) and ALT_DIR:
        npz_alt = os.path.join(ALT_DIR, os.path.basename(npz_path))
        parq_alt = os.path.join(ALT_DIR, os.path.basename(labels_parquet))
        if E is None and os.path.exists(npz_alt):
            E = load_npz_embeddings(npz_alt)
        if labels is None and os.path.exists(parq_alt):
            try: labels = pd.read_parquet(parq_alt)
            except Exception: labels = None
    if E is None or labels is None or len(labels) != E.shape[0]:
        st.warning(f"[Dicionários] Ausentes ou incompatíveis: {npz_path} / {labels_parquet}")
        return None, None
    return E, labels

@st.cache_data(show_spinner=False)
def select_ws_bank(lang: str):
    if lang == "en" and os.path.exists(WS_EN_NPZ):
        return load_dict_bank(WS_EN_NPZ, WS_EN_LBL_PARQ)
    return load_dict_bank(WS_PT_NPZ, WS_PT_LBL_PARQ)

@st.cache_data(show_spinner=False)
def select_prec_bank(lang: str):
    if lang == "en" and os.path.exists(PREC_EN_NPZ):
        return load_dict_bank(PREC_EN_NPZ, PREC_EN_LBL_PARQ)
    return load_dict_bank(PREC_PT_NPZ, PREC_PT_LBL_PARQ)

@st.cache_data(show_spinner=False)
def select_cp_bank():
    return load_dict_bank(CP_NPZ, CP_LBL_PARQ)

# ========================== Filtros Sphera ==========================

def get_sphera_location_col(df: pd.DataFrame) -> Optional[str]:
    """Preferir LOCATION; fallback para FPSO, Location, FPSO/Unidade, Unidade. NUNCA usar AREA/Setor.
    Se nenhuma existir, retorna None (UI mostrará "N/D")."""
    if df is None:
        return None
    preferred = ["LOCATION", "FPSO", "Location", "FPSO/Unidade", "Unidade"]
    for c in preferred:
        if c in df.columns:
            return c
    return None  # nunca cair para AREA

@st.cache_data(show_spinner=False)
def filter_sphera_by_date(df: pd.DataFrame, years: Optional[int]) -> pd.DataFrame:
    if df is None or years is None or "EVENT_DATE" not in df.columns:
        return df if df is not None else pd.DataFrame()
    d = df.copy()
    d["EVENT_DATE"] = pd.to_datetime(d["EVENT_DATE"], errors="coerce")
    cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=365*years))
    return d[d["EVENT_DATE"] >= cutoff]

@st.cache_data(show_spinner=False)
def apply_advanced_filters(base: pd.DataFrame, desc_contains: str, loc_list: List[str]) -> pd.DataFrame:
    d = base if base is not None else pd.DataFrame()
    # Location
    loc_col = get_sphera_location_col(d)
    if loc_col and loc_list:
        sel = [x.strip() for x in loc_list if x and x.strip()]
        d = d[d[loc_col].astype(str).isin(set(sel))]
    # Description contém (case-insensitive, regex escapado) — coluna correta
    desc_col = "Description" if "Description" in d.columns else ("DESCRIPTION" if "DESCRIPTION" in d.columns else None)
    if desc_col and desc_contains:
        pat = re.escape(desc_contains)
        d = d[d[desc_col].astype(str).str.contains(pat, case=False, na=False, regex=True)]
    return d

# ========================== Similaridade Sphera ==========================

def sphera_similar_to_text(query_text: str, min_sim: float, years: Optional[int], topk: int,
                           df_sph: pd.DataFrame, E_sph: np.ndarray,
                           desc_contains: str, loc_list: List[str]) -> List[Tuple[str, float, pd.Series]]:
    """Retorna lista de (event_id, sim, row) para Sphera, respeitando filtros e limiar de cos-sim."""
    if not query_text or df_sph is None or E_sph is None or E_sph.size == 0:
        return []
    base = df_sph
    if years is not None:
        base = filter_sphera_by_date(base, years)
    base = apply_advanced_filters(base, desc_contains, loc_list)

    # alinhar embeddings pelo índice filtrado (se índice for inteiro). Caso contrário, usar E_sph completo.
    try:
        idx_map = base.index.to_numpy()
        if np.issubdtype(idx_map.dtype, np.integer):
            E_view = E_sph[idx_map, :]
        else:
            raise TypeError
    except Exception:
        E_view = E_sph
        base = df_sph
        if years is not None:
            base = filter_sphera_by_date(base, years)
        base = apply_advanced_filters(base, desc_contains, loc_list)

    qv = encode_query(query_text)
    sims = (E_view @ qv).astype(float)
    ord_idx = np.argsort(-sims)

    id_col = "Event ID" if "Event ID" in base.columns else ("EVENT_NUMBER" if "EVENT_NUMBER" in base.columns else None)

    out = []
    kept = 0
    for i in ord_idx:
        s = float(sims[i])
        if s < min_sim:
            continue
        row = base.iloc[int(i)]
        evid = row.get(id_col, f"row{i}") if id_col else f"row{i}"
        out.append((str(evid), s, row))
        kept += 1
        if kept >= topk:
            break
    return out

# ========================== Agregação — WS / Precursores / CP ==========================

def aggregate_dict_matches_over_hits(
    hits: List[Tuple[str, float, pd.Series]],
    E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
    thr_ws: float, thr_prec: float, thr_cp: float,
    topn_ws: int, topn_prec: int, topn_cp: int,
    agg_mode: str = "max",
    per_event_thr: float = 0.30,
    min_support: int = 1,
) -> Dict[str, List[Tuple[str, float, int]]]:
    """Compara dicionários vs DESCRIPTIONS dos **hits** Sphera. Retorna listas (label, sim, suporte)."""
    if not hits:
        return {"ws": [], "prec": [], "cp": []}

    # Coleta descrições
    descs = []
    for _, _, row in hits:
        descs.append(str(row.get("Description", row.get("DESCRIPTION", ""))).strip())
    descs = [d for d in descs if d]
    if not descs:
        return {"ws": [], "prec": [], "cp": []}

    V_desc = encode_texts(descs, batch_size=32)
    V_desc_T = V_desc.T

    def _score(E_bank, labels_df, thr_global, topn_target):
        if E_bank is None or labels_df is None or len(labels_df) != (E_bank.shape[0] if hasattr(E_bank, "shape") else 0):
            return []
        S = (E_bank @ V_desc_T)  # N_terms x M_events
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

    return {
        "ws":   _score(E_ws,   L_ws,   thr_ws,   topn_ws),
        "prec": _score(E_prec, L_prec, thr_prec, topn_prec),
        "cp":   _score(E_cp,   L_cp,   thr_cp,   topn_cp),
    }

# ========================== Chat / Modelo ==========================

def ollama_chat(messages, model=None, temperature=0.2, stream=False, timeout=120):
    if not (OLLAMA_HOST and (model or OLLAMA_MODEL)):
        raise RuntimeError("Modelo não configurado. Defina OLLAMA_HOST e OLLAMA_MODEL.")
    import requests
    r = requests.post(f"{OLLAMA_HOST}/api/chat", headers=HEADERS_JSON, json={
        "model": model or OLLAMA_MODEL, "messages": messages, "temperature": float(temperature), "stream": bool(stream)
    }, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ========================== Sidebar ==========================
st.sidebar.subheader("Assistente de Prompts")
prompts_bank = load_prompts_md(PROMPTS_MD_PATH)

# Dois combos simultâneos: Texto e Upload
col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    titles_texto = [it["title"] for it in prompts_bank.get("Texto", [])]
    sel_texto = st.selectbox("Texto", options=["(vazio)"] + titles_texto, index=0, key="sel_texto")
with col_p2:
    titles_upload = [it["title"] for it in prompts_bank.get("Upload", [])]
    sel_upload = st.selectbox("Upload", options=["(vazio)"] + titles_upload, index=0, key="sel_upload")

if st.sidebar.button("Carregar no rascunho", use_container_width=True, key="btn_load_prompt"):
    draft = []
    if sel_texto != "(vazio)":
        body = next((it["body"] for it in prompts_bank["Texto"] if it["title"] == sel_texto), "")
        if body: draft.append(body)
    if sel_upload != "(vazio)":
        body = next((it["body"] for it in prompts_bank["Upload"] if it["title"] == sel_upload), "")
        if body: draft.append(body)
    st.session_state.draft_prompt = ("\n\n".join(draft)).strip()
    st.sidebar.success("Modelo(s) carregado(s) no rascunho.")
    st.rerun()

st.sidebar.header("Recuperação – Sphera")
k_sph      = st.sidebar.slider("Top-K Sphera", 1, 100, 20, 1)
thr_sph    = st.sidebar.slider("Limiar Sphera (cos)", 0.0, 1.0, 0.30, 0.01)
apply_tf   = st.sidebar.checkbox("Filtrar últimos N anos", True)
years_back = st.sidebar.slider("N (anos)", 1, 10, 3, 1)

st.sidebar.subheader("Filtros avançados – Sphera")
# LOCATION list (entrada livre separada por ;) para não limitar opções
sph_loc_selected  = st.sidebar.text_input("Filtrar LOCATION (lista ;)", "")
sph_desc_contains = st.sidebar.text_input("Description contém (substring)", "")

st.sidebar.subheader("Agregação sobre eventos recuperados (Sphera)")
agg_mode     = st.sidebar.selectbox("Agregação", ["max", "mean"], index=0)
per_ev_thr   = st.sidebar.slider("Limiar por evento (dicionários)", 0.0, 1.0, 0.30, 0.01)
min_support  = st.sidebar.slider("Suporte mínimo (nº de eventos)", 1, 20, 1, 1)
thr_ws       = st.sidebar.slider("Limiar global WS", 0.0, 1.0, 0.25, 0.01)
thr_prec     = st.sidebar.slider("Limiar global Precursores", 0.0, 1.0, 0.25, 0.01)
thr_cp       = st.sidebar.slider("Limiar global CP", 0.0, 1.0, 0.25, 0.01)
topn_ws      = st.sidebar.slider("Top-N WS", 3, 90, 10, 1)
topn_prec    = st.sidebar.slider("Top-N Precursores", 3, 90, 10, 1)
topn_cp      = st.sidebar.slider("Top-N CP", 3, 90, 10, 1)

st.sidebar.subheader("Upload")
upload_raw_max = st.sidebar.slider("Tamanho máx. de UPLOAD_RAW (chars)", 300, 20000, 2500, 100)

# Utilidades (sem disparo de prompts)
c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("Limpar uploads", use_container_width=True, key="btn_clear_upl"):
        st.session_state.upld_texts = []
        st.session_state.upld_meta  = []
        st.session_state.upld_emb   = None
        st.session_state.pop("last_upload_digest", None)
        st.experimental_rerun()
with c2:
    if st.button("Limpar chat", use_container_width=True, key="btn_clear_chat"):
        st.session_state.chat = []
        st.experimental_rerun()

# ========================== UI central ==========================
st.title("ESO • CHAT (Somente Sphera)")

st.text_area("Conteúdo do prompt", key="draft_prompt", height=180, placeholder="Digite ou carregue um modelo de prompt…")

user_text = st.text_area("Texto de análise (para Sphera)", height=200, placeholder="Cole aqui a descrição/evento a analisar…")

uploaded = st.file_uploader("Anexar arquivo (opcional)", type=["txt","md","pdf","docx","csv","xlsx"])
if uploaded is not None:
    raw = uploaded.read()
    # leitura básica (sem libs pesadas) — trata como texto bruto/csv simples
    try:
        as_text = raw.decode("utf-8", errors="ignore")
    except Exception:
        as_text = ""
    if as_text:
        if len(as_text) > upload_raw_max:
            as_text = as_text[:upload_raw_max]
        st.session_state.upld_texts.append(as_text)
        st.success(f"Upload recebido: {uploaded.name} (armazenado no contexto local).")

col_run1, col_run2 = st.columns([1,1])
go_btn      = col_run1.button("Enviar para o chat", type="primary", use_container_width=True, key="btn_send")
clear_draft = col_run2.button("Limpar rascunho", use_container_width=True, key="btn_clear_draft")
if clear_draft:
    st.session_state.draft_prompt = ""
    st.experimental_rerun()

# ========================== Execução ==========================

def render_hits_table(hits: List[Tuple[str, float, pd.Series]], df_all: Optional[pd.DataFrame]) -> str:
    if not hits:
        return ""
    lines = ["| Event ID | Similaridade | LOCATION | Descrição |", "|---|---:|---|---|"]
    loc_col = get_sphera_location_col(df_all) if df_all is not None else None
    for evid, s, row in hits[:min(10, len(hits))]:
        loc_val = str(row.get(loc_col, "N/D")) if loc_col else "N/D"
        desc    = str(row.get("Description", row.get("DESCRIPTION", ""))).replace("\n", " ").strip()[:240]
        lines.append(f"| {evid} | {s:.3f} | {loc_val} | {desc} |")
    return "\n".join(lines)


def push_model(messages: List[Dict[str, str]], pergunta: str, contexto_md: str):
    # Injeta contexto calculado pelo app como apoio, sem impor formato fixo
    messages.append({"role": "user", "content": "DADOS DE APOIO (não responda aqui):\n" + contexto_md})
    qt = pergunta or st.session_state.draft_prompt or "Analise os dados fornecidos e sintetize as lições."
    messages.append({"role": "user", "content": f"Pergunta: {qt}"})
    try:
        resp = ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2, stream=False)
        content = ""
        if isinstance(resp, dict):
            content = resp.get("message", {}).get("content", "") or resp.get("content", "")
        if not content:
            content = "(Sem conteúdo do modelo)"
        with st.chat_message("assistant"):
            st.markdown(content)
        st.session_state.chat.append({"role": "assistant", "content": content})
    except Exception as e:
        st.error(f"Falha ao consultar modelo: {e}")

if go_btn:
    # 1) Monta blocos do usuário (rascunho + texto + uploads)
    blocks = []
    if st.session_state.draft_prompt.strip():
        blocks.append("PROMPT:\n" + st.session_state.draft_prompt.strip())
    if (user_text or "").strip():
        blocks.append("TEXTO:\n" + user_text.strip())
    for i, t in enumerate(st.session_state.upld_texts or []):
        blocks.append(f"UPLOAD[{i+1}]:\n" + t.strip())

    messages = [{"role": "system", "content": st.session_state.system_prompt}]
    messages.append({"role": "user", "content": "\n\n".join(blocks) if blocks else "Sem prompt/texto. Explique como devo proceder."})

    # 2) Busca Sphera (somente Sphera)
    loc_list = [x.strip() for x in sph_loc_selected.split(";")] if sph_loc_selected.strip() else []
    hits = sphera_similar_to_text(
        query_text=(user_text or st.session_state.draft_prompt),
        min_sim=thr_sph,
        years=(years_back if apply_tf else None),
        topk=k_sph,
        df_sph=df_sph,
        E_sph=E_sph,
        desc_contains=sph_desc_contains,
        loc_list=loc_list,
    )

    # 3) Renderiza hits
    table_md = ""
    if hits:
        table_md = render_hits_table(hits, df_sph)
        st.markdown("**Eventos do Sphera (Top-10)**\n\n" + table_md)
        st.session_state.chat.append({"role": "assistant", "content": "Eventos Sphera listados."})
    else:
        st.info("Nenhum evento do Sphera atingiu o limiar de similaridade com os filtros atuais.")

    # 4) Dicionários sobre os hits
    # Heurística simples de idioma (PT/EN) — preferir PT
    lang = "pt"
    E_ws,   L_ws   = select_ws_bank(lang)
    E_prec, L_prec = select_prec_bank(lang)
    E_cp,   L_cp   = select_cp_bank()

    dict_matches = aggregate_dict_matches_over_hits(
        hits, E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
        thr_ws=thr_ws, thr_prec=thr_prec, thr_cp=thr_cp,
        topn_ws=topn_ws, topn_prec=topn_prec, topn_cp=topn_cp,
        agg_mode=agg_mode, per_event_thr=per_ev_thr, min_support=min_support,
    )

    if hits:
        md2 = []
        # WS
        ws = dict_matches.get("ws") or []
        md2 += ["**WS (≥ limiar, calculado no app)**"]
        if ws:
            md2 += ["| Rank | Termo | Similaridade | Suporte |", "|---:|---|---:|---:|"]
            for r, (label, s, sup) in enumerate(ws, 1):
                md2.append(f"| {r} | {label} | {s:.3f} | {sup} |")
        else:
            md2 += ["Nenhum WS ≥ limiar."]
        # Precursores
        prec = dict_matches.get("prec") or []
        md2 += ["", "**Precursores (≥ limiar, calculado no app)**"]
        if prec:
            md2 += ["| Rank | Termo | Similaridade | Suporte |", "|---:|---|---:|---:|"]
            for r, (label, s, sup) in enumerate(prec, 1):
                md2.append(f"| {r} | {label} | {s:.3f} | {sup} |")
        else:
            md2 += ["Nenhum Precursor ≥ limiar."]
        # CP
        cp = dict_matches.get("cp") or []
        md2 += ["", "**CP (≥ limiar, calculado no app)**"]
        if cp:
            md2 += ["| Rank | Fator | Similaridade | Suporte |", "|---:|---|---:|---:|"]
            for r, (label, s, sup) in enumerate(cp, 1):
                md2.append(f"| {r} | {label} | {s:.3f} | {sup} |")
        else:
            md2 += ["Nenhum Fator CP ≥ limiar."]
        st.markdown("\n".join(md2))

    # 5) Síntese pelo modelo (sem heurística fixa)
    ctx_chunks = [
        f"Sphera_hits={len(hits)}, thr_sph={thr_sph:.2f}, years={'all' if not apply_tf else years_back}",
    ]
    if hits and table_md:
        ctx_chunks.append("HITS_TOP10_MD:\n" + table_md)
        def _b(lst, name):
            if not lst:
                return f"{name}: nenhum ≥ limiar"
            rows = [f"- {lab} (sim={s:.3f}, sup={sup})" for lab, s, sup in lst[:10]]
            return name + ":\n" + "\n".join(rows)
        ctx_chunks.append(_b(dict_matches.get("ws"),   "WS selecionados"))
        ctx_chunks.append(_b(dict_matches.get("prec"), "Precursores selecionados"))
        ctx_chunks.append(_b(dict_matches.get("cp"),   "CP selecionados"))

    model_context = "\n\n".join(ctx_chunks)
    push_model([{ "role": "system", "content": st.session_state.system_prompt }], user_text, model_context)

# ========================== Histórico ==========================
if st.session_state.chat:
    st.divider()
    st.subheader("Histórico")
    for m in st.session_state.chat[-10:]:
        role = m.get("role","assistant")
        with st.chat_message("assistant" if role != "user" else "user"):
            st.markdown(m.get("content",""))
