# -*- coding: utf-8 -*- v. 05/11/2025 - 19 hs
"""
app_chat_novo_correto.py — Sphera + RAG + DIC (WS/Precursores/CP)

Patches desta versão:
- Reativado filtro de Location (multiselect na sidebar) com opções derivadas do Sphera.
- Filtro de Location aplicado em filter_sphera(...) e passado para sphera_similar_to_text(...).
- OCR removido (mantida extração de texto nativa para PDF; upload aceita txt, md, csv, pdf, docx, xlsx).
- Demais funcionalidades preservadas (prompts, contexto datasets, limpeza de estado, WS/Prec/CP, depuração).

"""

import os
import re
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ========================== Config ==========================
st.set_page_config(page_title="SAFETY • CHAT", page_icon="💬", layout="wide")

DATA_DIR = Path("data")
AN_DIR   = DATA_DIR / "analytics"
XLSX_DIR = DATA_DIR / "xlsx"
DATASETS_CONTEXT_PATH = DATA_DIR / "datasets_context.md"
PROMPTS_MD_PATH       = DATA_DIR / "prompts" / "prompts.md"

SPH_PQ_PATH  = AN_DIR / "sphera.parquet"
SPH_NPZ_PATH = AN_DIR / "sphera_embeddings.npz"
XLSX_LOCATION_PATH = XLSX_DIR / "TRATADO_safeguardOffShore.xlsx"

OLLAMA_HOST    = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", ""))
OLLAMA_MODEL   = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", ""))
OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY"))
HEADERS_JSON   = {"Authorization": f"Bearer {OLLAMA_API_KEY}", "Content-Type": "application/json"} if OLLAMA_API_KEY else {"Content-Type": "application/json"}

ST_MODEL_NAME = os.getenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# ========================== Helpers ==========================

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
def load_npz_embeddings(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        with np.load(str(path), allow_pickle=True) as z:
            for key in ("embeddings", "E", "X", "vectors", "vecs", "arr_0"):
                if key in z:
                    E = np.array(z[key]).astype(np.float32, copy=False)
                    n = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
                    return (E / n).astype(np.float32)
            # fallback: maior matriz 2D
            best_k, best_n = None, -1
            for k in z.files:
                arr = z[k]
                if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] > best_n:
                    best_k, best_n = k, arr.shape[0]
            if best_k is None:
                return None
            E = np.array(z[best_k]).astype(np.float32, copy=False)
            n = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
            return (E / n).astype(np.float32)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_prompts_md(md_path: Path) -> Dict[str, List[Dict[str, str]]]:
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
            if not p:
                continue
            title, _, body = p.partition("\n")
            data[first.strip()].append({"title": title.strip(), "body": body.strip()})
    return data

@st.cache_data(show_spinner=False)
def load_file_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"[AVISO] Não consegui ler {p}: {e} (continuando sem este contexto)"

# ========================== Helpers (Location) ==========================

def get_sphera_location_col(df: pd.DataFrame) -> Optional[str]:
    """Retorna a melhor coluna de localização, nunca AREA.
    Prioridade: LOCATION → FPSO → Location → FPSO/Unidade → Unidade."""
    if df is None or df.empty:
        return None
    for c in ["LOCATION", "FPSO", "Location", "FPSO/Unidade", "Unidade"]:
        if c in df.columns and df[c].notna().any():
            return c
    return None

@st.cache_data(show_spinner=False)
def _location_options_from(df_full: pd.DataFrame) -> Tuple[Optional[str], List[str]]:
    col = get_sphera_location_col(df_full)
    if not col:
        return None, []
    s = df_full[col].astype(str).str.strip()
    s = s[(~s.isna()) & (s.str.len() > 0)]
    bad = {"nan", "none", "n/d", "nd"}
    s = s[~s.str.lower().isin(bad)]
    # de-duplicar preservando primeira grafia
    seen = {}
    for v in s:
        k = v.lower()
        if k not in seen:
            seen[k] = v
    return col, sorted(seen.values())

# ========================== Carregamento de dados ==========================
if not SPH_PQ_PATH.exists():
    st.error(f"Parquet do Sphera não encontrado em {SPH_PQ_PATH}")

df_sph = pd.read_parquet(SPH_PQ_PATH) if SPH_PQ_PATH.exists() else pd.DataFrame()
E_sph  = load_npz_embeddings(SPH_NPZ_PATH)
# coluna exibida para Location
LOC_DISPLAY_COL = get_sphera_location_col(df_sph)

# --- WS/Precursores ---
WS_NPZ,   WS_LBL   = AN_DIR / "ws_embeddings_pt.npz",   AN_DIR / "ws_embeddings_pt.parquet"
PREC_NPZ, PREC_LBL = AN_DIR / "prec_embeddings_pt.npz", AN_DIR / "prec_embeddings_pt.parquet"
E_ws   = load_npz_embeddings(WS_NPZ) if WS_NPZ.exists() else None
L_ws   = (pd.read_parquet(WS_LBL) if WS_LBL.exists() else None)
E_prec = load_npz_embeddings(PREC_NPZ) if PREC_NPZ.exists() else None
L_prec = (pd.read_parquet(PREC_LBL) if PREC_LBL.exists() else None)

# --- CP (loader robusto com fallbacks) ---
CP_NPZ_MAIN   = AN_DIR / "cp_embeddings.npz"
CP_NPZ_ALT    = AN_DIR / "cp_vectors.npz"        # fallback
CP_LBL_PARQ   = AN_DIR / "cp_labels.parquet"
CP_LBL_JSONL  = AN_DIR / "cp_labels.jsonl"       # fallback

@st.cache_data(show_spinner=False)
def _load_npz_any(path: Path):
    if not path.exists():
        return None
    try:
        with np.load(str(path), allow_pickle=True) as z:
            for k in ("embeddings", "E", "X", "vectors", "vecs", "arr_0"):
                if k in z:
                    A = np.array(z[k])
                    if isinstance(A, np.ndarray) and A.ndim == 2 and A.shape[0] > 0:
                        A = A.astype(np.float32, copy=False)
                        A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
                        return A
            # maior matriz 2D
            best = None
            for k in z.files:
                A = z[k]
                if isinstance(A, np.ndarray) and A.ndim == 2 and (best is None or A.shape[0] > best.shape[0]):
                    best = A
            if best is not None:
                A = best.astype(np.float32, copy=False)
                A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
                return A
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False)
def _load_cp_labels() -> Optional[pd.DataFrame]:
    df = None
    if CP_LBL_PARQ.exists():
        try:
            df = pd.read_parquet(CP_LBL_PARQ)
        except Exception:
            df = None
    if df is None and CP_LBL_JSONL.exists():
        try:
            df = pd.read_json(CP_LBL_JSONL, lines=True)
        except Exception:
            df = None
    if df is None:
        return None
    label_col = next((c for c in ["label","LABEL","text","name","CP","cp"] if c in df.columns), None)
    if not label_col:
        return None
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})
    return df[["label"]] if "label" in df.columns else None

E_cp = _load_npz_any(CP_NPZ_MAIN)
if E_cp is None:
    E_cp = _load_npz_any(CP_NPZ_ALT)

L_cp = _load_cp_labels()

if E_cp is None or L_cp is None:
    st.warning("[Dicionários/CP] Não foi possível carregar completamente os dados de CP. Continuando sem CP.")
else:
    if L_cp.shape[0] != E_cp.shape[0]:
        st.warning(f"[Dicionários/CP] Desalinhamento: labels={L_cp.shape[0]} vs embeddings={E_cp.shape[0]}. Ignorando CP para evitar inconsistências.")
        E_cp, L_cp = None, None

# ========================== Estado ==========================
if "system_prompt" not in st.session_state:
    pre = (
        "Você é o ESO-CHAT para segurança operacional (óleo e gás). "
        "Responda em PT-BR, cite IDs/sim quando usar buscas locais, e não invente dados fora dos contextos fornecidos.\n\n"
    )
    sys_ctx = (load_file_text(DATASETS_CONTEXT_PATH) if DATASETS_CONTEXT_PATH.exists() else "")
    st.session_state.system_prompt = pre + ("=== DATASETS_CONTEXT ===\n" + sys_ctx if sys_ctx else "")
if "chat" not in st.session_state:
    st.session_state.chat = []
if "draft_prompt" not in st.session_state:
    st.session_state.draft_prompt = ""
if "_clear_draft_flag" not in st.session_state:
    st.session_state._clear_draft_flag = False
if "st_encoder" not in st.session_state:
    st.session_state.st_encoder = ensure_st_encoder()
if "upld_texts" not in st.session_state:
    st.session_state.upld_texts = []

# ========================== Encode ==========================
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

# ========================== Filtros / Similaridade ==========================
@st.cache_data(show_spinner=False)
def filter_sphera(df: pd.DataFrame, locations: List[str], substr: str, years: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()

    # Janela temporal
    if "EVENT_DATE" in out.columns:
        out["EVENT_DATE"] = pd.to_datetime(out["EVENT_DATE"], errors="coerce")
        cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=365 * years))
        out = out[out["EVENT_DATE"] >= cutoff]

    # Filtro por Location (string exata, preservando grafia exibida)
    loc_col = get_sphera_location_col(out)
    if loc_col and locations:
        selected = set([str(x).strip() for x in locations if str(x).strip()])
        out = out[out[loc_col].astype(str).isin(selected)]

    # Description contém (case-insensitive)
    desc_col = "Description" if "Description" in out.columns else ("DESCRIPTION" if "DESCRIPTION" in out.columns else None)
    if desc_col and substr:
        pat = re.escape(substr)
        out = out[out[desc_col].astype(str).str.contains(pat, case=False, na=False, regex=True)]

    return out

@st.cache_data(show_spinner=False)
def sphera_similar_to_text(query_text: str, min_sim: float, years: int, topk: int,
                           df_base: pd.DataFrame, E_base: Optional[np.ndarray],
                           substr: str, locations: List[str]) -> List[Tuple[str, float, pd.Series]]:
    if not query_text or df_base is None or df_base.empty or E_base is None or E_base.size == 0:
        return []
    base = filter_sphera(df_base, locations, substr, years)
    if base.empty:
        return []
    try:
        idx_map = base.index.to_numpy()
        if np.issubdtype(idx_map.dtype, np.integer):
            E_view = E_base[idx_map, :]
        else:
            E_view = E_base
            base = df_base
    except Exception:
        E_view = E_base
        base = df_base
    qv = encode_query(query_text)
    sims = (E_view @ qv).astype(float)
    ord_idx = np.argsort(-sims)
    id_col = "Event ID" if "Event ID" in base.columns else ("EVENT_NUMBER" if "EVENT_NUMBER" in base.columns else ("EVENTID" if "EVENTID" in base.columns else None))
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

# ========================== Agregação dicionários ==========================
@st.cache_data(show_spinner=False)
def aggregate_dict_matches_over_hits(
    hits: List[Tuple[str, float, pd.Series]],
    E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
    thr_ws_sim: float, thr_prec_sim: float, thr_cp_sim: float,
    topn_ws: int, topn_prec: int, topn_cp: int,
    agg_mode: str = "max",
    per_event_thr: float = 0.30,
    min_support: int = 1,
) -> Dict[str, List[Tuple[str, float, int]]]:
    if not hits:
        return {"ws": [], "prec": [], "cp": []}
    descs = [str(r.get("Description", r.get("DESCRIPTION", "")).strip()) for _, _, r in hits]
    descs = [d for d in descs if d]
    if not descs:
        return {"ws": [], "prec": [], "cp": []}
    V_desc = encode_texts(descs, batch_size=32).T  # (D x M)

    def _score(E_bank, labels_df, thr_sim, topn_target):
        if E_bank is None or labels_df is None or len(labels_df) != E_bank.shape[0]:
            return []
        S = (E_bank @ V_desc)                 # (N_terms, M_events)
        support = (S >= per_event_thr).sum(axis=1)
        sims = S.mean(axis=1) if agg_mode == "mean" else S.max(axis=1)
        mask = (support >= min_support) & (sims >= thr_sim)
        idx = np.where(mask)[0]
        if idx.size == 0:
            return []
        order = idx[np.argsort(sims[idx])[::-1]]
        out_terms = []
        for i in order[:topn_target]:
            label = str(labels_df.iloc[i].get("label", labels_df.iloc[i].get("text", f"TERM_{i}")))
            out_terms.append((label, float(sims[i]), int(support[i])))
        return out_terms

    return {
        "ws":   _score(E_ws,   L_ws,   thr_ws_sim,   topn_ws),
        "prec": _score(E_prec, L_prec, thr_prec_sim, topn_prec),
        "cp":   _score(E_cp,   L_cp,   thr_cp_sim,   topn_cp),
    }

# ===== Depuração: Top-N "brutos" (ignora thresholds) =====
def _topk_raw_for_bank(E_bank: np.ndarray, labels_df: pd.DataFrame, V_desc_T: np.ndarray, topk: int = 10):
    if E_bank is None or labels_df is None or (len(labels_df) != (E_bank.shape[0] if hasattr(E_bank,'shape') else 0)) or V_desc_T is None:
        return pd.DataFrame()
    S = (E_bank @ V_desc_T)
    sims = S.max(axis=1)
    order = np.argsort(sims)[::-1][:topk]
    labels_col = next((c for c in ["label","text","name","CP","cp"] if c in labels_df.columns), None)
    rows = []
    for i in order:
        lab = str(labels_df.iloc[i].get(labels_col, f"TERM_{i}"))
        rows.append({"Termo": lab, "Similaridade(max)": float(sims[i])})
    return pd.DataFrame(rows)

def debug_preview_dicts(hits, E_ws, L_ws, E_prec, L_prec, E_cp, L_cp, topk=10):
    if not hits:
        st.info("Sem hits Sphera para depuração de dicionários.")
        return
    descs = [str(r.get("Description", r.get("DESCRIPTION",""))).strip() for _,_,r in hits]
    descs = [d for d in descs if d]
    if not descs:
        st.info("Sem descrições válidas para depuração.")
        return
    V_desc = encode_texts(descs, batch_size=32).T  # (D x M)
    with st.expander("🔎 Depuração — Top-N brutos (ignora thresholds)", expanded=False):
        if E_ws is not None and L_ws is not None:
            st.markdown("**WS (max entre eventos, sem limiares)**")
            st.dataframe(_topk_raw_for_bank(E_ws, L_ws, V_desc, topk), use_container_width=True, hide_index=True)
        if E_prec is not None and L_prec is not None:
            st.markdown("**Precursores (max entre eventos, sem limiares)**")
            st.dataframe(_topk_raw_for_bank(E_prec, L_prec, V_desc, topk), use_container_width=True, hide_index=True)
        if E_cp is not None and L_cp is not None:
            st.markdown("**CP (max entre eventos, sem limiares)**")
            st.dataframe(_topk_raw_for_bank(E_cp, L_cp, V_desc, topk), use_container_width=True, hide_index=True)

# ===== Hints por evento =====
def build_event_hints(
    hits: List[Tuple[str, float, pd.Series]],
    E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
    per_event_thr: float,
    top_per_family: int = 3,
) -> Tuple[str, np.ndarray | None]:
    if not hits:
        return "=== EVENT_HINTS === [NENHUM HIT]", None
    descs = [str(r.get("Description", r.get("DESCRIPTION",""))).strip() for _,_,r in hits]
    descs = [d for d in descs if d]
    if not descs:
        return "=== EVENT_HINTS === [SEM DESCRIÇÕES VÁLIDAS]", None
    V_desc = encode_texts(descs, batch_size=32).T  # (D x M)
    def _labels_col(df):
        return next((c for c in ["label","text","name","CP","cp"] if (df is not None and c in df.columns)), None)
    lines = ["=== EVENT_HINTS ==="]
    families = [("WS",E_ws,L_ws), ("PRE",E_prec,L_prec), ("CP",E_cp,L_cp)]
    M = V_desc.shape[1]
    for ev_idx, (evid, _, row) in enumerate(hits[:M]):
        ev_terms = []
        for fam_name, E_bank, L_bank in families:
            if E_bank is None or L_bank is None or len(L_bank) != E_bank.shape[0]:
                continue
            S = (E_bank @ V_desc[:, [ev_idx]]).squeeze(axis=1)
            idx = np.where(S >= per_event_thr)[0]
            if idx.size == 0:
                continue
            order = idx[np.argsort(S[idx])[::-1]][:top_per_family]
            labcol = _labels_col(L_bank)
            fam_lines = []
            for i in order:
                lab = str(L_bank.iloc[i].get(labcol, f"TERM_{i}"))
                fam_lines.append(f"{lab} (sim={float(S[i]):.3f})")
            if fam_lines:
                ev_terms.append(f"{fam_name}: " + "; ".join(fam_lines))
        lines.append(f"[EventID={evid}] " + (" | ".join(ev_terms) if ev_terms else "—"))
    return "".join(lines) + "", V_desc

# ===== Atribuição determinística por evento =====
def assign_terms_per_event(
    hits: List[Tuple[str, float, pd.Series]],
    V_desc: Optional[np.ndarray],
    E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
    per_event_thr: float,
    max_per_event: int = 2,
    max_global_frac: float = 0.5,
) -> str:
    if not hits or V_desc is None:
        return "=== EVENT_ASSIGNMENTS === [NENHUM HIT]"
    families = [("WS",E_ws,L_ws), ("PRE",E_prec,L_prec), ("CP",E_cp,L_cp)]
    def _labels_col(df):
        return next((c for c in ["label","text","name","CP","cp"] if (df is not None and c in df.columns)), None)
    M = V_desc.shape[1]
    candidates, label_pool = [], set()
    for ev_idx in range(min(M, len(hits))):
        ev_cands = []
        for fam_name, E_bank, L_bank in families:
            if E_bank is None or L_bank is None or len(L_bank) != E_bank.shape[0]:
                continue
            S = (E_bank @ V_desc[:, [ev_idx]]).squeeze(axis=1)
            idx = np.where(S >= per_event_thr)[0]
            if idx.size == 0:
                continue
            order = idx[np.argsort(S[idx])[::-1]]
            labcol = _labels_col(L_bank)
            for i in order[:10]:
                lab = str(L_bank.iloc[i].get(labcol, f"TERM_{i}"))
                ev_cands.append((lab, float(S[i]), fam_name))
                label_pool.add(lab)
        ev_cands.sort(key=lambda t: t[1], reverse=True)
        candidates.append(ev_cands)
    n_events = len(candidates)
    max_global = max(1, int(np.ceil(max_global_frac * n_events)))
    used_count: Dict[str, int] = {lab: 0 for lab in label_pool}
    lines = ["=== EVENT_ASSIGNMENTS ==="]
    for ev_idx, (evid, _, _row) in enumerate(hits[:n_events]):
        picked, seen_fams = [], set()
        for lab, sim, fam in candidates[ev_idx]:
            if used_count.get(lab, 0) >= max_global:
                continue
            if fam in seen_fams and len(seen_fams) < 3:
                continue
            picked.append((lab, sim, fam))
            used_count[lab] = used_count.get(lab, 0) + 1
            seen_fams.add(fam)
            if len(picked) >= max_per_event:
                break
        lines.append(f"[EventID={evid}] " + ("; ".join([p[0] for p in picked]) if picked else "—"))
    return "".join(lines) + ""

# ========================== Modelo ==========================
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

col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    titles_texto = [it["title"] for it in prompts_bank.get("Texto", [])]
    sel_texto = st.selectbox("Texto", options=["(vazio)"] + titles_texto, index=0)
with col_p2:
    titles_upload = [it["title"] for it in prompts_bank.get("Upload", [])]
    sel_upload = st.selectbox("Upload", options=["(vazio)"] + titles_upload, index=0)

if st.sidebar.button("Carregar no rascunho", use_container_width=True):
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
k_sph   = st.sidebar.slider("Top-K Sphera", 1, 100, 20, 1)
thr_sph = st.sidebar.slider("Limiar Sphera (cos)", 0.0, 1.0, 0.30, 0.01)
years   = st.sidebar.slider("Últimos N anos", 1, 10, 3, 1)

st.sidebar.subheader("Filtros avançados – Sphera")
# NOVO: multiselect de Location
_loc_col_sidebar, _loc_options = _location_options_from(df_sph)
locations = st.sidebar.multiselect(
    f"Location (coluna: {_loc_col_sidebar or 'N/D'})",
    options=_loc_options,
    default=[],
    help="Opções extraídas do Sphera (LOCATION → FPSO → Location → FPSO/Unidade → Unidade)."
)
substr = st.sidebar.text_input("Description contém (substring)", "")

st.sidebar.subheader("Agregação sobre eventos recuperados (Sphera)")
agg_mode    = st.sidebar.selectbox("Agregação", ["max", "mean"], index=0)
per_ev_thr  = st.sidebar.slider("Limiar por evento (dicionários)", 0.0, 1.0, 0.15, 0.01)  # default ajustado
min_support = st.sidebar.slider("Suporte mínimo (nº de eventos)", 1, 20, 1, 1)

thr_ws_sim   = st.sidebar.slider("Limiar de similaridade WS",        0.0, 1.0, 0.25, 0.01)
thr_prec_sim = st.sidebar.slider("Limiar de similaridade Precursor", 0.0, 1.0, 0.25, 0.01)
thr_cp_sim   = st.sidebar.slider("Limiar de similaridade CP",        0.0, 1.0, 0.25, 0.01)

topn_ws   = st.sidebar.slider("Top-N WS",          3, 90, 10, 1)
topn_prec = st.sidebar.slider("Top-N Precursores", 3, 90, 10, 1)
topn_cp   = st.sidebar.slider("Top-N CP",          3, 90, 10, 1)

uc1, uc2 = st.sidebar.columns(2)
with uc1:
    if st.button("Limpar uploads", use_container_width=True):
        st.session_state.pop("upld_texts", None)
        st.session_state.upld_texts = []
        st.rerun()
with uc2:
    if st.button("Limpar chat", use_container_width=True):
        st.session_state.chat = []
        st.rerun()

# ========================== UI central ==========================
if st.session_state._clear_draft_flag:
    st.session_state.draft_prompt = ""
    st.session_state._clear_draft_flag = False

st.title("SAFETY • CHAT (Somente Sphera)")

st.text_area("Conteúdo do prompt", key="draft_prompt", height=180, placeholder="Digite ou carregue um modelo de prompt…")
user_text = st.text_area("Texto de análise (para Sphera)", height=200, placeholder="Cole aqui a descrição/evento a analisar…")

# ---------- Upload (txt, md, csv, pdf, docx, xlsx) ----------
uploaded = st.file_uploader(
    "Anexar arquivo (opcional)",
    type=["txt", "md", "csv", "pdf", "docx", "xlsx"]
)  # upload não dispara

def extract_pdf_text(file_like: io.BytesIO) -> str:
    """
    Extrai texto de PDF. Tenta PyPDF2 -> PyMuPDF (fitz) -> pdfminer.six.
    Retorna string (pode ser vazia se o PDF for apenas imagem/scaneado).
    """
    # 1) PyPDF2
    try:
        import PyPDF2
        file_like.seek(0)
        reader = PyPDF2.PdfReader(file_like)
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts).strip()
    except Exception:
        pass
    # 2) PyMuPDF
    try:
        import fitz  # PyMuPDF
        file_like.seek(0)
        doc = fitz.open(stream=file_like.read(), filetype="pdf")
        parts = [page.get_text() for page in doc]
        return "\n".join(parts).strip()
    except Exception:
        pass
    # 3) pdfminer.six
    try:
        from pdfminer.high_level import extract_text
        file_like.seek(0)
        return (extract_text(file_like) or "").strip()
    except Exception:
        pass
    return ""

def extract_docx_text(file_like: io.BytesIO) -> str:
    """Extrai texto de um .docx (python-docx)."""
    try:
        from docx import Document
        file_like.seek(0)
        doc = Document(file_like)
        parts = [p.text for p in doc.paragraphs if p.text]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" ".join(cell.text for cell in row.cells if cell.text))
        return "\n".join(parts).strip()
    except Exception:
        return ""

def extract_xlsx_text(file_like: io.BytesIO) -> str:
    """Extrai texto de um .xlsx (pandas + openpyxl)."""
    try:
        file_like.seek(0)
        sheets = pd.read_excel(file_like, sheet_name=None, engine="openpyxl")
        lines = []
        for name, df in sheets.items():
            if df is None or df.empty:
                continue
            df = df.astype(str).fillna("")
            lines.append(f"=== SHEET: {name} ===")
            lines.extend(df.apply(lambda r: " ".join(r.values), axis=1).tolist())
        return "\n".join(lines).strip()
    except Exception:
        return ""

if uploaded is not None:
    raw = uploaded.read()
    name = uploaded.name.lower()
    as_text = ""

    if name.endswith(".pdf"):
        as_text = extract_pdf_text(io.BytesIO(raw))
    elif name.endswith(".docx"):
        as_text = extract_docx_text(io.BytesIO(raw))
    elif name.endswith(".xlsx"):
        as_text = extract_xlsx_text(io.BytesIO(raw))
    else:
        # textos puros e CSV
        try:
            as_text = raw.decode("utf-8", errors="ignore")
        except Exception:
            as_text = ""
        if name.endswith(".csv") and as_text:
            try:
                dfcsv = pd.read_csv(io.StringIO(as_text))
                as_text = "\n".join(
                    dfcsv.astype(str).fillna("")
                    .apply(lambda r: " ".join(r.values), axis=1)
                    .tolist()
                )
            except Exception:
                pass

    if as_text:
        st.success(f"Upload recebido: {uploaded.name} (armazenado no contexto local).")
        st.session_state.upld_texts.append(as_text)
    else:
        st.warning(
            f"Não foi possível extrair texto de {uploaded.name}. "
            "Se for PDF escaneado (imagem), poderá exigir OCR externo."
        )
# ---------- /Upload ----------

col_run1, col_run2, col_run3 = st.columns([1, 1, 1])
go_btn      = col_run1.button("Enviar para o chat", type="primary", use_container_width=True)
clear_draft = col_run2.button("Limpar rascunho", use_container_width=True)
clear_chat  = col_run3.button("Limpar chat", use_container_width=True)

if clear_draft:
    st.session_state._clear_draft_flag = True
    st.rerun()
if clear_chat:
    st.session_state.chat = []
    st.rerun()

# ========================== Execução ==========================

def render_hits_table(hits: List[Tuple[str, float, pd.Series]], topk_display: int):
    if not hits:
        return
    rows = []
    for evid, s, row in hits[: min(topk_display, len(hits))]:
        loc_val = (str(row.get(LOC_DISPLAY_COL, row.get('LOCATION', 'N/D'))) if 'LOC_DISPLAY_COL' in globals() and LOC_DISPLAY_COL else str(row.get('LOCATION','N/D')))
        desc    = str(row.get("Description", row.get("DESCRIPTION", ""))).strip()
        # normalizar quebras de linha e artefatos _x000D_
        desc = desc.replace("\r", " ").replace("\n", " ").replace("_x000D_", " ")
        desc = re.sub(r"\s+", " ", desc).strip()
        rows.append({"Event ID": evid, "Similaridade": round(s, 3), "LOCATION": loc_val, "Description": desc})
    st.markdown(f"**Eventos do Sphera (Top-{min(topk_display, len(hits))})**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def push_model(messages: List[Dict[str, str]], pergunta: str, contexto_md: str, dic_matches_md: str):
    # Guardrails para impedir invenção de termos fora dos dicionários
    guardrails = (
        "REGRAS PARA WS/PRECURSORES/CP:\n"
        "- Use EXCLUSIVAMENTE os termos listados em DIC_MATCHES para nomear WS/Precursores/CP.\n"
        "- NÃO crie categorias novas nem traduza/alterar rótulos.\n"
        "- Se a lista estiver vazia, escreva 'nenhum termo ≥ limiar'.\n"
        "- Quando citar um termo, mantenha o rótulo exatamente como fornecido.\n"
        "\n"
        "REGRAS PARA A COLUNA 'Observações/Precursores relevantes':\n"
        "- Para cada EventID, escolha no máximo 2 termos dentre os sugeridos em EVENT_HINTS (quando houver).\n"
        "- Não repita o mesmo termo em mais de 50% dos eventos; prefira diversidade baseada em EVENT_HINTS.\n"
        "- Se um evento não tiver termos em EVENT_HINTS, use '—' nessa coluna (não invente).\n"
    )

    messages.append({"role": "user", "content": "DADOS DE APOIO (não responda aqui):\n" + contexto_md + "\n\n" + dic_matches_md})
    q = pergunta or st.session_state.draft_prompt or "Faça a síntese conforme regras."
    messages.append({"role": "user", "content": guardrails + "\nPergunta/objetivo:\n" + q})

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
        st.session_state["_just_replied"] = True
    except Exception as e:
        st.error(f"Falha ao consultar modelo: {e}")

if go_btn:
    blocks = []
    if st.session_state.draft_prompt.strip():
        blocks.append("PROMPT:\n" + st.session_state.draft_prompt.strip())
    if (user_text or "").strip():
        blocks.append("TEXTO:\n" + user_text.strip())
    for i, t in enumerate(st.session_state.upld_texts or []):
        blocks.append(f"UPLOAD[{i+1}]:\n" + t.strip())

    hits = sphera_similar_to_text(
        query_text=(user_text or st.session_state.draft_prompt),
        min_sim=thr_sph, years=years, topk=k_sph,
        df_base=df_sph, E_base=E_sph, substr=substr, locations=locations,  # <— aplica filtro de Location
    )

    if hits:
        render_hits_table(hits, k_sph)
    else:
        st.info("Nenhum evento do Sphera atingiu o limiar/filtros atuais.")

    dict_matches = aggregate_dict_matches_over_hits(
        hits, E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
        thr_ws_sim=thr_ws_sim, thr_prec_sim=thr_prec_sim, thr_cp_sim=thr_cp_sim,
        topn_ws=topn_ws, topn_prec=topn_prec, topn_cp=topn_cp,
        agg_mode=agg_mode, per_event_thr=per_ev_thr, min_support=min_support,
    )

    # Depuração (opcional): mostra Top-N brutos para confirmar que o espaço vetorial está ok
    debug_preview_dicts(hits, E_ws, L_ws, E_prec, L_prec, E_cp, L_cp, topk=10)

    # Hints por evento (para ancorar as observações do modelo)
    EVENT_HINTS_MD, _Vdesc = build_event_hints(
        hits, E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
        per_event_thr=per_ev_thr,
        top_per_family=3,
    )

    # Bloco estruturado com os termos encontrados para passar ao LLM
    def _fmt_list(name, arr):
        if not arr:
            return f"{name}: NENHUM_TERMO_ACIMA_DO_LIMIAR\n"
        lines = [f"{name}:"]
        for lab, sim, sup in arr:
            lines.append(f"- termo={lab} | sim={sim:.3f} | suporte={sup}")
        return "\n".join(lines) + "\n"

    ws_list   = dict_matches.get("ws")   or []
    prec_list = dict_matches.get("prec") or []
    cp_list   = dict_matches.get("cp")   or []

    DIC_MATCHES_MD = (
        "=== DIC_MATCHES ===\n"
        + _fmt_list("WS", ws_list)
        + _fmt_list("PRECURSORES", prec_list)
        + _fmt_list("CP", cp_list)
    )

    # Contexto Sphera em texto
    table_ctx_rows = []
    for evid, s, row in hits[: min(k_sph, len(hits))]:
        loc_val = (str(row.get(LOC_DISPLAY_COL, row.get('LOCATION', 'N/D'))) if 'LOC_DISPLAY_COL' in globals() and LOC_DISPLAY_COL else str(row.get('LOCATION','N/D')))
        desc    = str(row.get("Description", row.get("DESCRIPTION", ""))).strip()
        desc = desc.replace("\r", " ").replace("\n", " ").replace("_x000D_", " ")
        desc = re.sub(r"\s+", " ", desc).strip()
        table_ctx_rows.append(f"EventID={evid} | sim={s:.3f} | LOCATION={loc_val} | Description={desc}")

    ctx_chunks = [
        f"Sphera_hits={len(hits)}, thr_sph={thr_sph:.2f}, years={years}",
        "\n".join(table_ctx_rows),
        EVENT_HINTS_MD,
    ]

    messages = [
        {"role": "system", "content": st.session_state.system_prompt},
        {"role": "user", "content": "".join([b for b in blocks if b])},
    ]
    ctx_full = "".join([x for x in ctx_chunks if x])
    push_model(messages, user_text, ctx_full, DIC_MATCHES_MD)

# ========================== Histórico ==========================
if st.session_state.get("_just_replied"):
    st.session_state["_just_replied"] = False
else:
    if st.session_state.chat:
        st.divider()
        st.subheader("Histórico")
        for m in st.session_state.chat[-10:]:
            role = m.get("role", "assistant")
            with st.chat_message("assistant" if role != "user" else "user"):
                st.markdown(m.get("content", ""))
