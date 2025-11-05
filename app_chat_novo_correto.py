
# -*- coding: utf-8 -*-
# app_chat_novo.py — ESO • CHAT (modelo-first, com Sphera + Dicionários)
# - Corrige contagens Sphera/WS/Precursores/CP (agregação sobre hits).
# - Remove "Modo de Saída" e estatísticas fixas (modelo faz a síntese).
# - Injeção SEMPRE de docs/contexto_eso_chat.md (sem checkbox).
# - Sidebar Upload: apenas "Tamanho máx. de UPLOAD_RAW (chars)".
# - Mantém Assistente de Prompts (Texto/Upload) + rascunho.
# - Mantém Filtros avançados – Sphera + Agregação sobre eventos.
# - "Description contém" corrigido (case-insensitive, regex escapado).
# - "Limpar chat" e "Limpar uploads" com rerun.

import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ========================== Config inicial ==========================
st.set_page_config(page_title="SAFETY • CHAT", page_icon="💬", layout="wide")

DATA_DIR = "data"
AN_DIR   = os.path.join(DATA_DIR, "analytics")
ALT_DIR  = "/mnt/data"  # fallback
DOCS_DIR = Path("docs")
CONTEXT_MD_REL_PATH = DOCS_DIR / "contexto_eso_chat.md"
PROMPTS_MD_PATH     = Path("data/prompts/prompts.md")

# Modelo (chat)
OLLAMA_HOST    = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", ""))
OLLAMA_MODEL   = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", ""))
OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY"))
HEADERS_JSON   = {"Authorization": f"Bearer {OLLAMA_API_KEY}", "Content-Type": "application/json"} if OLLAMA_API_KEY else {"Content-Type": "application/json"}

# Embeddings
ST_MODEL_NAME = os.getenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# ========================== Helpers base ==========================
def _fatal(msg: str):
    st.error(msg)
    st.stop()

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    _fatal(f"❌ sentence-transformers indisponível: {e}")

def l2norm(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(np.float32, copy=False)
    n = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return mat / n

def load_npz_embeddings(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=True) as z:
            # chaves comuns
            for key in ("embeddings", "E", "X", "vectors", "vecs"):
                if key in z:
                    E = np.array(z[key]).astype(np.float32, copy=False)
                    return l2norm(E)
            # fallback: maior 2D
            best_k, best_n = None, -1
            for k in z.files:
                arr = z[k]
                if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] > best_n:
                    best_k, best_n = k, arr.shape[0]
            if best_k is None:
                st.warning(f"{os.path.basename(path)} não contém matriz 2D.")
                return None
            E = np.array(z[best_k]).astype(np.float32, copy=False)
            return l2norm(E)
    except Exception as e:
        st.warning(f"Falha ao ler {path}: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_prompts_md(md_path: Path):
    """Lê prompts markdown e retorna {'Texto': [...], 'Upload': [...]} com títulos e corpos."""
    if not md_path.exists():
        return {"Texto": [], "Upload": []}
    raw = md_path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^##\s+", raw)
    data = {"Texto": [], "Upload": []}
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        first_line, _, rest = sec.partition("\n")
        section_name = first_line.strip()
        if section_name not in ("Texto", "Upload"):
            continue
        parts = re.split(r"(?m)^###\s+", rest)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            title_line, _, body = p.partition("\n")
            data[section_name].append({"title": title_line.strip(), "body": body.strip()})
    # ordena por número se houver "1) ..."
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
    # SEMPRE injetar datasets_contexto.md (contexto_eso_chat.md)
    pre = (
        "Você é o ESO-CHAT para segurança operacional (óleo e gás). "
        "Responda em PT-BR, cite IDs/similaridade quando usar buscas locais, "
        "e não invente dados fora dos contextos fornecidos.\n\n"
    )
    ctx_md = load_file_text(CONTEXT_MD_REL_PATH)
    return pre + "=== CONTEXTO ESO-CHAT (.md) ===\n" + ctx_md

def ollama_chat(messages, model=None, temperature=0.2, stream=False, timeout=120):
    if not (OLLAMA_HOST and model):
        raise RuntimeError("Modelo não configurado. Defina OLLAMA_HOST e OLLAMA_MODEL.")
    r = requests.post(f"{OLLAMA_HOST}/api/chat", headers=HEADERS_JSON, json={
        "model": model, "messages": messages, "temperature": float(temperature), "stream": bool(stream)
    }, timeout=timeout)
    r.raise_for_status()
    return r.json()

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
    st.session_state.st_encoder = None

# ========================== Encoder ==========================
def ensure_st_encoder():
    if st.session_state.st_encoder is None:
        try:
            st.session_state.st_encoder = SentenceTransformer(ST_MODEL_NAME)
        except Exception as e:
            _fatal(f"❌ Não foi possível carregar o encoder: {e}")

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

# Dicionários (PT por padrão)
WS_PT_NPZ        = os.path.join(AN_DIR, "ws_embeddings_pt.npz")
WS_PT_LBL_PARQ   = os.path.join(AN_DIR, "ws_embeddings_pt.parquet")
PREC_PT_NPZ      = os.path.join(AN_DIR, "prec_embeddings_pt.npz")
PREC_PT_LBL_PARQ = os.path.join(AN_DIR, "prec_embeddings_pt.parquet")
CP_NPZ           = os.path.join(AN_DIR, "cp_embeddings.npz")
CP_LBL_PARQ      = os.path.join(AN_DIR, "cp_labels.parquet")

def load_dict_bank(npz_path: str, labels_parquet: str):
    """Carrega (E, labels) com fallback para ALT_DIR, e valida tamanhos."""
    E = load_npz_embeddings(npz_path)
    labels = None
    if os.path.exists(labels_parquet):
        try: labels = pd.read_parquet(labels_parquet)
        except Exception: labels = None

    if (E is None or labels is None or (labels is not None and E is not None and len(labels) != E.shape[0])):
        base_npz = os.path.basename(npz_path)
        base_lbl = os.path.basename(labels_parquet)
        npz_alt  = os.path.join(ALT_DIR, base_npz)
        lbl_alt  = os.path.join(ALT_DIR, base_lbl)
        if E is None and os.path.exists(npz_alt):
            E = load_npz_embeddings(npz_alt)
        if labels is None and os.path.exists(lbl_alt):
            try: labels = pd.read_parquet(lbl_alt)
            except Exception: labels = None

    if E is None or labels is None:
        st.warning(f"[Dicionários] Arquivos ausentes/ilegíveis: {npz_path} / {labels_parquet}")
        return None, None
    if len(labels) != E.shape[0]:
        st.warning(f"[Dicionários] Mismatch: labels={len(labels)} vs embeddings={E.shape[0]}")
        return None, None
    return E, labels

E_ws,   L_ws   = load_dict_bank(WS_PT_NPZ,   WS_PT_LBL_PARQ)
E_prec, L_prec = load_dict_bank(PREC_PT_NPZ, PREC_PT_LBL_PARQ)
E_cp,   L_cp   = load_dict_bank(CP_NPZ,      CP_LBL_PARQ)

# ========================== Funções Sphera e Filtros ==========================
def get_sphera_location_col(df: pd.DataFrame) -> str | None:
    """Preferência por LOCATION; fallback para AREA/Area/Setor."""
    if df is None:
        return None
    preferred = ["LOCATION", "Location", "LOCAL", "Local", "FPSO", "FPSO/Unidade", "Unidade"]
    fallback  = ["AREA", "Area", "Setor"]
    for c in preferred:
        if c in df.columns:
            return c
    for c in fallback:
        if c in df.columns:
            st.warning(f"⚠️ Usando '{c}' como fallback de Location.")
            return c
    return None

def filter_sphera_by_date(df: pd.DataFrame, years: int | None) -> pd.DataFrame:
    if df is None or years is None or "EVENT_DATE" not in df.columns:
        return df
    try:
        d = df.copy()
        d["EVENT_DATE"] = pd.to_datetime(d["EVENT_DATE"], errors="coerce")
        cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=365*years))
        return d[d["EVENT_DATE"] >= cutoff]
    except Exception:
        return df

def apply_advanced_filters(base: pd.DataFrame, desc_contains: str, loc_list: list[str]) -> pd.DataFrame:
    d = base
    # Location
    loc_col = get_sphera_location_col(d)
    if loc_col and loc_list:
        d = d[d[loc_col].astype(str).isin(set([x.strip() for x in loc_list if x.strip()]))]
    # Description contém (case-insensitive, regex escapado)
    desc_col = "Description" if "Description" in d.columns else ("DESCRIPTION" if "DESCRIPTION" in d.columns else None)
    if desc_col and desc_contains:
        pat = re.escape(desc_contains)
        d = d[d[desc_col].astype(str).str.contains(pat, case=False, na=False, regex=True)]
    return d

def sphera_similar_to_text(query_text: str, min_sim: float, years: int | None, topk: int,
                           df_sph: pd.DataFrame, E_sph: np.ndarray,
                           desc_contains: str, loc_list: list[str]):
    """Retorna [(event_id, sim, row)] com sim >= min_sim (cosine) e filtros aplicados."""
    if df_sph is None or E_sph is None or E_sph.size == 0:
        return []
    base = df_sph
    if years is not None:
        base = filter_sphera_by_date(base, years)
    base = apply_advanced_filters(base, desc_contains, loc_list)

    # alinhar embeddings pelo índice
    try:
        idx_map = base.index.to_numpy()
        if np.issubdtype(idx_map.dtype, np.integer):
            E_view = E_sph[idx_map, :]
        else:
            raise TypeError
    except Exception:
        # fallback: usa df original
        base = df_sph
        if years is not None:
            base = filter_sphera_by_date(base, years)
        base = apply_advanced_filters(base, desc_contains, loc_list)
        E_view = E_sph

    # Query vector
    if not (query_text or "").strip():
        return []
    vq = encode_query(query_text)
    sims = (E_view @ vq).astype(float)
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
        out.append((evid, s, row))
        kept += 1
        if kept >= topk:
            break
    return out

# ========================== Agregação dicionários sobre hits ==========================
def aggregate_dict_matches_over_hits(
    hits,
    E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
    thr_ws: float, thr_prec: float, thr_cp: float,
    topn_ws: int, topn_prec: int, topn_cp: int,
    agg_mode: str = "max",
    per_event_thr: float = 0.30,
    min_support: int = 2,
):
    """
    Compara dicionários (WS/Precursores/CP) vs DESCRIPTIONS dos hits Sphera.
    Agrega por 'max' ou 'mean', aplica limiar por evento e suporte mínimo.
    Retorna {'ws': [(label, sim, suporte), ...], ...}
    """
    try:
        if not hits:
            return {"ws": [], "prec": [], "cp": []}

        descs = []
        for _, _, row in hits:
            descs.append(str(row.get("Description", row.get("DESCRIPTION", ""))).strip())
        descs = [d for d in descs if d]
        if not descs:
            return {"ws": [], "prec": [], "cp": []}

        V_desc = encode_texts(descs, batch_size=32)
        V_desc_T = V_desc.T

        def _score(E_bank, labels_df, thr_global, topn_target):
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

        return {
            "ws":   _score(E_ws,   L_ws,   thr_ws,   topn_ws),
            "prec": _score(E_prec, L_prec, thr_prec, topn_prec),
            "cp":   _score(E_cp,   L_cp,   thr_cp,   topn_cp),
        }
    except Exception as e:
        try:
            st.warning(f"[Dict/Hits] Falha: {e}")
        except Exception:
            pass
        return {"ws": [], "prec": [], "cp": []}

# ========================== Sidebar ==========================
st.sidebar.subheader("Assistente de Prompts")
prompts_bank = load_prompts_md(PROMPTS_MD_PATH)

ptype = st.sidebar.selectbox("Tipo", options=["Texto", "Upload"], index=0, key="ptype_sel")
titles = [it["title"] for it in prompts_bank.get(ptype, [])]
if titles:
    sel = st.sidebar.selectbox("Modelo de prompt", options=titles, index=0, key=f"sel_{ptype}")
    body = next((it["body"] for it in prompts_bank[ptype] if it["title"] == sel), "")
    if st.sidebar.button("Carregar no rascunho", use_container_width=True, key="btn_load_prompt"):
        st.session_state.draft_prompt = body
        st.sidebar.success("Modelo carregado no rascunho.")
        st.rerun()
else:
    st.sidebar.info(f"Nenhum prompt encontrado em {PROMPTS_MD_PATH} ({ptype}).")

st.sidebar.header("Configurações")
st.sidebar.write("Host:", OLLAMA_HOST or "(não definido)")
st.sidebar.write("Modelo:", OLLAMA_MODEL or "(não definido)")

st.sidebar.subheader("Recuperação – Sphera")
k_sph      = st.sidebar.slider("Top-K Sphera", 0, 50, 20, 1)
thr_sph    = st.sidebar.slider("Limiar Sphera (cos)", 0.0, 1.0, 0.30, 0.01)
apply_tf   = st.sidebar.checkbox("Filtrar últimos N anos", True)
years_back = st.sidebar.slider("N (anos)", 1, 10, 3, 1)

st.sidebar.subheader("Filtros avançados – Sphera")
sph_desc_contains = st.sidebar.text_input("Description contém (substring)", "")
sph_loc_selected  = st.sidebar.text_input("Filtrar LOCATION (lista separada por ;)", "")

st.sidebar.subheader("Agregação sobre eventos recuperados (Sphera)")
agg_mode     = st.sidebar.selectbox("Agregação", ["max", "mean"], index=0)
per_ev_thr   = st.sidebar.slider("Limiar por evento (dicionários)", 0.0, 1.0, 0.30, 0.01)
min_support  = st.sidebar.slider("Suporte mínimo (nº de eventos)", 1, 20, 2, 1)
thr_ws       = st.sidebar.slider("Limiar global WS", 0.0, 1.0, 0.25, 0.01)
thr_prec     = st.sidebar.slider("Limiar global Precursores", 0.0, 1.0, 0.25, 0.01)
thr_cp       = st.sidebar.slider("Limiar global CP", 0.0, 1.0, 0.25, 0.01)
topn_ws      = st.sidebar.slider("Top-N WS", 3, 90, 10, 1)
topn_prec    = st.sidebar.slider("Top-N Precursores", 3, 90, 10, 1)
topn_cp      = st.sidebar.slider("Top-N CP", 3, 90, 10, 1)

st.sidebar.subheader("Upload")
upload_raw_max = st.sidebar.slider("Tamanho máx. de UPLOAD_RAW (chars)", 300, 20000, 2500, 100)

# utilidade
c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("Limpar uploads", use_container_width=True, key="btn_clear_upl"):
        st.session_state.upld_texts = []
        st.session_state.upld_meta  = []
        st.session_state.upld_emb   = None
        st.session_state.pop("last_upload_digest", None)
        try:
            st.experimental_rerun()
        except Exception:
            st.rerun()
with c2:
    if st.button("Limpar chat", use_container_width=True, key="btn_clear_chat"):
        st.session_state.chat = []
        try:
            st.experimental_rerun()
        except Exception:
            st.rerun()

# ========================== UI central ==========================
st.title("SAFETY • CHAT")

# Rascunho de prompt
st.text_area("Conteúdo do prompt", key="draft_prompt", height=180, placeholder="Digite ou carregue um modelo de prompt...")

# Texto principal (para Sphera)
user_text = st.text_area("Texto de análise (para Sphera)", height=200, placeholder="Cole aqui o relato/descrição do evento a analisar...")

# Upload opcional (conteúdo bruto limitado)
uploaded = st.file_uploader("Anexar arquivo (opcional)", type=["txt","md","pdf","docx","csv","xlsx"])
if uploaded is not None:
    raw = uploaded.read()
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
    try:
        st.experimental_rerun()
    except Exception:
        st.rerun()

# ========================== Execução ==========================
def render_hits_table(hits, df_all):
    if not hits:
        return ""
    lines = ["| Event ID | Similaridade | LOCATION | Descrição |", "|---|---:|---|---|"]
    loc_col = get_sphera_location_col(df_all)
    for evid, s, row in hits[:min(10, len(hits))]:
        loc_val = str(row.get(loc_col, "")) if loc_col else ""
        desc    = str(row.get("Description", row.get("DESCRIPTION", ""))).replace("\n", " ").strip()[:240]
        lines.append(f"| {evid} | {s:.3f} | {loc_val} | {desc} |")
    return "\n".join(lines)

def messages_with_context(user_blocks: list[str]):
    msgs = [{"role": "system", "content": st.session_state.system_prompt}]
    if user_blocks:
        msgs.append({"role": "user", "content": "\n\n".join(user_blocks)})
    else:
        msgs.append({"role": "user", "content": "Sem prompt ou texto. Explique como devo proceder."})
    return msgs

def push_model(messages, user_text_for_question: str, context_md: str):
    messages.append({"role": "user", "content": "DADOS DE APOIO (não responda aqui):\n" + context_md})
    qt = user_text_for_question or st.session_state.draft_prompt or "Analise os dados fornecidos e sintetize as lições aprendidas."
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
    # 1) Monta a mensagem do usuário (prompt draft + texto e/ou uploads)
    blocks = []
    if st.session_state.draft_prompt.strip():
        blocks.append("PROMPT:\n" + st.session_state.draft_prompt.strip())
    if (user_text or "").strip():
        blocks.append("TEXTO:\n" + user_text.strip())
    for i, t in enumerate(st.session_state.upld_texts or []):
        blocks.append(f"UPLOAD[{i+1}]:\n" + t.strip())

    msgs = messages_with_context(blocks)

    # 2) Busca Sphera
    loc_list = [x.strip() for x in sph_loc_selected.split(";")] if sph_loc_selected.strip() else []
    hits = sphera_similar_to_text(
        query_text=user_text or st.session_state.draft_prompt,
        min_sim=thr_sph,
        years=(years_back if apply_tf else None),
        topk=k_sph,
        df_sph=df_sph,
        E_sph=E_sph,
        desc_contains=sph_desc_contains,
        loc_list=loc_list,
    )

    # 3) Mostra hits (Top-10)
    table_md = ""
    if hits:
        table_md = render_hits_table(hits, df_sph)
        st.markdown("**Eventos do Sphera (Top-10)**\n\n" + table_md)
        st.session_state.chat.append({"role": "assistant", "content": "Eventos Sphera listados."})

    # 4) WS/Precursores/CP (somente se houver hits)
    dict_matches = aggregate_dict_matches_over_hits(
        hits, E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
        thr_ws=thr_ws, thr_prec=thr_prec, thr_cp=thr_cp,
        topn_ws=topn_ws, topn_prec=topn_prec, topn_cp=topn_cp,
        agg_mode=agg_mode, per_event_thr=per_ev_thr, min_support=min_support,
    )

    if hits:
        md2 = []
        if dict_matches["ws"]:
            md2 += [
                "**WS (≥ limiar, calculado no app)**",
                "| Rank | Termo | Similaridade | Suporte |",
                "|---:|---|---:|---:|",
            ]
            for r, (label, s, sup) in enumerate(dict_matches["ws"], 1):
                md2.append(f"| {r} | {label} | {s:.3f} | {sup} |")
        else:
            md2 += ["**WS (≥ limiar, calculado no app)**", "Nenhum WS ≥ limiar."]

        if dict_matches["prec"]:
            md2 += [
                "",
                "**Precursores (≥ limiar, calculado no app)**",
                "| Rank | Termo | Similaridade | Suporte |",
                "|---:|---|---:|---:|",
            ]
            for r, (label, s, sup) in enumerate(dict_matches["prec"], 1):
                md2.append(f"| {r} | {label} | {s:.3f} | {sup} |")
        else:
            md2 += ["", "**Precursores (≥ limiar, calculado no app)**", "Nenhum Precursor ≥ limiar."]

        if dict_matches["cp"]:
            md2 += [
                "",
                "**CP (≥ limiar, calculado no app)**",
                "| Rank | Fator | Similaridade | Suporte |",
                "|---:|---|---:|---:|",
            ]
            for r, (label, s, sup) in enumerate(dict_matches["cp"], 1):
                md2.append(f"| {r} | {label} | {s:.3f} | {sup} |")
        else:
            md2 += ["", "**CP (≥ limiar, calculado no app)**", "Nenhum Fator CP ≥ limiar."]
        st.markdown("\n".join(md2))

    # 5) Modelo faz a síntese
    ctx_chunks = []
    ctx_chunks.append(f"Sphera_hits={len(hits)}, thr_sph={thr_sph:.2f}, years={'all' if not apply_tf else years_back}")
    if hits:
        if table_md:
            ctx_chunks.append("HITS_TOP10_MD:\n" + table_md)
        def _md(lst, header):
            if not lst:
                return header + ": nenhum."
            rows = [f"- {lab} (sim={s:.3f}, sup={sup})" for lab, s, sup in lst[:10]]
            return header + ":\n" + "\n".join(rows)
        ctx_chunks.append(_md(dict_matches["ws"],   "WS selecionados"))
        ctx_chunks.append(_md(dict_matches["prec"], "Precursores selecionados"))
        ctx_chunks.append(_md(dict_matches["cp"],   "CP selecionados"))

    model_context = "\n\n".join(ctx_chunks)
    push_model(msgs, user_text, model_context)

# Histórico simples
if st.session_state.chat:
    st.divider()
    st.subheader("Histórico")
    for m in st.session_state.chat[-10:]:
        role = m.get("role","assistant")
        with st.chat_message("assistant" if role!="user" else "user"):
            st.markdown(m.get("content",""))
