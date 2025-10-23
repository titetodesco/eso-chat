# -*- coding: utf-8 -*-
"""
app_chat_novo.py — FINAL (corrigido)

Correções desta entrega (além do que já havia):
1) LOCATION no multiselect agora lê **exclusivamente** de `LOCATION` do Sphera (data/analytics/sphera.parquet). Se a coluna não existir, o filtro mostra aviso e permanece vazio (sem quebrar o fluxo).
2) Caminhos ajustados: **tudo** em `data/analytics/` (parquet e npz). Mensagens de erro e loaders sincronizados.
3) Execução só ocorre após clicar **“Enviar para o chat”** — upload/inputs não disparam a pipeline sozinhos.
4) Limiar de similaridade **WS / Precursor / CP** mantidos/renomeados na UI como solicitado (sem retirar funcionalidades). Também mantidos: limiar por evento e suporte mínimo.

Mantido do release anterior:
- Recuperação **somente Sphera** por similaridade coseno com `Description`/`DESCRIPTION` (query ou trecho de upload).
- `datasets_context.md` é **sempre injetado** no system; sem toggle.
- WS/Precursores/CP **apenas** dos dicionários existentes (.npz/.parquet), agregação `max/mean`, limiar por evento e suporte mínimo, sem inventar termos.
- “Description contém (substring)” case-insensitive + segura; Location somente `LOCATION`.
- Seletor de prompts (Texto/Upload) lido de `data/prompts/prompts.md` + “Carregar no rascunho”; rascunho editável e “Enviar para o chat”.
- “Limpar uploads” e “Limpar chat” limpam estado e chamam `st.rerun()` sem disparar o chat.

Requisitos: streamlit, pandas, numpy, sentence-transformers, requests.
Modelo: OLLAMA_HOST / OLLAMA_MODEL (ou adapte para OpenAI se desejar).
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

DATA_DIR = Path("data")
AN_DIR   = DATA_DIR / "analytics"
DOCS_DIR = Path("docs")
DATASETS_CONTEXT_PATH = DATA_DIR / "datasets_context.md"
PROMPTS_MD_PATH       = DATA_DIR / "prompts" / "prompts.md"

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
def load_npz_embeddings(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        with np.load(str(path), allow_pickle=True) as z:
            for key in ("embeddings", "E", "X", "vectors", "vecs"):
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
                st.warning(f"{path.name} não contém matriz 2D.")
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
    pre = (
        "Você é o ESO-CHAT para segurança operacional (óleo e gás). "
        "Responda em PT-BR, cite IDs/sim quando usar buscas locais, e não invente dados fora dos contextos fornecidos.\n\n"
    )
    ctx = []
    if DATASETS_CONTEXT_PATH.exists():
        ctx.append("=== DATASETS_CONTEXT ===\n" + load_file_text(DATASETS_CONTEXT_PATH))
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

if "st_encoder" not in st.session_state:
    st.session_state.st_encoder = ensure_st_encoder()

# ========================== Dados Sphera & Dicionários ==========================
SPH_PQ_PATH  = AN_DIR / "sphera.parquet"
SPH_EMB_PATH = AN_DIR / "sphera_embeddings.npz"

df_sph = None
if SPH_PQ_PATH.exists():
    try:
        df_sph = pd.read_parquet(SPH_PQ_PATH)
    except Exception as e:
        st.error(f"Falha ao ler {SPH_PQ_PATH}: {e}")
else:
    st.error(f"Parquet do Sphera não encontrado em {SPH_PQ_PATH}")

E_sph = load_npz_embeddings(SPH_EMB_PATH)
if E_sph is None:
    st.error(f"Embeddings do Sphera não encontrados em {SPH_EMB_PATH}")

# Dicionários PT/EN (ajuste se usar EN)
WS_NPZ,   WS_LBL   = AN_DIR / "ws_embeddings_pt.npz",   AN_DIR / "ws_embeddings_pt.parquet"
PREC_NPZ, PREC_LBL = AN_DIR / "prec_embeddings_pt.npz", AN_DIR / "prec_embeddings_pt.parquet"
CP_NPZ,   CP_LBL   = AN_DIR / "cp_embeddings.npz",      AN_DIR / "cp_labels.parquet"

E_ws = load_npz_embeddings(WS_NPZ)
L_ws = pd.read_parquet(WS_LBL) if WS_LBL.exists() else None
E_prec = load_npz_embeddings(PREC_NPZ)
L_prec = pd.read_parquet(PREC_LBL) if PREC_LBL.exists() else None
E_cp = load_npz_embeddings(CP_NPZ)
L_cp = pd.read_parquet(CP_LBL) if CP_LBL.exists() else None

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

# ========================== Filtros / Similaridade ==========================
@st.cache_data(show_spinner=False)
def filter_sphera(df: pd.DataFrame, locations: List[str], substr: str, years: int) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    # Janela temporal (se houver col EVENT_DATE)
    if "EVENT_DATE" in out.columns:
        out["EVENT_DATE"] = pd.to_datetime(out["EVENT_DATE"], errors="coerce")
        cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=365*years))
        out = out[out["EVENT_DATE"] >= cutoff]
    # LOCATION obrigatório
    if "LOCATION" in out.columns and locations:
        sel = set([str(x).strip() for x in locations if str(x).strip()])
        out = out[out["LOCATION"].astype(str).isin(sel)]
    elif "LOCATION" not in out.columns:
        st.info("Coluna LOCATION não encontrada no parquet. O filtro de Location ficará vazio.")
    # Description contém
    desc_col = "Description" if "Description" in out.columns else ("DESCRIPTION" if "DESCRIPTION" in out.columns else None)
    if desc_col and substr:
        pat = re.escape(substr)
        out = out[out[desc_col].astype(str).str.contains(pat, case=False, na=False, regex=True)]
    return out

@st.cache_data(show_spinner=False)
def sphera_similar_to_text(query_text: str, min_sim: float, years: int, topk: int,
                           df_base: pd.DataFrame, E_base: np.ndarray,
                           substr: str, locations: List[str]) -> List[Tuple[str, float, pd.Series]]:
    if not query_text or df_base is None or E_base is None or E_base.size == 0:
        return []
    base = filter_sphera(df_base, locations, substr, years)
    # Alinhamento índice→vetor (supõe embeddings na mesma ordem do parquet)
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
    # Descrições dos hits
    descs = [str(r.get("Description", r.get("DESCRIPTION", ""))).strip() for _,_,r in hits]
    descs = [d for d in descs if d]
    if not descs:
        return {"ws": [], "prec": [], "cp": []}
    V_desc = encode_texts(descs, batch_size=32).T  # MxD -> D x M (transpose ao final)

    def _score(E_bank, labels_df, thr_sim, topn_target):
        if E_bank is None or labels_df is None or len(labels_df) != E_bank.shape[0]:
            return []
        S = (E_bank @ V_desc)  # N_terms x M_events (cos-sim)
        support = (S >= per_event_thr).sum(axis=1)
        sims = S.mean(axis=1) if agg_mode == "mean" else S.max(axis=1)
        mask = (support >= min_support) & (sims >= thr_sim)
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
        "ws":   _score(E_ws,   L_ws,   thr_ws_sim,   topn_ws),
        "prec": _score(E_prec, L_prec, thr_prec_sim, topn_prec),
        "cp":   _score(E_cp,   L_cp,   thr_cp_sim,   topn_cp),
    }

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
# LOCATION a partir da coluna LOCATION
loc_options = []
if isinstance(df_sph, pd.DataFrame) and not df_sph.empty and "LOCATION" in df_sph.columns:
    loc_options = sorted([x for x in df_sph["LOCATION"].dropna().astype(str).unique().tolist() if x])
else:
    st.sidebar.info("Coluna LOCATION não encontrada — o filtro ficará vazio.")
locations = st.sidebar.multiselect("Location", options=loc_options, default=[])
substr    = st.sidebar.text_input("Description contém (substring)", "")

st.sidebar.subheader("Agregação sobre eventos recuperados (Sphera)")
agg_mode    = st.sidebar.selectbox("Agregação", ["max", "mean"], index=0)
per_ev_thr  = st.sidebar.slider("Limiar por evento (dicionários)", 0.0, 1.0, 0.30, 0.01)
min_support = st.sidebar.slider("Suporte mínimo (nº de eventos)", 1, 20, 1, 1)

# Limiares de similaridade por família (mantidos)
thr_ws_sim   = st.sidebar.slider("Limiar de similaridade WS", 0.0, 1.0, 0.25, 0.01)
thr_prec_sim = st.sidebar.slider("Limiar de similaridade Precursor", 0.0, 1.0, 0.25, 0.01)
thr_cp_sim   = st.sidebar.slider("Limiar de similaridade CP", 0.0, 1.0, 0.25, 0.01)

topn_ws   = st.sidebar.slider("Top-N WS", 3, 90, 10, 1)
topn_prec = st.sidebar.slider("Top-N Precursores", 3, 90, 10, 1)
topn_cp   = st.sidebar.slider("Top-N CP", 3, 90, 10, 1)

# Utilidades
cc1, cc2 = st.sidebar.columns(2)
with cc1:
    if st.button("Limpar uploads", use_container_width=True):
        st.session_state.upld_texts = []
        st.rerun()
with cc2:
    if st.button("Limpar chat", use_container_width=True):
        st.session_state.chat = []
        st.rerun()

# ========================== UI central ==========================
st.title("ESO • CHAT (Somente Sphera)")

st.text_area("Conteúdo do prompt", key="draft_prompt", height=180, placeholder="Digite ou carregue um modelo de prompt…")

user_text = st.text_area("Texto de análise (para Sphera)", height=200, placeholder="Cole aqui a descrição/evento a analisar…")

uploaded = st.file_uploader("Anexar arquivo (opcional)", type=["txt","md","csv"])
if uploaded is not None:
    raw = uploaded.read()
    try:
        as_text = raw.decode("utf-8", errors="ignore")
    except Exception:
        as_text = ""
    if uploaded.name.lower().endswith(".csv") and as_text:
        try:
            dfcsv = pd.read_csv(io.StringIO(as_text))
            as_text = "\n".join(dfcsv.astype(str).fillna("").apply(lambda r: " ".join(r.values), axis=1).tolist())
        except Exception:
            pass
    if as_text:
        st.session_state.upld_texts.append(as_text)
        st.success(f"Upload recebido: {uploaded.name} (armazenado no contexto local).")

col_run1, col_run2 = st.columns([1,1])
go_btn      = col_run1.button("Enviar para o chat", type="primary", use_container_width=True)
clear_draft = col_run2.button("Limpar rascunho", use_container_width=True)
if clear_draft:
    st.session_state.draft_prompt = ""
    st.rerun()

# ========================== Execução (somente ao clicar) ==========================

def render_hits_table(hits: List[Tuple[str, float, pd.Series]]) -> str:
    if not hits:
        return ""
    lines = ["| Event ID | Similaridade | LOCATION | Descrição |", "|---|---:|---|---|"]
    for evid, s, row in hits[:min(10, len(hits))]:
        loc_val = str(row.get("LOCATION", "N/D"))
        desc    = str(row.get("Description", row.get("DESCRIPTION", ""))).replace("\n", " ").strip()[:240]
        lines.append(f"| {evid} | {s:.3f} | {loc_val} | {desc} |")
    return "\n".join(lines)


def push_model(messages: List[Dict[str, str]], pergunta: str, contexto_md: str):
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
    # 1) Monta blocos do usuário
    blocks = []
    if st.session_state.draft_prompt.strip():
        blocks.append("PROMPT:\n" + st.session_state.draft_prompt.strip())
    if (user_text or "").strip():
        blocks.append("TEXTO:\n" + user_text.strip())
    for i, t in enumerate(st.session_state.upld_texts or []):
        blocks.append(f"UPLOAD[{i+1}]:\n" + t.strip())

    # 2) Busca Sphera
    hits = sphera_similar_to_text(
        query_text=(user_text or st.session_state.draft_prompt),
        min_sim=thr_sph,
        years=years,
        topk=k_sph,
        df_base=df_sph,
        E_base=E_sph,
        substr=substr,
        locations=locations,
    )

    # 3) Renderiza hits
    if hits:
        st.markdown("**Eventos do Sphera (Top-10)**\n\n" + render_hits_table(hits))
        st.session_state.chat.append({"role": "assistant", "content": "Eventos Sphera listados."})
    else:
        st.info("Nenhum evento do Sphera atingiu o limiar/filtros atuais.")

    # 4) Dicionários sobre os hits
    dict_matches = aggregate_dict_matches_over_hits(
        hits, E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
        thr_ws_sim=thr_ws_sim, thr_prec_sim=thr_prec_sim, thr_cp_sim=thr_cp_sim,
        topn_ws=topn_ws, topn_prec=topn_prec, topn_cp=topn_cp,
        agg_mode=agg_mode, per_event_thr=per_ev_thr, min_support=min_support,
    )

    if hits:
        md2 = []
        for title, key in [("WS", "ws"), ("Precursores", "prec"), ("CP", "cp")]:
            arr = dict_matches.get(key) or []
            md2 += [f"**{title} (≥ limiares)**"]
            if arr:
                md2 += ["| Rank | Termo | Similaridade | Suporte |", "|---:|---|---:|---:|"]
                for r, (label, s, sup) in enumerate(arr, 1):
                    md2.append(f"| {r} | {label} | {s:.3f} | {sup} |")
            else:
                md2 += [f"Nenhum {title} ≥ limiar."]
            md2 += [""]
        st.markdown("\n".join(md2))

    # 5) Síntese pelo modelo
    table_md = render_hits_table(hits)
    ctx_chunks = [
        f"Sphera_hits={len(hits)}, thr_sph={thr_sph:.2f}, years={years}",
        ("HITS_TOP10_MD:\n" + table_md) if table_md else "",
    ]
    # Resumo curto dos dicionários
    def _b(lst, name):
        if not lst:
            return f"{name}: nenhum ≥ limiar"
        rows = [f"- {lab} (sim={s:.3f}, sup={sup})" for lab, s, sup in lst[:10]]
        return name + ":\n" + "\n".join(rows)
    ctx_chunks.append(_b(dict_matches.get("ws"), "WS selecionados"))
    ctx_chunks.append(_b(dict_matches.get("prec"), "Precursores selecionados"))
    ctx_chunks.append(_b(dict_matches.get("cp"), "CP selecionados"))

    messages = [{"role": "system", "content": st.session_state.system_prompt}, {"role": "user", "content": "\n\n".join([b for b in blocks if b])}]
    push_model(messages, user_text, "\n\n".join([x for x in ctx_chunks if x]))

# ========================== Histórico ==========================
if st.session_state.chat:
    st.divider()
    st.subheader("Histórico")
    for m in st.session_state.chat[-10:]:
        role = m.get("role","assistant")
        with st.chat_message("assistant" if role != "user" else "user"):
            st.markdown(m.get("content",""))
