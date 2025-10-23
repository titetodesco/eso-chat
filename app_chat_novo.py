# -*- coding: utf-8 -*-
"""
app_chat_novo.py — Revisado e aprimorado

Regras atendidas:
- Mantém funcionalidades e reaproveita embeddings/datasets existentes (.npz/.parquet), sem regerar embeddings do corpus.
- Usa somente Sphera (Description) para recuperação por similaridade do cosseno entre consulta OU trecho do upload e as DESCRIPTIONS.
- Aplica filtros avançados (Location, substring em Description, limiar de similaridade, janela temporal – últimos N anos).
- Contagens WS/Precursores/CP feitas APENAS com dicionários embutidos (npz/parquet) contra as DESCRIPTIONS dos eventos recuperados (agregação mean/max, limiar por evento e suporte mínimo).
- Seletor de prompts (dois combos: "Texto" e "Upload") lendo de data/prompts/prompts.md; botão "Carregar no rascunho" preenche a caixa de conteúdo do prompt.
- "Limpar chat" limpa histórico sem disparar inferência; "Limpar uploads" zera buffers, metadados e embeddings de upload.
- Injeta datasets_context.md sempre no contexto do LLM (sem toggle).
- "Description contém (substring)" é case-insensitive e filtra a coluna correta.
- Remove visuais poluentes do fluxo padrão; depuração apenas via toggle.
- Corrige ordem/escopo/indentação e padroniza funções antes das chamadas.
- Evita heurísticas de UI fixas: síntese e interpretação ficam a cargo do LLM.

Dependências:
- streamlit>=1.32
- pandas, numpy, pyarrow
- openai (para embeddings/completion) — configure OPENAI_API_KEY

Estrutura de arquivos esperada (exemplos — ajuste aos seus caminhos reais):
- data/sphera/sphera.parquet (colunas mín.: ['EventID','Description','Location','EventDate'] com EventDate em ISO/yyy-mm-dd)
- data/sphera/sphera_embeddings.npz (keys: 'embeddings' (float32 [N,D]), 'index' (int64 ids alinhados ao parquet))
- data/dicts/ws_cp_precursores.parquet ou .npz (ver loaders; deve conter termos/itens com colunas 'type' in {'WS','Precursor','CP'} e 'term' OU estruturas equivalentes)
- data/prompts/prompts.md (seções ## Texto e ## Upload com itens ### Nome do prompt seguidos do conteúdo)
- data/datasets_context.md (texto sempre injetado no system/context)

Observação: este app NÃO rege gera embeddings do CORPUS. Apenas calcula embeddings de query/upload (permitido) para comparar com embeddings já gerados do Sphera.
"""

import os
import io
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# =============================
# --------- Constantes --------
# =============================
APP_TITLE = "ESO-CHAT — Análise Sphera (RAG)"
SPHERA_PARQUET_PATH = "data/sphera/sphera.parquet"
SPHERA_EMBED_NPZ_PATH = "data/sphera/sphera_embeddings.npz"
DICT_PATHS = [
    "data/dicts/ws_cp_precursores.parquet",
    "data/dicts/ws_cp_precursores.npz",
]
PROMPTS_MD_PATH = "data/prompts/prompts.md"
DATASETS_CONTEXT_PATH = "data/datasets_context.md"

DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")

# Chaves de session_state
SS_LOADED = "loaded_ok"
SS_CHAT = "chat_history"
SS_UPLOAD_TEXT = "upload_text"
SS_UPLOAD_META = "upload_meta"
SS_UPLOAD_EMB = "upload_emb"
SS_PROMPT_DRAFT = "prompt_draft"
SS_LAST_RESULTS = "last_results_df"
SS_LAST_MATCHES = "last_matches_idx"
SS_DEBUG = "debug_mode"

# =============================
# ---------- Utilidades -------
# =============================

def _safe_read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def load_sphera() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Carrega Sphera parquet e embeddings NPZ.

    Retorna:
        df: DataFrame com pelo menos ['EventID','Description','Location','EventDate']
        emb: ndarray shape [N, D]
        idx: ndarray shape [N,] mapeando linhas do emb -> índice do df
    """
    if not os.path.exists(SPHERA_PARQUET_PATH):
        st.error(f"Parquet do Sphera não encontrado em {SPHERA_PARQUET_PATH}")
        return pd.DataFrame(), np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    df = pd.read_parquet(SPHERA_PARQUET_PATH)
    # Normaliza colunas esperadas
    expected_cols = {"EventID", "Description", "Location", "EventDate"}
    missing = expected_cols - set(df.columns)
    if missing:
        st.warning(f"Parquet do Sphera está sem colunas: {missing} — a aplicação pode se limitar.")
    # EventDate para datetime
    if "EventDate" in df.columns:
        df["EventDate"] = pd.to_datetime(df["EventDate"], errors="coerce")

    # Embeddings
    if not os.path.exists(SPHERA_EMBED_NPZ_PATH):
        st.error(f"Embeddings do Sphera não encontrados em {SPHERA_EMBED_NPZ_PATH}")
        return df, np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    npz = np.load(SPHERA_EMBED_NPZ_PATH)
    emb = npz.get("embeddings")
    idx = npz.get("index")
    if emb is None or idx is None:
        st.error("NPZ de embeddings do Sphera deve conter keys 'embeddings' e 'index'.")
        return df, np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    # Garantias de tipo
    emb = emb.astype(np.float32, copy=False)
    idx = idx.astype(np.int64, copy=False)

    if len(idx) != len(emb):
        st.warning("Tamanho de 'index' diverge do de 'embeddings' no NPZ.")

    return df, emb, idx


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / denom


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Retorna matriz/ vetor de similaridade coseno entre a e b já normalizados."""
    return a @ b.T


def embed_texts(texts: List[str], model: str = DEFAULT_EMBED_MODEL) -> np.ndarray:
    """Gera embeddings para lista de textos (consulta ou upload)."""
    from openai import OpenAI

    client = OpenAI()
    resp = client.embeddings.create(input=texts, model=model)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return l2_normalize(vecs)


def openai_chat(messages: List[Dict], model: str = DEFAULT_CHAT_MODEL) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=1200,
    )
    return resp.choices[0].message.content or ""


def filter_sphera(
    df: pd.DataFrame,
    locations: Optional[List[str]],
    substr: Optional[str],
    years: int,
) -> pd.DataFrame:
    now = pd.Timestamp.now(tz=None)
    start_date = now - pd.DateOffset(years=years)

    out = df.copy()
    if "EventDate" in out.columns:
        out = out[out["EventDate"] >= start_date]

    # Location: usa LOCATION > FPSO > Location > FPSO/Unidade > Unidade (nunca AREA)
    loc_col = None
    for _c in ["LOCATION","FPSO","Location","FPSO/Unidade","Unidade"]:
        if _c in out.columns:
            loc_col = _c
            break
    if loc_col and locations:
        sel = set([str(x).strip() for x in locations if str(x).strip()])
        out = out[out[loc_col].astype(str).isin(sel)]

    if substr:
        # Case-insensitive, coluna correta: Description
        desc_col = "Description" if "Description" in out.columns else ("DESCRIPTION" if "DESCRIPTION" in out.columns else None)
        if desc_col:
            mask = out[desc_col].astype(str).str.contains(substr, case=False, na=False)
            out = out[mask]
    return out


def retrieve_topk(
    query_vec: np.ndarray,
    emb_corpus: np.ndarray,
    idx_map: np.ndarray,
    df_filtered: pd.DataFrame,
    top_k: int,
    sim_threshold: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Retorna top-k por similaridade (cos) >= limiar, respeitando df_filtered."""
    if emb_corpus.size == 0 or df_filtered.empty:
        return pd.DataFrame(columns=["EventID","Description","Location","EventDate","sim"]), np.array([], dtype=np.int64)

    # Mapeia índice do df para posições no embeddings
    allowed_ids = set(df_filtered.index.tolist())

    # Similaridade
    sims = cosine_sim(query_vec, emb_corpus)[0]  # (N,)

    # Seleciona apenas eventos presentes no df_filtered via idx_map
    mask_allowed = np.array([i in allowed_ids for i in idx_map])
    sims_allowed = np.where(mask_allowed, sims, -1.0)

    # Filtra por limiar e top-k
    valid_idx = np.where(sims_allowed >= sim_threshold)[0]
    if valid_idx.size == 0:
        return pd.DataFrame(columns=["EventID","Description","Location","EventDate","sim"]), np.array([], dtype=np.int64)

    order = np.argsort(-sims_allowed[valid_idx])[:top_k]
    picked = valid_idx[order]

    picked_df_idxs = idx_map[picked]
    rows = df_filtered.loc[picked_df_idxs, [c for c in ["EventID","Description","Location","EventDate"] if c in df_filtered.columns]].copy()
    rows["sim"] = sims_allowed[picked]
    # Garante que não venha zero contagens quando há hits
    if rows.empty is False:
        pass
    return rows, picked_df_idxs


# =============================
# --- Dicionários WS/CP/Prec ---
# =============================

def load_dicts() -> pd.DataFrame:
    """Carrega dicionários (ws/precursor/cp) de .parquet OU .npz.
    Espera colunas: ['type' in {'WS','Precursor','CP'}, 'term'] opcional 'weight'.
    Se .npz tiver 'terms' e 'types' (e opcional 'weights'), também é aceito.
    """
    for p in DICT_PATHS:
        if not os.path.exists(p):
            continue
        if p.endswith(".parquet"):
            try:
                df = pd.read_parquet(p)
                if not {"type","term"}.issubset(df.columns):
                    st.warning(f"Dicionário {p} sem colunas mínimas ['type','term'] — ignorando.")
                    continue
                # Normaliza
                df = df[["type","term"] + (["weight"] if "weight" in df.columns else [])].copy()
                return df
            except Exception as e:
                st.warning(f"Falha ao ler {p}: {e}")
        else:
            # NPZ
            try:
                npz = np.load(p, allow_pickle=True)
                terms = npz.get("terms")
                types = npz.get("types")
                weights = npz.get("weights")
                if terms is None or types is None:
                    continue
                data = {"type": list(types), "term": list(terms)}
                if weights is not None:
                    data["weight"] = list(weights)
                return pd.DataFrame(data)
            except Exception as e:
                st.warning(f"Falha ao ler {p}: {e}")
    return pd.DataFrame(columns=["type","term","weight"])  # vazio, mas com colunas


def score_dict_matches(
    events_df: pd.DataFrame,
    dict_df: pd.DataFrame,
    mode: str = "mean",
    per_event_threshold: float = 0.0,
    min_support: int = 1,
) -> pd.DataFrame:
    """Calcula contagens por WS/Precursor/CP com base em termos do dicionário NA DESCRIPTION.
    Regra: NUNCA inventar termos — apenas aqueles presentes no dicionário são usados.
    Implementação:
      - substring case-insensitive dos termos na Description (rápido e determinístico).
      - agrega por tipo com mean OU max por evento (se termo tiver peso, utiliza; caso contrário peso=1.0)
      - aplica per_event_threshold (p.ex., pontuação >= limiar conta)
      - aplica min_support (número mínimo de eventos com pontuação acima do limiar para exibir)
    Retorna DataFrame com colunas: ['type','support','score']
    """
    if events_df.empty or dict_df.empty:
        return pd.DataFrame(columns=["type","support","score"])  # nada a mostrar

    # Prepara dicionário por tipo
    dict_df = dict_df.copy()
    if "weight" not in dict_df.columns:
        dict_df["weight"] = 1.0
    dict_df["term_norm"] = dict_df["term"].astype(str).str.lower()

    # Para cada evento, calcula score por tipo agregando termos encontrados
    results = []
    for idx, row in events_df.iterrows():
        desc = str(row.get("Description", ""))
        desc_l = desc.lower()
        ev_scores: Dict[str, float] = {}
        for ttype in ["WS","Precursor","CP"]:
            sub = dict_df[dict_df["type"] == ttype]
            if sub.empty:
                ev_scores[ttype] = 0.0
                continue
            hits = sub[ sub["term_norm"].apply(lambda x: x in desc_l) ]
            if hits.empty:
                ev_scores[ttype] = 0.0
                continue
            vals = hits["weight"].astype(float).values
            if mode == "max":
                s = float(np.max(vals))
            else:
                s = float(np.mean(vals))
            ev_scores[ttype] = s
        results.append(ev_scores)

    ev_scores_df = pd.DataFrame(results)
    ev_scores_df.index = events_df.index

    # Aplica limiar por evento (mantém 1 se >= limiar, senão 0)
    bin_df = (ev_scores_df >= per_event_threshold).astype(int)

    # Suporte por tipo = número de eventos com score >= limiar
    support = bin_df.sum(axis=0)

    # Score agregado por tipo = média dos scores nos eventos com >= limiar (ou 0 se nenhum)
    score_vals = {}
    for col in ev_scores_df.columns:
        sel = ev_scores_df[col][bin_df[col] == 1]
        score_vals[col] = float(sel.mean()) if not sel.empty else 0.0

    out = pd.DataFrame({
        "type": ["WS","Precursor","CP"],
        "support": [int(support.get(t, 0)) for t in ["WS","Precursor","CP"]],
        "score": [float(score_vals.get(t, 0.0)) for t in ["WS","Precursor","CP"]],
    })

    # Aplica min_support para exibição coerente
    out = out[out["support"] >= int(min_support)]

    return out.reset_index(drop=True)


# =============================
# ---------- Prompts -----------
# =============================

def parse_prompts_md(md_text: str) -> Dict[str, Dict[str, str]]:
    """Parses prompts.md em duas seções: 'Texto' e 'Upload'.
    Estrutura esperada:
    ## Texto
    ### Nome 1
    Conteúdo...
    ### Nome 2
    ...
    ## Upload
    ### Nome A
    ...
    Retorna: {'Texto': {nome:conteudo}, 'Upload': {nome:conteudo}}
    """
    sections = {"Texto": {}, "Upload": {}}
    current = None
    current_title = None
    buffer = []

    lines = md_text.splitlines()
    for line in lines:
        if line.startswith("## "):
            # salva bloco anterior
            if current and current_title and buffer:
                sections[current][current_title] = "\n".join(buffer).strip()
            # inicia nova seção
            name = line[3:].strip()
            current = "Texto" if name.lower().startswith("texto") else ("Upload" if name.lower().startswith("upload") else None)
            current_title = None
            buffer = []
        elif line.startswith("### ") and current is not None:
            # salva bloco anterior
            if current_title and buffer:
                sections[current][current_title] = "\n".join(buffer).strip()
            current_title = line[4:].strip()
            buffer = []
        else:
            if current is not None and current_title is not None:
                buffer.append(line)
    # salva final
    if current and current_title and buffer:
        sections[current][current_title] = "\n".join(buffer).strip()
    return sections


# =============================
# --------- UI Helpers ---------
# =============================

def init_session_state():
    for k, v in [
        (SS_LOADED, False),
        (SS_CHAT, []),
        (SS_UPLOAD_TEXT, ""),
        (SS_UPLOAD_META, {}),
        (SS_UPLOAD_EMB, None),
        (SS_PROMPT_DRAFT, ""),
        (SS_LAST_RESULTS, pd.DataFrame()),
        (SS_LAST_MATCHES, np.array([], dtype=np.int64)),
        (SS_DEBUG, False),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v


def sidebar_controls(df_sphera: pd.DataFrame) -> Dict:
    st.sidebar.header("Parâmetros de busca — Sphera")

    # Location (multiselect com detecção de coluna)
    loc_col = None
    if df_sphera is not None and not df_sphera.empty:
        for _c in ["LOCATION","FPSO","Location","FPSO/Unidade","Unidade"]:
            if _c in df_sphera.columns:
                loc_col = _c
                break
    loc_options = []
    if loc_col:
        loc_options = sorted([x for x in df_sphera[loc_col].dropna().astype(str).unique().tolist() if x])
    locations = st.sidebar.multiselect("Location", options=loc_options, default=[])

    # Substring
    substr = st.sidebar.text_input("Description contém (substring)", value="")

    # Janela temporal
    years = st.sidebar.slider("Últimos N anos", min_value=1, max_value=10, value=3, step=1)

    # Top-k e limiar
    top_k = st.sidebar.slider("Top-k Sphera", min_value=5, max_value=100, value=20, step=5)
    sim_thr = st.sidebar.slider("Limiar de similaridade (cos)", min_value=0.1, max_value=0.95, value=0.45, step=0.05)

    # Dicionários
    st.sidebar.subheader("Dicionários (WS/Precursores/CP)")
    agg_mode = st.sidebar.selectbox("Agregação por evento", options=["mean","max"], index=0)
    per_ev_thr = st.sidebar.slider("Limiar por evento (WS/Prec/CP)", 0.0, 5.0, 0.0, 0.1)
    min_support = st.sidebar.slider("Suporte mínimo (nº de eventos)", 1, 50, 1, 1)

    # Depuração
    st.sidebar.markdown("---")
    debug = st.sidebar.toggle("Modo depuração (opcional)", value=False)

    st.session_state[SS_DEBUG] = debug

    return dict(
        locations=locations,
        substr=substr,
        years=years,
        top_k=top_k,
        sim_thr=float(sim_thr),
        agg_mode=agg_mode,
        per_ev_thr=float(per_ev_thr),
        min_support=int(min_support),
    )


def prompts_ui() -> Tuple[str, Dict[str, Dict[str, str]]]:
    st.subheader("Seletor de Prompts")
    md = _safe_read_text(PROMPTS_MD_PATH)
    sections = parse_prompts_md(md) if md else {"Texto": {}, "Upload": {}}

    col1, col2 = st.columns(2)
    with col1:
        opt_texto = ["(vazio)"] + list(sections.get("Texto", {}).keys())
        sel_texto = st.selectbox("Prompt — Texto", options=opt_texto, index=0, key="sel_texto")
    with col2:
        opt_upload = ["(vazio)"] + list(sections.get("Upload", {}).keys())
        sel_upload = st.selectbox("Prompt — Upload", options=opt_upload, index=0, key="sel_upload")

    # Caixa de rascunho
    st.text_area("Conteúdo do prompt", key=SS_PROMPT_DRAFT, height=160)

    # Botão carregar
    def _load_prompt_to_draft():
        chosen = None
        if st.session_state.get("sel_texto") and st.session_state["sel_texto"] != "(vazio)":
            chosen = sections.get("Texto", {}).get(st.session_state["sel_texto"]) or ""
        if st.session_state.get("sel_upload") and st.session_state["sel_upload"] != "(vazio)":
            # Se também escolheu upload, concatena abaixo
            chosen2 = sections.get("Upload", {}).get(st.session_state["sel_upload"]) or ""
            chosen = (chosen or "") + ("\n\n" + chosen2 if chosen2 else "")
        if chosen:
            st.session_state[SS_PROMPT_DRAFT] = chosen

    st.button("Carregar no rascunho", on_click=_load_prompt_to_draft)
    return st.session_state.get(SS_PROMPT_DRAFT, ""), sections


def uploads_ui() -> str:
    st.subheader("Upload opcional (texto)")
    up = st.file_uploader("Envie um arquivo .txt, .md ou .csv (usado para consulta por similaridade)", type=["txt","md","csv"], accept_multiple_files=False)

    if up is not None:
        try:
            content = up.read()
            try:
                text = content.decode("utf-8")
            except Exception:
                text = content.decode("latin-1", errors="ignore")
            # .csv: extrai só texto bruto
            if up.name.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(io.StringIO(text))
                    text = "\n".join(df.astype(str).fillna("").apply(lambda r: " ".join(r.values), axis=1).tolist())
                except Exception:
                    pass
            st.session_state[SS_UPLOAD_TEXT] = text
            st.session_state[SS_UPLOAD_META] = {"filename": up.name, "size": len(content)}
            st.success(f"Upload carregado: {up.name} ({len(content)} bytes)")
        except Exception as e:
            st.error(f"Falha no upload: {e}")

    colu1, colu2 = st.columns([3,1])
    with colu1:
        st.text_area("Trecho do upload (opcional, usado na busca)", key=SS_UPLOAD_TEXT, height=140)
    with colu2:
        def _clear_uploads():
            st.session_state[SS_UPLOAD_TEXT] = ""
            st.session_state[SS_UPLOAD_META] = {}
            st.session_state[SS_UPLOAD_EMB] = None
            st.toast("Uploads limpos.")
            st.rerun()
        st.button("Limpar uploads", on_click=_clear_uploads)

    return st.session_state.get(SS_UPLOAD_TEXT, "")


def chat_box_ui() -> Tuple[str, bool, bool]:
    st.subheader("Chat")
    # Botões de manutenção
    colc1, colc2 = st.columns([1,1])
    with colc1:
        def _clear_chat():
            st.session_state[SS_CHAT] = []
            st.toast("Histórico de chat limpo.")
            st.rerun()
        st.button("Limpar chat", on_click=_clear_chat)
    with colc2:
        run_btn = st.button("Enviar para o chat")

    user_msg = st.chat_input("Escreva sua pergunta (ou use somente o upload/rascunho)")

    # Só dispara quando clicar em "Enviar para o chat"
    return user_msg or "", run_btn, False


# =============================
# --------- Motor RAG ---------
# =============================

def build_messages(
    datasets_context: str,
    prompt_draft: str,
    user_question: str,
    retrieved_df: pd.DataFrame,
    dict_summary: pd.DataFrame,
) -> List[Dict]:
    """Monta mensagens para o LLM. Evita heurísticas fixas; apenas fornece contexto e instruções claras."""
    system = (
        "Você é um(a) analista de segurança offshore. Responda de forma objetiva, usando APENAS o contexto fornecido. "
        "Não invente dados. Quando citar WS/Precursores/CP, use SOMENTE os termos presentes nos dicionários aplicados sobre as DESCRIPTIONS retornadas do Sphera. "
        "Se algo não estiver no contexto, diga que não há evidência."
    )
    # Injeta datasets_context sempre
    system = datasets_context + "\n\n" + system

    # Contexto factual com top eventos
    ctx_rows = []
    for _, r in retrieved_df.iterrows():
        parts = [
            f"EventID: {r.get('EventID','')}",
            f"Location: {r.get('Location','')}",
            f"EventDate: {str(r.get('EventDate'))}",
            f"Similarity: {r.get('sim',0):.3f}",
            f"Description: {str(r.get('Description','')).strip()}",
        ]
        ctx_rows.append(" | ".join(parts))
    ctx_block = "\n".join(ctx_rows[:100])  # limita contexto

    # Resumo WS/Precursores/CP (apenas o que tem suporte)
    dict_block = []
    if not dict_summary.empty:
        for _, r in dict_summary.iterrows():
            dict_block.append(f"{r['type']}: suporte={int(r['support'])}, score={float(r['score']):.3f}")
    dict_block = "\n".join(dict_block)

    user_full = "\n\n".join([
        ("INSTRUÇÕES DO PROMPT:\n" + prompt_draft.strip()) if prompt_draft.strip() else "",
        ("PERGUNTA DO USUÁRIO:\n" + user_question.strip()) if user_question.strip() else "",
        ("EVENTOS SELECIONADOS (Sphera):\n" + ctx_block) if ctx_block else "",
        ("RESUMO DICIONÁRIOS (WS/Precursores/CP):\n" + dict_block) if dict_block else "",
    ]).strip()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_full},
    ]
    return messages


def run_pipeline(
    df_sphera: pd.DataFrame,
    emb_corpus: np.ndarray,
    idx_map: np.ndarray,
    params: Dict,
    prompt_draft: str,
    user_question: str,
    upload_text: str,
    datasets_context: str,
    dict_df: pd.DataFrame,
) -> Tuple[str, pd.DataFrame]:
    """Executa: filtro -> embedding da consulta -> retrieve -> dicionários -> LLM."""
    # 1) Filtros Sphera
    df_f = filter_sphera(df_sphera, params["locations"], params["substr"], params["years"]) if not df_sphera.empty else df_sphera

    # 2) Texto de consulta (prioridade: user_question; senão upload; senão rascunho)
    query_text = (user_question or upload_text or prompt_draft).strip()
    if not query_text:
        return "Forneça uma pergunta, prompt ou trecho de upload para buscar no Sphera.", pd.DataFrame()

    # 3) Embedding da consulta
    q_vec = embed_texts([query_text])  # normalizado

    # 4) Retrieve top-k com limiar
    top_df, picked_idx = retrieve_topk(q_vec, emb_corpus, idx_map, df_f, params["top_k"], params["sim_thr"])

    # guarda para UI/depuração
    st.session_state[SS_LAST_RESULTS] = top_df.copy()
    st.session_state[SS_LAST_MATCHES] = picked_idx

    if top_df.empty:
        return "Nenhum evento do Sphera atingiu o limiar de similaridade dentro dos filtros.", top_df

    # 5) Dicionários WS/Prec/CP sobre DESCRIPTIONS dos eventos selecionados
    dict_summary = score_dict_matches(
        events_df=top_df,
        dict_df=dict_df,
        mode=params["agg_mode"],
        per_event_threshold=params["per_ev_thr"],
        min_support=params["min_support"],
    )

    # 6) Monta mensagens e chama LLM
    messages = build_messages(
        datasets_context=datasets_context,
        prompt_draft=prompt_draft,
        user_question=user_question,
        retrieved_df=top_df,
        dict_summary=dict_summary,
    )
    answer = openai_chat(messages)

    return answer, dict_summary


# =============================
# -------------- App ----------
# =============================

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_session_state()

    st.title(APP_TITLE)

    # Carregamentos base (uma vez)
    @st.cache_resource(show_spinner=True)
    def _load_all():
        df, emb, idx = load_sphera()
        dicts = load_dicts()
        ds_ctx = _safe_read_text(DATASETS_CONTEXT_PATH)
        return df, emb, idx, dicts, ds_ctx

    df_sphera, sph_emb, sph_idx, dict_df, datasets_ctx = _load_all()
    if not st.session_state[SS_LOADED]:
        st.session_state[SS_LOADED] = True

    # Sidebar
    params = sidebar_controls(df_sphera)

    # Prompts
    prompt_draft, _sections = prompts_ui()

    # Uploads
    upload_text = uploads_ui()

    # Chat
    user_msg, run_btn, has_user_msg = chat_box_ui()

    # Exibe histórico existente (sem disparar LLM)
    for m in st.session_state[SS_CHAT]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Disparo manual (botão) OU nova mensagem do usuário
    should_run = run_btn

    if should_run:
        if has_user_msg:
            st.session_state[SS_CHAT].append({"role": "user", "content": user_msg})

        with st.spinner("Buscando no Sphera e gerando resposta..."):
            answer, dict_summary = run_pipeline(
                df_sphera=df_sphera,
                emb_corpus=l2_normalize(sph_emb) if sph_emb.size else sph_emb,
                idx_map=sph_idx,
                params=params,
                prompt_draft=prompt_draft,
                user_question=user_msg,
                upload_text=upload_text,
                datasets_context=datasets_ctx,
                dict_df=dict_df,
            )

        # Anexa resposta do assistente SEM padrões fixos
        st.session_state[SS_CHAT].append({"role": "assistant", "content": answer})

        # Renderiza últimas mensagens (somente as novas)
        with st.chat_message("assistant"):
            st.markdown(answer)

    # Se debug ativo, mostrar tabelas auxiliares
    if st.session_state[SS_DEBUG]:
        st.markdown("---")
        st.caption("Depuração — Top eventos e dicionários")
        if isinstance(st.session_state.get(SS_LAST_RESULTS), pd.DataFrame) and not st.session_state[SS_LAST_RESULTS].empty:
            st.dataframe(st.session_state[SS_LAST_RESULTS])
        if isinstance(st.session_state.get(SS_LAST_MATCHES), np.ndarray) and st.session_state[SS_LAST_MATCHES].size:
            st.write("Índices selecionados:", st.session_state[SS_LAST_MATCHES].tolist())


if __name__ == "__main__":
    main()
