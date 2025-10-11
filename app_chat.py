# -*- coding: utf-8 -*-
"""
app_chat.py
---------------------------------------
Pipeline unificado de RAG local para Sphera / GoSee / Docs + Dicionários
(WS / Precursores / CP) com seleção automática de idioma e respeito às
diretivas do prompt do usuário.

Patches inclusos:
- Parsing robusto de diretivas (limiares, janela temporal, fontes, idioma).
- Gates de fontes (Only Sphera / Ignore GoSee/Docs/Upload).
- Respeito a thresholds e anos informados no prompt.
- Uso do upload somente quando solicitado e existente.
- WS/Precursores/CP "a partir do upload" usam apenas upload → índices.
- Limpeza de DESCRIPTION (x000D → \n).
- Mensagens curtas quando não há matches.
- Carregamento tolerante a arquivos ausentes.

Requer: numpy, pandas, scikit-learn (pairwise cosine_similarity).
"""

from __future__ import annotations

import os
import re
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# Configuração de diretórios
# =============================================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
AN_DIR = os.path.join(DATA_DIR, "analytics")
ALT_DIR = os.path.join("/mnt", "data")  # caminho alternativo (colab/sandbox/etc.)
XLSX_DIR = os.path.join(DATA_DIR, "xlsx")

os.makedirs(AN_DIR, exist_ok=True)


# =============================================================================
# Utilitários
# =============================================================================

def path_first_existing(*candidates: str) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _try_read_parquet_csv_jsonl(path: str) -> Optional[pd.DataFrame]:
    try:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        if path.endswith(".csv"):
            return pd.read_csv(path)
        if path.endswith(".jsonl"):
            # jsonlines em pandas
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return pd.DataFrame(rows)
    except Exception:
        return None
    return None


def load_labels_any(*candidates: str) -> Optional[pd.DataFrame]:
    for p in candidates:
        if p and os.path.exists(p):
            df = _try_read_parquet_csv_jsonl(p)
            if df is not None and len(df) > 0:
                return df
    return None


def load_npz_embeddings(*candidates_npz: str) -> Tuple[Optional[np.ndarray], Optional[List[Dict[str, Any]]]]:
    """
    Lê um .npz com array 'embeddings' e, se existir 'meta.json' ao lado ou
    AN equivalent (parquet/jsonl), tenta sincronizar o mesmo número de linhas.
    """
    npz_path = path_first_existing(*candidates_npz)
    if not npz_path:
        return None, None
    try:
        data = np.load(npz_path)
        if "embeddings" in data:
            embs = data["embeddings"]
        else:
            # compat: alguns índices antigos usam chave 'vectors'
            embs = data.get("vectors", None)
        if embs is None:
            return None, None
        # tenta meta ao lado (parquet preferencial; jsonl fallback)
        base = os.path.dirname(npz_path)
        stem = os.path.splitext(os.path.basename(npz_path))[0]

        # heurísticas de meta adjacente
        meta_candidates = [
            os.path.join(base, stem.replace("_embeddings", "") + ".parquet"),
            os.path.join(base, stem.replace("_embeddings", "") + ".jsonl"),
            os.path.join(AN_DIR, stem.replace("_embeddings", "") + ".parquet"),
        ]
        meta_df = None
        for mc in meta_candidates:
            if os.path.exists(mc):
                meta_df = _try_read_parquet_csv_jsonl(mc)
                if meta_df is not None:
                    break

        meta: Optional[List[Dict[str, Any]]] = None
        if meta_df is not None and len(meta_df) == embs.shape[0]:
            meta = meta_df.to_dict(orient="records")

        return embs, meta
    except Exception:
        return None, None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """cosine_similarity compatível com 2D; garante float32->float64 safety"""
    if a is None or b is None:
        return np.zeros((0, 0), dtype=np.float64)
    return cosine_similarity(a.astype(np.float64), b.astype(np.float64))


def clean_desc(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    return txt.replace("x000D", "\n").strip()


def _to_float(x, default=None):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default


# =============================================================================
# Parsing de diretivas do prompt (PATCH #1)
# =============================================================================

PROMPT_FLOAT = r'(?P<val>\d+(?:[.,]\d+)?)'

def parse_user_directives(txt: str) -> Dict[str, Any]:
    t = (txt or "").lower()

    # idioma forçado
    force_en = ('use english' in t) or ('only en' in t)
    force_pt = ('responda em pt' in t) or ('pt-br' in t) or ('apenas pt' in t) or ('responda em português' in t)

    # fontes
    only_sphera = ('only [sphera' in t) or ('somente sphera' in t) or ('use apenas [sphera' in t)
    ignore_gosee = ('ignore gosee' in t) or ('não use gosee' in t)
    ignore_docs  = ('ignore docs' in t)  or ('não use docs' in t)
    ignore_upload= ('ignore upload' in t) or ('não use upload' in t)

    # janelas temporais (anos)
    years = None
    m = re.search(r'(anos|years)\s*=?\s*' + PROMPT_FLOAT, t)
    if m:
        years = int(float(m.group('val').replace(',', '.')))
    m2 = re.search(r'(hoje|today)\s*-\s*' + PROMPT_FLOAT + r'\s*(anos|years)', t)
    if m2 and years is None:
        years = int(float(m2.group(1).replace(',', '.')))  # fallback; não deve ocorrer

    # limiares gerais
    sphera_th = None
    m = re.search(r'(limiar|threshold)\s*=?\s*' + PROMPT_FLOAT, t)
    if m:
        sphera_th = _to_float(m.group('val'))

    # limiares específicos
    ws_th = None
    m = re.search(r'(limiar\s*ws|ws\s*=\s*)' + PROMPT_FLOAT, t)
    if m:
        ws_th = _to_float(m.group('val'))

    prec_th = None
    m = re.search(r'(limiar\s*prec|prec\s*=\s*)' + PROMPT_FLOAT, t)
    if m:
        prec_th = _to_float(m.group('val'))

    cp_th = None
    m = re.search(r'(limiar\s*cp|cp\s*=\s*)' + PROMPT_FLOAT, t)
    if m:
        cp_th = _to_float(m.group('val'))

    # tarefas específicas
    # Obs: tolerante a variações; basta conter "a partir do upload" e um dos termos
    want_ws_only_from_upload   = ('a partir do upload' in t) and (('weak signal' in t) or ('weak signals' in t) or (' ws' in t) or (' ws=' in t))
    want_prec_only_from_upload = ('a partir do upload' in t) and (('precursor' in t) or ('precursors' in t) or (' prec=' in t))
    want_cp_only_from_upload   = ('a partir do upload' in t) and (('cp' in t) or ('fatores' in t) or (' fatores' in t))

    return {
        'force_en': force_en,
        'force_pt': force_pt,
        'only_sphera': only_sphera,
        'ignore_gosee': ignore_gosee,
        'ignore_docs': ignore_docs,
        'ignore_upload': ignore_upload,
        'years': years,
        'sphera_th': sphera_th,
        'ws_th': ws_th,
        'prec_th': prec_th,
        'cp_th': cp_th,
        'want_ws_only_from_upload': want_ws_only_from_upload,
        'want_prec_only_from_upload': want_prec_only_from_upload,
        'want_cp_only_from_upload': want_cp_only_from_upload,
    }


# =============================================================================
# Seleção de fontes (PATCH #2)
# =============================================================================

def select_sources(flags: Dict[str, Any], have_upload_chunks: bool) -> Tuple[bool, bool, bool, bool]:
    use_sphera = True
    use_gosee = True
    use_docs = True
    use_upload = have_upload_chunks

    if flags.get('only_sphera'):
        use_gosee = False
        use_docs = False
        # upload só entra se explicitamente não foi ignorado e for necessário para a tarefa
        if flags.get('ignore_upload'):
            use_upload = False

    if flags.get('ignore_gosee'):
        use_gosee = False
    if flags.get('ignore_docs'):
        use_docs = False
    if flags.get('ignore_upload'):
        use_upload = False

    return use_sphera, use_gosee, use_docs, use_upload


# =============================================================================
# Carregamento dos índices / labels
# =============================================================================

# Sphera
SPHERA_NPZ = path_first_existing(
    os.path.join(AN_DIR, "sphera_embeddings.npz"),
    os.path.join(ALT_DIR, "sphera_embeddings.npz"),
)
SPHERA_DF = load_labels_any(
    os.path.join(AN_DIR, "sphera.parquet"),
    os.path.join(ALT_DIR, "sphera.parquet"),
)

sphera_embs, sphera_meta = load_npz_embeddings(SPHERA_NPZ) if SPHERA_NPZ else (None, None)
if (sphera_meta is None) and (SPHERA_DF is not None):
    # fallback de meta
    sphera_meta = SPHERA_DF.to_dict(orient="records") if len(SPHERA_DF) else None

# GoSee (opcional)
GOSEE_NPZ = path_first_existing(
    os.path.join(AN_DIR, "gosee_embeddings.npz"),
    os.path.join(ALT_DIR, "gosee_embeddings.npz"),
)
GOSEE_DF = load_labels_any(
    os.path.join(AN_DIR, "gosee.parquet"),
    os.path.join(ALT_DIR, "gosee.parquet"),
)
gosee_embs, gosee_meta = load_npz_embeddings(GOSEE_NPZ) if GOSEE_NPZ else (None, None)
if (gosee_meta is None) and (GOSEE_DF is not None):
    gosee_meta = GOSEE_DF.to_dict(orient="records") if len(GOSEE_DF) else None

# Docs/History (opcional – pode ser ausente)
HIST_NPZ = path_first_existing(
    os.path.join(AN_DIR, "history_embeddings.npz"),
    os.path.join(ALT_DIR, "history_embeddings.npz"),
)
HIST_JSONL = path_first_existing(
    os.path.join(AN_DIR, "history_texts.jsonl"),
    os.path.join(ALT_DIR, "history_texts.jsonl"),
)
hist_embs, hist_meta = load_npz_embeddings(HIST_NPZ) if HIST_NPZ else (None, None)
if (hist_meta is None) and HIST_JSONL:
    try:
        rows = [json.loads(x) for x in open(HIST_JSONL, "r", encoding="utf-8") if x.strip()]
        hist_meta = rows
    except Exception:
        hist_meta = None

# WS PT/EN
WS_PT_NPZ = path_first_existing(
    os.path.join(AN_DIR, "ws_embeddings_pt.npz"),
    os.path.join(ALT_DIR, "ws_embeddings_pt.npz"),
)
WS_EN_NPZ = path_first_existing(
    os.path.join(AN_DIR, "ws_embeddings_en.npz"),
    os.path.join(ALT_DIR, "ws_embeddings_en.npz"),
)
WS_PT_DF = load_labels_any(
    os.path.join(AN_DIR, "ws_embeddings_pt.parquet"),
    os.path.join(ALT_DIR, "ws_embeddings_pt.parquet"),
)
WS_EN_DF = load_labels_any(
    os.path.join(AN_DIR, "ws_embeddings_en.parquet"),
    os.path.join(ALT_DIR, "ws_embeddings_en.parquet"),
)

ws_pt_embs, ws_pt_meta = load_npz_embeddings(WS_PT_NPZ) if WS_PT_NPZ else (None, None)
if (ws_pt_meta is None) and (WS_PT_DF is not None):
    ws_pt_meta = WS_PT_DF.to_dict(orient="records")

ws_en_embs, ws_en_meta = load_npz_embeddings(WS_EN_NPZ) if WS_EN_NPZ else (None, None)
if (ws_en_meta is None) and (WS_EN_DF is not None):
    ws_en_meta = WS_EN_DF.to_dict(orient="records")

# Prec PT/EN  (com labels já “HTO/Precursor” conforme fix_prec_labels.py)
PREC_PT_NPZ = path_first_existing(
    os.path.join(AN_DIR, "prec_embeddings_pt.npz"),
    os.path.join(ALT_DIR, "prec_embeddings_pt.npz"),
)
PREC_EN_NPZ = path_first_existing(
    os.path.join(AN_DIR, "prec_embeddings_en.npz"),
    os.path.join(ALT_DIR, "prec_embeddings_en.npz"),
)
PREC_PT_DF = load_labels_any(
    os.path.join(AN_DIR, "prec_embeddings_pt.parquet"),
    os.path.join(ALT_DIR, "prec_embeddings_pt.parquet"),
)
PREC_EN_DF = load_labels_any(
    os.path.join(AN_DIR, "prec_embeddings_en.parquet"),
    os.path.join(ALT_DIR, "prec_embeddings_en.parquet"),
)

prec_pt_embs, prec_pt_meta = load_npz_embeddings(PREC_PT_NPZ) if PREC_PT_NPZ else (None, None)
if (prec_pt_meta is None) and (PREC_PT_DF is not None):
    prec_pt_meta = PREC_PT_DF.to_dict(orient="records")

prec_en_embs, prec_en_meta = load_npz_embeddings(PREC_EN_NPZ) if PREC_EN_NPZ else (None, None)
if (prec_en_meta is None) and (PREC_EN_DF is not None):
    prec_en_meta = PREC_EN_DF.to_dict(orient="records")

# CP (labels/embeddings opcionais; muitos times só usam labels)
CP_LABELS_DF = load_labels_any(
    os.path.join(AN_DIR, "cp_labels.parquet"),
    os.path.join(ALT_DIR, "cp_labels.parquet"),
    os.path.join(AN_DIR, "cp_labels.jsonl"),
    os.path.join(ALT_DIR, "cp_labels.jsonl"),
)
CP_NPZ = path_first_existing(
    os.path.join(AN_DIR, "cp_embeddings.npz"),
    os.path.join(ALT_DIR, "cp_embeddings.npz"),
)
cp_embs, cp_meta = load_npz_embeddings(CP_NPZ) if CP_NPZ else (None, None)
if (cp_meta is None) and (CP_LABELS_DF is not None):
    cp_meta = CP_LABELS_DF.to_dict(orient="records")


# =============================================================================
# Filtros de tempo e ranking (PATCH #3)
# =============================================================================

def filter_by_years_records(records: List[Dict[str, Any]], years: Optional[int]) -> List[Dict[str, Any]]:
    """Filtra 'records' (dicts) pela chave EVENT_DATE dentro de anos."""
    if (records is None) or (years is None):
        return records or []
    now = datetime.now().date()
    cutoff = now - timedelta(days=365 * years)
    out = []
    for r in records:
        d = r.get("EVENT_DATE", None)
        if d is None:
            continue
        try:
            d2 = pd.to_datetime(d, errors="coerce").date()
            if d2 and (d2 >= cutoff):
                out.append(r)
        except Exception:
            pass
    return out


def rank_sphera_by_upload(sphera_emb: np.ndarray,
                          sphera_records: List[Dict[str, Any]],
                          upload_emb: np.ndarray,
                          upload_chunks: List[str],
                          th: float) -> List[Dict[str, Any]]:
    """
    sim(event) = max cosine(upload_chunk, event)
    upload_snippet = chunk do upload que atingiu o máximo
    """
    if (upload_emb is None) or (upload_emb.shape[0] == 0):
        return []
    if (sphera_emb is None) or (sphera_records is None) or (len(sphera_records) == 0):
        return []
    sims = cosine_sim(upload_emb, sphera_emb)  # [n_up, n_events]
    best_idx = sims.argmax(axis=0)             # por evento
    best_sim = sims[best_idx, range(sims.shape[1])]

    out = []
    for j, s in enumerate(best_sim):
        if s >= th:
            ev = dict(sphera_records[j])
            out.append({
                "event_id": ev.get("EVENT_NUMBER", ev.get("ID", j)),
                "sim": float(s),
                "description": clean_desc(ev.get("DESCRIPTION", "")),
                "upload_snippet": upload_chunks[best_idx[j]] if upload_chunks else None,
                "severity": ev.get("SEVERITY", None),
                "fpso": ev.get("FPSO", ev.get("LOCATION", "")),
                "event_date": ev.get("EVENT_DATE", None),
            })
    out.sort(key=lambda r: r["sim"], reverse=True)
    return out


def rank_sphera_by_query(sphera_emb: np.ndarray,
                         sphera_records: List[Dict[str, Any]],
                         query_emb: np.ndarray,
                         th: float) -> List[Dict[str, Any]]:
    """
    Modo "Only Sphera" sem upload: sim(event) = cosine(query, event)
    """
    if (query_emb is None) or (query_emb.ndim == 1 and query_emb.size == 0):
        return []
    if (sphera_emb is None) or (sphera_records is None) or (len(sphera_records) == 0):
        return []
    q = query_emb.reshape(1, -1)
    sims = cosine_sim(q, sphera_emb).ravel()
    out = []
    for j, s in enumerate(sims):
        if s >= th:
            ev = dict(sphera_records[j])
            out.append({
                "event_id": ev.get("EVENT_NUMBER", ev.get("ID", j)),
                "sim": float(s),
                "description": clean_desc(ev.get("DESCRIPTION", "")),
                "upload_snippet": None,
                "severity": ev.get("SEVERITY", None),
                "fpso": ev.get("FPSO", ev.get("LOCATION", "")),
                "event_date": ev.get("EVENT_DATE", None),
            })
    out.sort(key=lambda r: r["sim"], reverse=True)
    return out


# =============================================================================
# Dicionários a partir do upload (PATCH #4)
# =============================================================================

def detect_lang_for_dicts(flags: Dict[str, Any], default: str = "pt") -> str:
    if flags.get("force_en"):
        return "en"
    if flags.get("force_pt"):
        return "pt"
    return default


def _dict_label(meta_row: Dict[str, Any]) -> str:
    """
    Tenta retornar o melhor campo de label, priorizando:
    - 'label' (pós-fix precisa estar HTO/Precursor concatenado)
    - 'Label' / 'LABEL'
    - fallback em 'PT'/'EN' se existir.
    """
    for k in ("label", "Label", "LABEL"):
        if k in meta_row and isinstance(meta_row[k], str) and meta_row[k].strip():
            return meta_row[k]
    for k in ("PT", "EN", "term", "name"):
        if k in meta_row and isinstance(meta_row[k], str) and meta_row[k].strip():
            return meta_row[k]
    return str(meta_row)


def match_dict_from_upload(upload_emb: np.ndarray,
                           upload_texts: List[str],
                           dict_emb: np.ndarray,
                           dict_meta: List[Dict[str, Any]],
                           th: float) -> List[Dict[str, Any]]:
    """
    Para cada item do dicionário, score = max cosine(upload_chunk, dict_item)
    Retorna itens com score >= th.
    """
    if (upload_emb is None) or (upload_emb.shape[0] == 0):
        return []
    if (dict_emb is None) or (dict_meta is None) or (len(dict_meta) == 0):
        return []

    sims = cosine_sim(upload_emb, dict_emb)   # [n_up, n_dict]
    best = sims.max(axis=0)                   # por item do dicionário
    src  = sims.argmax(axis=0)

    out = []
    for k, s in enumerate(best):
        if s >= th:
            label = _dict_label(dict_meta[k])
            out.append({
                "label": label,
                "sim": float(s),
                "snippet": upload_texts[src[k]] if upload_texts else None,
            })
    out.sort(key=lambda r: r["sim"], reverse=True)
    return out


# =============================================================================
# Encoder "mock" para query/upload (o app real já possui o encoder ativo)
# =============================================================================

def encode_texts_mock(texts: List[str], dim: int = 384, seed: int = 13) -> np.ndarray:
    """
    Placeholder de encoder caso você queira rodar standalone.
    No seu app real, substitua este encoder por sentence-transformers.
    """
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    rng = np.random.RandomState(seed)
    X = rng.rand(len(texts), dim).astype(np.float32)
    # normaliza para unit norm para produzir cosines realistas
    X /= np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    return X


# =============================================================================
# API principal
# =============================================================================

DEFAULT_SPHERA_TH = 0.50
DEFAULT_WS_TH = 0.50
DEFAULT_PREC_TH = 0.50
DEFAULT_CP_TH = 0.50
DEFAULT_YEARS = None  # sem filtro temporal se não especificado

def run_query(
    user_prompt: str,
    query_text: Optional[str] = None,
    upload_sentences: Optional[List[str]] = None,
    encoder_fn = encode_texts_mock,   # troque pelo seu encode real
) -> Dict[str, Any]:
    """
    Executa a consulta conforme diretivas.
    Retorna um dicionário com as chaves possíveis:
      - sphera_events: List[dict]
      - ws_from_upload: List[dict]
      - prec_from_upload: List[dict]
      - cp_from_upload: List[dict]
      - meta: info de diagnóstico (quais fontes ativas, shapes, thresholds, anos)
    """
    flags = parse_user_directives(user_prompt or "")
    upload_sentences = upload_sentences or []

    use_sphera, use_gosee, use_docs, use_upload = select_sources(flags, have_upload_chunks=bool(upload_sentences))

    # thresholds
    sph_th = flags.get("sphera_th")
    ws_th  = flags.get("ws_th")
    pr_th  = flags.get("prec_th")
    cp_th  = flags.get("cp_th")

    sph_th = sph_th if isinstance(sph_th, (int, float)) else DEFAULT_SPHERA_TH
    ws_th  = ws_th  if isinstance(ws_th,  (int, float)) else DEFAULT_WS_TH
    pr_th  = pr_th  if isinstance(pr_th,  (int, float)) else DEFAULT_PREC_TH
    cp_th  = cp_th  if isinstance(cp_th,  (int, float)) else DEFAULT_CP_TH
    years  = flags.get("years", DEFAULT_YEARS)

    # idioma p/ dicionários
    dict_lang = detect_lang_for_dicts(flags, default="pt")

    # Encode query e upload (no app real, use seu encoder)
    q_emb = encoder_fn([query_text]) if query_text else None
    if q_emb is not None and q_emb.ndim == 2 and q_emb.shape[0] == 1:
        q_emb = q_emb[0]

    up_emb = encoder_fn(upload_sentences) if (use_upload and upload_sentences) else None

    # ---- Sphera ----
    sphera_events: List[Dict[str, Any]] = []
    if use_sphera and (sphera_embs is not None) and (sphera_meta is not None):
        # filtro temporal
        sph_records = sphera_meta
        if years is not None:
            sph_records = filter_by_years_records(sph_records, years)
            # filtra também os embeddings pelo mesmo mask (quando possível)
            if len(sph_records) != len(sphera_meta):
                # tenta mapear por EVENT_NUMBER ou índice posicional
                # (robusto: alinhamento por posição; em produção é melhor manter índices consistentes)
                idxs = []
                old_by_id = {}
                for i, r in enumerate(sphera_meta):
                    rid = r.get("EVENT_NUMBER", i)
                    old_by_id[rid] = i
                for r in sph_records:
                    rid = r.get("EVENT_NUMBER", None)
                    i = old_by_id.get(rid, None)
                    if i is None:
                        # fallback posicional — assume que a ordem foi preservada
                        # (se não, apenas não filtra embeddings)
                        idxs = None
                        break
                    idxs.append(i)
                if idxs is not None:
                    embs = sphera_embs[idxs, :]
                else:
                    embs = sphera_embs
            else:
                embs = sphera_embs
        else:
            sph_records = sphera_meta
            embs = sphera_embs

        # rankeamento
        if (up_emb is not None) and (len(upload_sentences) > 0):
            sphera_events = rank_sphera_by_upload(embs, sph_records, up_emb, upload_sentences, sph_th)
        elif q_emb is not None:
            sphera_events = rank_sphera_by_query(embs, sph_records, q_emb, sph_th)
        else:
            sphera_events = []  # sem base de consulta

    # ---- Dicionários a partir do upload ----
    ws_from_upload: List[Dict[str, Any]] = []
    prec_from_upload: List[Dict[str, Any]] = []
    cp_from_upload: List[Dict[str, Any]] = []

    if flags.get("want_ws_only_from_upload"):
        if dict_lang == "en":
            ws_from_upload = match_dict_from_upload(up_emb, upload_sentences, ws_en_embs, ws_en_meta, ws_th)
        else:
            ws_from_upload = match_dict_from_upload(up_emb, upload_sentences, ws_pt_embs, ws_pt_meta, ws_th)

    if flags.get("want_prec_only_from_upload"):
        if dict_lang == "en":
            prec_from_upload = match_dict_from_upload(up_emb, upload_sentences, prec_en_embs, prec_en_meta, pr_th)
        else:
            prec_from_upload = match_dict_from_upload(up_emb, upload_sentences, prec_pt_embs, prec_pt_meta, pr_th)

    if flags.get("want_cp_only_from_upload"):
        # alguns times não possuem embeddings de CP; se ausentes, a lista ficará vazia
        cp_from_upload = match_dict_from_upload(up_emb, upload_sentences, cp_embs, cp_meta, cp_th)

    # ---- Monta resposta ----
    info = {
        "sources": {
            "sphera": bool(use_sphera and (sphera_embs is not None) and (sphera_meta is not None)),
            "gosee":  bool(use_gosee and (gosee_embs is not None) and (gosee_meta is not None)),
            "docs":   bool(use_docs and (hist_embs is not None) and (hist_meta is not None)),
            "upload": bool(use_upload and (up_emb is not None) and (up_emb.shape[0] > 0)),
        },
        "thresholds": {"sphera": sph_th, "ws": ws_th, "prec": pr_th, "cp": cp_th},
        "years_filter": years,
        "dict_lang": dict_lang,
    }

    return {
        "sphera_events": sphera_events,
        "ws_from_upload": ws_from_upload,
        "prec_from_upload": prec_from_upload,
        "cp_from_upload": cp_from_upload,
        "meta": info,
    }


# =============================================================================
# Renderização simples (para debug/local)
# =============================================================================

def _fmt_events_table(rows: List[Dict[str, Any]], include_severity: bool = False, include_upload_col: bool = True) -> str:
    if not rows:
        return ""
    cols = ["Event Id", "Similarity", "Description"]
    if include_severity:
        cols.insert(2, "SEVERITY")
    if include_upload_col:
        cols.append("Upload sentence used")

    lines = ["\t".join(cols)]
    for r in rows:
        cells = [
            str(r.get("event_id", "")),
            f"{r.get('sim', 0.0):.3f}",
        ]
        if include_severity:
            cells.append("" if r.get("severity", None) is None else str(r.get("severity")))
        cells.append(clean_desc(r.get("description", "")))
        if include_upload_col:
            cells.append((r.get("upload_snippet") or "").replace("\n", " ")[:500])
        lines.append("\t".join(cells))
    return "\n".join(lines)


def _fmt_dict_table(rows: List[Dict[str, Any]], header: str = "Termos") -> str:
    if not rows:
        return ""
    lines = [f"{header}\tSimilarity\tUpload snippet"]
    for r in rows[:50]:
        lines.append(f"{r.get('label','')}\t{r.get('sim',0.0):.3f}\t{(r.get('snippet','') or '').replace(os.linesep,' ')[:300]}")
    return "\n".join(lines)


# =============================================================================
# Execução direta (exemplo mínimo)
# =============================================================================

if __name__ == "__main__":
    # Exemplo mínimo de uso local.
    # No app real, substitua o encoder_fn por seu encoder (ex.: all-MiniLM-L6-v2)
    prompt = (
        "Somente Sphera. Limiar = 0.30. "
        "Considerar EVENT_DATE ≥ HOJE-3 anos. "
        "Monte tabela: Event Id | Description | Similaridade. "
        "Ignore GoSee/Docs/Upload."
    )
    result = run_query(
        user_prompt=prompt,
        query_text="queda de objetos durante operação de carga",
        upload_sentences=[],  # ignorado
        encoder_fn=encode_texts_mock,
    )

    sph = result["sphera_events"]
    ws  = result["ws_from_upload"]
    prec= result["prec_from_upload"]
    cp  = result["cp_from_upload"]

    if sph:
        print("Eventos do Sphera (≥ limiar):")
        print(_fmt_events_table(sph, include_severity=True, include_upload_col=False))
    else:
        print(f"Nenhum evento do Sphera encontrado ≥ {result['meta']['thresholds']['sphera']} (com as fontes solicitadas).")

    # Exemplos de “a partir do upload…”
    prompt_ws = "A partir do upload, retorne somente Weak Signals (WS) (PT). Limiar WS=0.25."
    upload = [
        "Alarme recorrente no painel foi reconhecido repetidas vezes sem investigação.",
        "Durante a manutenção, um operador não certificado executou a tarefa.",
        "Correção temporária foi deixada sem follow-up após teste."
    ]
    result2 = run_query(
        user_prompt=prompt_ws,
        query_text=None,
        upload_sentences=upload,
        encoder_fn=encode_texts_mock,
    )
    if result2["ws_from_upload"]:
        print("\nWS (a partir do upload):")
        print(_fmt_dict_table(result2["ws_from_upload"], header="WS Label"))
    else:
        print(f"\nNenhum WS ≥ {result2['meta']['thresholds']['ws']} (com as fontes solicitadas).")
