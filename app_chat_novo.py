# -*- coding: utf-8 -*-
"""
App Chat Novo — v5 killer

Foco desta versão:
- **Filtro Location impecável**: detecção de coluna dinâmica (LOCATION → FPSO → Location → FPSO/Unidade → Unidade) com hidratação multi‑fonte
  (parquet → npz → excel por chave/índice). Multiselect sempre baseado no **dataframe completo já hidratado**.
- **Depuração embutida** (expander "Debug Location") quando forem encontradas menos de 2 opções de LOCATION.
- Mantidas todas funcionalidades anteriores (somente Sphera, dicionários, limiares, prompts, limpar chat/uploads/rascunho, execução só ao clicar,
  descrição completa, sem duplicações de saída).
"""

import os, re, io
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ========================== Config ==========================
st.set_page_config(page_title="SAFETY • CHAT", page_icon="💬", layout="wide")

DATA_DIR = Path("data"); AN_DIR = DATA_DIR/"analytics"; XLSX_DIR = DATA_DIR/"xlsx"
DATASETS_CONTEXT_PATH = DATA_DIR/"datasets_context.md"
PROMPTS_MD_PATH = DATA_DIR/"prompts"/"prompts.md"

SPH_PQ_PATH  = AN_DIR/"sphera.parquet"
SPH_NPZ_PATH = AN_DIR/"sphera_embeddings.npz"
XLSX_LOCATION_PATH = XLSX_DIR/"TRATADO_safeguardOffShore.xlsx"

# Modelo
OLLAMA_HOST    = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", ""))
OLLAMA_MODEL   = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", ""))
OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY"))
HEADERS_JSON   = {"Authorization": f"Bearer {OLLAMA_API_KEY}", "Content-Type": "application/json"} if OLLAMA_API_KEY else {"Content-Type": "application/json"}

ST_MODEL_NAME = os.getenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# ========================== Helpers ==========================

def _fatal(msg: str): st.error(msg); st.stop()

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    _fatal(f"❌ sentence-transformers indisponível: {e}")

@st.cache_resource(show_spinner=False)
def ensure_st_encoder():
    try: return SentenceTransformer(ST_MODEL_NAME)
    except Exception as e: _fatal(f"❌ Não foi possível carregar o encoder: {e}")

@st.cache_data(show_spinner=False)
def load_npz_embeddings(path: Path) -> Optional[np.ndarray]:
    if not path.exists(): return None
    try:
        with np.load(str(path), allow_pickle=True) as z:
            for key in ("embeddings","E","X","vectors","vecs"):
                if key in z:
                    E = np.array(z[key]).astype(np.float32, copy=False)
                    n = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
                    return (E/n).astype(np.float32)
            best_k, best_n = None, -1
            for k in z.files:
                arr = z[k]
                if isinstance(arr, np.ndarray) and arr.ndim==2 and arr.shape[0]>best_n:
                    best_k, best_n = k, arr.shape[0]
            if best_k is None: return None
            E = np.array(z[best_k]).astype(np.float32, copy=False)
            n = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
            return (E/n).astype(np.float32)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_prompts_md(md_path: Path) -> Dict[str, List[Dict[str,str]]]:
    if not md_path.exists(): return {"Texto":[],"Upload":[]}
    raw = md_path.read_text(encoding="utf-8"); sections = re.split(r"(?m)^##\s+", raw)
    out = {"Texto":[],"Upload":[]}
    for sec in sections:
        sec=sec.strip();
        if not sec: continue
        head, _, body = sec.partition("\n"); head=head.strip()
        if head not in ("Texto","Upload"): continue
        parts = re.split(r"(?m)^###\s+", body)
        for p in parts:
            p=p.strip();
            if not p: continue
            title, _, text = p.partition("\n")
            out[head].append({"title":title.strip(),"body":text.strip()})
    return out

@st.cache_data(show_spinner=False)
def load_file_text(p: Path) -> str:
    try: return p.read_text(encoding="utf-8")
    except Exception as e: return f"[AVISO] Não consegui ler {p}: {e} (continuando sem este contexto)"

# ========================== Location utils ==========================

def get_sphera_location_col(df: pd.DataFrame) -> Optional[str]:
    """Retorna a melhor coluna de Location. Prioridade: LOCATION → FPSO → Location → FPSO/Unidade → Unidade. Nunca AREA."""
    if df is None or df.empty: return None
    for c in ["LOCATION","FPSO","Location","FPSO/Unidade","Unidade"]:
        if c in df.columns: return c
    return None

@st.cache_data(show_spinner=False)
def hydrate_location(df: pd.DataFrame, npz_path: Path, xlsx_path: Path) -> pd.DataFrame:
    d = df.copy()
    # 1) Parquet já traz LOCATION?
    if "LOCATION" in d.columns and d["LOCATION"].notna().any():
        return d
    # 2) NPZ
    try:
        if npz_path.exists():
            with np.load(str(npz_path), allow_pickle=True) as z:
                loc_arr=None
                for k in ("LOCATION","location","locations"):
                    if k in z: loc_arr=z[k]; break
                if loc_arr is None and "meta" in z:
                    meta = z["meta"].item() if isinstance(z["meta"], np.ndarray) else z["meta"]
                    if isinstance(meta, dict): loc_arr = meta.get("LOCATION") or meta.get("location")
                if loc_arr is not None:
                    loc_arr=np.asarray(loc_arr)
                    if loc_arr.shape[0]==len(d):
                        d["LOCATION"]=pd.Series(loc_arr).astype(str); return d
    except Exception:
        pass
    # 3) Excel por chave/índice
    if xlsx_path.exists():
        try:
            xls=pd.ExcelFile(xlsx_path); candidate=None
            for sh in xls.sheet_names:
                tmp=xls.parse(sh)
                if "LOCATION" in tmp.columns and tmp["LOCATION"].notna().any():
                    candidate=tmp; break
            if candidate is not None:
                cand=candidate.copy()
                df_keys=[c for c in ["Event ID","EVENT_NUMBER","EVENTID"] if c in d.columns]
                xls_keys=[c for c in ["Event ID","EVENT_NUMBER","EVENTID"] if c in cand.columns]
                merged=None
                if df_keys and xls_keys:
                    kd, kx = df_keys[0], xls_keys[0]
                    a=d.copy(); a[kd]=a[kd].astype(str)
                    b=cand[[kx,"LOCATION"]].copy(); b[kx]=b[kx].astype(str)
                    merged=a.merge(b, left_on=kd, right_on=kx, how="left")
                    if "LOCATION_y" in merged.columns:
                        if "LOCATION" in merged.columns:
                            merged["LOCATION"] = merged["LOCATION"].fillna(merged["LOCATION_y"])
                        else:
                            merged.rename(columns={"LOCATION_y":"LOCATION"}, inplace=True)
                        merged.drop(columns=[c for c in ["LOCATION_y","LOCATION_x",kx] if c in merged.columns], inplace=True)
                if merged is None or "LOCATION" not in merged.columns or merged["LOCATION"].isna().all():
                    if len(cand)==len(d):
                        d["LOCATION"]=cand["LOCATION"].astype(str).values; return d
                    elif len(cand)>0 and len(d)>0:
                        d["LOCATION"]=cand["LOCATION"].astype(str).iloc[:len(d)].values; return d
                else:
                    d=merged
        except Exception:
            pass
    return d

# ========================== Carregamento base ==========================
if not SPH_PQ_PATH.exists(): st.error(f"Parquet do Sphera não encontrado em {SPH_PQ_PATH}")

df_sph = pd.read_parquet(SPH_PQ_PATH) if SPH_PQ_PATH.exists() else pd.DataFrame()
df_sph = hydrate_location(df_sph, SPH_NPZ_PATH, XLSX_LOCATION_PATH)

E_sph = load_npz_embeddings(SPH_NPZ_PATH)

WS_NPZ, WS_LBL = AN_DIR/"ws_embeddings_pt.npz", AN_DIR/"ws_embeddings_pt.parquet"
PREC_NPZ, PREC_LBL = AN_DIR/"prec_embeddings_pt.npz", AN_DIR/"prec_embeddings_pt.parquet"
CP_NPZ, CP_LBL = AN_DIR/"cp_embeddings.npz", AN_DIR/"cp_labels.parquet"

E_ws   = load_npz_embeddings(WS_NPZ) if WS_NPZ.exists() else None
L_ws   = (pd.read_parquet(WS_LBL) if WS_LBL.exists() else None)
E_prec = load_npz_embeddings(PREC_NPZ) if PREC_NPZ.exists() else None
L_prec = (pd.read_parquet(PREC_LBL) if PREC_LBL.exists() else None)
E_cp   = load_npz_embeddings(CP_NPZ) if CP_NPZ.exists() else None
L_cp   = (pd.read_parquet(CP_LBL) if CP_LBL.exists() else None)

# ========================== Estado ==========================
if "system_prompt" not in st.session_state:
    pre = ("Você é o ESO-CHAT para segurança operacional (óleo e gás). "
           "Responda em PT-BR, cite IDs/sim quando usar buscas locais, e não invente dados fora dos contextos fornecidos.\n\n")
    sys_ctx = (load_file_text(DATASETS_CONTEXT_PATH) if DATASETS_CONTEXT_PATH.exists() else "")
    st.session_state.system_prompt = pre + ("=== DATASETS_CONTEXT ===\n" + sys_ctx if sys_ctx else "")
if "chat" not in st.session_state: st.session_state.chat=[]
if "draft_prompt" not in st.session_state: st.session_state.draft_prompt=""
if "_clear_draft_flag" not in st.session_state: st.session_state._clear_draft_flag=False
if "st_encoder" not in st.session_state: st.session_state.st_encoder = ensure_st_encoder()
if "upld_texts" not in st.session_state: st.session_state.upld_texts=[]

# ========================== Encode ==========================
@st.cache_data(show_spinner=False)
def encode_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    M = st.session_state.st_encoder.encode(texts, batch_size=batch_size, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    return M

@st.cache_data(show_spinner=False)
def encode_query(q: str) -> np.ndarray:
    v = st.session_state.st_encoder.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
    v /= (np.linalg.norm(v)+1e-9); return v

# ========================== Filtros / Similaridade ==========================
@st.cache_data(show_spinner=False)
def filter_sphera(df: pd.DataFrame, locations: List[str], substr: str, years: int) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    out=df.copy()
    if "EVENT_DATE" in out.columns:
        out["EVENT_DATE"]=pd.to_datetime(out["EVENT_DATE"], errors="coerce")
        cutoff = pd.Timestamp(datetime.utcnow()-timedelta(days=365*years))
        out = out[out["EVENT_DATE"]>=cutoff]
    loc_col = get_sphera_location_col(out)
    if loc_col and locations:
        sel = set([str(x).strip() for x in locations if str(x).strip()])
        out = out[out[loc_col].astype(str).isin(sel)]
    desc_col = "Description" if "Description" in out.columns else ("DESCRIPTION" if "DESCRIPTION" in out.columns else None)
    if desc_col and substr:
        pat=re.escape(substr); out = out[out[desc_col].astype(str).str.contains(pat, case=False, na=False, regex=True)]
    return out

@st.cache_data(show_spinner=False)
def sphera_similar_to_text(query_text: str, min_sim: float, years: int, topk: int,
                           df_base: pd.DataFrame, E_base: Optional[np.ndarray],
                           substr: str, locations: List[str]) -> List[Tuple[str,float,pd.Series]]:
    if not query_text or df_base is None or df_base.empty or E_base is None or E_base.size==0: return []
    base = filter_sphera(df_base, locations, substr, years)
    if base.empty: return []
    try:
        idx_map = base.index.to_numpy()
        if np.issubdtype(idx_map.dtype, np.integer): E_view = E_base[idx_map,:]
        else: E_view = E_base; base = df_base
    except Exception: E_view = E_base; base = df_base
    qv = encode_query(query_text); sims = (E_view @ qv).astype(float)
    ord_idx = np.argsort(-sims)
    id_col = next((c for c in ["Event ID","EVENT_NUMBER","EVENTID"] if c in base.columns), None)
    out=[]; kept=0
    for i in ord_idx:
        s=float(sims[i]);
        if s<min_sim: continue
        row=base.iloc[int(i)]; evid = row.get(id_col, f"row{i}") if id_col else f"row{i}"
        out.append((str(evid), s, row)); kept+=1
        if kept>=topk: break
    return out

# ========================== Agregação dicionários ==========================
@st.cache_data(show_spinner=False)
def aggregate_dict_matches_over_hits(
    hits: List[Tuple[str,float,pd.Series]],
    E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
    thr_ws_sim: float, thr_prec_sim: float, thr_cp_sim: float,
    topn_ws: int, topn_prec: int, topn_cp: int,
    agg_mode: str="max", per_event_thr: float=0.30, min_support: int=1,
) -> Dict[str, List[Tuple[str,float,int]]]:
    if not hits: return {"ws":[],"prec":[],"cp":[]}
    descs=[str(r.get("Description", r.get("DESCRIPTION",""))).strip() for _,_,r in hits]
    descs=[d for d in descs if d];
    if not descs: return {"ws":[],"prec":[],"cp":[]}
    V_desc = encode_texts(descs, batch_size=32).T
    def _score(E_bank, labels_df, thr_sim, topn_target):
        if E_bank is None or labels_df is None or len(labels_df)!=E_bank.shape[0]: return []
        S=(E_bank @ V_desc); support=(S>=per_event_thr).sum(axis=1)
        sims = S.mean(axis=1) if agg_mode=="mean" else S.max(axis=1)
        mask=(support>=min_support) & (sims>=thr_sim); idx=np.where(mask)[0]
        if idx.size==0: return []
        order=idx[np.argsort(sims[idx])[::-1]]; out=[]
        for i in order[:topn_target]:
            label=str(labels_df.iloc[i].get("label", labels_df.iloc[i].get("text", f"TERM_{i}")))
            out.append((label, float(sims[i]), int(support[i])))
        return out
    return {"ws":_score(E_ws,L_ws,thr_ws_sim,topn_ws),
            "prec":_score(E_prec,L_prec,thr_prec_sim,topn_prec),
            "cp":_score(E_cp,L_cp,thr_cp_sim,topn_cp)}

# ========================== Modelo ==========================

def ollama_chat(messages, model=None, temperature=0.2, stream=False, timeout=120):
    if not (OLLAMA_HOST and (model or OLLAMA_MODEL)):
        raise RuntimeError("Modelo não configurado. Defina OLLAMA_HOST e OLLAMA_MODEL.")
    import requests
    r = requests.post(f"{OLLAMA_HOST}/api/chat", headers=HEADERS_JSON, json={
        "model": model or OLLAMA_MODEL, "messages": messages, "temperature": float(temperature), "stream": bool(stream)
    }, timeout=timeout)
    r.raise_for_status(); return r.json()

# ========================== Sidebar ==========================
@st.cache_data(show_spinner=False)
def _location_options_from(df_full: pd.DataFrame) -> Tuple[Optional[str], List[str]]:
    col = get_sphera_location_col(df_full)
    if not col: return None, []
    s = df_full[col].astype(str).str.strip()
    s = s[(~s.isna()) & (s.str.len()>0)]
    bad={"nan","none","n/d","nd"}; s=s[~s.str.lower().isin(bad)]
    seen={}
    for v in s:
        k=v.lower()
        if k not in seen: seen[k]=v
    return col, sorted(seen.values())

st.sidebar.subheader("Assistente de Prompts")
prompts_bank = load_prompts_md(PROMPTS_MD_PATH)
col_p1,col_p2 = st.sidebar.columns(2)
with col_p1:
    titles_texto=[it["title"] for it in prompts_bank.get("Texto", [])]
    sel_texto=st.selectbox("Texto", options=["(vazio)"]+titles_texto, index=0)
with col_p2:
    titles_upload=[it["title"] for it in prompts_bank.get("Upload", [])]
    sel_upload=st.selectbox("Upload", options=["(vazio)"]+titles_upload, index=0)
if st.sidebar.button("Carregar no rascunho", use_container_width=True):
    draft=[]
    if sel_texto!="(vazio)": draft.append(next((it["body"] for it in prompts_bank["Texto"] if it["title"]==sel_texto),""))
    if sel_upload!="(vazio)": draft.append(next((it["body"] for it in prompts_bank["Upload"] if it["title"]==sel_upload),""))
    st.session_state.draft_prompt=("\n\n".join([d for d in draft if d])).strip(); st.sidebar.success("Carregado no rascunho."); st.rerun()

st.sidebar.header("Recuperação – Sphera")
k_sph   = st.sidebar.slider("Top-K Sphera", 1, 100, 20, 1)
thr_sph = st.sidebar.slider("Limiar Sphera (cos)", 0.0, 1.0, 0.30, 0.01)
years   = st.sidebar.slider("Últimos N anos", 1, 10, 3, 1)

st.sidebar.subheader("Filtros avançados – Sphera")
loc_col_sidebar, loc_options = _location_options_from(df_sph)
locations = st.sidebar.multiselect(
    f"Location (coluna: {loc_col_sidebar or 'N/D'})", options=loc_options, default=[],
    help="Lista montada do dataframe completo (após hidratação)."
)
if len(loc_options) < 2:
    with st.sidebar.expander("Debug Location"):
        st.write({
            "tem_col": bool(loc_col_sidebar),
            "col": loc_col_sidebar,
            "n_options": len(loc_options),
            "n_unique_total": int(df_sph[loc_col_sidebar].astype(str).nunique()) if loc_col_sidebar else 0,
        })
        st.dataframe(df_sph[[loc_col_sidebar]].head(12) if loc_col_sidebar else pd.DataFrame(), use_container_width=True)
substr = st.sidebar.text_input("Description contém (substring)", "")

st.sidebar.subheader("Agregação sobre eventos recuperados (Sphera)")
agg_mode    = st.sidebar.selectbox("Agregação", ["max","mean"], index=0)
per_ev_thr  = st.sidebar.slider("Limiar por evento (dicionários)", 0.0, 1.0, 0.30, 0.01)
min_support = st.sidebar.slider("Suporte mínimo (nº de eventos)", 1, 20, 1, 1)

thr_ws_sim   = st.sidebar.slider("Limiar de similaridade WS",        0.0, 1.0, 0.25, 0.01)
thr_prec_sim = st.sidebar.slider("Limiar de similaridade Precursor", 0.0, 1.0, 0.25, 0.01)
thr_cp_sim   = st.sidebar.slider("Limiar de similaridade CP",        0.0, 1.0, 0.25, 0.01)

topn_ws   = st.sidebar.slider("Top-N WS",          3, 90, 10, 1)
topn_prec = st.sidebar.slider("Top-N Precursores", 3, 90, 10, 1)
topn_cp   = st.sidebar.slider("Top-N CP",          3, 90, 10, 1)

uc1,uc2 = st.sidebar.columns(2)
with uc1:
    if st.button("Limpar uploads", use_container_width=True):
        st.session_state.pop("upld_texts", None); st.session_state.upld_texts=[]; st.rerun()
with uc2:
    if st.button("Limpar chat", use_container_width=True):
        st.session_state.chat=[]; st.rerun()

# ========================== UI central ==========================
if st.session_state._clear_draft_flag:
    st.session_state.draft_prompt=""; st.session_state._clear_draft_flag=False

st.title("SAFETY • CHAT (Somente Sphera)")
st.text_area("Conteúdo do prompt", key="draft_prompt", height=180, placeholder="Digite ou carregue um modelo de prompt…")
user_text = st.text_area("Texto de análise (para Sphera)", height=200, placeholder="Cole aqui a descrição/evento a analisar…")

uploaded = st.file_uploader("Anexar arquivo (opcional)", type=["txt","md","csv"])  # upload não dispara
if uploaded is not None:
    raw=uploaded.read()
    try: as_text = raw.decode("utf-8", errors="ignore")
    except Exception: as_text=""
    if uploaded.name.lower().endswith(".csv") and as_text:
        try:
            dfcsv=pd.read_csv(io.StringIO(as_text))
            as_text="\n".join(dfcsv.astype(str).fillna("").apply(lambda r:" ".join(r.values), axis=1).tolist())
        except Exception: pass
    if as_text:
        st.session_state.upld_texts.append(as_text)
        st.success(f"Upload recebido: {uploaded.name} (armazenado no contexto local).")

col_run1,col_run2,col_run3 = st.columns([1,1,1])
go_btn      = col_run1.button("Enviar para o chat", type="primary", use_container_width=True)
clear_draft = col_run2.button("Limpar rascunho", use_container_width=True)
clear_chat  = col_run3.button("Limpar chat", use_container_width=True)
if clear_draft: st.session_state._clear_draft_flag=True; st.rerun()
if clear_chat:  st.session_state.chat=[]; st.rerun()

# ========================== Execução ==========================

def render_hits_table(hits: List[Tuple[str,float,pd.Series]]):
    if not hits: return
    first_row_df = hits[0][2].to_frame().T
    loc_col = get_sphera_location_col(first_row_df) or "LOCATION"
    rows=[]
    for evid,s,row in hits[:min(10,len(hits))]:
        loc_val = str(row.get(loc_col, row.get("LOCATION","N/D")))
        desc = str(row.get("Description", row.get("DESCRIPTION",""))).strip()
        rows.append({"Event ID":evid, "Similaridade":round(s,3), "LOCATION":loc_val, "Description":desc})
    st.markdown("**Eventos do Sphera (Top-10)**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def push_model(messages: List[Dict[str,str]], pergunta: str, contexto_md: str):
    messages.append({"role":"user","content":"DADOS DE APOIO (não responda aqui):\n"+contexto_md})
    qt = pergunta or st.session_state.draft_prompt or "Analise os dados fornecidos e sintetize as lições."
    messages.append({"role":"user","content":f"Pergunta: {qt}"})
    try:
        resp=ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2, stream=False)
        content=""
        if isinstance(resp,dict): content = resp.get("message",{}).get("content","") or resp.get("content","")
        if not content: content="(Sem conteúdo do modelo)"
        with st.chat_message("assistant"): st.markdown(content)
        st.session_state.chat.append({"role":"assistant","content":content})
        st.session_state["_just_replied"]=True
    except Exception as e:
        st.error(f"Falha ao consultar modelo: {e}")

if go_btn:
    blocks=[]
    if st.session_state.draft_prompt.strip(): blocks.append("PROMPT:\n"+st.session_state.draft_prompt.strip())
    if (user_text or "").strip(): blocks.append("TEXTO:\n"+user_text.strip())
    for i,t in enumerate(st.session_state.upld_texts or []): blocks.append(f"UPLOAD[{i+1}]:\n"+t.strip())

    hits = sphera_similar_to_text(
        query_text=(user_text or st.session_state.draft_prompt),
        min_sim=thr_sph, years=years, topk=k_sph,
        df_base=df_sph, E_base=E_sph, substr=substr, locations=locations,
    )

    if hits: render_hits_table(hits)
    else: st.info("Nenhum evento do Sphera atingiu o limiar/filtros atuais.")

    dict_matches = aggregate_dict_matches_over_hits(
        hits, E_ws, L_ws, E_prec, L_prec, E_cp, L_cp,
        thr_ws_sim=thr_ws_sim, thr_prec_sim=thr_prec_sim, thr_cp_sim=thr_cp_sim,
        topn_ws=topn_ws, topn_prec=topn_prec, topn_cp=topn_cp,
        agg_mode=agg_mode, per_event_thr=per_ev_thr, min_support=min_support,
    )

    if hits:
        for title,key in [("WS","ws"),("Precursores","prec"),("CP","cp")]:
            arr = dict_matches.get(key) or []
            st.markdown(f"**{title} (≥ limiares)**")
            if arr:
                df_out = pd.DataFrame([{ "Rank":r+1, "Termo":lab, "Similaridade":round(s,3), "Suporte":sup }
                                       for r,(lab,s,sup) in enumerate(arr)])
                st.dataframe(df_out, use_container_width=True, hide_index=True)
            else:
                st.write(f"Nenhum {title} ≥ limiar.")

    table_ctx_rows=[]
    for evid,s,row in hits[:min(10,len(hits))]:
        loccol = get_sphera_location_col(row.to_frame().T)
        loc_val = str(row.get(loccol, row.get("LOCATION","N/D"))) if loccol else str(row.get("LOCATION","N/D"))
        desc = str(row.get("Description", row.get("DESCRIPTION",""))).strip()
        table_ctx_rows.append(f"EventID={evid} | sim={s:.3f} | LOCATION={loc_val} | Description={desc}")
    ctx_chunks=[f"Sphera_hits={len(hits)}, thr_sph={thr_sph:.2f}, years={years}", "\n".join(table_ctx_rows)]

    messages=[{"role":"system","content":st.session_state.system_prompt}, {"role":"user","content":"\n\n".join([b for b in blocks if b])}]
    push_model(messages, user_text, "\n\n".join([x for x in ctx_chunks if x]))

# ========================== Histórico (sem duplicate no mesmo ciclo) ==========================
if st.session_state.get("_just_replied"):
    st.session_state["_just_replied"]=False
else:
    if st.session_state.chat:
        st.divider(); st.subheader("Histórico")
        for m in st.session_state.chat[-10:]:
            role=m.get("role","assistant")
            with st.chat_message("assistant" if role!="user" else "user"):
                st.markdown(m.get("content",""))
