# app_chat.py
# Chat RAG com Ollama Cloud (chat) + TF-IDF local duplo:
# - HIST (histórico) carregado na inicialização
# - UPLD (uploads) indexado a cada upload
# Combinação 70/30 (UPLD/HIST) + limiares separados + foco no último upload
# Usa secrets: OLLAMA_API_KEY (obrigatório), OLLAMA_HOST/OLLAMA_MODEL (opcionais)

import os
import io
import json
import re
import unicodedata
import requests
import numpy as np
import pandas as pd
import streamlit as st

# Parsers leves
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import docx
except Exception:
    docx = None

# -------------------------
# Configurações básicas
# -------------------------
st.set_page_config(page_title="ESO • CHAT (Ollama Cloud)", page_icon="💬", layout="wide")

# Segredos / env
OLLAMA_HOST  = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gpt-oss:20b"))
OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY", None))

if not OLLAMA_API_KEY:
    st.error("⚠️ OLLAMA_API_KEY não encontrado. Defina em **Settings → Secrets** no Streamlit Cloud.")
    st.stop()

HEADERS_JSON = {
    "Authorization": f"Bearer {OLLAMA_API_KEY}",
    "Content-Type": "application/json",
}

# Caminhos
DATA_DIR = "data"
ANALYTICS_DIR = os.path.join(DATA_DIR, "analytics")
DATASETS_CONTEXT_FILE = "datasets_context.md"
HISTORY_JSONL = os.path.join(ANALYTICS_DIR, "history_texts.jsonl")

# -------------------------
# Normalização / Tokenização
# -------------------------
STOP_PT = {
    "a","ao","aos","as","à","às","de","do","dos","da","das","e","é","em","no","nos","na","nas","para","por",
    "o","os","um","uns","uma","umas","que","se","com","sem","como","mais","menos","muito","muita","muitos",
    "muitas","já","não","sim","ser","ter","foi","são","pela","pelas","pelos","pelo","entre","sobre","até",
    "também","porque","quando","onde","qual","quais","quem","qualquer","toda","todo","todas","todos"
}
STOP_EN = {
    "the","a","an","and","or","but","if","then","else","for","of","in","on","at","to","from","by","with","as",
    "is","are","was","were","be","been","being","that","this","these","those","it","its","they","them","their",
    "there","here","not","no","yes","do","does","did","done","than","so","such","any","all","each","every"
}
STOPWORDS = STOP_PT | STOP_EN

def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

def tokenize(text: str):
    if not text:
        return []
    t = strip_accents(text.lower())
    toks = TOKEN_RE.findall(t)
    toks = [w for w in toks if w not in STOPWORDS and len(w) > 1]
    return toks

# -------------------------
# TF-IDF (leve)
# -------------------------
def tfidf_fit_transform(texts):
    # constrói vocab a partir de tokens
    from collections import Counter
    tokens_list = [tokenize(t) for t in texts]
    vocab = {}
    for toks in tokens_list:
        for w in toks:
            if w not in vocab:
                vocab[w] = len(vocab)

    N = len(texts)
    D = len(vocab)
    X = np.zeros((N, D), dtype=np.float32)

    # TF
    for i, toks in enumerate(tokens_list):
        c = Counter(toks)
        if not c:
            continue
        max_tf = max(c.values())
        for w, f in c.items():
            j = vocab.get(w)
            if j is not None:
                X[i, j] = f / max_tf

    # IDF
    df = np.zeros(D, dtype=np.int32)
    inv_index = {w: j for w, j in vocab.items()}
    for j in range(D):
        df[j] = 1  # evita div por zero
    for toks in tokens_list:
        seen = set()
        for w in toks:
            if w in vocab and w not in seen:
                df[vocab[w]] += 1
                seen.add(w)
    idf = np.log((N + 1) / (df)) + 1.0  # suavizado
    X *= idf[None, :]
    return X, vocab, idf

def tfidf_transform(texts, vocab, idf):
    from collections import Counter
    N = len(texts)
    D = len(vocab)
    X = np.zeros((N, D), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = tokenize(t)
        c = Counter(toks)
        if not c: 
            continue
        max_tf = max(c.values())
        for w, f in c.items():
            j = vocab.get(w)
            if j is not None:
                X[i, j] = (f / max_tf) * idf[j]
    return X

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1: a = a[None, :]
    if b.ndim == 1: b = b[None, :]
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    return (a @ b.T) / (a_norm * b_norm)

# -------------------------
# Leitura de arquivos
# -------------------------
def read_pdf(file: bytes) -> str:
    if pypdf is None:
        raise RuntimeError("pypdf não está instalado. Adicione `pypdf` ao requirements.txt.")
    reader = pypdf.PdfReader(io.BytesIO(file))
    txt = []
    for page in reader.pages:
        try:
            txt.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(txt)

def read_docx(file: bytes) -> str:
    if docx is None:
        raise RuntimeError("python-docx não está instalado. Adicione `python-docx` ao requirements.txt.")
    f = io.BytesIO(file)
    document = docx.Document(f)
    return "\n".join(p.text for p in document.paragraphs)

def read_xlsx(file: bytes) -> str:
    f = io.BytesIO(file)
    xls = pd.ExcelFile(f)
    frames = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        frames.append(df.astype(str))
    if frames:
        df_all = pd.concat(frames, axis=0, ignore_index=True)
        return df_all.to_csv(index=False)
    return ""

def read_csv(file: bytes) -> str:
    f = io.BytesIO(file)
    df = pd.read_csv(f)
    return df.astype(str).to_csv(index=False)

def load_file_to_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        return read_pdf(data)
    if name.endswith(".docx"):
        return read_docx(data)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return read_xlsx(data)
    if name.endswith(".csv"):
        return read_csv(data)
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def summarize_local(text: str, max_chars: int = 800) -> str:
    # Heurística simples: primeiras 3-5 sentenças ou até N caracteres
    if not text:
        return ""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sents:
        return text[:max_chars]
    take = min(len(sents), 5)
    summary = " ".join(sents[:take])
    return summary[:max_chars]

# -------------------------
# Ollama Cloud (chat)
# -------------------------
def ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2, stream=False, timeout=120):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "stream": bool(stream),
    }
    r = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        headers=HEADERS_JSON,
        json=payload,
        timeout=timeout
    )
    r.raise_for_status()
    return r.json()

# -------------------------
# Estado
# -------------------------
def init_state():
    if "hist" not in st.session_state:
        st.session_state.hist = {
            "texts": [], "metas": [],
            "vocab": None, "idf": None, "X": None
        }
    if "upld" not in st.session_state:
        st.session_state.upld = {
            "chunks": [], "metas": [],
            "vocab": None, "idf": None, "X": None,
            "active_summary": "", "active_name": "", "fresh_turns": 0
        }
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "dataset_context" not in st.session_state:
        ctx = ""
        try:
            if os.path.exists(DATASETS_CONTEXT_FILE):
                with open(DATASETS_CONTEXT_FILE, "r", encoding="utf-8") as f:
                    ctx = f.read()
        except Exception:
            ctx = ""
        st.session_state.dataset_context = ctx

init_state()

# -------------------------
# Carrega HIST na inicialização
# -------------------------
def load_history_jsonl(path: str):
    texts, metas = [], []
    if not os.path.exists(path):
        return texts, metas
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                txt = obj.get("text", "")
                meta = obj.get("meta", {})
                if txt:
                    texts.append(str(txt))
                    metas.append(meta)
            except Exception:
                continue
    return texts, metas

if st.session_state.hist["X"] is None:
    h_texts, h_metas = load_history_jsonl(HISTORY_JSONL)
    if h_texts:
        X, vocab, idf = tfidf_fit_transform(h_texts)
        st.session_state.hist = {
            "texts": h_texts,
            "metas": h_metas,
            "vocab": vocab,
            "idf": idf,
            "X": X
        }

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Configurações")
with st.sidebar.expander("Ollama Cloud", expanded=True):
    st.write("Host:", OLLAMA_HOST)
    st.write("Modelo (chat):", OLLAMA_MODEL)

topk_total = st.sidebar.slider("Top-K total (contexto)", 2, 12, 6, 1)
weight_upload = st.sidebar.slider("Peso Upload (%)", 50, 100, 70, 5)  # 70% default
thr_upload = st.sidebar.slider("Limiar UPLD (cosseno TF-IDF)", 0.05, 0.95, 0.25, 0.01)
thr_hist   = st.sidebar.slider("Limiar HIST (cosseno TF-IDF)", 0.05, 0.95, 0.35, 0.01)

chunk_size = st.sidebar.slider("Tamanho do chunk (caracteres)", 600, 2000, 1200, 50)
chunk_overlap = st.sidebar.slider("Overlap do chunk", 50, 600, 200, 10)
use_catalog_ctx = st.sidebar.checkbox("Injetar contexto do catálogo (datasets_context.md)", value=True)
st.sidebar.divider()

uploaded_files = st.sidebar.file_uploader(
    "Upload (PDF, DOCX, XLSX, CSV, TXT/MD)",
    type=["pdf","docx","xlsx","xls","csv","txt","md"],
    accept_multiple_files=True
)

col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Limpar uploads", use_container_width=True):
        st.session_state.upld = {
            "chunks": [], "metas": [],
            "vocab": None, "idf": None, "X": None,
            "active_summary": "", "active_name": "", "fresh_turns": 0
        }
with col_b:
    if st.button("Limpar chat", use_container_width=True):
        st.session_state.chat = []

with st.sidebar.expander("Diagnóstico /api/chat", expanded=False):
    if st.button("Testar /api/chat", use_container_width=True):
        try:
            r = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                headers=HEADERS_JSON,
                json={"model": OLLAMA_MODEL, "messages":[{"role":"user","content":"diga OK"}], "stream": False},
                timeout=30,
            )
            st.write("Status:", r.status_code)
            if r.headers.get("content-type","").startswith("application/json"):
                st.json(r.json())
            else:
                st.text(r.text[:1000])
        except Exception as e:
            st.error(f"Falhou: {e}")

with st.sidebar.expander("Catálogo carregado (analytics)", expanded=False):
    ok_any = False
    for fname in ["history_texts.jsonl"]:
        path = os.path.join(ANALYTICS_DIR, fname)
        if os.path.exists(path):
            st.write("✅", fname)
            ok_any = True
    if not ok_any:
        st.info("Suba `data/analytics/history_texts.jsonl` no repositório (JSONL com {text, meta}).")

# -------------------------
# Indexação de uploads (UPLD)
# -------------------------
def rebuild_upld_index():
    chunks = st.session_state.upld["chunks"]
    if not chunks:
        st.session_state.upld.update({"vocab": None, "idf": None, "X": None})
        return
    X, vocab, idf = tfidf_fit_transform(chunks)
    st.session_state.upld.update({"vocab": vocab, "idf": idf, "X": X})

if uploaded_files:
    with st.spinner("Lendo arquivos e indexando (TF-IDF)…"):
        new_chunks, new_metas = [], []
        active_name = ""
        for uf in uploaded_files:
            try:
                text = load_file_to_text(uf)
                parts = []
                # chunking com sobreposição
                text = text.replace("\r\n","\n").replace("\r","\n")
                start = 0
                L = len(text)
                ov = max(0, min(chunk_overlap, chunk_size-1))
                while start < L:
                    end = min(L, start + chunk_size)
                    part = text[start:end].strip()
                    if part:
                        parts.append(part)
                    if end >= L:
                        break
                    start = max(0, end - ov)

                for i, p in enumerate(parts):
                    new_chunks.append(p)
                    new_metas.append({"file": uf.name, "chunk_id": i})

                # define upload ativo
                active_name = uf.name
                st.session_state.upld["active_summary"] = summarize_local(text, max_chars=800)
                st.session_state.upld["active_name"] = active_name
                st.session_state.upld["fresh_turns"] = 3  # injeta resumo por 3 turnos
            except Exception as e:
                st.warning(f"Falha ao processar {uf.name}: {e}")

        if new_chunks:
            st.session_state.upld["chunks"].extend(new_chunks)
            st.session_state.upld["metas"].extend(new_metas)
            rebuild_upld_index()
            st.success(f"Indexados {len(new_chunks)} chunks (UPLD). Upload ativo: {active_name or '(desconhecido)'}")

# -------------------------
# UI principal
# -------------------------
st.title("ESO • CHAT — HIST + UPLD (TF-IDF) • Ollama Cloud para respostas")
st.caption("RAG local (TF-IDF) com foco no último upload + histórico. Catálogo opcional injetado como contexto (datasets_context.md).")

# histórico visual
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Digite sua pergunta…")
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context_blocks = []
    diag = {"upld_hits":0, "hist_hits":0}

    # 1) Upload: similaridade e seleção
    upld_X = st.session_state.upld["X"]
    if upld_X is not None and len(st.session_state.upld["chunks"]) > 0:
        vocab, idf = st.session_state.upld["vocab"], st.session_state.upld["idf"]
        qX = tfidf_transform([prompt], vocab, idf)
        sims = cosine_sim(qX, upld_X)[0]
        order = np.argsort(-sims)
        # quota 70% (ou conforme slider)
        k_u = max(1, int(np.ceil(topk_total * (weight_upload/100.0))))
        u_hits = []
        for idx in order:
            if len(u_hits) >= k_u:
                break
            if sims[idx] >= thr_upload:
                meta = st.session_state.upld["metas"][idx]
                txt = st.session_state.upld["chunks"][idx]
                u_hits.append((float(sims[idx]), meta, txt))
        # Se não bateu quota, ainda assim garanta pelo menos 1 do upload se existir
        if not u_hits and len(st.session_state.upld["chunks"]) > 0:
            best = order[0]
            meta = st.session_state.upld["metas"][best]
            txt = st.session_state.upld["chunks"][best]
            u_hits = [(float(sims[best]), meta, txt)]
        # injeta em ordem
        for s, meta, txt in u_hits:
            context_blocks.append(f"[UPLOAD {meta['file']} / {meta['chunk_id']}] (sim={s:.3f})\n{txt}")
        diag["upld_hits"] = len(u_hits)

    # 2) Histórico: similaridade e seleção
    hist_X = st.session_state.hist["X"]
    if hist_X is not None and len(st.session_state.hist["texts"]) > 0:
        vocab, idf = st.session_state.hist["vocab"], st.session_state.hist["idf"]
        qX = tfidf_transform([prompt], vocab, idf)
        sims = cosine_sim(qX, hist_X)[0]
        order = np.argsort(-sims)
        k_h = max(0, topk_total - len(context_blocks))
        h_hits = []
        for idx in order:
            if len(h_hits) >= k_h:
                break
            if sims[idx] >= thr_hist:
                meta = st.session_state.hist["metas"][idx] if idx < len(st.session_state.hist["metas"]) else {}
                txt  = st.session_state.hist["texts"][idx]
                src  = meta.get("source","HIST")
                tag  = meta.get("tag","")
                context_blocks.append(f"[HIST {src} {tag}] (sim={sims[idx]:.3f})\n{txt}")
                h_hits.append(idx)
        diag["hist_hits"] = len(h_hits)

    # 3) Resumo do upload ativo (cola por alguns turnos)
    if st.session_state.upld.get("active_summary") and st.session_state.upld.get("fresh_turns",0) > 0:
        context_blocks.insert(0, f"[RESUMO UPLOAD {st.session_state.upld.get('active_name','')}] {st.session_state.upld['active_summary']}")
        st.session_state.upld["fresh_turns"] -= 1

    # Mensagens para o modelo
    SYSTEM_PROMPT = (
        "Você é um assistente para gestão de segurança operacional.\n"
        "- Responda de forma objetiva e prática.\n"
        "- Cite os trechos de CONTEXTO quando usar algo deles.\n"
        "- Não gere código (Python/SQL/etc.) a menos que o usuário peça explicitamente.\n"
        "- Se não houver informação suficiente nos CONTEXTOS, diga isso claramente.\n"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if use_catalog_ctx and st.session_state.dataset_context:
        messages.append({"role": "system", "content": st.session_state.dataset_context})

    if context_blocks:
        ctx = "\n\n".join(context_blocks)
        messages.append({"role": "user", "content": f"CONTEXTOS RELEVANTES:\n{ctx}"})
        messages.append({"role": "user", "content": f"PERGUNTA: {prompt}"})
    else:
        messages.append({"role": "user", "content": prompt})

    # Chamada ao modelo
    with st.chat_message("assistant"):
        with st.spinner("Consultando o modelo na nuvem…"):
            try:
                resp = ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.2, stream=False)
                content = resp.get("message", {}).get("content", "").strip() or json.dumps(resp)[:1000]
            except Exception as e:
                content = f"Falha ao consultar o modelo: {e}"
            st.markdown(content)

    st.session_state.chat.append({"role": "assistant", "content": content})

    # Diagnóstico do contexto usado
    with st.expander("🔎 Contexto usado nesta resposta", expanded=False):
        st.write(f"Trechos do **Upload** injetados: {diag['upld_hits']}")
        st.write(f"Trechos do **Histórico** injetados: {diag['hist_hits']}")
        st.write(f"Resumo do upload ativo {'incluído' if st.session_state.upld.get('fresh_turns',0)>=0 and st.session_state.upld.get('active_summary') else 'não incluído'}")
        st.code("\n\n".join(context_blocks[:3])[:2000], language="markdown")

# -------------------------
# Painel: status dos índices
# -------------------------
with st.expander("📚 Status do índice — HIST (somente leitura)"):
    H = st.session_state.hist
    n = len(H["texts"])
    st.write(f"Textos no histórico: **{n}**")
    if n > 0:
        sample = [{"meta": H["metas"][i] if i < len(H["metas"]) else {}, "preview": H["texts"][i][:200]} for i in range(min(n, 20))]
        st.dataframe(pd.DataFrame(sample), use_container_width=True)

with st.expander("📄 Status do índice — UPLD (uploads)"):
    U = st.session_state.upld
    n = len(U["chunks"])
    st.write(f"Chunks indexados (UPLD): **{n}**")
    st.write(f"Upload ativo: **{U.get('active_name','(nenhum)')}** — turns restantes de resumo: {U.get('fresh_turns',0)}")
    if n > 0:
        df = pd.DataFrame(U["metas"])
        st.dataframe(df.head(50), use_container_width=True)
        if st.button("Baixar índice (CSV de chunks)", use_container_width=True):
            out = pd.DataFrame({
                "file": [m["file"] for m in U["metas"]],
                "chunk_id": [m["chunk_id"] for m in U["metas"]],
                "text": U["chunks"],
            })
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="rag_chunks_uploads.csv", mime="text/csv", use_container_width=True)
