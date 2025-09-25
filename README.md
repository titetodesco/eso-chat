https://eso-chat.streamlit.app/ 

1) Estrutura recomendada de pastas
eso-chat/
├─ app_chat.py                 # app principal (Streamlit)
├─ requirements.txt            # deps leves (streamlit, requests, numpy, pandas, pypdf, python-docx)
├─ README.md
├─ .gitignore
├─ .gitattributes              # opcional (Git LFS p/ PDFs/DOCX/NPZ)
└─ data/
   └─ analytics/               # opcional (csv/npz estáticos, se quiser versionar)

2) Pré-requisitos
Conta no Streamlit Cloud (grátis).
Conta no Ollama Cloud: https://ollama.com
Em Settings → API Keys, crie uma API key.
Repositório no GitHub com os arquivos acima.

3) Configurar secrets no Streamlit Cloud

No dashboard do Streamlit Cloud:

Deploy app a partir do GitHub (seu repo → app_chat.py).

Em App → Settings → Secrets, cole:

# Obrigatório
OLLAMA_API_KEY = "coloque_sua_api_key_aqui"

# Opcionais (deixe assim se usar o Ollama Cloud)

OLLAMA_HOST  = "https://api.ollama.com" 

OLLAMA_MODEL = "llama3.1"    # ou outro modelo disponível no Ollama Cloud

Salve os secrets.

4) requirements.txt (exemplo)
streamlit==1.36.0
requests
numpy
pandas
pypdf
python-docx

Não adicionamos torch ou sentence-transformers.

5) .gitignore (sugestão)
# Python / Streamlit
__pycache__/
*.pyc
.venv/
env/
.conda/
.ipynb_checkpoints/
.streamlit/secrets.toml

# Artefatos locais
data/storage/
data/tmp/
tmp/
*.log

# Se decidir gerar em runtime:
*.npz
*.npy

# SO / IDE
.DS_Store
Thumbs.db
.vscode/
.idea/

6) (Opcional) Git LFS para arquivos grandes

Se quiser versionar PDFs/DOCX/NPZ:
Instale Git LFS e rode:

git lfs install
git lfs track "*.pdf"
git lfs track "*.docx"
git lfs track "*.npz"


Isto criará/atualizará .gitattributes com algo como:

*.pdf  filter=lfs diff=lfs merge=lfs -text
*.docx filter=lfs diff=lfs merge=lfs -text
*.npz  filter=lfs diff=lfs merge=lfs -text
Commit e push normalmente.

7) Como funciona o app
Upload: PDF, DOCX, XLSX, CSV, TXT/MD.
O app faz chunking do texto e gera embeddings no Ollama Cloud (/api/embeddings, modelo nomic-embed-text).
Na pergunta do usuário, calcula similaridade cosseno vs. índice e envia contexto relevante ao modelo de chat (/api/chat).
Ajustes no sidebar:
Tamanho/overlap de chunk
Top-K de contexto
Limiar de similaridade

8) Teste rápido da API (local, opcional)
cURL
curl -s https://api.ollama.com/api/chat \
  -H "Authorization: Bearer $OLLAMA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.1","messages":[{"role":"user","content":"diga ok"}],"stream":false}'

Python
import os, requests
host = os.getenv("OLLAMA_HOST","https://api.ollama.com")
key  = os.getenv("OLLAMA_API_KEY")  # export OLLAMA_API_KEY=...

r = requests.post(
    f"{host}/api/chat",
    headers={"Authorization": f"Bearer {key}", "Content-Type":"application/json"},
    json={"model":"llama3.1", "messages":[{"role":"user","content":"diga ok"}], "stream":False},
    timeout=60
)
print(r.status_code, r.json())

9) Deploy
Faça push para o GitHub.
No Streamlit Cloud, crie o app apontando para app_chat.py.
Configure os Secrets (seção 3).
Clique em Deploy.

10) Solução de problemas
401 Unauthorized: verifique OLLAMA_API_KEY nos Secrets.
Time-out embeddings: uploads muito grandes? Faça em lotes menores; ajuste chunk size.
PDF vazio: alguns PDFs “scanneados” não têm texto extraível. Use OCR antes.
Erros de pacote: confira requirements.txt e reimplante.

11) Próximos passos (sugestões)
Persistir índice em data/analytics (CSV/NPZ) e permitir download/upload do índice.
Implementar fonte citada (arquivo e chunk_id) no rodapé da resposta.
Adicionar controles avançados (temperature, max tokens, penalidades).
Incluir métricas básicas (latência, tokens estimados).

12) Licença
Defina a licença que preferir (por exemplo, MIT) no repositório.
