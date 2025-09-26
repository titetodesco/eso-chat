# ESO – Catálogo de fontes para o Chat

Você tem acesso a estes conjuntos **já versionados no projeto**:

## 1) Histórico de eventos (texto plano)
- `data/analytics/history_texts.jsonl` — cada linha: `{"id": "<ID|arquivo>", "text": "<conteúdo textual>"}`.
- Use para comparar “evento atual (upload)” com eventos passados.

## 2) Dicionário de Weak Signals (WS)
- `data/analytics/ws_labels.jsonl` — cada linha: `{"label": "WS::<termo_em_inglês_ou_pt>"}`
- Objetivo: detectar sinais fracos no documento (upload) e cruzar com histórico.

## 3) Dicionário de Precursores (PREC)
- `data/analytics/precursors.csv` — colunas: **HTO**, **Precursor**.
- `data/analytics/prec_labels.jsonl` — cada linha: `{"label":"PREC::<nome>", "hto":"Humano/..."}`.
- Objetivo: detectar precursores no documento (upload) e mapear para HTO.

## 4) Mapa WS → Precursor
- `data/analytics/ws_precursors_edges.csv` — colunas: **weak signal**, **precursor**, **hto**.
- Objetivo: construir grafo (WS em vermelho, PRECs em azul) e sugerir investigações.

## 5) Taxonomia CP (Fatores Humanos)
- `data/analytics/cp_labels.jsonl` — cada linha:  
  `{"dim":"<Dimensão>", "fator":"<Fator>", "sub1":"<Subfator1>", "sub2":"<Subfator2>", "bag_pt":"a;b;c", "bag_en":"a;b;c"}`
- Objetivo: classificar trechos por fatores humanos.

## 6) Bases operacionais (consultas por similaridade textual)
- **SpheraCloud**: planilha em `data/xlsx/` com coluna **description** (descrição do evento).  
  Use para: “traga até 5 eventos do Sphera semelhantes ao do arquivo atual”.
- **GOSee**: planilha(s) em `data/xlsx/` com colunas de descrição/observações.

### Como responder
- Quando o usuário pedir “eventos similares no Sphera”, busque no **RAG do histórico** e/ou, se houver, no índice específico dessas planilhas (coluna **description**).  
- Quando pedir “WS/Precursores/Fatores”, use o dicionário correspondente e trechos do arquivo **uploadado nesta sessão**.  
- Sempre mostre até **3–5 trechos** mais similares do documento para justificar.  
- Quando não houver dado, explique claramente o que falta.

### Convenções
- **WS**: prefixo `WS::` nos rótulos.  
- **Precursor**: prefixo `PREC::`.  
- **HTO**: `Humano`, `Tarefa`, `Organização` (quando disponível).  
- **Cores no grafo**: WS = vermelho; Precursor = azul.
