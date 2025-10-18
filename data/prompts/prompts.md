# ESO-CHAT • Modelos de Prompts

## Texto

### 1) Somente Sphera — similaridade por cosseno
Baseado no texto a seguir, busque eventos semelhantes na base **Sphera** utilizando o cálculo do **cosseno entre o embedding do texto e a coluna Description**.
Aplique o limiar de similaridade definido no menu lateral.
Apresente para cada evento: **EVENT ID**, **similaridade (cos)**, **Description** e **LOCATION**.
Em seguida, apresente recomendações para investigação e lições aprendidas de eventos semelhantes.
**Texto:**
(cole aqui o relato do evento)

### 2) Sphera + GoSee + Docs + Upload
A partir do relato abaixo, realize uma **busca combinada** nas bases **Sphera, GoSee, Docs** e **arquivos enviados via upload**.
Aplique os Top-K e limiares definidos na barra lateral.
Mostre os eventos com maior similaridade e identifique padrões comuns de causas e consequências.
**Texto:**
(cole aqui o relato)

### 3) Weak Signals (sinais fracos)
Analise o texto a seguir buscando **sinais fracos (Weak Signals)** de comportamento/processo e correlacione com a base Sphera (Description).
Informe os WS mais correlacionados e descreva medidas preventivas.
**Texto:**
(cole aqui o relato)

### 4) Precursores + CP
Analise o evento descrito, recupere **eventos Sphera semelhantes** (≥ limiar) e identifique **Precursores** e **Fatores CP** correlatos.
Explique a relação com o caso e proponha recomendações de mitigação.
**Texto:**
(cole aqui o relato)

### 5) Combinado (Sphera + WS + Precursores + CP)
Realize uma análise integrada usando **Sphera** e dicionários de **Weak Signals, Precursores e CP**.
Destaque: eventos Sphera (≥ limiar), WS/Precursores/CP correspondentes e um conjunto de lições aprendidas.
**Texto:**
(cole aqui o relato)


## Upload

### 1) Somente Sphera — arquivo(s)
Analise o(s) arquivo(s) enviado(s) e recupere os eventos da base **Sphera** com maior similaridade de cosseno (arquivo × Description).
Aplique o limiar definido e apresente: **EVENT ID**, **similaridade**, **Description**, **LOCATION**.
Conclua com recomendações e boas práticas.

### 2) Análise completa — Sphera + GoSee + Docs + Upload
Integre **Sphera**, **GoSee**, **Docs** e o conteúdo dos **uploads** (Top-K e limiares da barra lateral).
Apresente os principais similares e um resumo técnico com recomendações e lições aprendidas.

### 3) Weak Signals com uploads
Procure **sinais fracos (Weak Signals)** nos arquivos enviados e correlacione com descrições da Sphera.
Apresente ranking de WS e sugestões de melhoria.

### 4) Precursores + Fatores CP (sobre uploads)
A partir dos arquivos, identifique **eventos Sphera** (≥ limiar) e mapeie **Precursores** e **Fatores CP** associados.
Mostre a ligação e recomende ações preventivas.

### 5) Combinado (Sphera + WS + Precursores + CP) — uploads
Com base nos uploads, faça uma análise integrada: **Sphera**, **WS**, **Precursores**, **CP**.
Exiba resultados por categoria e um resumo descritivo com recomendações.
