# ESO-CHAT • Modelos de Prompts

## Texto

### 1) Somente Sphera — similaridade por cosseno
Atue como um especialista em segurança operacional offshore empregando o texto a seguir, busque eventos semelhantes na base Sphera utilizando o cálculo do cosseno entre o embedding do texto e a coluna Description.
Aplique os parâmetros definidos no menu lateral, em especial o limiar de similaridade e os Top-k.
Apresente para cada evento: EVENT ID, EVENT TYPE, similaridade (cos), Description e LOCATION.
Em seguida, apresente recomendações para investigação e lições aprendidas de eventos semelhantes.

### 2) Sphera + GoSee + Docs + Upload
Atue como um especialista em segurança operacional offshore. A partir do relato abaixo, realize uma busca combinada nas bases Sphera, GoSee, Docs e arquivos enviados via upload.
Aplique os Top-K e limiares definidos na barra lateral.
Mostre os eventos com maior similaridade e identifique padrões comuns de causas e consequências realizando um comparativo entre as bases Sphera e GoSee.

### 3) Weak Signals (sinais fracos)
Atue como um especialista em segurança operacional offshore. Analise o texto a seguir buscando sinais fracos (Weak Signals) de comportamento/processo e correlacione com a base Sphera (Description).
Informe os WS mais correlacionados e descreva medidas preventivas bem como possíveis aprendizados relacionados ao histórico de eventos recuperados.

### 4) Precursores + CP
Atue como um especialista em segurança operacional offshore. Analise o evento descrito, recupere eventos Sphera semelhantes (≥ limiar) e identifique Precursores e Fatores CP correlatos ocorridos nos eventos recuperados.
Explique a relação com o caso e proponha recomendações de mitigação de riscos de incidentes futuros.

### 5) Combinado (Sphera + WS + Precursores + CP)
Atue como um especialista em segurança operacional offshore. Realize uma análise integrada usando Sphera e dicionários de Weak Signals, Precursores e CP.
Destaque: eventos Sphera (≥ limiar), WS/Precursores/CP correspondentes e um conjunto de lições aprendidas e evidencie os tipos de eventos do Sphera, ou seja, observation, near miss e incidents, apresentando a correlação entre os sinais fracos, precursores e CP x tipos de eventos.


## Upload

### 1) Somente Sphera — similaridade por cosseno
Atue como um especialista em segurança operacional offshore empregando o texto a seguir, busque eventos semelhantes na base **Sphera** utilizando o cálculo do cosseno entre o embedding do texto e a coluna Description.
Aplique os parâmetros definidos no menu lateral, em especial o limiar de similaridade e os Top-k.
Apresente para cada evento: EVENT ID, EVENT TYP, similaridade (cos), Description e LOCATION.
Em seguida, apresente recomendações para investigação e lições aprendidas de eventos semelhantes.

### 2) Sphera + GoSee + Docs + Upload
Atue como um especialista em segurança operacional offshore. A partir do relato abaixo, realize uma busca combinada nas bases Sphera, GoSee, Docs e arquivos enviados via upload.
Aplique os Top-K e limiares definidos na barra lateral.
Mostre os eventos com maior similaridade e identifique padrões comuns de causas e consequências realizando um comparativo entre as bases Sphera e GoSee.

### 3) Weak Signals (sinais fracos)
Atue como um especialista em segurança operacional offshore. Analise o texto a seguir buscando sinais fracos (Weak Signals) de comportamento/processo e correlacione com a base Sphera (Description).
Informe os WS mais correlacionados e descreva medidas preventivas bem como possíveis aprendizados relacionados ao histórico de eventos recuperados.

### 4) Precursores + CP
Atue como um especialista em segurança operacional offshore. Analise o evento descrito, recupere eventos Sphera semelhantes (≥ limiar) e identifique Precursores e Fatores CP correlatos ocorridos nos eventos recuperados.
Explique a relação com o caso e proponha recomendações de mitigação de riscos de incidentes futuros.

### 5) Combinado (Sphera + WS + Precursores + CP)
Atue como um especialista em segurança operacional offshore. Realize uma análise integrada usando Sphera e dicionários de Weak Signals, Precursores e CP.
Destaque: eventos Sphera (≥ limiar), WS/Precursores/CP correspondentes e um conjunto de lições aprendidas e evidencie os tipos de eventos do Sphera, ou seja, observation, near miss e incidents, apresentando a correlação entre os sinais fracos, precursores e CP x tipos de eventos.
