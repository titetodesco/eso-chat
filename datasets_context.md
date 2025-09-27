version: 1
description: >
  Contexto descritivo das bases usadas pelo chat/RAG e pelos módulos de
  análise (WS, Precursores, Taxonomia CP, histórico de eventos).
  Este arquivo descreve caminhos, colunas e como cada dataset deve ser
  interpretado na aplicação.

defaults:
  encoding: utf-8
  sep: ","

datasets:

  # -------------------------------------------------------------------
  # DICIONÁRIO DE WEAK SIGNALS (PT/EN)
  # -------------------------------------------------------------------
  - name: weak_signals_dict
    path: data/xlsx/DicionarioWaekSignals.xlsx
    type: excel
    sheet: Dicionário_WS
    primary_key: null
    language:
      pt: "Termo (PT)"
      en: "Termo (EN)"
    text_fields:
      - "Termo (PT)"
      - "Termo (EN)"
    columns:
      - name: "Termo (EN)"
        description: Termo do weak signal em inglês (rótulo EN).
      - name: "Termo (PT)"
        description: Termo do weak signal em português (rótulo PT).
    semantics:
      role: labels_ws
      label_column_pt: "Termo (PT)"
      label_column_en: "Termo (EN)"
      # dica: durante a criação de embeddings, você pode optar por PT, EN ou uma fusão PT+EN (concatenar termos).

  # -------------------------------------------------------------------
  # PRECURSORES (EXPANDIDO)
  # -------------------------------------------------------------------
  - name: precursors_dict
    path: data/xlsx/precursores_expandido.xlsx
    type: excel
    sheet: Precursores
    primary_key: null
    language:
      pt: "Precursores_PT"
      en: "Precursores_EN"
    extra:
      hto: "HTO"
    text_fields:
      - "Precursores_PT"
      - "Precursores_EN"
    columns:
      - name: "HTO"
        description: Categoria macro do precursor (Humano, Tecnologia ou Organização).
      - name: "Precursores_EN"
        description: Rótulo do precursor em inglês.
      - name: "Precursores_PT"
        description: Rótulo do precursor em português.
    semantics:
      role: labels_precursors
      label_column_pt: "Precursores_PT"
      label_column_en: "Precursores_EN"
      hto_column: "HTO"

  # -------------------------------------------------------------------
  # TAXONOMIA CP (Fatores Humanos)
  # -------------------------------------------------------------------
  - name: taxonomy_cp
    path: data/xlsx/TaxonomiaCP_Por.xlsx
    type: excel
    sheet: TaxonomiaCP_Por
    primary_key: null
    hierarchy:
      dimension: "Dimensão"
      factor: "Fatores"
      subfactor1: "Subfator 1"
      subfactor2: "Subfator 2"
    bags:
      pt: "Bag de termos"   # termos separados por ';'
      en: "Bag of terms"    # termos separados por ';'
      delimiter: ";"
    text_fields:
      - "Bag de termos"
      - "Bag of terms"
    columns:
      - name: "Dimensão"
        description: Dimensão de alto nível da Taxonomia CP.
      - name: "Fatores"
        description: Fator humano dentro da Dimensão.
      - name: "Subfator 1"
        description: Primeiro nível de subfator.
      - name: "Subfator 2"
        description: Segundo nível de subfator.
      - name: "Bag de termos"
        description: Lista (PT) de termos representativos para embeddings (separados por ';').
      - name: "Bag of terms"
        description: Lista (EN) de termos representativos para embeddings (separados por ';').
      - name: "Recomendação 1"
        description: Recomendações associadas (campo descritivo).
      - name: "Recomendação 2"
        description: Recomendações associadas (campo descritivo).
    semantics:
      role: labels_cp
      label_hierarchy:
        - "Dimensão"
        - "Fatores"
        - "Subfator 1"
        - "Subfator 2"
      bag_pt_column: "Bag de termos"
      bag_en_column: "Bag of terms"
      use_bag_language: "pt"  # pt|en|both
      bag_delimiter: ";"

  # -------------------------------------------------------------------
  # MAPA TRIPLO TRATADO (para grafo WS → Precursor)
  # -------------------------------------------------------------------
  - name: mapa_triplo
    path: data/xlsx/MapaTriplo_tratado.xlsx
    type: excel
    sheet: MapaTriplo_tratado
    primary_key: null
    columns:
      - name: "HTO"
        description: Categoria HTO do precursor.
      - name: "Precursor"
        description: Precursor (rótulo, geralmente PT).
      - name: "WeakSignal"
        description: Weak signal (rótulo, geralmente em EN).
    semantics:
      role: ws_precursor_graph_source
      hto_column: "HTO"
      precursor_column: "Precursor"
      ws_column: "WeakSignal"

  # -------------------------------------------------------------------
  # EDGES WS → PREC (CSV auxiliar para grafo)
  # Observação: você tem dois CSVs, um com cabeçalho 'weaksignal, precursor, hto'
  # e outro 'WeakSignal, Precursor, HTO'. Uniformizamos aqui.
  # -------------------------------------------------------------------
  - name: edges_ws_prec
    path: data/analytics/ws_precursors_edges.csv
    type: csv
    primary_key: null
    expected_columns:
      - "WeakSignal"
      - "Precursor"
      - "HTO"
    semantics:
      role: ws_precursor_edges
      hto_column: "HTO"
      precursor_column: "Precursor"
      ws_column: "WeakSignal"

  # -------------------------------------------------------------------
  # PRECURSORS (CSV auxiliar simples)
  # -------------------------------------------------------------------
  - name: precursors_csv
    path: data/analytics/precursors.csv
    type: csv
    primary_key: null
    expected_columns:
      - "HTO"
      - "Precursor"
    semantics:
      role: precursors_aux
      hto_column: "HTO"
      precursor_column: "Precursor"

  # -------------------------------------------------------------------
  # Gosee (inspeções/observações)
  # -------------------------------------------------------------------
  - name: gosee
    path: data/xlsx/GoSee.xlsx
    type: excel
    sheet: "query (1)"
    primary_key: "ID"
    text_fields:
      - "Observation"
    date_fields:
      - "Date"
    columns:
      - name: "ID"
        description: Identificador do registro.
      - name: "Title"
        description: Título curto da observação/visita.
      - name: "Date"
        description: Data da observação.
      - name: "Observation"
        description: Texto livre com observações (campo principal para similaridade).
      - name: "Area"
        description: Área (contexto de localização).
      - name: "Equipment"
        description: Equipamento envolvido (se houver).
      - name: "Category of Factor"
        description: Categoria de fator (pode mapear HTO/Fator quando aplicável).
      - name: "Action Required"
        description: Ação requerida (se registrada).
      - name: "Status"
        description: Estado do item (aberto, fechado, etc.).
    semantics:
      role: history_aux_gosee
      text_for_similarity: "Observation"

  # -------------------------------------------------------------------
  # SpheraCloud (tratado) — histórico de eventos
  # -------------------------------------------------------------------
  - name: sphera_cloud
    path: data/xlsx/TRATADO_safeguardOffShore.xlsx
    type: excel
    sheet: TRATADO_safeguardOffShore
    primary_key: "EVENT_NUMBER"
    text_fields:
      - "DESCRIPTION"
    date_fields:
      - "EVENT_DATE"
    columns:
      - name: "EVENT_NUMBER"
        description: Identificador do evento no Sphera.
      - name: "FPSO"
        description: Unidade (ex.: P-36, Espírito Santo, etc.).
      - name: "EVENT_DATE"
        description: Data do evento.
      - name: "DESCRIPTION"
        description: Descrição narrativa do evento (campo principal para similaridade RAG/histórico).
      - name: "REPORTABLE"
        description: Indicador de reportabilidade.
      - name: "SEVERITY"
        description: Severidade/gravidade registrada.
      - name: "EQUIPMENT"
        description: Equipamento envolvido (quando aplicável).
      - name: "AREA"
        description: Área/local do evento.
      - name: "CATEGORY"
        description: Categoria do evento (quando existente).
    semantics:
      role: history_main_sphera
      text_for_similarity: "DESCRIPTION"

analytics:
  # Convenções para criação/uso de embeddings
  ws:
    label_source: weak_signals_dict
    label_column_prefer: "Termo (EN)"   # ou "Termo (PT)" conforme sua decisão
  precursors:
    label_source: precursors_dict
    label_column_prefer: "Precursores_PT"
    hto_column: "HTO"
  cp:
    label_source: taxonomy_cp
    label_columns_hierarchy:
      - "Dimensão"
      - "Fatores"
      - "Subfator 1"
      - "Subfator 2"
    bag_column: "Bag de termos"        # use PT por padrão
    bag_delimiter: ";"
  graph_edges:
    source: edges_ws_prec
    ws_column: "WeakSignal"
    precursor_column: "Precursor"
    hto_column: "HTO"
  history:
    primary:
      name: sphera_cloud
      text_field: "DESCRIPTION"
      id_field: "EVENT_NUMBER"
      date_field: "EVENT_DATE"
    secondary:
      - name: gosee
        text_field: "Observation"
        id_field: "ID"
        date_field: "Date"
