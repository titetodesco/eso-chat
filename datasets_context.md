catalog:
  spreadsheets:
    - name: sphera_cloud
      path: "data/xlsx/TRATADO_safeguardOffShore.xlsx"
      sheet: "TRATADO_safeguardOffShore"
      description: "Eventos do SpheraCloud tratados para offshore."
      key_columns: ["EVENT_NUMBER"]
      text_for_similarity: "DESCRIPTION"
      fields:
        - name: "EVENT_NUMBER"
          description: "Identificador único do evento."
        - name: "FPSO"
          description: "Unidade (por exemplo, 'P-36' ou 'Espírito Santo')."
        - name: "EVENT_DATE"
          description: "Data do evento."
        - name: "AREA"
          description: "Área/locação no ativo."
        - name: "DESCRIPTION"
          description: "Descrição livre do evento (PT/EN)."
        - name: "SEVERITY"
          description: "Severidade (numérica ou categórica)."

    - name: gosee
      path: "data/xlsx/GoSee.xlsx"
      sheet: "GoSee"
      description: "Registros de Go & See / observações de segurança."
      key_columns: []
      text_for_similarity: "Observation"
      fields:
        - name: "Observation"
          description: "Texto da observação (PT)."
        - name: "Area"
          description: "Área local da observação."
        - name: "Date"
          description: "Data do registro."

    - name: ws_dict
      path: "data/xlsx/DicionarioWaekSignals.xlsx"
      sheet: "Sheet1"
      description: "Dicionário de Weak Signals PT/EN."
      key_columns: []
      fields:
        - name: "PT"
          description: "Termo do Weak Signal em português."
        - name: "EN"
          description: "Termo do Weak Signal em inglês."

    - name: precursors
      path: "data/xlsx/precursores_expandido.xlsx"
      sheet: "Sheet1"
      description: "Lista expandida de precursores e HTO."
      key_columns: []
      fields:
        - name: "HTO"
          description: "Dimensão Humano / Tecnologia / Organização."
        - name: "Precursor"
          description: "Nome do precursor (PT)."
        - name: "EN"
          description: "Nome do precursor (EN) — se existir."

    - name: taxonomy_cp
      path: "data/xlsx/TaxonomiaCP_Por.xlsx"
      sheet: "Sheet1"
      description: "Taxonomia CP de fatores humanos."
      key_columns: []
      fields:
        - name: "Dimensão"
          description: "Dimensão da taxonomia (ex.: Humano, Organização, Tecnologia)."
        - name: "Fatores"
          description: "Fator principal."
        - name: "Subfator 1"
          description: "Subfator nível 1."
        - name: "Subfator 2"
          description: "Subfator nível 2."
        - name: "Bag de termos"
          description: "Lista de termos PT separados por ponto e vírgula."
        - name: "Bag of terms"
          description: "Lista de termos EN separados por ponto e vírgula."
        - name: "Recomendação 1"
          description: "Recomendação associada (opcional)."
        - name: "Recomendação 2"
          description: "Recomendação associada (opcional)."

  csvs:
    - name: ws_prec_edges
      path: "data/analytics/ws_precursors_edges.csv"
      description: "Relação WeakSignal → Precursor com HTO."
      columns_any_order: ["HTO", "WeakSignal", "Precursor"]

    - name: ws_prec_edges_alt
      path: "data/analytics/edges_ws_prec.csv"
      description: "Arquivo alternativo de arestas WS → Precursor."
      columns_any_order: ["HTO", "WeakSignal", "Precursor"]

    - name: precursors_csv
      path: "data/analytics/precursors.csv"
      description: "Tabela auxiliar de precursores."
      columns_any_order: ["HTO", "Precursor"]

  reports:
    folder: "data/docs"
    include_glob: ["*.pdf", "*.docx"]
    description: "Relatórios históricos (PDF/DOCX) para indexação."
    chunking:
      max_chars: 1200
      overlap: 200
