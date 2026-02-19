



## Visão Geral e Objetivo de Negócio

### 1) Declaração formal do objetivo
O objetivo deste projeto é desenvolver um modelo de Machine Learning capaz de prever o risco de um estudante apresentar defasagem escolar no próximo ano letivo (`t+1`), utilizando exclusivamente informações disponíveis até o ano corrente (`t`). A previsão tem caráter preventivo e visa apoiar decisões educacionais da Associação Passos Mágicos, priorizando alunos com maior risco.

### 2) Enquadramento do problema de Machine Learning
- Problema de `classificação binária` (risco vs. não risco).
- Foco em `estimativa de risco futuro`, e não em explicação retrospectiva.
- Uso de dados futuros é proibido para evitar `data leakage`.

### 3) Interpretação de negócio da defasagem escolar
No contexto da instituição, defasagem escolar representa desalinhamento entre o nível educacional esperado e o nível efetivamente observado no estudante. Valores negativos indicam maior atraso em relação ao esperado e, portanto, maior risco educacional. O interesse de negócio está em antecipar essa condição no ano seguinte para permitir intervenção preventiva.
No dataset operacional, essa condição é representada nos campos `Defas`/`Defasagem`, usados como referência de risco educacional no recorte temporal.

### 4) Contexto de uso da previsão
- Usuários potenciais: coordenação pedagógica, equipe psicopedagógica e gestão educacional.
- Uso principal: priorização de acompanhamento preventivo e alocação de suporte para alunos em risco.
- Decisão de risco: falsos negativos têm custo maior que falsos positivos, pois deixam de sinalizar alunos que precisariam de intervenção.
O modelo tem caráter preditivo e não causal, sendo utilizado exclusivamente como ferramenta de apoio à decisão humana.

### 5) Implicações técnicas assumidas nesta fase
- Horizonte temporal adotado: `t -> t+1`.
- Tipo de problema: `classificação binária`.
- Métrica prioritária: `Recall` (minimização de falsos negativos).
- Saída esperada do modelo: `probabilidade de risco` (com posterior aplicação de threshold operacional).
- Coorte temporal: pares válidos consideram estudantes com `RA` presente em anos consecutivos (`t` e `t+1`).

### Escopo aberto do Datathon e motivação do problema escolhido
As orientações mais recentes do Datathon caracterizam o desafio como escopo aberto: a equipe pode propor abordagens distintas (como classificação, clusterização ou soluções com LLM), desde que a motivação do problema seja bem justificada no contexto de negócio e operação.

Decidimos manter o problema de classificação de risco de defasagem escolar em `t+1` porque ele é diretamente acionável para intervenção preventiva, priorização pedagógica e alocação de suporte. Essa escolha preserva alinhamento com a dor operacional da instituição e com o uso prático esperado pelos stakeholders.

#### Justificativa: por que “risco de defasagem em t+1” é acionável

Escolhemos prever o risco de um estudante apresentar defasagem no próximo ano (`t+1`) porque este é um sinal diretamente operacionalizável para intervenção preventiva. Diferente de análises puramente descritivas, o score de risco permite transformar dados em uma lista priorizada de alunos que devem receber atenção antes que a defasagem se consolide.

Na prática, a probabilidade prevista pode ser usada para:
- Priorizar acompanhamento pedagógico e psicopedagógico (triagem) quando a capacidade de atendimento é limitada.
- Direcionar reforço e monitoria para grupos de maior risco, com ações antes do fechamento do próximo ciclo letivo.
- Padronizar critérios de priorização (reduzindo subjetividade) e registrar evidências de decisão para acompanhamento.

O custo de erro é assimétrico: falsos negativos (não sinalizar um aluno que ficará defasado) têm impacto maior do que falsos positivos. Por isso, definimos `Recall` como métrica primária e operamos com threshold orientado a minimizar casos não detectados, aceitando um aumento controlado de alertas.

Do ponto de vista de sistema, este problema também é adequado para produção porque:
- A geração de features usa apenas dados do ano corrente (`t`), evitando leakage.
- O rótulo (`Defasagem_{t+1}`) chega com atraso, o que é compatível com estratégia de mensuração offline (pós-fato) e monitoramento online via drift/distribuição de scores.
- O pipeline permite retreinamento periódico conforme novos dados anuais/semestrais entram, mantendo o modelo atualizado para mudanças de população e de processo.

#### Alternativas consideradas e por que não adotamos como escopo principal

**1) Clusterização (segmentação de perfis de alunos)**  
A clusterização poderia identificar perfis como “alto engajamento com dificuldade”, “baixo engajamento e queda persistente” ou “bom desempenho e estabilidade”, apoiando ações diferenciadas por grupo. Não adotamos como escopo principal porque exige validação qualitativa forte com especialistas (interpretação dos clusters), definição de métricas de utilidade (não há “ground truth” direto) e tende a aumentar o risco de subjetividade na entrega acadêmica. A abordagem pode ser incorporada futuramente como camada complementar (ex.: cluster + risco) para orientar tipos de intervenção.

**2) Solução com LLM (assistente/relatórios pedagógicos)**  
Uma solução com LLM poderia gerar relatórios individualizados e recomendações pedagógicas a partir do histórico do aluno e apoiar professores na tomada de decisão. Não adotamos como escopo principal porque traz dependências fora da stack proposta, maior complexidade de governança (alucinações, segurança, privacidade e auditoria), e requer validação operacional e critérios de qualidade diferentes dos exigidos para um modelo supervisionado. O projeto atual mantém foco em previsões reproduzíveis, mensuráveis e auditáveis com dados tabulares.

**3) Classificação de risco de evasão (t -> t+1)**  
Prever evasão escolar seria altamente relevante para retenção e planejamento de acompanhamento. Não adotamos como escopo principal porque, no dataset atual, a “evasão” não está rotulada de forma explícita e consistente: a ausência de `RA` em `t+1` pode refletir evasão, transferência, mudança de cadastro ou outras causas (ambiguidade de rótulo). Sem um contrato claro do processo, o target ficaria ruidoso e poderia induzir conclusões erradas. Ainda assim, as estatísticas de coorte e interseção por `RA` já implementadas são base para explorar essa hipótese com validação institucional.

**4) Prever melhora/piora de defasagem (delta de defasagem entre anos)**  
Modelar a variação (melhora/piora) da defasagem poderia apoiar identificação de trajetórias e eficácia de intervenções. Não adotamos como escopo principal porque a formulação depende de escolhas adicionais (regra do delta, discretização, classes e interpretação) e pode ser mais sensível a ruído e mudanças de medição entre anos. Para esta entrega, preferimos um target mais direto e acionável (“estar defasado em t+1”) com decisão de risco clara e prioridade em Recall. A previsão de delta pode ser explorada como extensão, reaproveitando o pareamento temporal já implementado.

O modelo continua com caráter preditivo de apoio à decisão humana: não é causal nem prescritivo. O foco de engenharia deste projeto é o sistema de ML em produção, incluindo entrada de alunos novos, validação por contrato, inferência, mensuração em produção, monitoramento de drift, retreinamento e promoção/rollback de versões.

## Definição do Target

- Regra formal do target binário:
  - `y = 1` se `Defasagem_{t+1} < 0`
  - `y = 0` caso contrário
- Comparador adotado: estritamente `< 0`.
- Recorte temporal oficial:
  - Treino: `X(2022) -> y(2023)`
  - Holdout final: `X(2023) -> y(2024)`
- Política para qualidade do target em `t+1`:
  - Tokens inválidos (ex.: `#N/A`, `#DIV/0!`, `INCLUIR`) são convertidos para `NaN` antes da definição de `y`.
  - Pares com target ausente/inválido são excluídos.
  - As contagens de exclusão por `missing` e `invalid` são registradas em log.
- Regra de coorte por `RA`:
  - Apenas estudantes presentes em ambos os anos consecutivos (`t` e `t+1`) entram nos pares temporais.
- Regra anti-leakage:
  - `X` usa somente variáveis de `t`.
  - `y` é calculado exclusivamente com `Defasagem` de `t+1`.
  - `RA` é usado apenas como identificador/auditoria, nunca como feature.
  - O dataset de pares temporais implementa validações anti-leakage e falha caso colunas do ano `t+1` vazem para `X` (ex.: sufixos de merge).
- A métrica primária de sucesso é Recall (minimizar falsos negativos). Como métricas secundárias de acompanhamento e trade-off, reportamos PR-AUC (Average Precision), Precision, F1-score e ROC-AUC.

## Análise das Bases e Dicionário

A análise detalhada do dicionário de dados e das bases `2022`, `2023` e `2024` está documentada em:

- [docs/analise_bases_e_dicionario.md](docs/analise_bases_e_dicionario.md)
- Regra de ingestão aplicada: `Defas` (2022) é padronizada para `Defasagem` para manter schema único entre anos.

## Dados e Ingestão

- O arquivo XLSX do projeto contém as abas `PEDE2022`, `PEDE2023` e `PEDE2024`.
- O caminho do arquivo pode ser configurado via `DATASET_PATH`.
- A leitura raw foi separada da padronização:
  - `load_pede_workbook_raw` / `load_year_sheet_raw`: apenas leitura.
  - `load_pede_workbook` / `load_year_sheet`: wrappers com padronização.
- A harmonização de schema usa nomes canônicos entre anos, incluindo:
  - `Defas -> Defasagem`
  - `Matem -> Mat`, `Portug -> Por`, `Inglês -> Ing`
  - `Idade 22 -> Idade`
  - `Fase ideal/Fase Ideal -> Fase_Ideal`
  - `Nome/Nome Anonimizado -> Nome_Anon`
  - `Ano nasc/Data de Nasc -> Data_Nasc`
- Regras de fallback para colunas canônicas derivadas:
  - `INDE` por ano:
    - 2022: `INDE 22`
    - 2023: `INDE 2023` -> `INDE 23` -> `INDE 22`
    - 2024: `INDE 2024` -> `INDE 23` -> `INDE 22`
  - `Pedra_Ano` por ano:
    - 2022: `Pedra 22` -> `Pedra 21` -> `Pedra 20`
    - 2023: `Pedra 2023` -> `Pedra 23` -> `Pedra 22`
    - 2024: `Pedra 2024` -> `Pedra 23` -> `Pedra 22`
- Duplicadas de planilha (`.1`, `.2`, ...) são tratadas de forma determinística como `__dupN`, sem perda silenciosa.
- O crosswalk de equivalência de colunas é centralizado em `src/column_mapping.py` e documentado em `docs/column_mapping.md` / `docs/column_mapping.json`.
  - Aplicação ocorre antes do alinhamento entre anos, com resolução de colisões por `combine_first` e auditoria em `metadata["column_mapping_report"]`.
  - Quando múltiplas colunas candidatas mapeiam para a mesma canônica (ex.: `Defasagem` + `Defasagem__dup1`), elas são combinadas (`combine_first`) e as colunas fonte são descartadas para manter o schema canônico.
- Nota semântica importante:
  - `Ano nasc` e `Data de Nasc` não são semanticamente idênticos (ano vs data completa). Nesta fase harmonizamos header e aplicamos uma normalização mínima de conteúdo para garantir tipo `datetime` e evitar quebra do pipeline. A interpretação semântica fina (ex.: precisão de data vs apenas ano) continua sendo uma limitação conhecida.
  - `Nome` e `Nome Anonimizado` são harmonizados para `Nome_Anon` apenas para alinhamento de schema; isso não garante anonimização no dado de 2022.
- Padronização de tipos após harmonização/alinhamento:
  - `Data_Nasc` é padronizada para `datetime` com desambiguação explícita:
    - valores numéricos em `1900..2100` são interpretados como ano (`YYYY-01-01`)
    - demais numéricos são interpretados como serial Excel (`origin=1899-12-30`)
  - `Idade` é sanitizada para remover valores datetime (ex.: `1900-01-...`, que viram `NaN`) e convertida para `Int64` (nullable).
  - Nota técnica (2023): quando `Idade` aparece como artefato de planilha no formato `1900-01-XX`, o pipeline interpreta `XX` como idade para recuperar o valor numérico. Essa recuperação é aplicada apenas quando o padrão indica claramente artefato de planilha e passa por validação de faixa plausível (`3..30`).
  - Colunas numéricas usam dtypes nulos estáveis (`Float64`/`Int64`) com coerção robusta (`to_numeric(..., errors=\"coerce\")`), incluindo tratamento do token `INCLUIR`.
  - Colunas categóricas são padronizadas para `string` com `strip`.
- Normalização de categorias textuais:
  - `Gênero`: `Menina/Menino` -> `Feminino/Masculino`
  - `Instituição de ensino`: `Escola Pública/Publica` -> `Pública`; unificação de variações de capitalização
  - `Pedra*` e `Pedra_Ano`: `Agata` -> `Ágata`; `INCLUIR` -> `NA`
  - `Turma`: normalização para `UPPERCASE`
  - `Fase` e `Fase_Ideal`: tratadas como categóricas textuais (não numéricas) por instabilidade semântica em 2024
  - Auditoria da normalização em `artifacts/category_normalization_report.json`

## Validação de Consistência

- O validador checa automaticamente:
  - sanidade de `RA` (nulo, branco, duplicado)
  - missing por coluna com distinção entre:
    - missing estrutural (coluna não existia no ano e foi adicionada no alinhamento)
    - missing real (coluna existia no ano e está vazia)
  - coerção/tipos inválidos por coluna com base no relatório de tipagem
  - validação automática contra data contracts versionados em `docs/contracts/` (nome, tipo, missing e domínio por coluna)
  - respeito a `presence` (`original` vs `structural_optional`) e `enforcement` (`error`, `warning`, `info`)
- Execução padrão:

```bash
python -m src.validate
```

- Relatórios gerados em:
  - `artifacts/data_quality_report.json` (sempre)
  - `artifacts/data_quality_report.md` (opcional)
- Modo de execução:
  - `strict=False` por padrão (gera relatório sem quebrar o fluxo)
  - use `--strict` para falhar o pipeline quando houver `ERROR`
  - use `--contracts-dir` para apontar um diretório customizado de contratos versionados

## Coorte e Interseção por RA

- O projeto calcula estatísticas de interseção de `RA` entre anos para rastreabilidade do recorte temporal.
- O relatório agrega apenas contagens e percentuais (sem listas de identificadores).
- Execução:

```bash
python -m src.cohort_stats
```

- Artefatos gerados:
  - `artifacts/ra_intersections.json` (sempre)
  - `artifacts/ra_intersections.md` (opcional)

Resumo atual do dataset:

| Par | Interseção | % do 1º ano | % do 2º ano | União | Jaccard |
|---|---:|---:|---:|---:|---:|
| 2022 ∩ 2023 | 600 | 69.77% | 59.17% | 1274 | 0.4710 |
| 2023 ∩ 2024 | 765 | 75.44% | 66.18% | 1405 | 0.5445 |
| 2022 ∩ 2024 | 472 | 54.88% | 40.83% | 1544 | 0.3057 |

## Data Contract

- O projeto mantém contratos de dados versionados por ano (`2022`, `2023`, `2024`) em:
  - `docs/contracts/data_contract_2022.json`
  - `docs/contracts/data_contract_2023.json`
  - `docs/contracts/data_contract_2024.json`
- Cada contrato define:
  - `dtype` esperado por coluna
  - `presence` da coluna no ano (`original` vs `structural_optional`)
  - `enforcement` por regra (`error`, `warning`, `info`)
  - domínios plausíveis (`range`, `set`, `date_range`, `none`)
- O contrato também marca campos sensíveis (`pii`) e metadados de linhagem (`contract_version`, `generated_at`, `dataset_basename`, `dataset_sha256`).
- Exportação dos contratos:

```bash
python -m src.contracts --export
```

## 📁 Estrutura do Projeto

O repositório é organizado para separar claramente ingestão e tratamento de dados, treinamento do modelo, disponibilização via API, monitoramento e testes, garantindo manutenibilidade, reprodutibilidade e facilidade de deploy.

```
fiap-techchalenge-f5/
├── README.md                     # documentação principal do projeto
├── .gitignore                    # regras de versionamento/artefatos ignorados
├── requirements.txt              # dependências de runtime
├── requirements-dev.txt          # dependências de desenvolvimento e testes
├── agents.md                     # convenções operacionais para agentes LLM
├── app/                          # camada de aplicação (API/serving)
│   └── model/                    # diretório de artefatos de modelo para inferência
│       └── .gitkeep
├── artifacts/                    # artefatos gerados (modelos, metadados, etc.)
│   └── .gitkeep
├── dashboards/                   # dashboards de monitoramento/visualização
├── docs/                         # documentação técnica complementar
│   ├── .gitkeep
│   ├── analise_bases_e_dicionario.md  # análise das bases e dicionário de dados
│   ├── column_mapping.md         # tabela de equivalência de colunas entre anos
│   ├── column_mapping.json       # espelho JSON do crosswalk de colunas
│   └── contracts/                # contratos versionados por ano
│       ├── data_contract_2022.json
│       ├── data_contract_2023.json
│       └── data_contract_2024.json
├── logs/                         # logs locais da aplicação/pipeline
│   └── .gitkeep
├── notebooks/                    # exploração e experimentos
│   └── .gitkeep
├── src/                          # código-fonte do pipeline de dados e ML
│   ├── __init__.py
│   ├── categories.py             # normalização textual de categorias e auditoria
│   ├── column_mapping.py         # crosswalk e harmonização de colunas equivalentes
│   ├── cohort_stats.py           # estatísticas de interseção de RA por ano
│   ├── config.py                 # constantes globais (ex.: RANDOM_STATE)
│   ├── contracts.py              # definição/export de data contracts por ano
│   ├── data.py                   # ingestão XLSX e geração de pares temporais
│   ├── dtypes.py                 # padronização de tipos e auditoria de coerção
│   ├── feature_pruning.py        # plano determinístico de remoção de colunas irrelevantes/leakage
│   ├── features.py               # seleção de features e split num/cat/datetime
│   ├── imputation.py             # plano de imputação de missing para treino/inferência
│   ├── leakage.py                # detecção/assert explícito de data leakage
│   ├── pipeline_components.py    # transformer sklearn serializável (raw -> model frame)
│   ├── preprocessing.py          # ColumnTransformer com imputação + one-hot + escalonamento numérico opcional
│   ├── contract_validate.py      # validação automática dos data contracts
│   ├── schema.py                 # harmonização/alinhamento de schema entre anos
│   ├── smoke_pipeline.py         # smoke test oficial da pipeline (com/sem sklearn)
│   ├── train_pipeline.py         # fábrica da Pipeline sklearn completa (raw_to_model + preprocessor + model)
│   ├── utils.py                  # utilitários compartilhados (ex.: logging)
│   └── validate.py               # validação de consistência e geração de relatórios
└── tests/                        # suíte de testes automatizados
    ├── __init__.py
    ├── test_categories.py        # testes de normalização de categorias
    ├── test_cohort_stats.py      # testes de interseção por RA e privacidade
    ├── conftest.py               # configuração compartilhada dos testes
    ├── test_column_mapping.py    # testes do crosswalk e harmonização de equivalências
    ├── test_config.py            # testes de configuração global
    ├── test_contract_validate.py # testes da validação automática de contratos
    ├── test_contracts.py         # testes dos contratos de dados por ano
    ├── test_data.py              # testes de ingestão e pares temporais
    ├── test_dtypes.py            # testes da padronização de tipos
    ├── test_feature_pruning.py   # testes do feature pruning plan (fit-only treino, apply-only inferência)
    ├── test_features.py          # testes da seleção/split de features
    ├── test_imputation.py        # testes da política e plano de imputação
    ├── test_inference_reusability.py # testes do contrato de entrada e reuso do pré-processamento na inferência
    ├── test_leakage.py           # testes da lista negra e asserts temporais anti-leakage
    ├── test_logging.py           # testes de logging centralizado
    ├── test_preprocessing_bundle.py # testes de integração do bundle (raw -> engineered -> preprocessor)
    ├── test_preprocessing.py     # testes do ColumnTransformer e OneHotEncoder
    ├── test_pipeline_build.py    # testes da pipeline end-to-end serializável (fit/predict/joblib)
    ├── test_raw_to_model_transformer.py # testes do transformer RAW->MODEL sem dependência de sklearn
    ├── test_schema.py            # testes de harmonização/alinhamento de schema
    └── test_validate.py          # testes do validador de consistência
```

## Ambiente Local (.venv)

### macOS / Linux

1) Criar e ativar ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Instalar dependências

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Reprodutibilidade: `RANDOM_STATE = 42` é usado globalmente no projeto.

### Execução de Testes

1) Instalar dependências de desenvolvimento

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
```

2) Rodar suíte de testes

```bash
pytest -q
```

3) Rodar testes com cobertura (comando oficial do projeto)

```bash
pytest --cov=src --cov-report=term-missing --cov-fail-under=80
```

Observação: mantenha este comando de cobertura sempre documentado no `README.md` para padronizar validação local e evidência técnica da entrega.

Observação sobre SKIP: alguns testes de pipeline usam `pytest.importorskip("sklearn")`.
Se o ambiente não tiver sklearn disponível, esses testes serão marcados como `SKIPPED`.
No CI da Fase 7, manter instalação via `requirements-dev.txt` para evitar skip indevido.

### Smoke da Pipeline

Executar smoke oficial:

```bash
python -m src.smoke_pipeline
```

Com sklearn disponível: roda fit/predict_proba + roundtrip `joblib`.
Sem sklearn: roda apenas `RawToModelFrameTransformer` e valida o caminho `RAW -> MODEL`.

## Logging

- Logging básico centralizado em `src/utils.py` com `setup_logging()` e `get_logger()`.
- Nível padrão: `INFO`. Para ajustar em runtime:
  - `LOG_LEVEL=DEBUG` (ou `INFO`, `WARNING`, `ERROR`)
  - Valores inválidos de `LOG_LEVEL` fazem fallback para `INFO` (com warning em log).
- Saída padrão: `stdout`.
- Opcional: habilitar arquivo em `logs/app.log` com:
  - `LOG_TO_FILE=1`
- Idempotência: `setup_logging()` pode ser chamado múltiplas vezes sem duplicar handlers/logs.
- Privacidade operacional:
  - Não logar `RA`, listas de identificadores, payloads completos ou dados pessoais.
  - Logar apenas métricas agregadas e contadores operacionais.

## Separação de Features (Fase 4)

- A seleção de features exclui PII por padrão a partir de `PII_COLUMNS` do contrato (`Nome_Anon` e `Avaliador*`), sem depender de listas duplicadas.
- As features são separadas por dtypes em três grupos: `numeric`, `categorical` e `datetime` (com `Data_Nasc` tratada explicitamente como datetime nesta etapa).
- O split pode ser auditado no pareamento temporal:
  - `make_temporal_pairs(..., persist_feature_split=True)` gera `artifacts/feature_split_report.json`.
  - por padrão (`persist_feature_split=False`), não há side effect de escrita em disco.

## Imputação de Missing (Fase 4)

- A política de imputação é definida como plano auditável em `src/imputation.py`, para uso dentro de `Pipeline/ColumnTransformer` na etapa de treino.
- Estratégias padrão:
  - numéricas: `median` + `add_indicator=True`
  - categóricas: `most_frequent` + `add_indicator=True`
  - datetime (`Data_Nasc`): não imputado nesta etapa; permanece em `datetime_cols_excluded` para tratamento posterior de feature engineering.
- Colunas 100% missing no recorte real de treino (`2022->2023`) são removidas do conjunto de imputação quando `drop_all_missing_columns=True`:
  - `Ativo/ Inativo`, `Ativo/ Inativo__dup1`, `Destaque IPV__dup1`, `Escola`, `INDE 2023`, `INDE 2024`, `INDE 23`, `IPP`, `Pedra 2023`, `Pedra 2024`, `Pedra 23`, `Rec Psicologia`
- Evidência local:
  - `artifacts/imputation_plan.json` (gerado por `persist_imputation_plan(...)`).

## Codificação Categórica (Fase 4)

- A codificação categórica é feita com `OneHotEncoder(handle_unknown="ignore")` para tolerar categorias novas em produção sem quebrar inferência.
- O bloco categórico é aplicado após imputação (`SimpleImputer(strategy="most_frequent", add_indicator=True)`), dentro de `ColumnTransformer` em `src/preprocessing.py`.
- O bloco numérico usa `SimpleImputer(strategy="median", add_indicator=True)`.
- `Fase` e `Fase_Ideal` permanecem categóricas nesta etapa.
- `Data_Nasc` (datetime) não entra na codificação nesta fase; fica para feature engineering posterior.

## Escalonamento Numérico (Fase 4)

- O pré-processador agora suporta escalonamento numérico configurável em `src/preprocessing.py`.
- Regra adotada:
  - baseline linear (`Logistic Regression`): usar `numeric_scaler="standard"` (preset `DEFAULT_SCALER_FOR_LINEAR`);
  - modelos de árvore (ex.: `HistGradientBoosting`): usar `numeric_scaler="none"` (preset `DEFAULT_SCALER_FOR_TREE`).
- O escalonador pode ser configurado entre `standard`, `robust` e `none`, com validação explícita de parâmetro.

## Reuso do Pré-processamento na Inferência (Fase 4)

- O contrato de entrada para inferência é exposto por `get_expected_raw_feature_columns()` e deriva diretamente de `get_feature_columns_for_model()` (fonte única de verdade).
- `validate_inference_frame(...)` valida:
  - tipo de entrada (`pandas.DataFrame`);
  - colunas mínimas esperadas (falha com erro claro para colunas faltantes);
  - colunas extras são permitidas por padrão (registradas apenas por nome/contagem).
- `build_preprocessing_bundle(...)` entrega um bundle reutilizável para treino/API contendo:
  - `expected_raw_cols` (contrato da API),
  - `expected_model_cols` (raw + engineered),
  - `excluded_cols`, `numeric_scaler` e `preprocessor`,
  - `transform_raw_to_model_frame(...)` para aplicar feature engineering internamente antes do `ColumnTransformer`.
- O contrato da API valida somente colunas raw; as features derivadas são detalhe interno do pipeline.

## Feature Engineering (Fase 4)

- Features derivadas simples e anti-leakage (somente dados de `t`) são criadas em `src/features.py` por `add_engineered_features(...)`.
- Numéricas:
  - `avg_grades`, `min_grade`, `max_grade`, `grade_std`, `missing_grades_count`
  - `missing_indicators_count`
  - `defasagem_abs`, `defasagem_neg_flag`, `age_is_missing_flag`
- Categórica opcional:
  - `age_bucket` (`07_10`, `11_14`, `15_18`, `19_plus`)
- A engenharia é opt-in no bundle (`enable_feature_engineering=True/False`) e pode incluir/excluir `age_bucket` (`enable_age_bucket`).

## Gate Anti-Leakage (Fase 4)

- A validação explícita de leakage fica em `src/leakage.py` com `assert_no_leakage(...)`.
- O gate roda em três pontos:
  - construção de pares temporais (`make_temporal_pairs`) no contexto `TRAIN`, com assert temporal `t -> t+1`;
  - validação de inferência (`validate_inference_frame`) no contexto `RAW`, bloqueando payloads com colunas suspeitas (ex.: `_y`, `_t1`, `target`, `t+1`);
  - transformação para model frame (`transform_raw_to_model_frame`) no contexto `MODEL`, antes do `ColumnTransformer`.
- Colunas suspeitas e 100% ausentes (artefato estrutural de alinhamento) são toleradas apenas quando não têm sinal (`n_non_null = 0`) e podem ser removidas no fluxo de treino, mantendo logs apenas agregados por nome de coluna.

## Feature Pruning (Fase 4)

- A remoção de colunas irrelevantes/leakage é feita por plano determinístico em `src/feature_pruning.py`.
- O plano (`compute_feature_pruning_plan`) é calculado apenas no treino, após feature engineering.
- Regras auditáveis do pruning:
  - `all-missing`
  - constante (`n_unique <= 1`)
  - alta cardinalidade categórica (threshold absoluto e por taxa)
  - exclusões explícitas (ex.: PII)
  - colunas bloqueadas por leakage (já detectadas pelos gates)
- Na inferência, o plano é somente aplicado (`apply_feature_pruning_plan`) sem recalcular critérios no payload de produção.
- Artefato local de auditoria: `artifacts/feature_pruning_report.json`.

## Decisões de Feature Engineering (Fase 4)

Esta fase consolida as decisões necessárias para transformar dados crus (já validados na Fase 3) em um model frame consistente e reutilizável em treino e inferência.

### 1) Princípios e escopo
- Feature engineering ocorre após ingestão/qualidade (Fase 3) e antes do treinamento (Fase 5).
- As mesmas transformações são reaplicáveis em inferência via `build_preprocessing_bundle(...)`.
- O modelo é preditivo (não causal) e opera apenas com dados disponíveis em `t`.

### 2) Exclusões por privacidade e identificação
- `RA` nunca é feature (somente ID/auditoria).
- Colunas sensíveis (`PII_COLUMNS` em `src/contracts.py`) são excluídas do model frame, incluindo `Nome_Anon` e `Avaliador1..Avaliador6`.
- `Nome_Anon` é tratado como sensível porque em 2022 pode representar nome real.

### 3) Split numérico / categórico / datetime (canônico vs snapshot)
- O preprocessor usa listas canônicas estáveis em `src/preprocessing.py`:
  - `NUMERIC_COLS`: 18
  - `CATEGORICAL_COLS`: 18
  - `DATETIME_COLS`: 1
- `Data_Nasc` é classificada como datetime, mas não entra no model frame nesta fase.

Snapshots agregados do recorte temporal (após exclusão de PII e drops estruturais de leakage):
- `2022->2023`: numeric=20, categorical=22, datetime=1, total_features no recorte=43, all_missing remanescente=6, leakage estrutural dropado=6 (`INDE 2023`, `INDE 2024`, `INDE 23`, `Pedra 2023`, `Pedra 2024`, `Pedra 23`).
- `2023->2024`: numeric=22, categorical=24, datetime=1, total_features no recorte=47, all_missing=19, leakage estrutural dropado=2 (`INDE 2024`, `Pedra 2024`).

### 4) Missing e imputação
- Numéricas: `SimpleImputer(strategy="median", add_indicator=True)`.
- Categóricas: `SimpleImputer(strategy="most_frequent", add_indicator=True)`.
- `add_indicator=True` aumenta dimensionalidade após `fit`, comportamento esperado da pipeline.

### 5) Codificação de categóricas
- `OneHotEncoder(handle_unknown="ignore")` para robustez com categorias novas na inferência.
- Compatibilidade de versão do sklearn é tratada em helper único (`sparse_output` vs `sparse`).

### 6) Escalonamento numérico (quando necessário)
- Baseline linear (`LogisticRegression`): `StandardScaler` (opção `RobustScaler`).
- Modelo de árvore (`HistGradientBoostingClassifier`): sem scaling por padrão.

### 7) Features derivadas (quando habilitadas)
- `add_engineered_features(...)` aplica derivação determinística e NA-safe antes do `ColumnTransformer`.
- Quando habilitadas, as novas colunas entram explicitamente no conjunto esperado do model frame.
- Não há uso de informação futura nas features derivadas.

### 8) Anti-leakage (RAW + MODEL/TRAIN)
- `RAW`: bloqueio estrito de colunas extras future-like/target-like.
- `MODEL/TRAIN`: tolera suspeitas 100% missing (ruído estrutural), mas falha se houver qualquer sinal não nulo.
- Detecção semântica usa padrões específicos (`INDE/Pedra` + ano, sufixos `_t1`, `target`, etc.) sem regex genérica de ano.

### 9) Pruning (fit no treino, apply na inferência)
- Pruning remove colunas sem sinal/instáveis com regras configuráveis (all-missing, constante, alta cardinalidade, exclusões explícitas).
- O plano é fitado no treino e somente aplicado na inferência.
- Relatórios são agregados (nomes/contagens), sem valores de célula ou IDs.

### 10) Reutilização em inferência
- A API valida schema raw, aplica feature engineering interno (quando habilitado), aplica seleção/model frame e gate anti-leakage, depois transforma com o mesmo preprocessor do treino.
- Isso garante consistência treino=inferência.

Referências:
- `src/preprocessing.py`
- `src/leakage.py`
- `src/contracts.py`
- `src/features.py`

## ColumnTransformer para Pré-processamento (Fase 5)

- O núcleo do pré-processamento do modelo é um `ColumnTransformer`, construído por `build_preprocessing_bundle(...)` em `src/preprocessing.py`.
- Pipeline numérico:
  - `SimpleImputer(strategy="median", add_indicator=True)`
  - scaler configurável por `numeric_scaler`: `"none" | "standard" | "robust"`
- Pipeline categórico:
  - `SimpleImputer(strategy="most_frequent", add_indicator=True)`
  - `OneHotEncoder(handle_unknown="ignore")` com fallback explícito no helper `_build_ohe`:
    - tenta `sparse_output=False` (sklearn mais novo)
    - se não suportado, usa `sparse=False` (sklearn mais antigo)
- `Data_Nasc` permanece fora do model frame nesta etapa.
- Estratégia de scaler:
  - default do bundle: `"none"` (caminho típico para árvores)
  - baseline linear: usar `DEFAULT_SCALER_FOR_LINEAR = "standard"` explicitamente.
  - na fábrica da pipeline completa (`build_model_pipeline`), o default é `standard` para baseline linear; para árvore, usar `scaler_strategy="none"`.

## Pipeline End-to-End (Fase 5)

- A unidade de treino/inferência serializável é uma `Pipeline` sklearn completa:
  - `raw_to_model` (`RawToModelFrameTransformer`)
  - `preprocessor` (`ColumnTransformer`)
  - `model` (estimador)
- O estágio `raw_to_model` aplica o mesmo contrato do bundle (`validate_inference_frame`, feature engineering opcional, pruning e gate anti-leakage de model frame).
- O `feature_pruning_plan` é fitado no treino (fora da `Pipeline`) e passado como plano fixo para treino/inferência, sem recalcular em produção.
- Após `fit`, o transformer expõe `expected_model_cols_` como contrato interno de consistência do model frame.
- A pipeline recebe `DataFrame` cru no `fit` e no `predict`/`predict_proba`, mantendo consistência com o contrato da API.
- O contrato raw da API usa `bundle["expected_raw_cols"]` como schema mínimo; o smoke valida `fit`/`predict_proba` diretamente a partir desse frame raw.
- Treino oficial e fixo em `2022->2023`; holdout `2023->2024` e reservado para avaliação e é bloqueado por padrão nos CLIs de treino.
- Overrides de par temporal exigem flags explicitas (`--allow-nontrain-pair`; e para holdout, também `--allow-holdout-training`).
- A construção fica centralizada em `src/train_pipeline.py` (`build_model_pipeline(...)`), sem closures no transformer para garantir serialização via `joblib`.
- Smoke oficial para esse fluxo: `python -m src.smoke_pipeline`.

## Treino Baseline (Fase 5)

- CLI oficial para baseline:
  - `python -m src.train_baseline --year-t 2022 --year-t1 2023 --out-dir artifacts/models/baseline_logreg --scaler standard --variants none --enable-feature-engineering 1 --enable-age-bucket 1`
- Artefatos por variante:
  - `artifacts/models/baseline_logreg/<variant>/model.joblib`
  - `artifacts/models/baseline_logreg/<variant>/metadata.json`
- O feature pruning e fitado no treino e aplicado na inferencia sem recalculo:
  - evita drift de schema entre treino e producao;
  - mantem compatibilidade com o `ColumnTransformer` e com o contrato de colunas do modelo.

## Treino Não-Linear (Fase 5)

- CLI oficial para modelo não-linear:
  - `python -m src.train_hgb --file-path <xlsx> --year-t 2022 --year-t1 2023 --out-dir artifacts/models/nonlinear_hgb --variants default,tuned --enable-feature-engineering 1 --enable-age-bucket 0`
- O treino utiliza `X_raw_train` (contrato raw da API) e `y_train` pareado via coorte `RA`.
- Validação interna opcional (CV estratificada) no treino oficial:
  - `--cv 1 --cv-splits 5 --cv-repeat 1`
  - a seção `cv` é anexada no `metadata.json` de cada variante (métricas agregadas por fold e mean/std).
- Artefatos por variante:
  - `artifacts/models/nonlinear_hgb/<variant>/model.joblib`
  - `artifacts/models/nonlinear_hgb/<variant>/metadata.json`

## Estratégia de Decisão e Desbalanceamento (Fase 5)

- Prevalência observada no pipeline oficial:
  - treino `2022->2023`: `n=600`, `n_pos=366`, prevalência `0.6100`
  - holdout `2023->2024`: `n=765`, `n_pos=308`, prevalência `0.4026`
- Decisão operacional padrão:
  - `class_weight=none` como default de treino
  - política de decisão padrão: `top_k` com `k_frac=0.10` (alternativa documentada: `0.20`)
  - `threshold=0.50` mantido como alternativa para comparabilidade
- Justificativa:
  - no cenário atual, `class_weight="balanced"` piorou Recall no holdout em relação a `class_weight=none`.
  - otimização de threshold no treino não generalizou melhor no holdout.
  - como a intervenção é limitada por capacidade, ranking `top-k` é mais acionável que um threshold fixo.
- Implementação:
  - utilitários agregados em `src/metrics.py` (`threshold` e `top-k`), sem persistir scores/IDs.
  - `metadata.json` dos modelos inclui:
    - `class_imbalance_strategy` (prevalência, decisão, alternativas e evidências agregadas)
    - `prediction_policy` (config padrão consumível pela camada de serviço da API)

## Comparação de Modelos (Fase 5)

- Comando oficial:
  - `python -m src.compare_models --models-root artifacts/models --out-json artifacts/model_comparison.json --out-md artifacts/model_comparison.md`
- A comparação lê apenas `metadata.json` dos artefatos de treino (sem re-treinar modelos e sem recalcular métricas no comparador).
- Política de ranking:
  - primária: Recall holdout@0.5
  - secundária: PR-AUC holdout
  - terciária: menor positive_rate holdout@0.5
- O relatório é agregado e privacy-safe: sem listas de `RA`/IDs e sem valores de célula.

## Avaliação Holdout Temporal (Fase 5)

- A avaliação `2023->2024` é estritamente read-only: o modelo é treinado em `2022->2023` e apenas inferido no holdout.
- Nos CLIs de treino (`src.train_baseline` e `src.train_hgb`), o bloco `evaluation_holdout` é incluído no `metadata.json` quando `--eval-holdout 1`.
- CLI dedicado para reavaliar artefatos serializados:
  - `python -m src.evaluate_holdout --models-root artifacts/models --dataset-path <xlsx> --output artifacts/holdout_evaluation.json`
  - o comando carrega `model.joblib` e avalia no holdout oficial sem refit.

## Métricas Oficiais (Fase 5)

- O cálculo oficial de métricas está centralizado em `src/metrics.py`, evitando lógica duplicada entre CLIs.
- Nesta fase, o threshold padrão é `0.5` (ajuste operacional de threshold é tarefa separada).
- Cada `metadata.json` salva:
  - `evaluation_train` (pair `2022->2023`)
  - `evaluation_holdout` (pair `2023->2024`, quando `--eval-holdout 1`)
  - bloco de métricas com `Recall`, `Precision`, `F1`, `ROC-AUC`, `PR-AUC` e `positive_rate`.

## Checklist do Projeto - Datathon Machine Learning Engineering

Este checklist foi elaborado considerando explicitamente as inconsistências reais do dataset fornecido (schemas distintos entre anos, colunas duplicadas, valores inválidos, mudanças semânticas de campos e interseção parcial de estudantes entre períodos). As etapas descritas adotam práticas de Data Engineering e MLOps para garantir robustez, reprodutibilidade e validade estatística do modelo em produção.

Status: `TODO` | `DOING` | `DONE` | `BLOCKED`

Progresso geral (barra visual):
`[🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜]`

`68 de 110 tarefas concluídas (61.8%)`

| Fase | Progresso |
|---|---|
| Fase 1 - Entendimento do Problema e Target | 13/13 |
| Fase 2 - Organização do Projeto e Ambiente | 7/7 |
| Fase 3 - Ingestão, Qualidade e Governança de Dados | 14/14 |
| Fase 4 - Pré-processamento e Engenharia de Features | 10/10 |
| Fase 5 - Pipeline, Treinamento e Avaliação | 12/17 |
| Fase 6 - Artefatos, API e Deploy | 0/15 |
| Fase 7 - Testes, Monitoramento e Dashboard | 2/13 |
| Fase 8 - Documentação e Entrega Final | 10/21 |
| Total | 68/110 |

### Fase 1 - Entendimento do Problema e Target [13/13]
- [x] Compreender o objetivo de negócio: prever o risco de defasagem escolar (t+1)
- [x] Estudar o dicionário de dados e as bases de 2022, 2023 e 2024
- [x] Padronizar a coluna de defasagem (`Defas` -> `Defasagem`)
- [x] Definir a formulação do target binário
- [x] Definir métrica primária de sucesso (`Recall`) e métricas secundárias (`PR-AUC`, `Precision`, `F1`, `ROC-AUC`) já na fase de desenho
- [x] Definir `y = 1` se `Defasagem_{t+1} < 0`
- [x] Definir `y = 0` caso contrário
- [x] Definir a estratégia de pares temporais
- [x] Definir treino: `X(2022) -> y(2023)`
- [x] Definir holdout final: `X(2023) -> y(2024)`
- [x] Garantir que `RA` seja usado apenas como ID, nunca como feature
- [x] Justificar o problema escolhido no contexto de escopo aberto (por que risco de defasagem `t+1` é acionável e útil)
- [x] Documentar alternativas consideradas (clusterização, LLM, evasão, prever melhora/piora) e por que não adotamos (1 parágrafo cada)

### Fase 2 - Organização do Projeto e Ambiente [7/7]
- [x] Configurar `.gitignore` inicial (ignorar `agents.md`, `dataset/` e `.DS_Store`)
- [x] Expandir `.gitignore` com padrões essenciais de Python/MLOps (cache, venv, cobertura, builds, logs e segredos locais)
- [x] Criar estrutura de diretórios do projeto
- [x] Criar `requirements.txt` com dependências mínimas
- [x] Fixar versões das dependências para garantir reprodutibilidade do ambiente de execução
- [x] Definir `random_state` global para reprodutibilidade
- [x] Configurar logging básico do projeto

### Fase 3 - Ingestão, Qualidade e Governança de Dados [14/14]
Camadas conceituais desta fase:
- Camada A - Pré-ingestão e Ingestão: contrato de dados, mapeamento de colunas equivalentes, tratamento de headers duplicados, normalização de valores inválidos, padronização de datas e normalização semântica.
- Camada B - Governança e Validação Contínua: coorte temporal por `RA`, validações de shift, versionamento de dataset e privacidade operacional.

Nota de coorte temporal:
> A construção dos pares temporais considera apenas estudantes presentes em ambos os anos consecutivos (`t` e `t+1`), evitando viés por evasão ou entrada tardia e garantindo consistência estatística na definição do target.

- [x] Implementar leitura do arquivo XLSX
- [x] Tratar diferenças de colunas entre os anos
- [x] Padronizar nomes e tipos de dados
- [x] Criar função de geração dos pares temporais (`t -> t+1`)
- [x] Validar consistência dos dados (missing, tipos inválidos)
- [x] Definir um data contract por ano (nome, tipo e domínio esperado por coluna)
- [x] Implementar validação automática do data contract (asserts de nome, tipo e domínio por coluna)
- [x] Criar tabela de mapeamento entre colunas equivalentes (`Matem/Portug/Inglês` <-> `Mat/Por/Ing`; `Defas` <-> `Defasagem`)
- [x] Tratar headers duplicados na ingestão com regra determinística
- [x] Normalizar valores inválidos em campos numéricos (ex.: `#N/A`, `#DIV/0!`, `INCLUIR`)
- [x] Padronizar datas de nascimento para formato único
- [x] Normalizar categorias textuais entre anos (`Menina/Menino` <-> `Feminino/Masculino`; `Escola Pública` <-> `Pública`)
- [x] Definir regra formal de coorte temporal por `RA` (entradas, saídas e interseções por ano)
- [x] Gerar e registrar estatísticas de interseção por `RA` entre anos (contagem absoluta e percentual)

### Fase 4 - Pré-processamento e Engenharia de Features [10/10]
- [x] Separar features numéricas e categóricas
- [x] Tratar valores ausentes (imputação)
- [x] Codificar variáveis categóricas (`OneHotEncoder` ou similar)
- [x] Escalonar variáveis numéricas (se necessário)
- [x] Garantir que o pré-processamento seja reutilizável na inferência
- [x] Criar novas features relevantes (se aplicável)
- [x] Implementar checagem explícita de data leakage (lista negra de colunas futuras + asserts temporais)
- [x] Remover colunas irrelevantes ou com leakage
- [x] Garantir que nenhuma feature use informação futura
- [x] Documentar as principais decisões de feature engineering

### Fase 5 - Pipeline, Treinamento e Avaliação [12/17]
Nota de shift temporal:
> Antes do treinamento final, é realizada uma análise de shift temporal do target e das features, uma vez que a prevalência da classe positiva varia significativamente entre os períodos analisados (aprox. `61%` para `40%`).

- [x] Criar `ColumnTransformer` para pré-processamento
- [x] Encapsular tudo em uma `Pipeline` do scikit-learn
- [x] Garantir consistência treino vs inferência
- [x] Validar que a pipeline aceita dados crus da API
- [x] Treinar modelo baseline (`Logistic Regression`)
- [x] Treinar modelo não-linear (ex.: `HistGradientBoosting`)
- [x] Usar apenas dados de treino (`2022 -> 2023`)
- [x] (Opcional) Validação interna (CV estratificada)
- [x] Definir estratégia explícita para desbalanceamento de classes (`class_weight`, ajuste de threshold ou decisão justificada de não tratar)
- [x] Comparar modelos com foco em Recall e PR-AUC
- [x] Avaliar desempenho no holdout temporal (`2023 -> 2024`)
- [x] Calcular métricas: Recall, Precision, F1-score, ROC-AUC, PR-AUC
- [ ] Gerar matriz de confusão
- [ ] Definir threshold operacional focado em Recall
- [ ] Definir critério objetivo formal de seleção do modelo final (ex.: maior Recall com PR-AUC acima de limiar mínimo)
- [ ] Justificar escolha do modelo final
- [ ] Incluir validação de shift temporal do target e das features antes do treinamento final

### Fase 6 - Artefatos, API e Deploy [0/15]
- [ ] Salvar pipeline completa em `model.joblib`
- [ ] Criar `metadata.json` com modelo, métricas, threshold, features esperadas, data do treino e versões das bibliotecas
- [ ] Salvar dados de referência para monitoramento de drift
- [ ] Versionar dataset de treino/validação (`hash/checksum` + versão usada no experimento)
- [ ] Definir schema formal de saída do modelo/API (probabilidade, classe prevista, threshold aplicado e versão do modelo)
- [ ] Criar aplicação FastAPI
- [ ] Implementar endpoint `POST /predict`
- [ ] Implementar `GET /health` e `GET /version`
- [ ] Validar entradas com Pydantic
- [ ] Garantir carregamento do modelo salvo
- [ ] Criar Dockerfile enxuto baseado em `python:slim`
- [ ] Documentar comandos de build e run no README
- [ ] Implementar versionamento de modelos local (ex.: `artifacts/models/<model_version>/` com `model.joblib` + `metadata.json`)
- [ ] Definir estratégia de promoção de modelo (staging -> prod local) com critério objetivo (Recall/PR-AUC/threshold)
- [ ] Documentar procedimento de atualização do modelo na API (troca de versão e rollback local)

### Fase 7 - Testes, Monitoramento e Dashboard [2/13]
- [x] Criar testes unitários e de integração com pytest
- [x] Garantir cobertura mínima de 80% com `pytest-cov`
- [ ] Adicionar CI automatizada (rodar `pytest`, coverage, `python -m src.validate` e `python -m src.cohort_stats`)
- [ ] Definir comportamento para alunos novos (sem histórico): validação de contrato, imputação/valores default e logging da taxa de campos ausentes
- [ ] Definir estratégia de mensuração em produção com "ground truth delay" (métricas online vs métricas offline quando o rótulo chega)
- [ ] Implementar logging agregado de inferência (distribuição de scores, taxa de positivos por threshold, taxa de erro de validação) sem PII
- [ ] Implementar rotina de avaliação pós-fato (quando labels `t+1` chegam) para medir Recall/PR-AUC em produção (mesmo que simulado)
- [ ] Definir política de retenção/limpeza de logs e artefatos locais (script simples + documentação)
- [ ] Implementar teste de não-regressão do modelo com limiares mínimos de métricas (ex.: Recall e/ou PR-AUC)
- [ ] Configurar logging estruturado
- [ ] Aplicar política de privacidade operacional (não logar identificadores sensíveis como `RA` em API e monitoramento)
- [ ] Implementar relatório de drift com Evidently
- [ ] Criar aplicação Streamlit para visualização do relatório de drift

### Fase 8 - Documentação e Entrega Final [10/21]
- [x] Documentar visão geral do problema e objetivo
- [ ] Documentar stack tecnológica
- [ ] Adicionar versionamento/changelog dos contratos (`docs/contracts`)
- [x] Documentar estrutura do projeto
- [ ] Documentar etapas do pipeline de Machine Learning
- [ ] Documentar ciclo de vida em produção: entrada de alunos novos, validação de contrato, inferência, logging, drift, retreino, promoção/rollback
- [ ] Documentar explicitamente contratos em produção (data contracts + contrato de payload da API + contrato de saída)
- [ ] Documentar estratégia de retreino (gatilhos por tempo e/ou por drift, e como executar)
- [ ] Documentar limitações conhecidas do modelo e riscos assumidos
- [ ] Documentar exemplos de chamadas à API
- [x] Documentar setup de ambiente local com `.venv` e instalação de dependências
- [ ] Publicar código organizado no GitHub
- [ ] Disponibilizar API acessível localmente
- [ ] Gravar vídeo gerencial (<= 5 minutos) explicando a solução
- [x] Criar `agents.md` com convenções operacionais para agentes LLM
- [x] Adicionar barra de progresso geral visual (`[🟩⬜...]`) no checklist
- [x] Atualizar `agents.md` com regra explícita de manutenção da barra visual e da contagem geral
- [x] Incorporar recomendações da revisão técnica do checklist (gaps de maturidade por fase)
- [x] Refinar redação do objetivo para "apresentar defasagem no t+1" (evita ambiguidade de transição vs estado)
- [x] Refinar visão geral com vínculo explícito a `Defas/Defasagem` e regra de coorte por `RA`
- [x] Adicionar menção explícita de não-causalidade do modelo na seção de contexto de uso

<details>
<summary>Notas de uso do checklist</summary>

- Atualize os contadores de progresso de cada fase ao concluir tarefas.
- Atualize a barra visual de progresso geral (`[🟩⬜...]`) com base na porcentagem concluída.
- Regra da barra: 40 blocos (`1 bloco = 2,5%`), com arredondamento para baixo.
- Marque uma tarefa como `DOING` no texto do item quando estiver em andamento.
- Promova para `DONE` apenas após evidência (teste, artefato, log ou documentação).
- Use `BLOCKED` quando depender de decisão, dado externo ou ajuste de escopo.

</details>
