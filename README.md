



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

<details>
<summary>Alternativas consideradas e por que não adotamos como escopo principal (expandir)</summary>

#### Alternativas consideradas e por que não adotamos como escopo principal

**1) Clusterização (segmentação de perfis de alunos)**  
A clusterização poderia identificar perfis como “alto engajamento com dificuldade”, “baixo engajamento e queda persistente” ou “bom desempenho e estabilidade”, apoiando ações diferenciadas por grupo. Não adotamos como escopo principal porque exige validação qualitativa forte com especialistas (interpretação dos clusters), definição de métricas de utilidade (não há “ground truth” direto) e tende a aumentar o risco de subjetividade na entrega acadêmica. A abordagem pode ser incorporada futuramente como camada complementar (ex.: cluster + risco) para orientar tipos de intervenção.

**2) Solução com LLM (assistente/relatórios pedagógicos)**  
Uma solução com LLM poderia gerar relatórios individualizados e recomendações pedagógicas a partir do histórico do aluno e apoiar professores na tomada de decisão. Não adotamos como escopo principal porque traz dependências fora da stack proposta, maior complexidade de governança (alucinações, segurança, privacidade e auditoria), e requer validação operacional e critérios de qualidade diferentes dos exigidos para um modelo supervisionado. O projeto atual mantém foco em previsões reproduzíveis, mensuráveis e auditáveis com dados tabulares.

**3) Classificação de risco de evasão (t -> t+1)**  
Prever evasão escolar seria altamente relevante para retenção e planejamento de acompanhamento. Não adotamos como escopo principal porque, no dataset atual, a “evasão” não está rotulada de forma explícita e consistente: a ausência de `RA` em `t+1` pode refletir evasão, transferência, mudança de cadastro ou outras causas (ambiguidade de rótulo). Sem um contrato claro do processo, o target ficaria ruidoso e poderia induzir conclusões erradas. Ainda assim, as estatísticas de coorte e interseção por `RA` já implementadas são base para explorar essa hipótese com validação institucional.

**4) Prever melhora/piora de defasagem (delta de defasagem entre anos)**  
Modelar a variação (melhora/piora) da defasagem poderia apoiar identificação de trajetórias e eficácia de intervenções. Não adotamos como escopo principal porque a formulação depende de escolhas adicionais (regra do delta, discretização, classes e interpretação) e pode ser mais sensível a ruído e mudanças de medição entre anos. Para esta entrega, preferimos um target mais direto e acionável (“estar defasado em t+1”) com decisão de risco clara e prioridade em Recall. A previsão de delta pode ser explorada como extensão, reaproveitando o pareamento temporal já implementado.

</details>

O modelo continua com caráter preditivo de apoio à decisão humana: não é causal nem prescritivo. O foco de engenharia deste projeto é o sistema de ML em produção, incluindo entrada de alunos novos, validação por contrato, inferência, mensuração em produção, monitoramento de drift, retreinamento e promoção/rollback de versões.

## Como Navegar Este README

- Leitura rápida (banca/gestão): `Visão Geral`, `Target`, `Stack Tecnológica`, `Etapas do Pipeline`, `Ciclo de Vida em Produção`, `Contratos em Produção`, `Estratégia de Retreino`, `Retreino Automatizado (Tempo + Drift)`, `Explainability Local do Campeão`, `Limitações Conhecidas e Riscos Assumidos`, `Exemplos de Chamadas à API`, `API Acessível Localmente`, `CI` e `Checklist`.
- Leitura técnica (engenharia/ML): `Dados e Ingestão`, `Data Contract`, `Contratos em Produção`, `Estratégia de Retreino`, `Retreino Automatizado (Tempo + Drift)`, `Explainability Local do Campeão`, `Limitações Conhecidas e Riscos Assumidos`, `Exemplos de Chamadas à API`, `API Acessível Localmente`, `Etapas do Pipeline` e os blocos detalhados (Fases 4 a 7), que estão em seções colapsáveis.
- Operação local: `Ambiente Local (.venv)`, `Rodar API Local`, `Exemplos de Chamadas à API`, `API Acessível Localmente`, `Docker`, `Drift (Evidently)`, `Dashboard de Drift (Streamlit)` e `Dashboard Operacional Consolidado (Streamlit)`.

<details>
<summary>Sumário de navegação (expandir)</summary>

- Fundamentos do problema:
  - `Visão Geral e Objetivo de Negócio`
  - `Definição do Target`
  - `Análise das Bases e Dicionário`
- Dados, contratos e pipeline:
  - `Dados e Ingestão`
  - `Validação de Consistência`
  - `Coorte e Interseção por RA`
  - `Data Contract`
  - `Stack Tecnológica`
  - `Etapas do Pipeline de Machine Learning`
- Operação em produção:
  - `Ciclo de Vida em Produção (Operação do Modelo)`
  - `Contratos em Produção (Dados + API + Saída)`
  - `Estratégia de Retreino (Gatilhos + Execução)`
  - `Retreino Automatizado (Tempo + Drift)`
  - `Explainability Local do Campeão`
  - `Limitações Conhecidas e Riscos Assumidos`
  - `Exemplos de Chamadas à API`
- Setup e execução:
  - `API Acessível Localmente (Run + Smoke)`
  - `Estrutura do Projeto`
  - `Ambiente Local (.venv)`
  - `Rodar API Local`
  - `Docker (Deploy Local)`
- Monitoramento e operação:
  - `Mensuração em Produção (Ground Truth Delay)`
  - `Retenção e Limpeza Local`
  - `Não-regressão do Modelo`
  - `Logging Estruturado`
  - `Privacidade Operacional`
  - `Drift (Evidently)`
  - `Dashboard de Drift (Streamlit)`
  - `Dashboard Operacional Consolidado (Streamlit)`
  - `CI (GitHub Actions)`
- Governança de execução:
  - `Checklist do Projeto - Datathon Machine Learning Engineering`

</details>

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
  - `Fase_Ideal` tem ambiguidade semântica/operacional no dataset (ex.: estudantes da mesma idade com valores diferentes como `Fase 1` e `Fase Alfa` em 2024). Embora a regra de negócio informada pela professora seja "fase ideal = ano escolar esperado pela idade", o projeto **não recalcula nem corrige `Fase_Ideal` pela `Idade`** nesta versão; a coluna é tratada como sinal categórico fornecido pela base e essa inconsistência permanece como limitação conhecida até formalização de regra institucional (incluindo corte etário e mapeamento de casos como `Fase Alfa`).
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

- O export agora também mantém um changelog dos contratos em `docs/contracts/`:
  - `docs/contracts/contracts_changelog.json` (resumo machine-readable de mudanças por ano)
  - `docs/contracts/CHANGELOG.md` (resumo humano)
- O diff estrutural do changelog ignora metadados voláteis como `generated_at` para evitar ruído.
- Use `--no-changelog` apenas se quiser exportar os contratos sem atualizar esse histórico (ex.: debug/local).

## Stack Tecnológica

### 1) Linguagem e runtime
- Linguagem principal: `Python 3.11.x`
- Versão recomendada local: `Python 3.11.9` (ver `.python-version`)
- Alinhamento de runtime:
  - Docker: `python:3.11-slim` (`Dockerfile`)
  - CI: `Python 3.11` (`.github/workflows/ci.yml`)
- Ambientes locais:
  - `.venv`: treino, API, CLIs e testes
  - `.venv-dashboard`: stack isolada para drift/dashboard (`Evidently + Streamlit`)

### 2) Bibliotecas principais (por área)

#### A) Dados e Machine Learning
- `pandas`, `numpy`
- `openpyxl` (leitura do dataset XLSX)
- `scikit-learn`:
  - `Pipeline`, `ColumnTransformer`
  - modelos: `LogisticRegression`, `HistGradientBoostingClassifier`
- `joblib` (serialização de artefatos do modelo)

#### B) API / Serving
- `FastAPI` (API de inferência)
- `Uvicorn` (ASGI server)
- `Pydantic` `v1.x` (validação de entrada/saída)

#### C) Qualidade, testes e reprodutibilidade
- `pytest`
- `pytest-cov`
- `logging` (stdlib) + logging estruturado JSON implementado no projeto (`src/utils.py`)

#### D) Monitoramento / Drift / Dashboard
- `Evidently` (relatório de drift em HTML, local)
- `Streamlit` (visualização local do relatório HTML)
- `prometheus-client` (endpoint `/metrics` na API)
- `Prometheus + Grafana` (observabilidade local opcional via `docker-compose.observability.yml`)
- Observação: a stack de drift/dashboard roda em ambiente isolado (`.venv-dashboard`) para evitar conflitos de dependências (ex.: `protobuf`)

### 3) Empacotamento, execução e automação
- Docker:
  - imagem base `python:3.11-slim`
  - execução local da API via `uvicorn` dentro do container
- Execução local (sem Docker):
  - API com `uvicorn`
  - CLIs de dados/ML em `src/*`
- CI (GitHub Actions):
  - `pytest` + coverage (`--cov-fail-under=80`)
  - `python -m src.regression_check` (não-regressão do campeão, modo CI-friendly)
  - `python -m src.validate --no-markdown --skip-dataset`
  - `python -m src.cohort_stats --no-markdown --skip-dataset`
- Workflow manual opcional:
  - `Dashboard Smoke (Manual)` para validar `src.drift` + dashboard Streamlit no GitHub Actions

### 4) Organização de artefatos e outputs
- Serving (API):
  - `app/model/model.joblib` + `app/model/metadata.json` (quando promovidos)
- Referência de drift (gerada por rotina específica):
  - `app/model/reference/reference_model_frame.csv`
  - `app/model/reference/reference_meta.json`
- Artefatos locais em `artifacts/`:
  - relatórios (`.json`, `.md`, `.html`)
  - versões de dataset (`artifacts/dataset_versions/*`)
  - modelos experimentais/releases locais (`artifacts/models/*`)
- Observação:
  - `artifacts/` contém artefatos gerados; parte pode aparecer versionada no repositório como evidência/saída de referência, enquanto outputs operacionais do dia a dia são locais e sujeitos à política de retenção (`python -m src.retention`)

### 5) Arquivos de dependências e configuração da stack
- `requirements.txt`: runtime principal (API + pipeline base)
- `requirements-dev.txt`: testes, coverage e ferramentas de desenvolvimento (inclui `Evidently` para drift local)
- `requirements-dashboard.txt`: stack isolada do dashboard (`Evidently + Streamlit + Playwright`)
- `Dockerfile`: empacotamento da API
- `.github/workflows/ci.yml`: automação de CI
- `.python-version`: versão recomendada para ambiente local (`pyenv`)

### 6) Nota de privacidade operacional
- Campos sensíveis como `RA`, `Nome_Anon` e `Avaliador1..Avaliador6` não são usados como features do modelo.
- Logs e monitoramento em produção/local usam apenas agregados (contagens, taxas e histogramas), sem payload raw, sem listas de IDs e sem probabilidades individuais por aluno.
- O monitoramento de drift opera sobre `MODEL frame` (sem PII), e os guardrails de privacidade/redaction estão centralizados no código (`src/privacy.py` + `src/utils.py`).

## Etapas do Pipeline de Machine Learning

Esta seção consolida o fluxo ponta a ponta do pipeline de ML (treino, avaliação, seleção, promoção e geração de referência de drift). O objetivo é conectar as peças já documentadas ao longo do README em um único mapa operacional.

### 1) Visão geral do fluxo (offline + artefatos)

```text
XLSX (PEDE2022/2023/2024)
  -> ingestão + harmonização + tipagem + normalização (RAW frame por ano)
  -> validação de qualidade + data contracts
  -> coorte temporal por RA + target t->t+1 (X_raw(t), y(t+1))
  -> anti-leakage + feature pruning
  -> RAW -> MODEL frame (transformer serializável)
  -> pré-processamento (imputação + one-hot + escalonamento opcional)
  -> treino (baseline + não-linear)
  -> avaliação holdout temporal + comparação
  -> seleção do campeão + threshold operacional
  -> versionamento/promoção (model.joblib + metadata.json)
  -> build_reference_data (referência de drift em app/model/reference)
```

### 2) Nomenclatura dos dados ao longo do pipeline

- `RAW frame`: dataframe harmonizado/alinhado por ano (após ingestão e padronização de schema/tipos).
- `X_raw`: features cruas do ano `t` (sem colunas futuras, antes do pré-processamento sklearn).
- `y`: target binário derivado de `Defasagem_{t+1}` (`1` se `< 0`, senão `0`).
- `MODEL frame`: saída do transformer `RAW -> MODEL` (features tabulares preparadas para o preprocessor/modelo, sem PII/leakage).
- `X_preprocessed`: matriz após `ColumnTransformer` (imputação + codificação categórica + escalonamento opcional).
- `Inference output`: `risk_proba`, `risk_class`, `threshold_applied`, `model_version` (contrato da API).

### 3) Etapas detalhadas (objetivo, módulos e saídas)

| Etapa | Objetivo | Módulos principais | Entradas | Saídas / artefatos |
|---|---|---|---|---|
| Ingestão e harmonização | Ler XLSX, padronizar headers, tipos e categorias entre anos | `src.data`, `src.schema`, `src.dtypes`, `src.categories`, `src.column_mapping` | `dataset/*.xlsx` | dataframes harmonizados por ano (`2022/2023/2024`) |
| Contratos e qualidade | Validar schema/tipos/missing/domínio com contratos versionados | `src.contracts`, `src.contract_validate`, `src.validate` | RAW frames + `docs/contracts/*` | `artifacts/data_quality_report.json/.md`, logs agregados |
| Coorte temporal e target | Construir pares `t -> t+1` com interseção de `RA` e target futuro | `src.data`, `src.cohort_stats` | frames por ano | pares temporais, `y`, estatísticas de coorte (`artifacts/ra_intersections.json/.md`) |
| Anti-leakage e pruning | Garantir que `X` use apenas variáveis de `t`; remover colunas inválidas/irrelevantes | `src.leakage`, `src.feature_pruning` | `X_raw` | `X_raw` seguro (sem leakage) + plano de pruning |
| Feature engineering (RAW -> MODEL) | Transformar features cruas em `MODEL frame` reusável em treino/inferência | `src.features`, `src.pipeline_components` | `X_raw` | `MODEL frame` com colunas estáveis esperadas |
| Pré-processamento sklearn | Imputar missing e codificar/transformar features para treino | `src.preprocessing`, `src.imputation` | `MODEL frame` | `X_preprocessed` + `ColumnTransformer` serializável |
| Treino e comparação | Treinar candidatos (baseline e não-linear) e comparar métricas | `src.train_baseline`, `src.train_hgb`, `src.compare_models` | treino `X(2022) -> y(2023)` | modelos candidatos + métricas + `artifacts/model_selection.json` |
| Avaliação temporal | Avaliar holdout `2023 -> 2024` e medir desempenho realista no tempo | `src.evaluate_holdout`, `src.temporal_shift`, `src.metrics` | modelo + holdout temporal | métricas (Recall/PR-AUC/etc), matriz de confusão, relatórios de shift |
| Seleção e gates | Aplicar critério objetivo (Recall primário + PR-AUC mínimo) e checar não-regressão | `src.model_selection`, `src.promotion_policy`, `src.regression_check` | metadados/relatórios dos candidatos | campeão + status `PASS/WARNING/FAIL` dos gates |
| Versionamento e promoção | Persistir release, promover para serving e manter rollback | `src.model_versioning`, `src.promote_model` | `model.joblib` + `metadata.json` do campeão | `artifacts/models/releases/*`, `app/model/model.joblib`, `app/model/metadata.json`, manifests |
| Referência para drift | Construir dataset de referência no `MODEL frame` do modelo promovido | `src.build_reference_data` | modelo promovido + dataset | `app/model/reference/reference_model_frame.csv` + `reference_meta.json` |

### 4) Sequência prática de execução (runbook resumido)

Observação: o diretório `dataset/` não é versionado no Git. Os comandos abaixo assumem o XLSX disponível localmente.

1. Validar qualidade e contratos dos dados:

```bash
python -m src.validate
```

2. Gerar estatísticas de coorte temporal por `RA`:

```bash
python -m src.cohort_stats
```

3. Treinar candidatos (baseline e HGB) e avaliar holdout:

```bash
python -m src.train_baseline
python -m src.train_hgb
python -m src.compare_models
python -m src.evaluate_holdout
```

4. Selecionar/prometer campeão e preparar serving:

```bash
python -m src.promote_model
```

5. Gerar referência para drift (após promoção):

```bash
python -m src.build_reference_data
```

6. (Opcional) Rodar checks complementares:

```bash
python -m src.regression_check
python -m src.temporal_shift
```

### 5) Artefatos gerados ao longo do pipeline (visão consolidada)

- Contratos e governança de dados:
  - `docs/contracts/data_contract_*.json/.md`
  - `docs/contracts/contracts_changelog.json`
  - `docs/contracts/CHANGELOG.md`
- Qualidade e coorte:
  - `artifacts/data_quality_report.json/.md`
  - `artifacts/ra_intersections.json/.md`
- Seleção/avaliação:
  - `artifacts/model_selection.json`
  - relatórios de holdout/shift em `artifacts/*`
- Versionamento de modelo:
  - `artifacts/models/releases/<model_version>/`
  - manifests de staging/promoção (`staging_manifest.json`, `promoted_model.json`, `release.json`)
- Serving:
  - `app/model/model.joblib`
  - `app/model/metadata.json`
- Drift (referência):
  - `app/model/reference/reference_model_frame.csv`
  - `app/model/reference/reference_meta.json`

### 6) Guardrails e decisões de desenho do pipeline

- Split temporal (não aleatório):
  - treino oficial `X(2022) -> y(2023)` e holdout final `X(2023) -> y(2024)` para refletir cenário real de produção.
- Anti-leakage:
  - `X` usa somente dados de `t`; `y` é derivado exclusivamente de `t+1`.
  - `RA` é usado para pareamento/auditoria de coorte, nunca como feature.
- Métrica de negócio priorizada:
  - `Recall` é a métrica primária (custo maior para falsos negativos), com `PR-AUC` como secundária para controlar trade-off.
- Reuso treino/inferência:
  - o caminho `RAW -> MODEL` e o pré-processamento são serializáveis e reutilizados pela API para manter consistência.
- Observabilidade e privacidade:
  - monitoramento online usa logs agregados (scores em histograma, taxas) e o monitoramento pós-fato é separado por causa do `ground truth delay`.
  - sem payload raw, sem IDs/`RA`, sem probabilidades individuais nos logs.

### 7) Relação com as seções específicas do README

- Detalhes de contratos de dados: ver **Data Contract**
- API/serving e payloads: ver seções de **API** / endpoints (`/predict`, `/health`, `/version`)
- Drift e dashboard: ver **Drift (Evidently)** e **Dashboard de Drift (Streamlit)**
- Retenção e limpeza local: ver **Retenção e Limpeza Local**

## Ciclo de Vida em Produção (Operação do Modelo)

Esta seção documenta o ciclo operacional do sistema em produção/local: da entrada de dados na API até monitoramento, mensuração pós-fato, retreino e promoção/rollback. O objetivo é mostrar o fluxo completo de operação sem repetir os detalhes já descritos nas seções específicas.

Observação de governança:
- O modelo é ferramenta de apoio à decisão humana (não substitui avaliação pedagógica/psicopedagógica).
- O rótulo de negócio (`Defasagem_{t+1}`) chega com atraso, então o ciclo separa monitoramento **online** (sem rótulo) e avaliação **offline** (pós-fato).

### 1) Visão geral do ciclo (entrada -> monitoramento -> melhoria)

```text
Payload de inferência
  -> FastAPI (/predict) + validação do body (Pydantic/FastAPI)
  -> validação de contrato raw + gate anti-leakage
  -> (opcional) payload parcial para aluno novo -> NA -> imputação
  -> pipeline de inferência (RAW -> MODEL -> preprocessor -> model)
  -> resposta (risk_proba, risk_class, threshold, model_version)
  -> logs estruturados + métricas online agregadas (sem PII)
  -> monitoramento de drift/qualidade de entrada
  -> chegada tardia do rótulo (t+1)
  -> avaliação offline (Recall/PR-AUC etc.)
  -> decisão operacional (manter / retreinar / promover / rollback)
```

### 2) Entrada em produção (incluindo alunos novos)

- A entrada operacional ocorre via `POST /predict` (single, batch ou envelope com `records`).
- O contrato base do payload usa `expected_raw_cols` do `metadata.json` do modelo em serving.
- Modo padrão (`ALLOW_PARTIAL_PAYLOAD=0`):
  - exige todas as colunas esperadas (`missing_columns` -> `400`)
- Modo para alunos novos / histórico incompleto (`ALLOW_PARTIAL_PAYLOAD=1`):
  - aceita payload parcial
  - colunas faltantes são reindexadas com `pd.NA`
  - imputação da pipeline (`SimpleImputer`) resolve missing (sem valores “mágicos”)
- Mesmo em payload parcial:
  - extras `leakage-like` continuam bloqueados (`target`, colunas futuras etc.)

Referências:
- seção `POST /predict`
- subseção `Alunos novos (sem histórico)`

### 3) Validação de contrato e respostas da API (visão operacional)

| Status | Situação típica | Ação operacional recomendada |
|---|---|---|
| `200` | Inferência concluída com sucesso | Monitorar taxas/métricas agregadas; seguir operação |
| `400` | Colunas faltantes, extras leakage-like, batch inválido | Corrigir payload/orquestração do cliente; revisar contrato raw |
| `422` | Body/JSON inválido (Pydantic/FastAPI) | Corrigir formato do request; verificar integração cliente |
| `503` | Modelo/metadata indisponíveis | Verificar `app/model/*`, promoção e restart da API |
| `500` | Erro interno excepcional | Inspecionar logs estruturados, corrigir e rerodar |

Notas importantes:
- O `422` do `/predict` é sanitizado (sem eco de payload/`input` do Pydantic).
- Respostas e logs não devem expor `RA`, IDs, payload raw ou probabilidades individuais.

### 4) Inferência e decisão em runtime

- A API reutiliza o mesmo caminho de treino:
  - `RAW -> MODEL frame -> ColumnTransformer -> model`
- A decisão de risco usa threshold operacional vindo do metadata de serving (`threshold_policy`), com fallback legado/default se necessário.
- A saída da API inclui metadados de rastreabilidade:
  - `risk_proba`, `risk_class`, `threshold_applied`, `model_version`, `model_family`, `variant`
- Em indisponibilidade de artefatos (`model.joblib` / `metadata.json`):
  - a API permanece saudável em `/health` e `/version`
  - `/predict` retorna `503` com notas diagnósticas agregadas

### 5) Logging e monitoramento online (sem rótulo)

- Logs estruturados JSON (stdout por padrão) com `request_id`, eventos e contexto agregado:
  - `src.utils.log_event(...)`
- Métricas online agregadas por request/batch:
  - `logs/online_metrics.jsonl`
  - histograma de scores (`bins`)
  - `positive_rate` no threshold operacional
  - `missing_cols_rate` / `missing_values_rate`
  - `status_family` (`2xx/4xx/5xx`)
- Erros de validação também entram no monitoramento agregado:
  - `400` (rota)
  - `422` (handler global)
- Política de privacidade operacional:
  - sem `RA`, sem payload/records, sem listas de IDs, sem scores individuais

Referências:
- `Logging Estruturado (Fase 7)`
- `Privacidade Operacional (Fase 7)`
- `Mensuração em Produção (Ground Truth Delay) (Fase 7)`

### 6) Drift e qualidade de entrada (monitoramento contínuo)

- O monitoramento de drift opera em `MODEL frame` (sem PII), usando referência do modelo promovido:
  - `app/model/reference/reference_model_frame.csv`
  - `app/model/reference/reference_meta.json`
- Relatório visual local com Evidently:
  - `python -m src.drift`
  - saída em `artifacts/drift_report.html` + `artifacts/drift_report_summary.json`
- Visualização local:
  - `streamlit run dashboards/streamlit_app.py`
  - `streamlit run dashboards/ops_dashboard.py` (consolidado: online + drift + offline)
- Além de drift estatístico, monitorar sinais operacionais:
  - explosão de `positive_rate`
  - aumento de `missing_*_rate`
  - crescimento de `4xx/422`

Referências:
- `Drift (Evidently) (Fase 7)`
- `Dashboard de Drift (Streamlit) (Fase 7)`
- `Dashboard Operacional Consolidado (Streamlit) (Fase 8/9)`

### 7) Mensuração offline quando o ground truth chega (`t+1`)

- Como o rótulo chega com atraso, métricas oficiais (Recall/PR-AUC etc.) não são avaliadas no request online.
- Quando dados de `t+1` chegam, o projeto roda avaliação pós-fato por replay local:
  - `python -m src.offline_evaluation ...`
- O fluxo offline:
  - reconstrói coorte temporal
  - reaplica o modelo (replay)
  - calcula métricas oficiais agregadas
- No contexto acadêmico/local, isso evita logar `RA` em produção e mantém privacidade operacional.

Referência:
- `Mensuração em Produção (Ground Truth Delay) (Fase 7)`

### 8) Gatilhos operacionais (sinais -> ação recomendada)

| Sinal observado | Interpretação possível | Ação recomendada |
|---|---|---|
| `positive_rate` muito acima/abaixo do padrão | mudança de distribuição de entrada / threshold descalibrado | investigar logs online e drift; revisar threshold/política |
| `missing_cols_rate` / `missing_values_rate` altos | degradação de qualidade de payload | acionar time integrador; revisar contrato raw; usar `ALLOW_PARTIAL_PAYLOAD` só quando necessário |
| `drift_report` em `WARNING/FAIL` | mudança de população/processo | investigar features com drift, comparar coortes e avaliar necessidade de retreino |
| queda de `Recall` / `PR-AUC` no offline | degradação real de performance | iniciar ciclo de retreino + comparação + seleção |
| `503` recorrente em `/predict` | problema de serving/promoção/artefato | validar `app/model/*`, promoção e restart; considerar rollback |

Observação:
- A estratégia formal de retreino (gatilhos/periodicidade) será detalhada em seção própria; aqui documentamos o ciclo operacional e os sinais práticos de ação.

### 9) Retreino, promoção e rollback (runbook operacional resumido)

Fluxo recomendado (alto nível):
1. Monitoramento aponta necessidade de reavaliar (`drift`, métricas offline, prevalência, erro operacional).
2. Executar ciclo offline de treino/avaliação/seleção:
   - `train_baseline` / `train_hgb` / `compare_models` / `evaluate_holdout` / `model_selection`
3. Validar gates e não-regressão (`src.regression_check` / `model_selection.status`)
4. Promover para `staging`, validar, depois promover para `prod local`
5. Reiniciar API (cache de modelo/metadata)
6. Validar `/health`, `/version` e um `POST /predict` de sanidade
7. Se houver regressão operacional, executar rollback por backup/release

Detalhes completos de promoção/rollback:
- `docs/pipeline_ml_deep_dives.md` (seção `Atualização do Modelo na API (Troca de Versão e Rollback) (Fase 6)`)

### 10) Runbook resumido por cenário

#### Cenário A: Operação normal (request de aluno novo)
1. Receber request no `/predict`
2. Se aluno novo/incompleto, habilitar `ALLOW_PARTIAL_PAYLOAD=1` conforme política operacional
3. Verificar `200` + resposta com `risk_proba/risk_class`
4. Monitorar `online_metrics.jsonl` (missing/positive_rate/status)

#### Cenário B: Drift detectado / mudança de distribuição
1. Gerar `drift_report.html` com `src.drift`
2. Inspecionar `share_drifted_features` + features afetadas
3. Confrontar sinais online (`positive_rate`, missing, erros)
4. Planejar/acionar retreino se impacto for persistente

#### Cenário C: Retreino e promoção
1. Treinar candidatos e avaliar holdout temporal
2. Selecionar campeão (`model_selection`)
3. Promover (`staging -> prod`)
4. Reiniciar API
5. Validar `/version`, `/health`, `/predict`

#### Cenário D: Rollback local
1. Identificar backup/release anterior
2. Restaurar `model.joblib` + `metadata.json`
3. Reiniciar API
4. Validar endpoints e retomar monitoramento

## Contratos em Produção (Dados + API + Saída)

Esta seção explicita os contratos usados em produção/local, separando claramente o que é contrato de dados (offline), contrato de entrada da API (online) e contrato de saída da API (consumo). O objetivo é evitar ambiguidade entre “schema do dataset”, “payload de inferência” e “resposta do modelo”.

### 1) Mapa dos contratos (fonte de verdade e ponto de validação)

| Contrato | Fonte de verdade | Onde é validado/aplicado | Natureza | Observações |
|---|---|---|---|---|
| Data contracts por ano (`2022/2023/2024`) | `docs/contracts/data_contract_*.json` + `src/contracts.py` | `src.contract_validate`, `src.validate`, fluxo de ingestão/qualidade | Estático por export | Focado em schema/qualidade do dataset (offline) |
| Changelog/versionamento dos contratos | `docs/contracts/contracts_changelog.json` + `docs/contracts/CHANGELOG.md` | Geração via `python -m src.contracts --export` | Histórico documental | Resume mudanças estruturais por ano (colunas/tipos/regras) |
| Contrato de payload da API (`POST /predict`) | `app/model/metadata.json` (`expected_raw_cols`) + regras de `app/routes.py` / `app/predict_utils.py` | Runtime do `/predict` | Dinâmico por modelo em serving | Mesma API pode exigir contratos diferentes conforme versão promovida |
| Contrato de saída da API | `app/schemas.py` (`PredictionResult`, `PredictResponse`) | `response_model` da FastAPI + validação Pydantic | Estável (versionado por código) | Campos de rastreabilidade e decisão (`threshold`, `model_version`) |

Observação de privacidade:
- Em todas as camadas, contratos e monitoramento operam sem expor `RA`, listas de IDs, payload raw ou probabilidades individuais.

### 2) Data contracts (dataset/ingestão offline)

Os data contracts descrevem o contrato estrutural e de qualidade das bases anuais (`PEDE2022`, `PEDE2023`, `PEDE2024`) e servem como referência de governança na ingestão.

O que os data contracts cobrem:
- nomes de colunas esperadas (já harmonizadas/documentadas por ano)
- tipo esperado (`dtype`) e observações de coerção
- missingness/domínio em nível de auditoria (quando aplicável)
- `presence` / `enforcement` (ex.: obrigatório, opcional, info)
- marcação de colunas sensíveis/PII

Arquivos principais:
- `docs/contracts/data_contract_2022.json`
- `docs/contracts/data_contract_2023.json`
- `docs/contracts/data_contract_2024.json`
- `docs/contracts/contracts_changelog.json`
- `docs/contracts/CHANGELOG.md`

Como atualizar/exportar:
```bash
python -m src.contracts --export
```

Limite importante (semântica vs estrutura):
- O data contract valida principalmente **estrutura e qualidade observável** (schema, tipos, missing, domínio).
- Nem toda regra de negócio institucional está formalizada no contrato (ex.: ambiguidade semântica `Idade x Fase_Ideal` registrada como limitação conhecida).

Referências:
- seção `Data Contract`
- seção `Validação de Consistência`

### 3) Contrato de payload da API (`POST /predict`)

#### Fonte de verdade do contrato de entrada

- A fonte de verdade do contrato de entrada em produção é o metadata do modelo em serving:
  - `app/model/metadata.json`
  - especialmente `expected_raw_cols`
- Isso significa que o contrato de payload é **dinâmico por versão de modelo promovida**.

#### Formatos aceitos de payload (estrutura HTTP)

O endpoint aceita três formatos equivalentes (normalizados internamente):
- registro único (objeto JSON)
- lista de registros (`[{...}, {...}]`)
- envelope com `records`: `{"records": [{...}, {...}]}`

Observação:
- a chave reservada `records` é aceita apenas no formato envelope (não dentro de um registro individual).

#### Regras de validação/aceitação no runtime

- Colunas faltantes em relação a `expected_raw_cols`:
  - `ALLOW_PARTIAL_PAYLOAD=0` (default): `400` com `missing_columns`
  - `ALLOW_PARTIAL_PAYLOAD=1`: permitido; colunas faltantes viram `pd.NA` e a imputação resolve
- Colunas extras:
  - extras comuns: toleradas (ignoradas ao reindexar para `expected_raw_cols`)
  - extras `leakage-like` (ex.: `target`, campos futuros): bloqueadas com `400`
- Limites operacionais:
  - batch size máximo validado na rota (`400` se exceder)
  - payload estrutural inválido cai em `422` (Pydantic/FastAPI)

Importante:
- `ALLOW_PARTIAL_PAYLOAD` **não altera o contrato base** (`expected_raw_cols`).
- Ele altera apenas a **política de aceitação de colunas faltantes** para cenários como alunos novos (sem histórico completo).

#### Status codes como violações/resultado do contrato de entrada

| Status | Significado (contrato/serving) | Exemplo de causa |
|---|---|---|
| `200` | Payload aceito + inferência concluída | Contrato estrutural válido e modelo disponível |
| `400` | Violação de regra de entrada da rota | `missing_columns`, leakage-like extras, batch inválido |
| `422` | Violação de schema HTTP/body (Pydantic/FastAPI) | body malformado / tipo inválido |
| `503` | Contrato/modelo indisponível no serving | `metadata.json` ausente ou `model.joblib` indisponível |
| `500` | Falha interna inesperada | erro interno de inferência/shape inválido |

Privacidade no contrato de entrada:
- logs e respostas não ecoam payload raw sensível
- `422` do `/predict` é sanitizado (sem `input` do Pydantic)
- não logar `RA`, listas de estudantes/IDs nem probabilidades individuais

Referências:
- seção `POST /predict`
- subseção `Alunos novos (sem histórico)`
- seção `Privacidade Operacional (Fase 7)`

### 4) Contrato de saída da API (resposta de inferência)

#### Fonte de verdade

- `app/schemas.py`
  - `PredictionResult` (linha por predição)
  - `PredictResponse` (resposta batch)

#### Estrutura de saída (alto nível)

- `PredictResponse`
  - `predictions`: lista de `PredictionResult`
  - `count`: contagem de predições (deve bater com `len(predictions)`)
  - `generated_at`: timestamp ISO8601
- `PredictionResult`
  - `risk_proba` (`[0,1]`)
  - `risk_class` (`0|1`, derivada de `risk_proba >= threshold_applied`)
  - `threshold_applied`
  - `model_version`, `model_family`, `variant`
  - `decision_policy`
  - `notes` (opcional)

Semântica operacional importante:
- `threshold_applied` é resolvido a partir do metadata (com fallback compatível para metadata legado).
- `risk_class` é validada como derivada de `risk_proba` e `threshold_applied`.
- `notes` é campo auxiliar para contexto operacional (ex.: fallback de metadata, resumo de missing agregado), sem PII.

#### Compatibilidade e evolução (saída)

Princípios de compatibilidade:
- evitar breaking changes no schema de resposta já consumido por clientes
- preferir extensões por:
  - novos campos opcionais, ou
  - `notes` (quando o conteúdo for apenas diagnóstico textual)
- mudanças de threshold/política de decisão impactam comportamento do modelo, mas não necessariamente o schema JSON

### 5) Política de mudança e compatibilidade entre contratos

#### O que tende a ser breaking vs non-breaking

- Data contract (offline):
  - mudança de nome/semântica de coluna relevante pode quebrar ingestão/validação e o pipeline de treino
- Payload contract (online):
  - mudar `expected_raw_cols` (metadata) pode quebrar clientes que montam payloads fixos
  - habilitar `ALLOW_PARTIAL_PAYLOAD=1` é mudança de política de aceitação (mais permissiva), não de contrato base
- Output contract (online):
  - adicionar campo opcional tende a ser non-breaking
  - remover/renomear campos obrigatórios é breaking

#### Como o projeto comunica/rastreia mudanças

- Data contracts e histórico:
  - `docs/contracts/contracts_changelog.json`
  - `docs/contracts/CHANGELOG.md`
- Metadata do modelo em serving:
  - `model_version`, `model_family`, `variant`, `threshold_policy`
- README + PRs:
  - documentação de mudanças operacionais e decisões de compatibilidade

### 6) Relação entre os contratos (camadas complementares)

- `Data contract` (offline):
  - garante governança e qualidade do dataset de origem
- `Payload contract` (online):
  - garante que a API receba um payload compatível com o modelo em serving
- `Output contract` (online):
  - garante estabilidade de consumo e rastreabilidade da inferência

Esses contratos coexistem e têm escopos diferentes:
- o contrato de dados não substitui o contrato da API
- o contrato da API não substitui a governança offline do dataset
- o contrato de saída não garante qualidade de entrada, apenas consistência da resposta

Referências rápidas:
- `Data Contract`
- `POST /predict`
- `Schema Formal de Saída do Modelo/API`
- `Ciclo de Vida em Produção (Operação do Modelo)`

## Estratégia de Retreino (Gatilhos + Execução)

Esta seção formaliza a estratégia de retreino do projeto, separando claramente:
- **sinais de investigação** (monitoramento online e drift),
- **decisão de retreinar** (ciclo offline),
- **decisão de promover** (gates + não-regressão).

Princípio central:
- **retreinar não implica promover**.
- O retreino gera candidatos; a promoção depende de avaliação objetiva (Recall/PR-AUC), seleção formal e validações de segurança operacional.

### 1) Objetivo da estratégia de retreino

A estratégia de retreino existe para responder a três cenários principais:
- mudança de distribuição dos dados de entrada (drift/qualidade de payload);
- queda de desempenho real quando o `ground truth` chega (`t+1`);
- atualização periódica do modelo quando há novo período de dados disponível.

Como o rótulo chega com atraso, o projeto usa uma estratégia em duas camadas:
- **online (sem rótulo)**: sinais/proxies para investigação
- **offline (com rótulo)**: decisão baseada em métricas oficiais (Recall/PR-AUC etc.)

### 2) Princípios operacionais (para evitar retreino por ruído)

- `drift` alto, sozinho, **não** é evidência suficiente para promover modelo novo.
- `422/400` altos indicam, em geral, problema de integração/contrato de entrada (não necessariamente problema do modelo).
- métricas **offline** (`src.offline_evaluation`) têm prioridade sobre proxies online para decidir retreino.
- mudanças devem ser tratadas com evidência e persistência:
  - evitar reagir a um único request/batch atípico
  - preferir confirmação por mais de uma observação/execução de monitoramento
- promoção exige gates e não-regressão:
  - `model_selection`
  - `src.regression_check`
  - critérios de `src.promotion_policy`

### 3) Gatilhos de retreino (tempo + drift + desempenho)

#### A) Gatilhos por tempo (cadência operacional)

Objetivo:
- garantir revisão periódica mesmo sem incidentes visíveis.

Recomendação documental (ajustável ao calendário de dados):
- **semanal**: revisar sinais online agregados (`online_metrics.jsonl`, logs estruturados)
- **quando novo período `t+1` chega** (ex.: anual/semestral conforme disponibilidade): rodar avaliação offline
- **após incorporação de novo ciclo de dados útil**: considerar rodada de retreino de candidatos

Observação:
- como o dataset institucional é anual no contexto atual (`2022/2023/2024`), o gatilho por tempo mais forte tende a ser a chegada de uma nova base/período.

#### B) Gatilhos por drift e qualidade de entrada (online)

Sinais típicos que justificam investigação e possível retreino:
- `drift_report` (`src.drift`) em `WARNING` ou `FAIL`
- `share_drifted_features` elevado (thresholds default documentados na seção de drift; ajustáveis por flags)
- `positive_rate` fora da faixa esperada
- aumento persistente de `missing_cols_rate` / `missing_values_rate`
- aumento persistente de `4xx/422` (após descartar falha de integração)

Importante:
- esses sinais justificam **investigação** e possivelmente preparação de retreino; não implicam promoção automática.

#### C) Gatilhos por desempenho real (offline, com ground truth)

Sinais prioritários para iniciar retreino:
- queda de `Recall` e/ou `PR-AUC` em `python -m src.offline_evaluation ...`
- degradação persistente em relação ao desempenho esperado do campeão
- resultados abaixo dos limiares mínimos usados nos gates de seleção/não-regressão

Esse é o gatilho de maior peso porque mede desempenho real após chegada do rótulo (`t+1`).

### 4) Matriz prática: sinal observado -> ação de retreino

| Sinal observado | Interpretação provável | Ação recomendada | Retreinar agora? |
|---|---|---|---|
| `422` alto / erros de body | cliente enviando payload inválido | corrigir integração / contrato HTTP | Não |
| `400` por `missing_columns` | quebra de contrato raw no cliente | corrigir payload/orquestração; revisar `ALLOW_PARTIAL_PAYLOAD` | Não (em geral) |
| `503` recorrente | problema de serving / artefato | incidente operacional; validar `app/model/*` e restart | Não |
| `missing_*_rate` alto + payload parcial frequente | degradação de qualidade de entrada / alunos novos | investigar integração/coleta; acompanhar impacto | Talvez (se persistir e afetar offline) |
| `drift_report` em `WARNING` | mudança moderada de distribuição | investigar features e sinais online; monitorar | Talvez |
| `drift_report` em `FAIL` + `positive_rate` anômala | mudança forte de população/processo | investigar e preparar rodada de retreino | Sim (avaliar) |
| queda de `Recall`/`PR-AUC` no offline | degradação real de performance | iniciar ciclo completo de retreino + seleção | Sim (prioritário) |

### 5) Retreino programado vs retreino emergencial

#### Retreino programado (cadenciado)

Quando usar:
- chegada de novo período de dados
- revisão periódica do modelo (governança)

Objetivo:
- testar se há ganho com dados mais recentes, mesmo sem incidente explícito

Fluxo:
- executar pipeline offline, comparar candidatos, manter campeão atual se não houver ganho/segurança suficiente

#### Retreino emergencial (orientado por sinal)

Quando usar:
- drift severo persistente
- queda de desempenho offline
- mudança operacional relevante na entrada (schema/qualidade/população)

Objetivo:
- responder a degradação real ou provável, preservando estabilidade de serving

Regra:
- ainda assim seguir seleção/gates/promoção (não pular governança por urgência)

### 6) Evidências mínimas antes de decidir retreinar/promover

Antes de decidir retreino/promoção, reunir evidências (preferencialmente em artefatos):
- `artifacts/drift_report_summary.json` (quando o gatilho for drift)
- `artifacts/offline_metrics_*.json` / `.md` (pós-fato)
- `artifacts/model_selection.json` (seleção formal do campeão)
- saída de `python -m src.regression_check`
- logs agregados (`logs/online_metrics.jsonl`) com comportamento observado

Isso melhora auditabilidade da decisão (banca/operação local).

### 7) Como executar o retreino (runbook resumido)

Fluxo recomendado de execução (alto nível):

1. Validar dados e contexto (se houver novo dataset/período)
```bash
python -m src.validate
python -m src.cohort_stats
```

2. Treinar candidatos (baseline + modelo não-linear)
```bash
python -m src.train_baseline
python -m src.train_hgb
```

3. Comparar e consolidar avaliação/seleção
```bash
python -m src.compare_models
python -m src.evaluate_holdout
python -m src.model_selection --models-root artifacts/models
```

4. Checar não-regressão do campeão selecionado (artefatos)
```bash
python -m src.regression_check \
  --selection-path artifacts/model_selection.json \
  --models-root artifacts/models
```

5. Promover campeão para serving local (com backup)
```bash
python -m src.promote_model \
  --selection-path artifacts/model_selection.json \
  --models-root artifacts/models \
  --out-dir app/model \
  --force 0 \
  --backup 1
```

6. Regenerar referência de drift após promoção
```bash
python -m src.build_reference_data
```

7. Reiniciar API e validar sanidade
- validar `GET /health`
- validar `GET /version`
- executar `POST /predict` de sanidade (single e/ou batch curto)

Observação:
- comandos e flags específicos já detalhados nas seções de pipeline/promoção; aqui o objetivo é registrar a sequência operacional de retreino.

### 8) Critério de promoção após retreino (retreinar != promover)

Após gerar candidatos, promover apenas se houver condição mínima de segurança/qualidade:
- seleção formal concluída (`artifacts/model_selection.json`)
- gates de qualidade compatíveis com a política (`Recall` primária + `PR-AUC` mínima)
- `src.regression_check` sem `FAIL`
- decisão humana explícita em caso de `WARNING` (com justificativa)

Se o campeão atual continuar melhor/mais seguro:
- manter versão atual em produção e registrar a decisão (sem promover o novo candidato)

### 9) Rollback após promoção (quando necessário)

Executar rollback se houver:
- regressão operacional após promoção (ex.: comportamento anômalo de `positive_rate`, erros recorrentes, incompatibilidade prática)
- indisponibilidade de artefato/serving após troca
- evidência rápida de piora após validação de sanidade

Procedimento resumido:
- restaurar release/backup anterior (`app/model/backups` ou `artifacts/models/releases/*`)
- reiniciar API
- validar `/health`, `/version` e `/predict`
- retomar monitoramento online e registrar incidente/decisão

Detalhes de rollback:
- `docs/pipeline_ml_deep_dives.md` (atualização do modelo na API e rollback)

### 10) Limitações da estratégia atual (escopo do projeto)

- O `ground truth` chega com atraso, então a decisão mais confiável depende de avaliação offline posterior.
- A avaliação em produção é simulada por replay local (`src.offline_evaluation`), sem join de IDs de produção (decisão de privacidade operacional).
- Não há automação de retreino com scheduler/gatilho automático nesta entrega.
  - A automação está registrada como backlog na `Fase 9`.

Referências rápidas:
- `Ciclo de Vida em Produção (Operação do Modelo)`
- `Mensuração em Produção (Ground Truth Delay) (Fase 7)`
- `Drift (Evidently) (Fase 7)`
- `Não-regressão do Modelo (Fase 7)`
- seções de promoção/versionamento do pipeline e `docs/pipeline_ml_deep_dives.md`

## Retreino Automatizado (Tempo + Drift)

A rotina automatizada local de retreino está disponível via:

```bash
python -m src.retrain_orchestrator
```

Política (versionável):
- arquivo: `docs/retrain_policy.json`
- gatilhos default:
  - tempo: `max_age_days=90` (base em `metadata.trained_at`)
  - drift: `FAIL => required`, `WARNING => recommended`
  - shift temporal: `FAIL => required`, `WARNING => recommended`

Saídas:
- `artifacts/retrain_decision.json` (sempre)
- `artifacts/retrain_run.json` (quando `--execute 1`)
- logs por etapa em `artifacts/retrain_logs/<run_id>/*.log`

Modo dry-run (apenas decisão):

```bash
python -m src.retrain_orchestrator \
  --policy docs/retrain_policy.json \
  --drift-summary artifacts/drift_report_summary.json \
  --shift-report artifacts/temporal_shift_report.json \
  --execute 0
```

Modo execução end-to-end (treino -> seleção -> staging -> produção -> referência):

```bash
python -m src.retrain_orchestrator \
  --dataset-path "dataset/DATATHON/BASE DE DADOS PEDE 2024 - DATATHON.xlsx" \
  --policy docs/retrain_policy.json \
  --execute 1 \
  --allow-recommended 1 \
  --allow-warning-promotion 1
```

Comportamento operacional:
- `RETRAIN_REQUIRED`: executa quando `--execute 1`
- `RETRAIN_RECOMMENDED`: executa somente com `--allow-recommended 1`
- `NOOP`: não executa retreino

Detalhes de robustez:
- modo padrão isola execução em `artifacts/models/runs/<run_id>` para evitar contaminação por artefatos antigos;
- em falha de qualquer etapa, o fluxo aborta e registra `failed_step` no manifesto;
- o manifesto salva apenas `dataset_basename` (não registra path absoluto).

Observação:
- após promoção/rollback, reinicie a API para recarregar modelo/metadata em memória.

Agendamento opcional (local):
- Linux/macOS (cron): executar `src.retrain_orchestrator --execute 0` diariamente e `--execute 1` conforme política da operação.
- Windows (Task Scheduler): configurar tarefa equivalente chamando `python -m src.retrain_orchestrator`.

## Explainability Local do Campeão

A explainability local é executada offline no holdout oficial (`2023->2024`) e gera:
- importâncias globais de features (top-k);
- análise de erro agregada por slices (sem registros individuais).

CLI:

```bash
python -m src.explainability \
  --model-dir app/model \
  --dataset-path "dataset/DATATHON/BASE DE DADOS PEDE 2024 - DATATHON.xlsx" \
  --year-t 2023 \
  --year-t1 2024 \
  --out-json artifacts/explainability_report.json \
  --out-md artifacts/explainability_report.md \
  --out-csv artifacts/feature_importance.csv \
  --top-k 20 \
  --max-rows 1000 \
  --seed 42
```

Saídas:
- `artifacts/explainability_report.json` (obrigatório)
- `artifacts/explainability_report.md` (resumo humano)
- `artifacts/feature_importance.csv` (opcional)

Métodos de importância:
- Modelos baseados em árvore com `feature_importances_`: usa importância nativa.
- Modelos lineares com `coef_`: usa `abs(coef)`.
- Fallback: `permutation_importance` (amostrado) quando o estimador não expõe importâncias nativas.

Privacidade e governança:
- sem `RA`, sem listas de IDs e sem registros individuais;
- apenas agregados por grupo/score e top-k de nomes de features;
- o relatório inclui nota explícita: importância global **não implica causalidade**.

## Limitações Conhecidas e Riscos Assumidos

Esta seção consolida limitações conhecidas do modelo/sistema e os riscos assumidos no escopo desta entrega. O objetivo é tornar explícitos os limites de interpretação e de operação, além das mitigações já implementadas.

Princípios de leitura:
- O modelo é **preditivo** (não causal) e deve apoiar a tomada de decisão humana.
- Limitações documentadas não invalidam a solução; elas delimitam o uso responsável e o plano de evolução.
- Parte dos riscos foi aceita por escopo acadêmico/local, com mitigação operacional onde possível.

### 1) Limitações conhecidas (modelo, dados e operação)

| Categoria | Limitação conhecida | Impacto potencial | Mitigação atual no projeto | Severidade | Status |
|---|---|---|---|---|---|
| Semântica de dados | Ambiguidade em `Idade x Fase_Ideal` (ex.: mesma idade aparecendo com `Fase Alfa` e `Fase 1`) | Ruído semântico em feature categórica e interpretação de consistência de dados | Limitação documentada; `Fase_Ideal` tratada como categoria da base (sem recálculo automático); sem regra institucional formalizada no pipeline | Média | Parcialmente mitigada (documentada) |
| Ground truth delay | O rótulo (`Defasagem_{t+1}`) chega com atraso | Não é possível medir Recall/PR-AUC no request online | Separação entre métricas online (proxies) e avaliação offline (`src.offline_evaluation`) | Alta | Mitigada por processo |
| Avaliação de produção (offline) | Avaliação pós-fato é feita por replay local, não por join com IDs de produção | Pode divergir do “histórico real” caso versão/threshold mudem ao longo do período | Replay documentado como escolha de privacidade operacional; uso de metadata/modelo promovido; artefatos de avaliação e versão | Média | Mitigada por transparência |
| Cobertura temporal de dados | Dataset local disponível é limitado a poucos períodos (`2022/2023/2024`) | Maior sensibilidade a mudanças de contexto e menor robustez estatística de generalização temporal | Holdout temporal explícito (`2023 -> 2024`), análise de shift e monitoramento de drift | Alta | Parcialmente mitigada |
| Coorte por interseção de `RA` | Pares temporais usam apenas estudantes presentes em `t` e `t+1` | Possível viés de cobertura (entradas tardias/saídas/evasão não rotulada) | Regra de coorte é explícita e auditada (`cohort_stats`); limitação reconhecida | Alta | Mitigada por governança |
| Payload parcial (alunos novos) | `ALLOW_PARTIAL_PAYLOAD=1` permite inferência com colunas faltantes imputadas | Pode reduzir qualidade da inferência em casos muito incompletos | Flag default `0`; logging agregado de missing; imputação sem valores “mágicos”; uso controlado | Média | Mitigada por guardrails |
| Drift monitorado localmente | Relatórios de drift e dashboard são locais (sem alertas automáticos contínuos) | Dependência de rotina manual de observação | `src.drift`, summary JSON, dashboard Streamlit e workflow smoke manual | Média | Parcialmente mitigada |
| Infra/escopo operacional | Apenas testes de carga básicos (sem HA/escala de produção) | Risco operacional parcial fora do escopo acadêmico/local | Testes básicos de carga da inferência (`tests/test_api_load_basic.py`), API local com health/version, Docker e logs estruturados | Baixa (no escopo atual) | Parcialmente mitigada |

### 2) Riscos assumidos (trade-offs do desenho)

| Risco assumido | Quando pode ocorrer | Impacto | Guardrails atuais | Ação recomendada |
|---|---|---|---|---|
| Maior taxa de falsos positivos (trade-off de Recall) | Threshold focado em `Recall` para reduzir falsos negativos | Mais alunos podem ser sinalizados para acompanhamento sem estarem em risco real | `PR-AUC`, `Precision`, `F1`, threshold explícito no metadata e monitoring de `positive_rate` | Ajustar threshold/política com base em evidências offline e contexto operacional |
| Mudança de população/processo antes do rótulo chegar | Alterações no perfil dos alunos, coleta ou processo escolar | Saída do modelo pode degradar antes da confirmação por métricas finais | Logs online agregados, drift report (Evidently), summary de missing, avaliação offline posterior | Investigar sinais online, gerar drift report e preparar retreino se persistente |
| Uso indevido como decisão automática | Consumo do score sem mediação humana | Risco de decisão injusta ou interpretação causal indevida | Documentação explícita de uso como apoio; metadados e notas de rastreabilidade | Reforçar política de uso e revisão humana em decisões sensíveis |
| Dependência de operação manual (retreino/promoção/rollback) | Falha de rotina, atraso operacional ou erro humano | Demora em reagir a drift/degradação ou promoção inadequada | Runbooks documentados, `model_selection`, `regression_check`, `promote_model`, backups/releases | Seguir checklist/runbook e registrar evidências antes de promover |
| Artefatos locais de serving | `app/model/*` inconsistente, ausente ou troca incompleta | `/predict` indisponível (`503`) ou comportamento inesperado | `/health`, `/version`, backups, releases versionadas, rollback local | Validar artefatos após promoção e usar rollback em caso de regressão |

### 3) Limites de uso do modelo (uso responsável)

- O score de risco **não** deve ser usado como decisão automática, punitiva ou causal.
- O modelo deve apoiar priorização de acompanhamento e triagem, com validação por equipe pedagógica/psicopedagógica.
- A interpretação do score deve considerar:
  - contexto institucional,
  - qualidade do payload de entrada,
  - versão do modelo (`model_version`) e threshold aplicado.

### 4) O que já está mitigado vs o que permanece estrutural

Mitigações implementadas no código/processo:
- anti-leakage explícito (`src.leakage`, pruning, checks temporais)
- contratos de dados e validação (`src.contracts`, `src.contract_validate`, `src.validate`)
- privacidade operacional (`src/privacy.py`, redaction, `422` sanitizado, logs agregados)
- monitoramento online + offline (`online_metrics.jsonl`, `src.offline_evaluation`)
- drift local em `MODEL frame` (Evidently + Streamlit)
- promoção/rollback com artefatos versionados e backups

Limitações estruturais (não resolvidas apenas com engenharia local):
- atraso na chegada do rótulo (`t+1`)
- baixa quantidade de períodos históricos disponíveis
- ambiguidades semânticas do dataset de origem (ex.: `Idade x Fase_Ideal`) sem regra institucional formalizada

### 5) Relação com roadmap / redução de risco futura

Itens já identificados para reduzir risco em evoluções futuras (Fase 9 / backlog opcional, já implementados nesta entrega):
- automação de retreino com gatilho por tempo + drift
- dashboard operacional consolidado (inferência + drift + métricas pós-fato)
- explainability local do campeão (importâncias globais/análise de erro agregada)
- testes de carga básicos para API de inferência

Referências rápidas:
- `Contratos em Produção (Dados + API + Saída)`
- `Ciclo de Vida em Produção (Operação do Modelo)`
- `Estratégia de Retreino (Gatilhos + Execução)`
- `Mensuração em Produção (Ground Truth Delay) (Fase 7)`
- `Privacidade Operacional (Fase 7)`
- `Drift (Evidently) (Fase 7)`

## Exemplos de Chamadas à API

Esta seção reúne exemplos práticos de uso da API (`/health`, `/version`, `/predict`) para teste local e troubleshooting rápido. Os exemplos são intencionalmente sintéticos e devem ser adaptados ao contrato de entrada do modelo atualmente promovido.

### 1) Pré-requisitos rápidos

- API rodando localmente (ex.: `uvicorn app.main:app --reload --port 8000`)
- base URL local: `http://127.0.0.1:8000`
- para `POST /predict` retornar `200`, é necessário haver modelo/metadata válidos em `app/model/`

Observação:
- mesmo sem modelo promovido, `GET /health` e `GET /version` funcionam; `/predict` tende a retornar `503`.

### 2) Checks básicos de sanidade (`/health` e `/version`)

Health check:

```bash
curl -s http://127.0.0.1:8000/health | jq
```

Exemplo de resposta:

```json
{
  "status": "ok"
}
```

Version check (útil para diagnosticar serving e contrato atual):

```bash
curl -s http://127.0.0.1:8000/version | jq
```

O que conferir em `/version`:
- `model_version`, `model_family`, `variant`
- `threshold_operational`
- `metadata_loaded`
- `model_loaded`
- `model_joblib_exists`

### 3) Como descobrir o contrato esperado antes de chamar `/predict`

O contrato de payload da API é **dinâmico por modelo promovido** e depende de `app/model/metadata.json` (`expected_raw_cols`).

Exemplos para inspecionar localmente:

```bash
jq '.expected_raw_cols' app/model/metadata.json
```

Ou com resumo:

```bash
jq '{model_version, expected_raw_cols_count: (.expected_raw_cols | length), expected_raw_cols}' app/model/metadata.json
```

Se `app/model/metadata.json` não existir:
- promova um modelo primeiro (ver `src.promote_model`), ou
- espere `/predict` retornar `503` enquanto `/health` e `/version` seguem disponíveis.

### 4) Exemplos de sucesso no `/predict` (ajustar ao `expected_raw_cols`)

Importante:
- Os payloads abaixo mostram **formato** (single/batch/envelope).
- Troque `coluna_1`, `coluna_2`, etc. pelas colunas reais do seu `expected_raw_cols`.
- Use dados sintéticos/sem PII.

#### A) Registro único (single object)

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "coluna_1": 1,
    "coluna_2": "categoria_a",
    "coluna_3": 10.5
  }' | jq
```

#### B) Batch (lista de registros)

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '[
    {"coluna_1": 1, "coluna_2": "categoria_a", "coluna_3": 10.5},
    {"coluna_1": 2, "coluna_2": "categoria_b", "coluna_3": 11.0}
  ]' | jq
```

#### C) Envelope com `records`

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {"coluna_1": 1, "coluna_2": "categoria_a", "coluna_3": 10.5},
      {"coluna_1": 2, "coluna_2": "categoria_b", "coluna_3": 11.0}
    ]
  }' | jq
```

Exemplo resumido de resposta esperada (`200`):

```json
{
  "predictions": [
    {
      "risk_proba": 0.21,
      "risk_class": 0,
      "threshold_applied": 0.3,
      "model_version": "2026-02-20T12-00-00Z",
      "model_family": "nonlinear_hgb",
      "variant": "default",
      "decision_policy": "fixed_threshold",
      "notes": null
    }
  ],
  "count": 1,
  "generated_at": "2026-02-20T13:20:00+00:00"
}
```

### 5) Exemplo para aluno novo / payload parcial (`ALLOW_PARTIAL_PAYLOAD`)

Padrão (`ALLOW_PARTIAL_PAYLOAD=0`):
- faltando colunas esperadas -> `400` com `missing_columns`

Modo opcional (`ALLOW_PARTIAL_PAYLOAD=1`):
- payload parcial é aceito
- colunas faltantes viram `pd.NA`
- a imputação da pipeline resolve os faltantes

Exemplo (subir API com payload parcial habilitado):

```bash
ALLOW_PARTIAL_PAYLOAD=1 uvicorn app.main:app --reload --port 8000
```

Exemplo de request parcial (formato ilustrativo; adapte às colunas reais):

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"coluna_1": 1, "coluna_2": "categoria_a"}' | jq
```

Observação:
- a API pode incluir resumo agregado de missing em `predictions[*].notes` (ex.: `missing_cols_rate`, `missing_values_rate`), sem alterar o schema principal.

### 6) Exemplos de erro comuns (úteis para integração)

#### A) `400` por colunas faltantes (`missing_columns`)

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"coluna_1": 1}' | jq
```

Resposta esperada (resumo):
- `status=400`
- `detail.missing_columns` com nomes das colunas ausentes

#### B) `400` por extra leakage-like (ex.: `target`)

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"target": 1}' | jq
```

Resposta esperada (resumo):
- `status=400`
- mensagem genérica de bloqueio de colunas leakage-like (sem ecoar payload sensível)

#### C) `422` por body inválido (schema HTTP/Pydantic)

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records":"x"}' | jq
```

Resposta esperada (resumo):
- `status=422`
- payload sanitizado (sem `input` bruto do Pydantic no caso de `/predict`)

#### D) `503` por modelo/metadata indisponíveis

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"coluna_1": 1, "coluna_2": "categoria_a"}' | jq
```

Resposta esperada (resumo):
- `status=503`
- `detail` indicando indisponibilidade de `model` e/ou `metadata`
- usar `GET /version` para confirmar `metadata_loaded` e `model_loaded`

### 7) Troubleshooting rápido (observabilidade)

Ao testar a API, verifique:
- header `X-Request-ID` (útil para correlacionar request com logs)
- logs estruturados JSON (`stdout`) com eventos da API
- `logs/online_metrics.jsonl` para métricas agregadas por request (`2xx/4xx/5xx/422`)
- `GET /version` para diagnosticar:
  - `model_loaded`
  - `metadata_loaded`
  - `model_joblib_exists`
  - `threshold_operational`

Exemplo para inspecionar headers:

```bash
curl -i http://127.0.0.1:8000/health
```

### 8) Nota de privacidade para testes locais

- Use dados sintéticos nos exemplos e testes manuais.
- Não enviar/logar identificadores sensíveis como `RA`, `Nome_Anon` e `Avaliador1..Avaliador6`.
- O projeto aplica guardrails de privacidade operacional (redaction em logs e monitoramento agregado), mas o uso responsável na entrada continua sendo obrigatório.

Referências rápidas:
- `POST /predict`
- `Contratos em Produção (Dados + API + Saída)`
- `Schema Formal de Saída (Fase 6)`
- `Privacidade Operacional (Fase 7)`

## API Acessível Localmente (Run + Smoke)

Esta seção consolida uma forma reprodutível de subir e validar a API localmente com scripts de apoio (sem depender apenas de comandos manuais soltos).

Objetivo da tarefa:
- demonstrar que a API está acessível localmente (`/health` e `/version`)
- fornecer smoke test rápido para `/predict` (comportamento com e sem modelo promovido)
- oferecer atalho opcional via Docker para demonstração/reprodução

### 1) Scripts adicionados (atalhos locais)

- `scripts/run_api_local.sh`
  - sobe `uvicorn` local usando `.venv` por padrão
  - aceita `HOST`, `PORT`, `RELOAD`, `LOG_LEVEL`, `ALLOW_PARTIAL_PAYLOAD`
- `scripts/smoke_api_local.sh`
  - valida `GET /health` e `GET /version`
  - valida rota `/predict` com `422` (body inválido) e probe funcional (`503` sem modelo ou `200/400` com modelo)
  - imprime `X-Request-ID` quando disponível
- `scripts/run_api_docker_local.sh` (opcional)
  - sobe a API em container com `docker run`
  - monta `app/model` por volume por padrão

### 2) Subir a API localmente (via `.venv`)

Execução padrão:

```bash
scripts/run_api_local.sh
```

Exemplos com override:

```bash
PORT=8010 LOG_LEVEL=INFO scripts/run_api_local.sh
```

```bash
ALLOW_PARTIAL_PAYLOAD=1 PORT=8010 scripts/run_api_local.sh
```

Notas:
- por padrão o script usa `./.venv/bin/python`
- se necessário, você pode apontar outro Python:

```bash
PYTHON_BIN=python3 scripts/run_api_local.sh
```

### 3) Smoke test local (runbook de validação)

Com a API já rodando, execute:

```bash
scripts/smoke_api_local.sh
```

Exemplo com porta customizada:

```bash
BASE_URL=http://127.0.0.1:8010 scripts/smoke_api_local.sh
```

Comportamento esperado do smoke:
- `GET /health` -> `200`
- `GET /version` -> `200`
- `POST /predict` com body inválido -> `422` (prova de acessibilidade da rota/validação HTTP)
- `POST /predict` com probe funcional:
  - sem modelo/metadata promovidos -> `503` (esperado)
  - com modelo/metadata disponíveis -> `200` **ou** `400` (dependendo do payload sintético vs contrato/política atual)

Modo estrito (exigir `200` no probe funcional do `/predict`):

```bash
REQUIRE_PREDICT_200=1 scripts/smoke_api_local.sh
```

Quando usar `REQUIRE_PREDICT_200=1`:
- após confirmar que a API está com modelo/metadata carregados
- e após adaptar/validar payload compatível com `expected_raw_cols`

### 3.1) Testes de carga básicos da inferência (pytest)

O projeto inclui testes de carga básicos (sem dependência de rede externa) em:
- `tests/test_api_load_basic.py`

Cenários cobertos:
- carga sequencial de requests pequenos (`batch_size=1`)
- carga no limite de batch da API (`batch_size=500`)

Como executar apenas esses testes:

```bash
source .venv/bin/activate
pytest -q tests/test_api_load_basic.py
```

Métricas verificadas nos testes:
- latência de p95 (limite generoso para CI)
- throughput mínimo (`requests/s` ou `records/s`)
- estabilidade funcional (`status_code=200` e contagem correta no retorno)

### 4) Como interpretar o resultado (sem modelo vs com modelo)

#### Cenário A: API acessível sem modelo promovido (válido para esta tarefa)

Sinais esperados:
- `/health` = `200`
- `/version` = `200`
- `/predict` = `503`

Isso indica:
- aplicação está acessível localmente
- serving de inferência ainda não está pronto (faltam artefatos em `app/model/*`)

#### Cenário B: API acessível com modelo promovido

Sinais esperados:
- `/health` = `200`
- `/version` = `200` com `model_loaded=true` e `metadata_loaded=true`
- `/predict` = `200` (payload compatível) ou `400` (payload não compatível / política strict)

Para aumentar a chance de `200`:
- inspecione `expected_raw_cols` em `app/model/metadata.json`
- use payload alinhado ao contrato (ver seção `Exemplos de Chamadas à API`)

### 5) Atalho opcional via Docker (demonstração/reprodutibilidade)

Subir via Docker (imagem já construída):

```bash
scripts/run_api_docker_local.sh
```

Com build automático + porta customizada:

```bash
AUTO_BUILD=1 HOST_PORT=8010 scripts/run_api_docker_local.sh
```

Observações:
- por padrão o script monta `app/model` em `/app/app/model` (para inferência local, quando houver artefatos)
- para desabilitar a montagem do modelo:

```bash
MODEL_MOUNT=0 scripts/run_api_docker_local.sh
```

Depois de subir via Docker, rode o smoke apontando a porta publicada:

```bash
BASE_URL=http://127.0.0.1:8010 scripts/smoke_api_local.sh
```

### 6) Troubleshooting rápido

Se o smoke falhar:
- verifique se a API está rodando e ouvindo na porta esperada
- confira `/version`:
  - `model_loaded`
  - `metadata_loaded`
  - `model_joblib_exists`
- confira logs estruturados no stdout (incluindo `request_id`)
- valide se `./.venv` contém `uvicorn` e dependências (`requirements-dev.txt`)

### 7) Nota de privacidade

- Os scripts de smoke usam payloads sintéticos e não exigem `RA`.
- Mantenha testes locais sem PII (`RA`, `Nome_Anon`, `Avaliador*`).
- Logs e monitoramento continuam agregados/privacy-safe por design.

Referências rápidas:
- `Rodar API Local (Fase 6)`
- `POST /predict (Fase 6)`
- `Exemplos de Chamadas à API`
- `Contratos em Produção (Dados + API + Saída)`

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

### Ambiente isolado para Drift + Dashboard (Evidently + Streamlit)

Para evitar conflitos locais de dependências (especialmente `protobuf`) entre versões antigas de `streamlit` instaladas globalmente e o `evidently` usado no relatório de drift, use um ambiente separado para a frente de dashboard/visualização:

Recomendação de ambiente:

- Python `3.11` (preferencial para a stack de dashboard: `Streamlit + Evidently`)
- venv dedicada (`.venv-dashboard`) para não misturar dependências com API/treino

```bash
scripts/bootstrap_dashboard_env.sh
source .venv-dashboard/bin/activate
# dashboards disponíveis:
streamlit run dashboards/streamlit_app.py
streamlit run dashboards/ops_dashboard.py
```

Esse script instala `requirements-dashboard.txt` em uma venv separada (`.venv-dashboard` por padrão), preservando o ambiente principal (`.venv`) da API/treino/CI.
Ele valida a versão do Python e falha com mensagem clara quando o interpretador local não é compatível com o pin atual do `Streamlit` (ex.: `Python 3.9.7`).

Se o seu `python3` local não for compatível, informe explicitamente o interpretador:

```bash
PYTHON_BIN=python3.11 scripts/bootstrap_dashboard_env.sh
# ou
scripts/bootstrap_dashboard_env.sh .venv-dashboard python3.11
```

Exemplo com `pyenv`:

```bash
PYENV_VERSION=3.11.9 scripts/bootstrap_dashboard_env.sh .venv-dashboard python3.11
```

Observação de reprodutibilidade (opcional):

- após validar a stack localmente, você pode gerar um lock do ambiente de dashboard (ex.: `requirements-dashboard.lock`) com `pip freeze` dentro da `.venv-dashboard`
- como isso pode variar por SO/arquitetura, trate esse lock como artefato de ambiente (ou documente o contexto em que foi gerado)

Se quiser outro diretório de venv:

```bash
scripts/bootstrap_dashboard_env.sh .venv-ui
source .venv-ui/bin/activate
```

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

## Deep Dives Técnicos (Fases 4 a 7)

As seções abaixo concentram detalhamento técnico de implementação. Para leitura executiva/técnica rápida, priorize as seções `Stack Tecnológica`, `Etapas do Pipeline de Machine Learning` e os resumos operacionais.

<details>
<summary>Detalhamento técnico das Fases 4 a 6 (feature engineering, treino, artefatos, API e deploy) — expandir</summary>

### Separação de Features (Fase 4)

- A seleção de features exclui PII por padrão a partir de `PII_COLUMNS` do contrato (`Nome_Anon` e `Avaliador*`), sem depender de listas duplicadas.
- As features são separadas por dtypes em três grupos: `numeric`, `categorical` e `datetime` (com `Data_Nasc` tratada explicitamente como datetime nesta etapa).
- O split pode ser auditado no pareamento temporal:
  - `make_temporal_pairs(..., persist_feature_split=True)` gera `artifacts/feature_split_report.json`.
  - por padrão (`persist_feature_split=False`), não há side effect de escrita em disco.

### Imputação de Missing (Fase 4)

- A política de imputação é definida como plano auditável em `src/imputation.py`, para uso dentro de `Pipeline/ColumnTransformer` na etapa de treino.
- Estratégias padrão:
  - numéricas: `median` + `add_indicator=True`
  - categóricas: `most_frequent` + `add_indicator=True`
  - datetime (`Data_Nasc`): não imputado nesta etapa; permanece em `datetime_cols_excluded` para tratamento posterior de feature engineering.
- Colunas 100% missing no recorte real de treino (`2022->2023`) são removidas do conjunto de imputação quando `drop_all_missing_columns=True`:
  - `Ativo/ Inativo`, `Ativo/ Inativo__dup1`, `Destaque IPV__dup1`, `Escola`, `INDE 2023`, `INDE 2024`, `INDE 23`, `IPP`, `Pedra 2023`, `Pedra 2024`, `Pedra 23`, `Rec Psicologia`
- Evidência local:
  - `artifacts/imputation_plan.json` (gerado por `persist_imputation_plan(...)`).

### Codificação Categórica (Fase 4)

- A codificação categórica é feita com `OneHotEncoder(handle_unknown="ignore")` para tolerar categorias novas em produção sem quebrar inferência.
- O bloco categórico é aplicado após imputação (`SimpleImputer(strategy="most_frequent", add_indicator=True)`), dentro de `ColumnTransformer` em `src/preprocessing.py`.
- O bloco numérico usa `SimpleImputer(strategy="median", add_indicator=True)`.
- `Fase` e `Fase_Ideal` permanecem categóricas nesta etapa.
- `Data_Nasc` (datetime) não entra na codificação nesta fase; fica para feature engineering posterior.

### Escalonamento Numérico (Fase 4)

- O pré-processador agora suporta escalonamento numérico configurável em `src/preprocessing.py`.
- Regra adotada:
  - baseline linear (`Logistic Regression`): usar `numeric_scaler="standard"` (preset `DEFAULT_SCALER_FOR_LINEAR`);
  - modelos de árvore (ex.: `HistGradientBoosting`): usar `numeric_scaler="none"` (preset `DEFAULT_SCALER_FOR_TREE`).
- O escalonador pode ser configurado entre `standard`, `robust` e `none`, com validação explícita de parâmetro.

### Reuso do Pré-processamento na Inferência (Fase 4)

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

### Feature Engineering (Fase 4)

- Features derivadas simples e anti-leakage (somente dados de `t`) são criadas em `src/features.py` por `add_engineered_features(...)`.
- Numéricas:
  - `avg_grades`, `min_grade`, `max_grade`, `grade_std`, `missing_grades_count`
  - `missing_indicators_count`
  - `defasagem_abs`, `defasagem_neg_flag`, `age_is_missing_flag`
- Categórica opcional:
  - `age_bucket` (`07_10`, `11_14`, `15_18`, `19_plus`)
- A engenharia é opt-in no bundle (`enable_feature_engineering=True/False`) e pode incluir/excluir `age_bucket` (`enable_age_bucket`).

### Gate Anti-Leakage (Fase 4)

- A validação explícita de leakage fica em `src/leakage.py` com `assert_no_leakage(...)`.
- O gate roda em três pontos:
  - construção de pares temporais (`make_temporal_pairs`) no contexto `TRAIN`, com assert temporal `t -> t+1`;
  - validação de inferência (`validate_inference_frame`) no contexto `RAW`, bloqueando payloads com colunas suspeitas (ex.: `_y`, `_t1`, `target`, `t+1`);
  - transformação para model frame (`transform_raw_to_model_frame`) no contexto `MODEL`, antes do `ColumnTransformer`.
- Colunas suspeitas e 100% ausentes (artefato estrutural de alinhamento) são toleradas apenas quando não têm sinal (`n_non_null = 0`) e podem ser removidas no fluxo de treino, mantendo logs apenas agregados por nome de coluna.

### Feature Pruning (Fase 4)

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

### Decisões de Feature Engineering (Fase 4)

Resumo (README):
- princípios de desenho: consistência treino=inferência, anti-leakage e privacidade;
- exclusão de PII (`RA`, `Nome_Anon`, `Avaliador*`) do model frame;
- política de imputação/codificação/escalonamento;
- regras de pruning e reaplicação na inferência.

Detalhamento técnico completo (incluindo snapshots de split, regras e referências):
- `docs/pipeline_ml_deep_dives.md` (seção `Decisões de Feature Engineering (Fase 4)`)

### ColumnTransformer para Pré-processamento (Fase 5)

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

### Pipeline End-to-End (Fase 5)

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

### Treino Baseline (Fase 5)

- CLI oficial para baseline:
  - `python -m src.train_baseline --year-t 2022 --year-t1 2023 --out-dir artifacts/models/baseline_logreg --scaler standard --variants none --enable-feature-engineering 1 --enable-age-bucket 1`
- Artefatos por variante:
  - `artifacts/models/baseline_logreg/<variant>/model.joblib`
  - `artifacts/models/baseline_logreg/<variant>/metadata.json`
- O feature pruning e fitado no treino e aplicado na inferencia sem recalculo:
  - evita drift de schema entre treino e producao;
  - mantem compatibilidade com o `ColumnTransformer` e com o contrato de colunas do modelo.

### Treino Não-Linear (Fase 5)

- CLI oficial para modelo não-linear:
  - `python -m src.train_hgb --file-path <xlsx> --year-t 2022 --year-t1 2023 --out-dir artifacts/models/nonlinear_hgb --variants default,tuned --enable-feature-engineering 1 --enable-age-bucket 0`
- O treino utiliza `X_raw_train` (contrato raw da API) e `y_train` pareado via coorte `RA`.
- Validação interna opcional (CV estratificada) no treino oficial:
  - `--cv 1 --cv-splits 5 --cv-repeat 1`
  - a seção `cv` é anexada no `metadata.json` de cada variante (métricas agregadas por fold e mean/std).
- Artefatos por variante:
  - `artifacts/models/nonlinear_hgb/<variant>/model.joblib`
  - `artifacts/models/nonlinear_hgb/<variant>/metadata.json`

### Estratégia de Decisão e Desbalanceamento (Fase 5)

- Prevalência observada no pipeline oficial:
  - treino `2022->2023`: `n=600`, `n_pos=366`, prevalência `0.6100`
  - holdout `2023->2024`: `n=765`, `n_pos=308`, prevalência `0.4026`
- Decisão operacional padrão:
  - `class_weight=none` como default de treino
  - política principal: `threshold` fixo, com alerta se `risk_proba >= 0.30`
  - política contingencial para restrição de capacidade: `top_k` com `k_frac=0.20`
  - recall alvo no treino para seleção de threshold: `>= 0.90` (evitar tuning por predição in-sample)
  - `threshold_calibration` (train-only) é evidência técnica e não substitui o `threshold` operacional fixo
- Justificativa:
  - no cenário atual, `class_weight="balanced"` piorou Recall no holdout em relação a `class_weight=none`.
  - para foco em Recall, threshold fixo performou melhor que `top-k` no holdout atual.
  - evidência no `nonlinear_hgb/tuned` (`2023->2024`): `threshold=0.30` -> Recall `0.8117`, Precision `0.4545`, `positive_rate=0.7190`.
  - evidência no `nonlinear_hgb/tuned` (`2023->2024`): `top_k=20%` -> Recall `0.4091`, Precision `0.8235`, `positive_rate=0.2000`.
  - como a prevalência cai de `0.61` (treino) para `0.40` (holdout), recalibração periódica do threshold é obrigatória.
- Implementação:
  - utilitários agregados em `src/metrics.py` (`threshold` e `top-k`), sem persistir scores/IDs.
  - `src/thresholding.py` centraliza avaliação por threshold e seleção calibrada por Recall no treino.
  - `metadata.json` dos modelos inclui:
    - `threshold_policy` (operação fixa `0.30` + fallback `top-k=20%`)
    - `threshold_calibration` (calibração no treino com `Recall>=0.90`, sem substituir política operacional)
    - `evaluation_train_at_0.5` / `evaluation_train_at_0.30` e equivalentes no holdout (agregado, sem IDs)
    - `class_imbalance_strategy` (prevalência, decisão, alternativas e evidências agregadas)
    - `prediction_policy` (config padrão consumível pela camada de serviço da API)

### Comparação de Modelos (Fase 5)

- Comando oficial:
  - `python -m src.compare_models --models-root artifacts/models --out-json artifacts/model_comparison.json --out-md artifacts/model_comparison.md`
- A comparação lê apenas `metadata.json` dos artefatos de treino (sem re-treinar modelos e sem recalcular métricas no comparador).
- Política de ranking:
  - primária: Recall holdout@0.5
  - secundária: PR-AUC holdout
  - terciária: menor positive_rate holdout@0.5
- O relatório é agregado e privacy-safe: sem listas de `RA`/IDs e sem valores de célula.

### Seleção do Modelo Campeão (Fase 5)

- A seleção formal do campeão usa holdout `2023->2024` no threshold operacional `0.30`:
  - `python -m src.model_selection --models-root artifacts/models --output-json artifacts/model_selection.json --output-md artifacts/model_selection.md`
- Gates mínimos de qualificação:
  - `Recall_holdout@0.30 >= 0.45`
  - `PR-AUC_holdout >= 0.60`
- Ranking determinístico entre qualificados:
  - maior `Recall`, depois maior `PR-AUC`, depois menor `positive_rate`, com desempate lexicográfico (`model_family/variant`).
- Se ninguém passar os gates, o processo escolhe o maior `Recall` com status `WARNING` e justificativa explícita no artefato.

### Justificativa da Escolha do Modelo Final (Fase 5)

- O modelo final é justificado a partir de `artifacts/model_selection.json` (fonte única da decisão formal).
- A justificativa para a banca fica versionada em `docs/model_final_justification.md` e pode ser regenerada por `python -m src.model_justification`.
- Critério oficial: `Recall` no holdout como métrica primária, `PR-AUC` como secundária e `positive_rate` como desempate, com threshold preferencial `0.30` (fallback `0.5` com `WARNING`).

### Avaliação Holdout Temporal (Fase 5)

- A avaliação `2023->2024` é estritamente read-only: o modelo é treinado em `2022->2023` e apenas inferido no holdout.
- Nos CLIs de treino (`src.train_baseline` e `src.train_hgb`), o bloco `evaluation_holdout` é incluído no `metadata.json` quando `--eval-holdout 1`.
- CLI dedicado para reavaliar artefatos serializados:
  - `python -m src.evaluate_holdout --models-root artifacts/models --dataset-path <xlsx> --output artifacts/holdout_evaluation.json`
  - o comando carrega `model.joblib` e avalia no holdout oficial sem refit.

### Métricas Oficiais (Fase 5)

- O cálculo oficial de métricas está centralizado em `src/metrics.py`, evitando lógica duplicada entre CLIs.
- Nesta fase, o threshold operacional padrão para foco em Recall é `0.30`.
- Cada `metadata.json` salva:
  - `evaluation_train` (pair `2022->2023`)
  - `evaluation_holdout` (pair `2023->2024`, quando `--eval-holdout 1`)
  - bloco de métricas com `Recall`, `Precision`, `F1`, `ROC-AUC`, `PR-AUC`, `positive_rate` e `confusion_matrix_at_0.5` (`tn/fp/fn/tp`).

### Shift Temporal (Fase 5)

- A validacao oficial de shift roda no **MODEL frame** (pos feature engineering + feature pruning), que representa exatamente o que o modelo consome.
- O relatorio inclui:
  - shift do target (prevalencia train vs holdout e deltas absoluto/relativo);
  - shift por feature com scores de drift (`PSI` para numericas e `TVD` para categoricas/binary), mudanca de missing e severidade por feature.
- Thresholds padrao:
  - target `|delta_abs|`: `WARNING>=0.15`, `FAIL>=0.25`
  - numericas (`PSI`): `WARNING>=0.10`, `FAIL>=0.25`
  - categoricas/binary (`TVD`): `WARNING>=0.10`, `FAIL>=0.25`
  - missing delta: `WARNING>=0.10`, `FAIL>=0.20`
- Regra de governanca:
  - `WARNING` nao bloqueia automaticamente;
  - `FAIL` e atribuido por regra objetiva de agregacao (target FAIL ou contagem de features FAIL acima do gate).
- Comando oficial (config do campeao atual):
  - `python -m src.temporal_shift --config winner`
- Artefatos:
  - `artifacts/temporal_shift_report.json` (obrigatorio)
  - `artifacts/temporal_shift_report.md` (opcional)

### Promoção do Modelo Campeão (Fase 6)

- O treino já persiste pipelines por variante em `artifacts/models/<family>/<variant>/model.joblib`.
- A promoção para serving copia deterministicamente o campeão selecionado para caminho fixo da API:
  - `app/model/model.joblib`
  - `app/model/metadata.json`
- Comando oficial:
  - `python -m src.promote_model --selection-path artifacts/model_selection.json --models-root artifacts/models --out-dir app/model --force 0 --backup 1`
- Rastreabilidade e rollback local:
  - `app/model/promoted_model.json` registra vencedor, source/dest, hashes `sha256` e timestamp;
  - `app/model/backups/<timestamp>/` guarda snapshot do modelo anterior quando `--backup 1`.

### Promoção (Staging -> Prod Local) (Fase 6)

- A promoção local agora aplica **policy objetiva** antes de copiar artefatos:
  - métrica principal: `Recall` no holdout
  - métrica secundária: `PR-AUC` no holdout
  - threshold oficial: `0.30` (fallback `0.5` com warning explícito)
- Regras operacionais:
  - `selection.status = PASS` => promoção permitida (`ALLOW`)
  - `selection.status = WARNING` => promoção permitida **somente** com override (`--allow-warning 1`)
  - `selection.status = FAIL` => promoção bloqueada (`BLOCK`)
- Staging e prod locais:
  - `app/model/staging/` (staging)
  - `app/model/` (prod local)
- Manifestos:
  - staging: `app/model/staging/staging_manifest.json`
  - prod: `app/model/promoted_model.json`
  - ambos incluem decisão da policy, threshold usado e métricas (`recall/pr_auc/positive_rate`) + hashes

Fluxo sugerido:

```bash
# 1) Treinar candidatos (baseline/hgb)
python -m src.train_baseline ...
python -m src.train_hgb ...

# 2) Selecionar campeão formalmente
python -m src.model_selection --models-root artifacts/models

# 3) Stage (não mexe no prod local)
python -m src.promote_model \
  --selection-path artifacts/model_selection.json \
  --models-root artifacts/models \
  --out-dir app/model/staging \
  --stage-only 1

# 4) Promote de staging para prod local
python -m src.promote_model \
  --selection-path artifacts/model_selection.json \
  --from-staging app/model/staging \
  --out-dir app/model \
  --promote 1
```

Override (quando `model_selection.status = WARNING`):

```bash
python -m src.promote_model \
  --selection-path artifacts/model_selection.json \
  --models-root artifacts/models \
  --out-dir app/model/staging \
  --stage-only 1 \
  --allow-warning 1
```

Rollback local:
- usar snapshots em `app/model/backups/<timestamp>/`
- promover novamente a versão desejada (ou restaurar `model.joblib`/`metadata.json` manualmente a partir do backup)

### Versionamento Local de Modelos (Releases) (Fase 6)

- O projeto mantém dois níveis de artefatos:
  - `artifacts/models/<family>/<variant>/` = **build artifacts** (saída de treino por variante)
  - `artifacts/models/releases/<model_version>/` = **release imutável** (cópia versionada para rastreabilidade/rollback)
- Cada release contém:
  - `model.joblib`
  - `metadata.json` (cópia enriquecida com `model_version`/`trained_at` quando necessário)
  - `release.json` (manifesto leve com identidade, hashes e paths)
- Comando oficial:
  - `python -m src.create_release --selection-path artifacts/model_selection.json --out-root artifacts/models/releases`
- Observação operacional:
  - a promoção da API continua usando o campeão do `model_selection`; o release versionado facilita rollback e auditoria local.

### Atualização do Modelo na API (Troca de Versão e Rollback) (Fase 6)

Resumo (README):
- fluxo operacional recomendado: `treino -> seleção -> staging -> promote`;
- validações de staging (`metadata_schema`, `/health`, `/version`);
- reinício obrigatório da API após troca por causa do cache (`lru_cache`);
- rollback por backup local (`app/model/backups/*`) ou release imutável (`artifacts/models/releases/*`).

Runbook detalhado (com comandos completos e observações operacionais):
- `docs/pipeline_ml_deep_dives.md` (seção `Atualização do Modelo na API (Troca de Versão e Rollback) (Fase 6)`)

### Metadata do Modelo (Serving) (Fase 6)

- O `metadata.json` de serving (`app/model/metadata.json`) segue schema mínimo validável para operação da API e monitoramento:
  - identidade/versionamento do modelo (`model_family`, `variant`, `model_version`, `trained_at`, `promoted_at`);
  - contrato de entrada e model frame (`expected_raw_cols`, `expected_model_cols`, `excluded_cols`);
  - política de decisão (`threshold` operacional, `top-k` de contingência e threshold calibrado);
  - métricas agregadas de treino/holdout e versões das bibliotecas.
- O processo de promoção enriquece e valida o metadata antes de finalizar cópia para `app/model/`.
- Comando oficial de validação:
  - `python -m src.metadata_schema --path app/model/metadata.json`

### Referência de Drift (Fase 6)

- A referência oficial para monitoramento de drift é o **MODEL frame** (pós feature engineering + pruning), que representa exatamente o que o modelo consome.
- O artefato salva uma amostra estratificada e determinística (até `1000` linhas por padrão), sem `RA`/PII:
  - `app/model/reference/reference_model_frame.csv`
  - `app/model/reference/reference_profile.json`
  - `app/model/reference/reference_meta.json`
- O processo suporta backup local para rollback em:
  - `app/model/reference/backups/<timestamp>/`
- Comando oficial:
  - `python -m src.build_reference_data --model-dir app/model --out-dir app/model/reference --max-rows 1000 --backup 1 --force 0`

### Versionamento do Dataset (Fase 6)

- Todo treino/avaliacao passa a registrar fingerprint do dataset (`SHA-256` em streaming), sem salvar conteudo de linhas.
- O `metadata.json` por variante inclui `dataset.path_hint`, `basename`, `bytes`, `mtime_utc` e `sha256`.
- Cada execucao gera um evento em `artifacts/dataset_versions/` com contexto (`train_baseline`, `train_hgb`, `evaluate_holdout`, `temporal_shift`, `build_reference_data`).
- Comando utilitario:
  - `python -m src.dataset_versioning --path dataset/PEDE_PASSOS_DATASET_FIAP.xlsx --context manual_check --out artifacts/dataset_versions/manual_check.json`

### Schema Formal de Saída (Fase 6)

- Contrato formal implementado em `app/schemas.py` com modelos Pydantic:
  - `PredictionResult` (`risk_proba`, `risk_class`, `threshold_applied`, `model_version`, `model_family`, `variant`, `decision_policy`, `notes`)
  - `PredictResponse` (`predictions`, `count`, `generated_at`)
- Fonte de verdade:
  - `threshold_applied` vem de `app/model/metadata.json` em `threshold_policy.operational_fixed_threshold` (fallback legado: `threshold_policy.operational.threshold`; default final `0.30`)
  - `model_version`, `model_family` e `variant` também vêm do metadata (fallback `unknown`)
- Regra de validação:
  - `risk_proba` deve estar em `[0,1]`
  - `risk_class` é sempre derivada de `risk_proba >= threshold_applied`

Exemplo de resposta (single):

```json
{
  "predictions": [
    {
      "risk_proba": 0.78,
      "risk_class": 1,
      "threshold_applied": 0.3,
      "model_version": "2026-02-20T12-00-00Z",
      "model_family": "nonlinear_hgb",
      "variant": "default",
      "decision_policy": "fixed_threshold",
      "notes": ["threshold_from_metadata"]
    }
  ],
  "count": 1,
  "generated_at": "2026-02-20T13:20:00+00:00"
}
```

### Rodar API Local (Fase 6)

- Subir aplicação:
  - `uvicorn app.main:app --reload --port 8000`
- Health check:
  - `curl -s http://127.0.0.1:8000/health`
- Version check:
  - `curl -s http://127.0.0.1:8000/version`

Exemplo `GET /health`:

```json
{
  "status": "ok"
}
```

Exemplo `GET /version` (sem metadata carregado):

```json
{
  "model_version": "unknown",
  "model_family": "unknown",
  "variant": "unknown",
  "threshold_operational": 0.3,
  "metadata_loaded": false,
  "model_loaded": false,
  "model_joblib_exists": false,
  "model_notes": [
    "model_file_missing",
    "metadata_json_not_found"
  ],
  "notes": [
    "metadata_missing_or_invalid",
    "fallback_unknown_model_version",
    "fallback_unknown_model_family",
    "fallback_unknown_variant",
    "fallback_default_threshold",
    "model_file_missing",
    "metadata_json_not_found"
  ]
}
```

### Docker (Deploy Local) (Fase 6)

- A imagem usa `python:3.11-slim`, instala dependências via `requirements.txt` e roda `uvicorn` como usuário não-root
- `dataset/` e `artifacts/` não entram no build context (ver `.dockerignore`)
- A API sobe mesmo sem modelo promovido (`/health` e `/version` ok; `/predict` retorna `503`)

Build:

```bash
docker build -t fiap-ml-api .
```

Run (sem modelo promovido):

```bash
docker run --rm -p 8000:8000 fiap-ml-api
```

Run (com modelo promovido montado em volume):

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/app/model:/app/app/model" \
  -e LOG_LEVEL=INFO \
  fiap-ml-api
```

Testar:

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/version
```

### Carregamento do Modelo (Serving) (Fase 6)

- O artefato servido é `app/model/model.joblib` (promovido via `python -m src.promote_model`)
- O carregamento é `lazy` + cache em `app/deps.py` (não recarrega a cada request)
- `GET /version` expõe `model_loaded` e `model_joblib_exists` para diagnóstico rápido
- `POST /predict` retorna `503` com `notes` quando o modelo não está disponível ou falha no load

### POST /predict (Fase 6)

- Formatos aceitos no body:
  - registro único (`{...}`)
  - lista de registros (`[{...},{...}]`)
  - envelope (`{"records":[{...},{...}]}`)
- Regras de validação:
  - base obrigatória: `expected_raw_cols` do metadata de serving
  - estrutura do body validada por Pydantic (single, batch, envelope)
  - extras não suspeitas: permitidas e ignoradas no reindex
  - extras leakage-like: bloqueadas com `400`
  - faltantes: `400` com `missing_columns`
  - batch acima do limite: `400` (`batch too large`)
  - body/JSON/tipo inválido: `422` (FastAPI/Pydantic)
  - metadata/model indisponíveis: `503`

### Alunos novos (sem histórico)

- Padrão (`ALLOW_PARTIAL_PAYLOAD=0`, default):
  - o payload deve conter todas as colunas de `expected_raw_cols`
  - colunas faltantes retornam `400` com `missing_columns`
- Modo opcional (`ALLOW_PARTIAL_PAYLOAD=1`):
  - payload parcial é aceito (mantendo o mesmo contrato base de colunas esperadas)
  - colunas faltantes são preenchidas com `NA` (`pd.NA`) no `reindex`
  - a imputação da pipeline (`SimpleImputer`) resolve os faltantes (sem valores "mágicos")
  - extras leakage-like continuam bloqueadas com `400` (mesmo com payload parcial)
- Observabilidade (sem PII):
  - `POST /predict` registra 1 log agregado por request/batch com `count_records`, `status_code`, `allow_partial_*`, taxas de missing e contagem de extras
  - erros estruturais `422` (FastAPI/Pydantic, antes do handler de rota) também são registrados de forma agregada via exception handler global, sem logar payload/valores
  - não loga payload, `RA`, IDs ou valores de célula
  - as taxas de missing usam como base `expected_raw_cols` do metadata de serving (contrato raw da API)
- Resposta:
  - sem breaking change no schema
  - a API pode incluir resumo de missing em `predictions[*].notes` (ex.: `missing_cols_rate`, `missing_values_rate`)
- Semântica de missing:
  - as métricas usam `isna()`/`pd.NA`; strings vazias (`""`) não são tratadas como missing nessa contabilização

Exemplo `curl` (single):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"a":1,"b":2}'
```

Exemplo `curl` (batch):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{"a":1,"b":2},{"a":3,"b":4}]'
```

Exemplo `curl` (envelope):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records":[{"a":1,"b":2},{"a":3,"b":4}]}'
```

Exemplo de resposta:

```json
{
  "predictions": [
    {
      "risk_proba": 0.21,
      "risk_class": 0,
      "threshold_applied": 0.3,
      "model_version": "2026-02-20T12-00-00Z",
      "model_family": "nonlinear_hgb",
      "variant": "default",
      "decision_policy": "fixed_threshold",
      "notes": null
    },
    {
      "risk_proba": 0.65,
      "risk_class": 1,
      "threshold_applied": 0.3,
      "model_version": "2026-02-20T12-00-00Z",
      "model_family": "nonlinear_hgb",
      "variant": "default",
      "decision_policy": "fixed_threshold",
      "notes": null
    }
  ],
  "count": 2,
  "generated_at": "2026-02-20T13:20:00+00:00"
}
```

</details>

<details>
<summary>Monitoramento, observabilidade, drift e dashboard (Fase 7) — expandir</summary>

### Mensuração em Produção (Ground Truth Delay) (Fase 7)

Como o rótulo de negócio (`Defasagem_{t+1}`) chega com atraso, a mensuração em produção é separada em dois blocos:

- ONLINE (sem rótulo, imediato):
  - objetivo: saúde do serviço + comportamento das predições + qualidade de entrada
  - fonte: eventos agregados por request/batch em `logs/online_metrics.jsonl`
  - cada evento registra apenas agregados (sem PII), por exemplo:
    - `generated_at`, `status_code`, `status_family`, `reason_code`
    - `model_version`, `model_family`, `variant`, `threshold`
    - `n_records`
    - `score_histogram` (histograma de `risk_proba`, sem probabilidades individuais)
    - `positive_rate_at_threshold` e `n_positive_at_threshold`
    - métricas de qualidade de input (`missing_cols_rate`, `missing_values_rate`, etc.)
  - observação:
    - o evento registra `status_family` (`2xx`/`4xx`/`5xx`)
    - a **taxa de erro agregada** deve ser calculada posteriormente a partir da agregação desses eventos (não por evento individual)
  - cobertura de erro:
    - requests `2xx/4xx/5xx` do `/predict`
    - erros estruturais `422` (FastAPI/Pydantic, antes da rota) via exception handler global

- OFFLINE (com rótulo, pós-fato):
  - objetivo: métricas oficiais de performance quando `t+1` chega
  - abordagem adotada: **replay determinístico** do período usando dataset completo (sem logs com `RA`)
  - por que replay:
    - evita PII em logs de produção (não armazenamos `RA`)
    - é aceitável no contexto acadêmico/local
  - comando:

```bash
python -m src.offline_evaluation \
  --dataset-path "dataset/DATATHON/BASE DE DADOS PEDE 2024 - DATATHON.xlsx" \
  --model-dir app/model \
  --year-t 2023 \
  --year-t1 2024 \
  --out-json artifacts/offline_metrics_2023_2024.json \
  --out-md artifacts/offline_metrics_2023_2024.md
```

- O relatório offline registra agregados como:
  - `model_version`, `model_family`, `variant`
  - par `year_t -> year_t1`
  - `threshold_operational` (extraído do metadata de serving)
  - `n`, `n_pos`, `prevalence`
  - `Recall`, `Precision`, `F1`, `ROC-AUC`, `PR-AUC`
  - matriz de confusão no threshold operacional

- Política operacional recomendada:
  - ONLINE: revisar periodicamente histograma, `positive_rate`, erros (`4xx/5xx/422`) e qualidade de input
  - OFFLINE: rodar quando os dados de `t+1` chegarem (ex.: ciclo anual)
  - Guardrails:
    - drift severo / `positive_rate` anômala: investigar inputs, threshold e versão
    - queda de Recall/PR-AUC no offline: avaliar retreino e promoção de nova versão

- Privacidade:
  - não logar `RA`, listas de IDs, payload raw, valores de célula ou probabilidades individuais
  - logs/artefatos de monitoramento devem conter apenas contagens, histogramas e taxas agregadas

### Retenção e Limpeza Local (Fase 7)

Para reduzir uso de disco e exposição desnecessária de artefatos locais, o projeto inclui uma rotina simples de retenção/limpeza baseada apenas em metadados de filesystem (path, `mtime`, tamanho), sem abrir conteúdo de arquivos.

- CLI: `python -m src.retention`
- comportamento padrão: `dry-run` (não deleta nada sem `--dry-run 0`)
- segurança:
  - não lê conteúdo dos arquivos
  - ignora symlinks
  - preserva `.gitkeep`

Política padrão:

- `logs/`
  - `*.jsonl` e `*.log`: TTL de `14` dias
- `artifacts/` (relatórios e saídas locais)
  - `artifacts/*.json` e `artifacts/*.md`: TTL de `30` dias
  - `artifacts/dataset_versions/*.json`: TTL de `30` dias
  - `artifacts/models/**`: **não** é limpo automaticamente por padrão (risco alto)
- `app/model/backups/`
  - mantém `10` diretórios de backup mais recentes
- `app/model/reference/backups/`
  - mantém `5` diretórios de backup mais recentes
- opcional (default OFF)
  - `artifacts/models/releases/`: limpeza por contagem via `--keep-model-releases N` (usar com cautela)

Comandos:

```bash
# Dry-run (padrão)
python -m src.retention

# Dry-run explícito
python -m src.retention --dry-run 1

# Executar limpeza de verdade
python -m src.retention --dry-run 0

# Customizar TTL/keep
python -m src.retention --dry-run 0 \
  --logs-ttl-days 7 \
  --artifacts-ttl-days 21 \
  --keep-model-backups 8 \
  --keep-reference-backups 3

# Opcional: limpar releases por contagem (default OFF)
python -m src.retention --dry-run 1 --keep-model-releases 3
```

Observações:

- A limpeza de `artifacts/models/**` fica desabilitada por padrão para evitar remoção acidental de artefatos de treino/releases.
- O TTL de `logs/online_metrics.jsonl` é por **arquivo** (baseado em `mtime`), não por linha/evento:
  - se o arquivo continuar recebendo append, o `mtime` fica recente e entradas antigas não serão removidas individualmente
  - para retenção fina por evento, o próximo passo recomendado é rotação de arquivos (ex.: diário)

### Não-regressão do Modelo (Fase 7)

O projeto inclui um check de não-regressão do modelo campeão baseado em artefatos JSON, sem depender do dataset no CI.

- CLI: `python -m src.regression_check`
- Fonte de verdade:
  - `artifacts/model_selection.json`
  - `metadata.json` do winner (`winner.path_metadata` ou fallback por `artifacts/models/<family>/<variant>/metadata.json`)
- Métricas verificadas no holdout:
  - `Recall` (primária)
  - `PR-AUC` (secundária)
- Limiares mínimos (alinhados com seleção/promoção do campeão):
  - `Recall >= 0.45`
  - `PR-AUC >= 0.60`
- Preferência de threshold:
  - usa `holdout@0.30` quando disponível
  - fallback para `holdout@0.5` é permitido e retorna `WARNING` (exit code `0`)

Comandos:

```bash
# Execução padrão (CI-friendly, sem dataset)
python -m src.regression_check

# Caminhos explícitos
python -m src.regression_check \
  --selection-path artifacts/model_selection.json \
  --models-root artifacts/models
```

Status e exit code:

- `PASS` -> exit `0`
- `WARNING` -> exit `0` (ex.: fallback para `0.5`)
- `SKIPPED` -> exit `0` (sem `model_selection.json`, CI não bloqueia)
- `FAIL` -> exit `1`

Observação:

- Este check não recalcula métricas; ele valida consistência mínima de qualidade a partir dos artefatos de seleção/metadata.
- Um modo local de recálculo com dataset real pode ser adicionado depois, mas fica fora do escopo deste check CI-friendly.

### Logging Estruturado (Fase 7)

O projeto usa logging estruturado em JSON por padrão (1 linha por evento), com foco em observabilidade local/Docker e sem vazamento de PII.

- Implementação central: `src.utils`
  - `JsonFormatter`
  - `setup_logging(...)`
  - `log_event(...)`
- Escopo:
  - pipeline (`src/*`) e API (`app/*`) usam a mesma configuração base
  - a API adiciona `request_id` via middleware e responde com header `X-Request-ID`

Formato padrão:

- `stdout` em JSON (bom para Docker e agregadores de logs)
- campos típicos por evento:
  - `ts`, `level`, `logger`, `msg`
  - `event` (quando emitido via `log_event`)
  - `context` (dict agregado/sanitizado)
  - `request_id` (API, quando houver)
  - `model_version` (quando aplicável)

Configuração por ambiente:

```bash
# Padrão (JSON em stdout)
export LOG_FORMAT=json

# Formato humano (plain) opcional
export LOG_FORMAT=plain

# Nível de log
export LOG_LEVEL=INFO

# Persistência opcional em arquivo local
export LOG_TO_FILE=1
export LOG_FILE_PATH=logs/app.log
```

Privacidade / anti-PII:

- `log_event(...)` aplica redaction de chaves sensíveis no `context` (ex.: `RA`, `Nome_Anon`, `Avaliador*`, `payload`, `records`)
- chaves removidas aparecem em `context.redacted_keys`
- não logar payload completo, IDs de alunos ou scores individuais
- logs da API usam métricas agregadas (rates/counts/histogramas) e eventos de erro sem body

Observações:

- A configuração é idempotente (não duplica handlers do projeto em chamadas repetidas de `setup_logging`)
- O projeto mantém handlers no `root logger` (com `propagate=True` nos loggers filhos) para compatibilidade com `pytest caplog`

### Privacidade Operacional (Fase 7)

Política operacional (logs, monitoramento e artefatos):

- Nunca logar / persistir em monitoramento:
  - `RA`
  - `Nome_Anon` (pode conter nome real em bases antigas)
  - `Avaliador1..Avaliador6`
  - payloads completos (`payload`, `records`)
  - listas de IDs/estudantes
  - probabilidades individuais por aluno (`probas`, `risk_probas`, `scores`)
- Permitido:
  - contagens agregadas
  - taxas (`missing_rate`, `positive_rate`)
  - histogramas agregados (`bin_edges`, `bin_counts`)
  - nomes de colunas (ex.: `missing_columns`, listas pequenas)

Guardrails implementados:

- `src/privacy.py`
  - centraliza definição de campos/chaves sensíveis
  - `redact_dict(...)` / `safe_log_extra(...)` para redaction em logs
  - `is_safe_json_payload(...)` para validação heurística de payloads JSON de monitoramento/artefatos
- `src.utils.log_event(...)`
  - aplica redaction automaticamente no `context`
  - adiciona `redacted_keys` quando remove campos sensíveis
- `src/online_metrics.append_online_event(...)`
  - valida privacidade do evento agregado antes de gravar em `logs/online_metrics.jsonl`
  - em caso inseguro: faz `warning` e **não grava** o evento
- API (`/predict`)
  - não loga payload/records
  - erro de extras leakage-like retorna mensagem genérica (sem listar campos sensíveis enviados)
  - `422` de `/predict` retorna resposta sanitizada (sem ecoar `input` do Pydantic)

### Drift (Evidently) (Fase 7)

Relatório visual local de drift em HTML (sem cloud) usando **Evidently**, comparando:

- referência: `app/model/reference/reference_model_frame.csv`
- atual: um `CSV` de **MODEL frame** fornecido por você (`--current-csv`)

Saídas padrão:

- `artifacts/drift_report.html` (obrigatório)
- `artifacts/drift_report_summary.json` (resumo agregado; opcional com `--no-json`)

Pré-requisito:

- gere a referência antes com `python -m src.build_reference_data`

Como rodar:

```bash
python -m src.drift \
  --reference-dir app/model/reference \
  --current-csv <caminho_para_current_model_frame.csv> \
  --out-html artifacts/drift_report.html \
  --out-json artifacts/drift_report_summary.json
```

Regras/contratos:

- o `current_csv` deve estar em **MODEL frame** (sem `RA`/PII, sem payload raw)
- colunas extras no `current_csv` são ignoradas
- colunas faltantes em relação à referência geram `FAIL` (erro claro com preview)
- amostragem determinística (quando necessário) com `--max-rows` e `--seed`

Status do resumo (`PASS/WARNING/FAIL`):

- `FAIL` se `share_drifted_features >= 0.30`
- `WARNING` se `>= 0.10`
- `PASS` caso contrário

Observações:

- `src.temporal_shift` e `src.drift` são **complementares**:
  - `src.temporal_shift`: relatório determinístico/gates do projeto (JSON/MD)
  - `src.drift`: relatório visual HTML com Evidently para inspeção local
- A geração automática de `current_csv` a partir do dataset XLSX (modo simulado local) fica fora do escopo desta tarefa; o CLI atual recebe `--current-csv`.
- O relatório opera apenas em **MODEL frame**, preservando privacidade operacional (sem `RA`, sem nomes, sem payload da API).

### Dashboards Streamlit (Drift + Operacional Consolidado)

Ambos os dashboards são locais (`read-only`, sem cloud) e exibem apenas agregados.

Pré-requisito comum:

```bash
scripts/bootstrap_dashboard_env.sh
source .venv-dashboard/bin/activate
```

#### A) Dashboard de Drift (Fase 7)

Entradas:
- `artifacts/drift_report.html`
- `artifacts/drift_report_summary.json` (opcional)

Execução:

```bash
streamlit run dashboards/streamlit_app.py
```

Uso:
- foco em inspeção visual do relatório Evidently HTML;
- path de HTML/summary configurável na sidebar;
- upload opcional de HTML alternativo (`.html`).

#### B) Dashboard Operacional Consolidado (Fase 8/9)

Entradas (padrão):
- online inference: `logs/online_metrics.jsonl`
- drift: `artifacts/drift_report.html` + `artifacts/drift_report_summary.json`
- pós-fato/offline: `artifacts/offline_metrics_*.json`

Execução:

```bash
streamlit run dashboards/ops_dashboard.py
```

Abas:
- `Online Inference`:
  - `n_events`, `n_records_total`, `positive_rate_avg` (ponderada por `n_records`)
  - `error_rate`, `validation_error_rate`, `model_unavailable_rate`
  - séries temporais de `positive_rate` e `error_rate`
  - histograma agregado de scores (quando `score_histogram` existir no JSONL)
- `Drift`:
  - resumo agregado (`status`, `share_drifted_features`, `drifted_features_count`)
  - embed do `drift_report.html`
- `Métricas Pós-Fato (Offline)`:
  - seleção de arquivo `offline_metrics_*.json`
  - `Recall`, `PR-AUC`, `Precision`, `F1`, `ROC-AUC`, `prevalence`, `matriz de confusão`
- `Runbook`:
  - comandos úteis para gerar drift, offline metrics, retenção e dry-run de retreino.

Privacidade:
- não carrega XLSX do dataset;
- não exibe payload/records/IDs/scores individuais;
- apenas agregados e artefatos já sanitizados.

Acesso local:
- `http://localhost:8501`

### Fase 9 (Opcional): Prometheus + Grafana (Local)

Observabilidade clássica de API/performance local, complementar ao monitoramento de ML já existente (`online_metrics.jsonl`, Evidently e dashboards Streamlit).

Escopo:
- endpoint de métricas em `GET /metrics` (Prometheus format)
- stack local via `docker-compose.observability.yml` com `api + prometheus + grafana`
- provisioning automático de datasource e dashboard Grafana

Arquivos:
- `docker-compose.observability.yml`
- `observability/prometheus/prometheus.yml`
- `observability/grafana/provisioning/datasources/datasource.yml`
- `observability/grafana/provisioning/dashboards/dashboard.yml`
- `observability/grafana/dashboards/api_observability.json`

Subir stack:

```bash
docker compose -f docker-compose.observability.yml up --build
```

URLs:
- API: `http://localhost:8000`
- Metrics: `http://localhost:8000/metrics`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

Validação recomendada:

```bash
# testes leves (pytest)
.venv/bin/pytest tests/test_metrics_endpoint.py tests/test_observability_contracts.py

# smoke local end-to-end (docker compose + probes API/Prometheus/Grafana)
bash scripts/smoke_observability.sh

# opcional: nao rebuildar imagem no smoke (exige imagem local ja criada)
NO_BUILD=1 bash scripts/smoke_observability.sh

# opcional: sobrescrever portas locais para evitar conflito
API_PORT=8010 PROMETHEUS_PORT=9099 GRAFANA_PORT=3300 bash scripts/smoke_observability.sh
```

Workflow manual opcional no GitHub Actions:
- `.github/workflows/observability-smoke.yml` (`workflow_dispatch`)
- executa `scripts/smoke_observability.sh`, coleta logs e derruba a stack ao final.

Métricas principais expostas:
- `http_requests_total{method,path,status}`
- `http_request_duration_seconds{method,path}` (histograma)
- `inference_records_total{endpoint="/predict"}`
- `inference_positive_total{endpoint="/predict",threshold}`
- `model_loaded` e `metadata_loaded` (gauges)

Notas operacionais:
- os painéis Grafana filtram `path="/metrics"` para evitar distorção de latência/erro por scrape.
- labels de `path` usam template de rota (`/predict`, `/health`, etc.); caminhos sem rota resolvida caem em `"/__unmatched__"` para controlar cardinalidade.
- sem PII: não há `RA`, payloads, IDs de alunos ou scores individuais nas métricas.
- `/metrics` não substitui monitoramento de drift/ground-truth delay; é complementar.

### CI (GitHub Actions) (Fase 7)

- Workflow em `.github/workflows/ci.yml`
- Executa em `push` e `pull_request` (`main`/`master`) com:
  - `pytest` + coverage (`--cov-fail-under=80`)
  - `python -m src.regression_check` (CI-friendly: `SKIPPED`/exit `0` se não houver artefatos)
  - `python -m src.validate --no-markdown --skip-dataset`
  - `python -m src.cohort_stats --no-markdown --skip-dataset`
- O modo `--skip-dataset` existe para CI porque `dataset/` não é versionado e não estará disponível no runner do GitHub Actions.
- Localmente (com dataset real), rode sem `--skip-dataset` para validação completa:
  - `python -m src.validate`
  - `python -m src.cohort_stats`

## Evidências Visuais para Banca (Local)

- As capturas de tela para apresentação estão em `artifacts/evidence_pack/screenshots/`.
- Este README apenas referencia os arquivos; as imagens não são exibidas inline.
- Mapeamento de uso (requisito -> screenshot): `docs/evidencias_banca.md`.
- Arquivos principais:
  - `artifacts/evidence_pack/screenshots/api_snapshot.png`
  - `artifacts/evidence_pack/screenshots/drift_report_html.png`
  - `artifacts/evidence_pack/screenshots/streamlit_drift_dashboard.png`
  - `artifacts/evidence_pack/screenshots/streamlit_ops_dashboard.png`
  - `artifacts/evidence_pack/screenshots/capture_manifest.json`

</details>

## Checklist do Projeto - Datathon Machine Learning Engineering

Este checklist foi elaborado considerando explicitamente as inconsistências reais do dataset fornecido (schemas distintos entre anos, colunas duplicadas, valores inválidos, mudanças semânticas de campos e interseção parcial de estudantes entre períodos). As etapas descritas adotam práticas de Data Engineering e MLOps para garantir robustez, reprodutibilidade e validade estatística do modelo em produção.

Status: `TODO` | `DOING` | `DONE` | `BLOCKED`

Progresso geral (barra visual):
`[🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩⬜]`

`115 de 116 tarefas concluídas (99.1%)`

| Fase | Progresso |
|---|---|
| Fase 1 - Entendimento do Problema e Target | 13/13 |
| Fase 2 - Organização do Projeto e Ambiente | 7/7 |
| Fase 3 - Ingestão, Qualidade e Governança de Dados | 14/14 |
| Fase 4 - Pré-processamento e Engenharia de Features | 10/10 |
| Fase 5 - Pipeline, Treinamento e Avaliação | 17/17 |
| Fase 6 - Artefatos, API e Deploy | 16/16 |
| Fase 7 - Testes, Monitoramento e Dashboard | 13/13 |
| Fase 8 - Documentação e Entrega Final | 25/26 |
| Total | 115/116 |

Nota:
- A `Fase 9` é opcional e fica fora da contagem oficial de progresso (`barra`, `X/Y` e `%`).

<details>
<summary>Checklist detalhado por fase (expandir)</summary>

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

### Fase 5 - Pipeline, Treinamento e Avaliação [17/17]
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
- [x] Gerar matriz de confusão
- [x] Definir threshold operacional focado em Recall
- [x] Definir critério objetivo formal de seleção do modelo final (ex.: maior Recall com PR-AUC acima de limiar mínimo)
- [x] Justificar escolha do modelo final
- [x] Incluir validação de shift temporal do target e das features antes do treinamento final

### Fase 6 - Artefatos, API e Deploy [16/16]
- [x] Salvar pipeline completa em `model.joblib`
- [x] Criar `metadata.json` com modelo, métricas, threshold, features esperadas, data do treino e versões das bibliotecas
- [x] Salvar dados de referência para monitoramento de drift
- [x] Versionar dataset de treino/validação (`hash/checksum` + versão usada no experimento)
- [x] Definir schema formal de saída do modelo/API (probabilidade, classe prevista, threshold aplicado e versão do modelo)
- [x] Criar aplicação FastAPI
- [x] Migrar startup da FastAPI para `lifespan` (remover depreciação de `on_event`)
- [x] Implementar endpoint `POST /predict`
- [x] Implementar `GET /health` e `GET /version`
- [x] Validar entradas com Pydantic
- [x] Garantir carregamento do modelo salvo
- [x] Criar Dockerfile enxuto baseado em `python:slim`
- [x] Documentar comandos de build e run no README
- [x] Implementar versionamento de modelos local (ex.: `artifacts/models/releases/<model_version>/` com `model.joblib` + `metadata.json`)
- [x] Definir estratégia de promoção de modelo (staging -> prod local) com critério objetivo (Recall/PR-AUC/threshold)
- [x] Documentar procedimento de atualização do modelo na API (troca de versão e rollback local)

### Fase 7 - Testes, Monitoramento e Dashboard [13/13]
- [x] Criar testes unitários e de integração com pytest
- [x] Garantir cobertura mínima de 80% com `pytest-cov`
- [x] Adicionar CI automatizada (rodar `pytest`, coverage, `python -m src.validate` e `python -m src.cohort_stats`)
- [x] Definir comportamento para alunos novos (sem histórico): validação de contrato, imputação/valores default e logging da taxa de campos ausentes
- [x] Definir estratégia de mensuração em produção com "ground truth delay" (métricas online vs métricas offline quando o rótulo chega)
- [x] Implementar logging agregado de inferência (distribuição de scores, taxa de positivos por threshold, taxa de erro de validação) sem PII (`logs/online_metrics.jsonl` + eventos agregados `2xx/4xx/5xx/422`)
- [x] Implementar rotina de avaliação pós-fato (quando labels `t+1` chegam) para medir Recall/PR-AUC em produção (mesmo que simulado) (`python -m src.offline_evaluation`)
- [x] Definir política de retenção/limpeza de logs e artefatos locais (script simples + documentação) (`python -m src.retention`)
- [x] Implementar teste de não-regressão do modelo com limiares mínimos de métricas (ex.: Recall e/ou PR-AUC) (`python -m src.regression_check` + `tests/test_model_regression.py`)
- [x] Configurar logging estruturado (JSON stdout por padrão, `log_event`, `request_id`, redaction anti-PII)
- [x] Aplicar política de privacidade operacional (não logar identificadores sensíveis como `RA` em API e monitoramento) (`src/privacy.py` + redaction no logger + `422` sanitizado no `/predict`)
- [x] Implementar relatório de drift com Evidently
- [x] Criar aplicação Streamlit para visualização do relatório de drift

### Fase 8 - Documentação e Entrega Final [25/26]
- [x] Documentar visão geral do problema e objetivo
- [x] Documentar stack tecnológica
- [x] Adicionar versionamento/changelog dos contratos (`docs/contracts`)
- [x] Documentar estrutura do projeto
- [x] Documentar etapas do pipeline de Machine Learning
- [x] Documentar ciclo de vida em produção: entrada de alunos novos, validação de contrato, inferência, logging, drift, retreino, promoção/rollback
- [x] Documentar explicitamente contratos em produção (data contracts + contrato de payload da API + contrato de saída)
- [x] Documentar estratégia de retreino (gatilhos por tempo e/ou por drift, e como executar)
- [x] Documentar limitações conhecidas do modelo e riscos assumidos (parcial: incluída limitação semântica `Idade x Fase_Ideal`)
- [x] Documentar exemplos de chamadas à API
- [x] Documentar setup de ambiente local com `.venv` e instalação de dependências
- [x] Publicar código organizado no GitHub (PR `#1` aberta via GitHub CLI)
- [x] Mesclar PR na `main` via GitHub CLI (PR `#1` mesclada)
- [x] Commitar checklist, abrir PR de atualização, mesclar na `main` e limpar branch local (PR `#2` mesclada)
- [x] Disponibilizar API acessível localmente
- [ ] Gravar vídeo gerencial (<= 5 minutos) explicando a solução
- [x] Registrar no README a localização das capturas de evidência para banca (sem embed das imagens) *(DOING -> DONE)*
- [x] Criar `docs/evidencias_banca.md` com mapeamento `requisito -> screenshot esperado` (sem embed) *(DOING -> DONE)*
- [x] Isolar `playwright` no `requirements-dashboard.txt` (remover de `requirements.txt`) para manter runtime da API enxuto *(DOING -> DONE)*
- [x] Criar `agents.md` com convenções operacionais para agentes LLM
- [x] Adicionar barra de progresso geral visual (`[🟩⬜...]`) no checklist
- [x] Atualizar `agents.md` com regra explícita de manutenção da barra visual e da contagem geral
- [x] Incorporar recomendações da revisão técnica do checklist (gaps de maturidade por fase)
- [x] Refinar redação do objetivo para "apresentar defasagem no t+1" (evita ambiguidade de transição vs estado)
- [x] Refinar visão geral com vínculo explícito a `Defas/Defasagem` e regra de coorte por `RA`
- [x] Adicionar menção explícita de não-causalidade do modelo na seção de contexto de uso

### Fase 9 - Opcional (Backlog Futuro, fora da contagem oficial)
- [x] Implementar explainability local do campeão (ex.: importâncias globais e análise de erro agregada)
- [x] Definir rotina de retreino automatizada com gatilho por tempo + gatilho por drift
- [x] Hardenizar rotina de retreino automatizada (cobertura de testes dos fluxos `NOOP`/`RECOMMENDED`/`FAIL`/`PASS` e CLI)
- [x] Publicar dashboard operacional consolidando inferência, drift e métricas pós-fato
- [x] Adicionar testes de carga básicos para API de inferência
- [x] Preparar pacote de evidências para banca (runbook + artefatos + checklist de auditoria) *(pacote local em `artifacts/evidence_pack/`; screenshots visuais ficam em `SCREENSHOTS_PENDENTES.md`)*
- [x] Implementar observabilidade mínima local com Prometheus + Grafana (`/metrics`, compose e provisioning) *(DOING -> DONE)*
- [x] Adicionar testes de observabilidade (pytest + contratos + smoke manual) *(DOING -> DONE)*

</details>

<details>
<summary>Notas de uso do checklist</summary>

- Atualize os contadores de progresso de cada fase ao concluir tarefas.
- Atualize a barra visual de progresso geral (`[🟩⬜...]`) com base na porcentagem concluída.
- Regra da barra: 40 blocos (`1 bloco = 2,5%`), com arredondamento para baixo.
- Marque uma tarefa como `DOING` no texto do item quando estiver em andamento.
- Promova para `DONE` apenas após evidência (teste, artefato, log ou documentação).
- Use `BLOCKED` quando depender de decisão, dado externo ou ajuste de escopo.

</details>
