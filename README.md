



📁 Project Structure

The repository is organized to clearly separate data handling, model training, API serving, monitoring, and tests, ensuring maintainability, reproducibility, and ease of deployment.

```
project-root/
│
├── app/                         # API layer (FastAPI)
│   ├── main.py                  # FastAPI application entrypoint
│   ├── routes.py                # API routes (/predict, /health, /version)
│   ├── schemas.py               # Pydantic request/response schemas
│   └── model/
│       ├── model.joblib         # Trained ML pipeline (serialized)
│       ├── metadata.json        # Model metadata (metrics, threshold, version)
│       └── reference_data.csv   # Reference dataset for drift monitoring
│
├── src/                         # Core ML pipeline
│   ├── data.py                  # Load XLSX, standardize columns, create t→t+1 pairs
│   ├── preprocessing.py         # Data cleaning, encoding, scaling
│   ├── feature_engineering.py   # Feature creation and selection
│   ├── train.py                 # Model training and internal validation
│   ├── evaluate.py              # Metrics, confusion matrix, threshold selection
│   ├── drift.py                 # Drift detection with Evidently
│   └── utils.py                 # Shared utilities (logging, configs, helpers)
│
├── dashboards/
│   └── streamlit_app.py         # Streamlit dashboard to visualize drift reports
│
├── tests/                       # Unit and integration tests (pytest)
│   ├── test_data.py             # Tests for data loading and temporal pairing
│   ├── test_preprocessing.py    # Tests for preprocessing steps
│   ├── test_feature_engineering.py
│   ├── test_train_smoke.py      # Smoke test for training pipeline
│   └── test_api_predict.py      # API endpoint tests (/predict)
│
├── notebooks/                   # (Optional) Exploratory analysis and experiments
│
├── Dockerfile                   # Docker image definition for API deployment
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore rules
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

## Checklist do Projeto - Datathon Machine Learning Engineering

Este checklist foi elaborado considerando explicitamente as inconsistências reais do dataset fornecido (schemas distintos entre anos, colunas duplicadas, valores inválidos, mudanças semânticas de campos e interseção parcial de estudantes entre períodos). As etapas descritas adotam práticas de Data Engineering e MLOps para garantir robustez, reprodutibilidade e validade estatística do modelo em produção.

Status: `TODO` | `DOING` | `DONE` | `BLOCKED`

Progresso geral (barra visual):
`[🟩🟩🟩⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜]`

`8 de 92 tarefas concluídas (8.7%)`

| Fase | Progresso |
|---|---|
| Fase 1 - Entendimento do Problema e Target | 0/11 |
| Fase 2 - Organização do Projeto e Ambiente | 2/7 |
| Fase 3 - Ingestão, Qualidade e Governança de Dados | 0/14 |
| Fase 4 - Pré-processamento e Engenharia de Features | 0/10 |
| Fase 5 - Pipeline, Treinamento e Avaliação | 0/17 |
| Fase 6 - Artefatos, API e Deploy | 0/12 |
| Fase 7 - Testes, Monitoramento e Dashboard | 0/7 |
| Fase 8 - Documentação e Entrega Final | 6/14 |
| Total | 8/92 |

### Fase 1 - Entendimento do Problema e Target [0/11]
- [ ] Compreender o objetivo de negócio: prever o risco de defasagem escolar (t+1)
- [ ] Estudar o dicionário de dados e as bases de 2022, 2023 e 2024
- [ ] Padronizar a coluna de defasagem (`Defas` -> `Defasagem`)
- [ ] Definir a formulação do target binário
- [ ] Definir métrica primária de sucesso (`Recall`) e métricas secundárias (`PR-AUC`, `Precision`, `F1`, `ROC-AUC`) já na fase de desenho
- [ ] Definir `y = 1` se `Defasagem_{t+1} < 0`
- [ ] Definir `y = 0` caso contrário
- [ ] Definir a estratégia de pares temporais
- [ ] Definir treino: `X(2022) -> y(2023)`
- [ ] Definir holdout final: `X(2023) -> y(2024)`
- [ ] Garantir que `RA` seja usado apenas como ID, nunca como feature

### Fase 2 - Organização do Projeto e Ambiente [2/7]
- [x] Configurar `.gitignore` inicial (ignorar `agents.md`, `dataset/` e `.DS_Store`)
- [x] Expandir `.gitignore` com padrões essenciais de Python/MLOps (cache, venv, cobertura, builds, logs e segredos locais)
- [ ] Criar estrutura de diretórios do projeto
- [ ] Criar `requirements.txt` com dependências mínimas
- [ ] Fixar versões das dependências para garantir reprodutibilidade do ambiente de execução
- [ ] Definir `random_state` global para reprodutibilidade
- [ ] Configurar logging básico do projeto

### Fase 3 - Ingestão, Qualidade e Governança de Dados [0/14]
Camadas conceituais desta fase:
- Camada A - Pré-ingestão e Ingestão: contrato de dados, mapeamento de colunas equivalentes, tratamento de headers duplicados, normalização de valores inválidos, padronização de datas e normalização semântica.
- Camada B - Governança e Validação Contínua: coorte temporal por `RA`, validações de shift, versionamento de dataset e privacidade operacional.

Nota de coorte temporal:
> A construção dos pares temporais considera apenas estudantes presentes em ambos os anos consecutivos (`t` e `t+1`), evitando viés por evasão ou entrada tardia e garantindo consistência estatística na definição do target.

- [ ] Implementar leitura do arquivo XLSX
- [ ] Tratar diferenças de colunas entre os anos
- [ ] Padronizar nomes e tipos de dados
- [ ] Criar função de geração dos pares temporais (`t -> t+1`)
- [ ] Validar consistência dos dados (missing, tipos inválidos)
- [ ] Definir um data contract por ano (nome, tipo e domínio esperado por coluna)
- [ ] Implementar validação automática do data contract (asserts de nome, tipo e domínio por coluna)
- [ ] Criar tabela de mapeamento entre colunas equivalentes (`Matem/Portug/Inglês` <-> `Mat/Por/Ing`; `Defas` <-> `Defasagem`)
- [ ] Tratar headers duplicados na ingestão com regra determinística
- [ ] Normalizar valores inválidos em campos numéricos (ex.: `#N/A`, `#DIV/0!`, `INCLUIR`)
- [ ] Padronizar datas de nascimento para formato único
- [ ] Normalizar categorias textuais entre anos (`Menina/Menino` <-> `Feminino/Masculino`; `Escola Pública` <-> `Pública`)
- [ ] Definir regra formal de coorte temporal por `RA` (entradas, saídas e interseções por ano)
- [ ] Gerar e registrar estatísticas de interseção por `RA` entre anos (contagem absoluta e percentual)

### Fase 4 - Pré-processamento e Engenharia de Features [0/10]
- [ ] Separar features numéricas e categóricas
- [ ] Tratar valores ausentes (imputação)
- [ ] Codificar variáveis categóricas (`OneHotEncoder` ou similar)
- [ ] Escalonar variáveis numéricas (se necessário)
- [ ] Garantir que o pré-processamento seja reutilizável na inferência
- [ ] Criar novas features relevantes (se aplicável)
- [ ] Implementar checagem explícita de data leakage (lista negra de colunas futuras + asserts temporais)
- [ ] Remover colunas irrelevantes ou com leakage
- [ ] Garantir que nenhuma feature use informação futura
- [ ] Documentar as principais decisões de feature engineering

### Fase 5 - Pipeline, Treinamento e Avaliação [0/17]
Nota de shift temporal:
> Antes do treinamento final, é realizada uma análise de shift temporal do target e das features, uma vez que a prevalência da classe positiva varia significativamente entre os períodos analisados (aprox. `61%` para `40%`).

- [ ] Criar `ColumnTransformer` para pré-processamento
- [ ] Encapsular tudo em uma `Pipeline` do scikit-learn
- [ ] Garantir consistência treino vs inferência
- [ ] Validar que a pipeline aceita dados crus da API
- [ ] Treinar modelo baseline (`Logistic Regression`)
- [ ] Treinar modelo não-linear (ex.: `HistGradientBoosting`)
- [ ] Usar apenas dados de treino (`2022 -> 2023`)
- [ ] (Opcional) Validação interna (CV estratificada)
- [ ] Definir estratégia explícita para desbalanceamento de classes (`class_weight`, ajuste de threshold ou decisão justificada de não tratar)
- [ ] Comparar modelos com foco em Recall e PR-AUC
- [ ] Avaliar desempenho no holdout temporal (`2023 -> 2024`)
- [ ] Calcular métricas: Recall, Precision, F1-score, ROC-AUC, PR-AUC
- [ ] Gerar matriz de confusão
- [ ] Definir threshold operacional focado em Recall
- [ ] Definir critério objetivo formal de seleção do modelo final (ex.: maior Recall com PR-AUC acima de limiar mínimo)
- [ ] Justificar escolha do modelo final
- [ ] Incluir validação de shift temporal do target e das features antes do treinamento final

### Fase 6 - Artefatos, API e Deploy [0/12]
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

### Fase 7 - Testes, Monitoramento e Dashboard [0/7]
- [ ] Criar testes unitários e de integração com pytest
- [ ] Garantir cobertura mínima de 80% com `pytest-cov`
- [ ] Implementar teste de não-regressão do modelo com limiares mínimos de métricas (ex.: Recall e/ou PR-AUC)
- [ ] Configurar logging estruturado
- [ ] Aplicar política de privacidade operacional (não logar identificadores sensíveis como `RA` em API e monitoramento)
- [ ] Implementar relatório de drift com Evidently
- [ ] Criar aplicação Streamlit para visualização do relatório de drift

### Fase 8 - Documentação e Entrega Final [6/14]
- [x] Documentar visão geral do problema e objetivo
- [ ] Documentar stack tecnológica
- [ ] Documentar estrutura do projeto
- [ ] Documentar etapas do pipeline de Machine Learning
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

<details>
<summary>Notas de uso do checklist</summary>

- Atualize os contadores de progresso de cada fase ao concluir tarefas.
- Atualize a barra visual de progresso geral (`[🟩⬜...]`) com base na porcentagem concluída.
- Marque uma tarefa como `DOING` no texto do item quando estiver em andamento.
- Promova para `DONE` apenas após evidência (teste, artefato, log ou documentação).
- Use `BLOCKED` quando depender de decisão, dado externo ou ajuste de escopo.

</details>
