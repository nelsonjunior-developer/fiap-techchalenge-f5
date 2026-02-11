



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
- Nota semântica importante:
  - `Ano nasc` e `Data de Nasc` não são semanticamente idênticos (ano vs data completa). Nesta fase harmonizamos apenas header; normalização de conteúdo será feita depois.
  - `Nome` e `Nome Anonimizado` são harmonizados para `Nome_Anon` apenas para alinhamento de schema; isso não garante anonimização no dado de 2022.
- Padronização de tipos após harmonização/alinhamento:
  - `Data_Nasc` é padronizada para `datetime` com desambiguação explícita:
    - valores numéricos em `1900..2100` são interpretados como ano (`YYYY-01-01`)
    - demais numéricos são interpretados como serial Excel (`origin=1899-12-30`)
  - `Idade` é sanitizada para remover valores datetime (ex.: `1900-01-...`, que viram `NaN`) e convertida para `Int64` (nullable).
  - Colunas numéricas usam dtypes nulos estáveis (`Float64`/`Int64`) com coerção robusta (`to_numeric(..., errors=\"coerce\")`), incluindo tratamento do token `INCLUIR`.
  - Colunas categóricas são padronizadas para `string` com `strip`.

## 📁 Estrutura do Projeto

O repositório é organizado para separar claramente ingestão e tratamento de dados, treinamento do modelo, disponibilização via API, monitoramento e testes, garantindo manutenibilidade, reprodutibilidade e facilidade de deploy.

```
raiz-do-projeto/
├── README.md
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── agents.md
├── app/
│   └── model/
│       └── .gitkeep
├── artifacts/
│   └── .gitkeep
├── dashboards/
├── docs/
│   ├── .gitkeep
│   └── analise_bases_e_dicionario.md
├── logs/
│   └── .gitkeep
├── notebooks/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── dtypes.py
│   ├── schema.py
│   └── utils.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_config.py
    ├── test_data.py
    ├── test_dtypes.py
    ├── test_logging.py
    └── test_schema.py
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

## Checklist do Projeto - Datathon Machine Learning Engineering

Este checklist foi elaborado considerando explicitamente as inconsistências reais do dataset fornecido (schemas distintos entre anos, colunas duplicadas, valores inválidos, mudanças semânticas de campos e interseção parcial de estudantes entre períodos). As etapas descritas adotam práticas de Data Engineering e MLOps para garantir robustez, reprodutibilidade e validade estatística do modelo em produção.

Status: `TODO` | `DOING` | `DONE` | `BLOCKED`

Progresso geral (barra visual):
`[🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜]`

`34 de 95 tarefas concluídas (35.8%)`

| Fase | Progresso |
|---|---|
| Fase 1 - Entendimento do Problema e Target | 11/11 |
| Fase 2 - Organização do Projeto e Ambiente | 7/7 |
| Fase 3 - Ingestão, Qualidade e Governança de Dados | 5/14 |
| Fase 4 - Pré-processamento e Engenharia de Features | 0/10 |
| Fase 5 - Pipeline, Treinamento e Avaliação | 0/17 |
| Fase 6 - Artefatos, API e Deploy | 0/12 |
| Fase 7 - Testes, Monitoramento e Dashboard | 1/7 |
| Fase 8 - Documentação e Entrega Final | 10/15 |
| Total | 34/95 |

### Fase 1 - Entendimento do Problema e Target [11/11]
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

### Fase 2 - Organização do Projeto e Ambiente [7/7]
- [x] Configurar `.gitignore` inicial (ignorar `agents.md`, `dataset/` e `.DS_Store`)
- [x] Expandir `.gitignore` com padrões essenciais de Python/MLOps (cache, venv, cobertura, builds, logs e segredos locais)
- [x] Criar estrutura de diretórios do projeto
- [x] Criar `requirements.txt` com dependências mínimas
- [x] Fixar versões das dependências para garantir reprodutibilidade do ambiente de execução
- [x] Definir `random_state` global para reprodutibilidade
- [x] Configurar logging básico do projeto

### Fase 3 - Ingestão, Qualidade e Governança de Dados [5/14]
Camadas conceituais desta fase:
- Camada A - Pré-ingestão e Ingestão: contrato de dados, mapeamento de colunas equivalentes, tratamento de headers duplicados, normalização de valores inválidos, padronização de datas e normalização semântica.
- Camada B - Governança e Validação Contínua: coorte temporal por `RA`, validações de shift, versionamento de dataset e privacidade operacional.

Nota de coorte temporal:
> A construção dos pares temporais considera apenas estudantes presentes em ambos os anos consecutivos (`t` e `t+1`), evitando viés por evasão ou entrada tardia e garantindo consistência estatística na definição do target.

- [x] Implementar leitura do arquivo XLSX
- [x] Tratar diferenças de colunas entre os anos
- [x] Padronizar nomes e tipos de dados
- [x] Criar função de geração dos pares temporais (`t -> t+1`)
- [ ] Validar consistência dos dados (missing, tipos inválidos)
- [ ] Definir um data contract por ano (nome, tipo e domínio esperado por coluna)
- [ ] Implementar validação automática do data contract (asserts de nome, tipo e domínio por coluna)
- [ ] Criar tabela de mapeamento entre colunas equivalentes (`Matem/Portug/Inglês` <-> `Mat/Por/Ing`; `Defas` <-> `Defasagem`)
- [ ] Tratar headers duplicados na ingestão com regra determinística
- [ ] Normalizar valores inválidos em campos numéricos (ex.: `#N/A`, `#DIV/0!`, `INCLUIR`)
- [ ] Padronizar datas de nascimento para formato único
- [ ] Normalizar categorias textuais entre anos (`Menina/Menino` <-> `Feminino/Masculino`; `Escola Pública` <-> `Pública`)
- [x] Definir regra formal de coorte temporal por `RA` (entradas, saídas e interseções por ano)
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

### Fase 7 - Testes, Monitoramento e Dashboard [1/7]
- [ ] Criar testes unitários e de integração com pytest
- [x] Garantir cobertura mínima de 80% com `pytest-cov`
- [ ] Implementar teste de não-regressão do modelo com limiares mínimos de métricas (ex.: Recall e/ou PR-AUC)
- [ ] Configurar logging estruturado
- [ ] Aplicar política de privacidade operacional (não logar identificadores sensíveis como `RA` em API e monitoramento)
- [ ] Implementar relatório de drift com Evidently
- [ ] Criar aplicação Streamlit para visualização do relatório de drift

### Fase 8 - Documentação e Entrega Final [10/15]
- [x] Documentar visão geral do problema e objetivo
- [ ] Documentar stack tecnológica
- [x] Documentar estrutura do projeto
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
- [x] Refinar redação do objetivo para "apresentar defasagem no t+1" (evita ambiguidade de transição vs estado)
- [x] Refinar visão geral com vínculo explícito a `Defas/Defasagem` e regra de coorte por `RA`
- [x] Adicionar menção explícita de não-causalidade do modelo na seção de contexto de uso

<details>
<summary>Notas de uso do checklist</summary>

- Atualize os contadores de progresso de cada fase ao concluir tarefas.
- Atualize a barra visual de progresso geral (`[🟩⬜...]`) com base na porcentagem concluída.
- Marque uma tarefa como `DOING` no texto do item quando estiver em andamento.
- Promova para `DONE` apenas após evidência (teste, artefato, log ou documentação).
- Use `BLOCKED` quando depender de decisão, dado externo ou ajuste de escopo.

</details>
