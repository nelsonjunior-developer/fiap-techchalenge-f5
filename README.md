



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

## Análise das Bases e Dicionário

A análise detalhada do dicionário de dados e das bases `2022`, `2023` e `2024` está documentada em:

- [docs/analise_bases_e_dicionario.md](docs/analise_bases_e_dicionario.md)
- Regra de ingestão aplicada: `Defas` (2022) é padronizada para `Defasagem` para manter schema único entre anos.

## 📁 Estrutura do Projeto

O repositório é organizado para separar claramente ingestão e tratamento de dados, treinamento do modelo, disponibilização via API, monitoramento e testes, garantindo manutenibilidade, reprodutibilidade e facilidade de deploy.

```
raiz-do-projeto/
│
├── app/                         # Camada da API (FastAPI)
│   ├── main.py                  # Ponto de entrada da aplicação FastAPI
│   ├── routes.py                # Rotas da API (/predict, /health, /version)
│   ├── schemas.py               # Schemas de requisição/resposta (Pydantic)
│   └── model/
│       ├── model.joblib         # Pipeline de ML treinada (serializada)
│       ├── metadata.json        # Metadados do modelo (métricas, threshold, versão)
│       └── reference_data.csv   # Dataset de referência para monitoramento de drift
│
├── src/                         # Pipeline principal de ML
│   ├── data.py                  # Carrega XLSX, padroniza colunas e cria pares t→t+1
│   ├── preprocessing.py         # Limpeza, codificação e escalonamento de dados
│   ├── feature_engineering.py   # Criação e seleção de atributos
│   ├── train.py                 # Treinamento do modelo e validação interna
│   ├── evaluate.py              # Métricas, matriz de confusão e seleção de threshold
│   ├── drift.py                 # Detecção de drift com Evidently
│   └── utils.py                 # Utilitários compartilhados (logging, configs, helpers)
│
├── dashboards/
│   └── streamlit_app.py         # Dashboard Streamlit para visualizar relatórios de drift
│
├── tests/                       # Testes unitários e de integração (pytest)
│   ├── test_data.py             # Testes da carga de dados e pareamento temporal
│   ├── test_preprocessing.py    # Testes das etapas de pré-processamento
│   ├── test_feature_engineering.py
│   ├── test_train_smoke.py      # Smoke test da pipeline de treinamento
│   └── test_api_predict.py      # Testes do endpoint da API (/predict)
│
├── notebooks/                   # (Opcional) Análises exploratórias e experimentos
│
├── Dockerfile                   # Definição da imagem Docker para deploy da API
├── requirements.txt             # Dependências Python
├── README.md                    # Documentação do projeto
└── .gitignore                   # Regras de arquivos ignorados no Git
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

## Checklist do Projeto - Datathon Machine Learning Engineering

Este checklist foi elaborado considerando explicitamente as inconsistências reais do dataset fornecido (schemas distintos entre anos, colunas duplicadas, valores inválidos, mudanças semânticas de campos e interseção parcial de estudantes entre períodos). As etapas descritas adotam práticas de Data Engineering e MLOps para garantir robustez, reprodutibilidade e validade estatística do modelo em produção.

Status: `TODO` | `DOING` | `DONE` | `BLOCKED`

Progresso geral (barra visual):
`[🟩🟩🟩🟩🟩🟩🟩🟩⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜]`

`18 de 95 tarefas concluídas (18.9%)`

| Fase | Progresso |
|---|---|
| Fase 1 - Entendimento do Problema e Target | 3/11 |
| Fase 2 - Organização do Projeto e Ambiente | 4/7 |
| Fase 3 - Ingestão, Qualidade e Governança de Dados | 0/14 |
| Fase 4 - Pré-processamento e Engenharia de Features | 0/10 |
| Fase 5 - Pipeline, Treinamento e Avaliação | 0/17 |
| Fase 6 - Artefatos, API e Deploy | 0/12 |
| Fase 7 - Testes, Monitoramento e Dashboard | 1/7 |
| Fase 8 - Documentação e Entrega Final | 10/15 |
| Total | 18/95 |

### Fase 1 - Entendimento do Problema e Target [3/11]
- [x] Compreender o objetivo de negócio: prever o risco de defasagem escolar (t+1)
- [x] Estudar o dicionário de dados e as bases de 2022, 2023 e 2024
- [x] Padronizar a coluna de defasagem (`Defas` -> `Defasagem`)
- [ ] Definir a formulação do target binário
- [ ] Definir métrica primária de sucesso (`Recall`) e métricas secundárias (`PR-AUC`, `Precision`, `F1`, `ROC-AUC`) já na fase de desenho
- [ ] Definir `y = 1` se `Defasagem_{t+1} < 0`
- [ ] Definir `y = 0` caso contrário
- [ ] Definir a estratégia de pares temporais
- [ ] Definir treino: `X(2022) -> y(2023)`
- [ ] Definir holdout final: `X(2023) -> y(2024)`
- [ ] Garantir que `RA` seja usado apenas como ID, nunca como feature

### Fase 2 - Organização do Projeto e Ambiente [4/7]
- [x] Configurar `.gitignore` inicial (ignorar `agents.md`, `dataset/` e `.DS_Store`)
- [x] Expandir `.gitignore` com padrões essenciais de Python/MLOps (cache, venv, cobertura, builds, logs e segredos locais)
- [ ] Criar estrutura de diretórios do projeto
- [x] Criar `requirements.txt` com dependências mínimas
- [x] Fixar versões das dependências para garantir reprodutibilidade do ambiente de execução
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
