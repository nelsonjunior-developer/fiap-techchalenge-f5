# Deep Dives do Pipeline de Machine Learning

Este documento concentra detalhamentos técnicos mais extensos do pipeline que foram resumidos no `README.md` para melhorar a navegação e reduzir duplicação.

Use junto com:
- `README.md` (visão geral, stack, etapas do pipeline e runbook)
- `docs/contracts/` (contratos de dados e changelog)
- módulos em `src/` citados em cada seção

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

## Atualização do Modelo na API (Troca de Versão e Rollback) (Fase 6)

Esta seção documenta o **procedimento operacional** para trocar a versão do modelo servido pela API e executar rollback local com segurança, reutilizando apenas os mecanismos já implementados no projeto.

Pré-requisitos:
- API FastAPI disponível (local ou em container).
- Caminho de serving padrão:
  - `app/model/model.joblib`
  - `app/model/metadata.json`
- O endpoint `GET /version` deve refletir `model_version`, `model_family`, `variant` e `threshold_operational`.

Fluxo recomendado (staging -> prod):

```bash
# A) Treinar e avaliar candidatos (treino oficial 2022->2023, holdout 2023->2024)
python -m src.train_baseline ...
python -m src.train_hgb ...

# B) Selecionar campeão com critério formal
python -m src.model_selection \
  --models-root artifacts/models \
  --output-json artifacts/model_selection.json \
  --output-md artifacts/model_selection.md

# C) (Opcional) Criar release imutável para rastreabilidade/rollback
python -m src.create_release \
  --selection-path artifacts/model_selection.json \
  --out-root artifacts/models/releases

# D) Stage (não altera produção local)
python -m src.promote_model \
  --selection-path artifacts/model_selection.json \
  --models-root artifacts/models \
  --out-dir app/model/staging \
  --stage-only 1 \
  --backup 1 \
  --force 0 \
  --allow-warning 0
```

Se `artifacts/model_selection.json` estiver em `WARNING` (decision `ALLOW_WITH_OVERRIDE`), repetir com override explícito:

```bash
python -m src.promote_model \
  --selection-path artifacts/model_selection.json \
  --models-root artifacts/models \
  --out-dir app/model/staging \
  --stage-only 1 \
  --backup 1 \
  --force 0 \
  --allow-warning 1
```

Validação antes de promover (staging):
- Verificar manifesto: `app/model/staging/staging_manifest.json`
- Validar contrato do metadata staged:
  - `python -m src.metadata_schema --path app/model/staging/metadata.json`
- Sanity check opcional do pipeline/dataset:
  - `python -m src.smoke_pipeline`
  - Observação: `src.smoke_pipeline` é um smoke check do pipeline/projeto; **não valida diretamente** o artefato em `app/model/staging/`.
- Se a API estiver rodando em Docker com volume montado, validar endpoints:
  - `curl http://localhost:8000/version`
  - `curl http://localhost:8000/health`

Promover staging -> prod local (troca efetiva):

```bash
python -m src.promote_model \
  --selection-path artifacts/model_selection.json \
  --from-staging app/model/staging \
  --out-dir app/model \
  --promote 1 \
  --backup 1 \
  --force 0 \
  --allow-warning 0
```

Se o `selection.status` continuar em `WARNING`, pode ser necessário repetir com `--allow-warning 1` também no `promote`.

Verificação pós-troca (obrigatória):
1. Reiniciar a API/processo (ou reiniciar o container) após o `promote`.
2. Consultar `GET /version`:
  - `curl http://localhost:8000/version`
  - validar `model_version`, `model_family`, `variant`, `threshold_operational`
3. (Opcional) Testar `POST /predict` com payload válido mínimo:
  - deve retornar `200` quando `model_loaded=true` e `metadata_loaded=true`

Observação importante sobre reinício:
- O serving usa cache em `app/deps.py` (`lru_cache`) para metadata/modelo/contexto.
- Sem reiniciar o processo, a API pode continuar servindo a versão anterior mesmo após copiar novos arquivos para `app/model/`.

Rollback local rápido (backup automático):
- Backups ficam em `app/model/backups/<timestamp>/`
- Cada snapshot inclui `model.joblib` e `metadata.json` anteriores

Procedimento:

```bash
# A) Inspecionar backups disponíveis
ls -1 app/model/backups

# B) Restaurar arquivos do backup escolhido
cp app/model/backups/<timestamp>/model.joblib app/model/model.joblib
cp app/model/backups/<timestamp>/metadata.json app/model/metadata.json

# C) Reiniciar API/container (obrigatório por causa do cache)
# D) Validar versão restaurada
curl http://localhost:8000/version
```

Rollback alternativo (via release imutável):
- Se existir release em `artifacts/models/releases/<model_version>/`, é possível restaurar manualmente:
  - copiar `model.joblib` e `metadata.json` da release para `app/model/`
  - reiniciar API/container
  - validar em `GET /version`

Notas operacionais:
- `--allow-warning` (governança) **não é** `--force` (sobrescrita de arquivos).
- `--force 1` apenas permite sobrescrever destino existente; mantenha `--backup 1` para preservar rollback local.
- `src.promote_model --promote 1` usa `--selection-path` (default: `artifacts/model_selection.json`) para reavaliar policy; se seu arquivo estiver em outro local, informe o path explicitamente.
- Não versionar `app/model/model.joblib` nem `app/model/metadata.json` no git; versione apenas documentação/manifests quando fizer sentido.
- Manifestos (`staging_manifest.json`, `promoted_model.json`, `release.json`) devem permanecer sem `RA`/IDs/PII.
