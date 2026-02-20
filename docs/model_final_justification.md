# Justificativa do Modelo Final

- generated_at: `2026-02-20T17:52:47.046038+00:00`
- status: **WARNING**
- selection_generated_at: `2026-02-20T17:30:39.507056+00:00`

## Status da selecao

- Status da selecao: **WARNING**
- Motivo: parte dos candidatos nao tinha avaliacao em threshold=0.30 (fallback para 0.5)
- Decisao aplicada: selecionado o melhor por ranking (Recall principal, PR-AUC desempate), com transparencia de fallback e trade-offs.

## Decisao

- Modelo campeao: `nonlinear_hgb/default`
- path_model: `artifacts/models/nonlinear_hgb/default/model.joblib`
- path_metadata: `artifacts/models/nonlinear_hgb/default/metadata.json`

## Por que este modelo

- nonlinear_hgb/default ganhou por maior Recall no holdout (0.7857 vs 0.7370). Runner-up: nonlinear_hgb/tuned.
- Threshold aplicado na selecao: `0.30` (preferred_operational_threshold_0.30)
- Gates minimos: Recall >= `0.45` e PR-AUC >= `0.6` | passed_gates=`True`
- ALERTA: a selecao formal retornou `WARNING`; ver secoes de trade-offs e riscos.

## Metricas no holdout (2023->2024)

| Metrica | Valor |
|---|---:|
| Recall | 0.7857 |
| PR-AUC | 0.6905 |
| Positive rate | 0.7059 |
| Precision | 0.4481 |
| F1 | 0.5708 |
| ROC-AUC | 0.6926 |
| Confusion matrix (tn/fp/fn/tp) | `159/298/66/242` |

## Criterio de selecao

- Primaria: maior Recall no holdout no threshold operacional.
- Secundaria: maior PR-AUC no holdout.
- Desempate: menor positive_rate; empate final lexicografico por modelo/variante.
- Threshold preferencial da politica: `0.30` (fallback para `0.5` com warning quando necessario).

## Trade-offs operacionais

- Operacao padrao: threshold fixo 0.30 (alerta se proba >= 0.30). Contingencia de capacidade: top-k 20% em processamento em lote (ranking de score).
- Positive_rate do campeao: `0.7059`. Maior recall tende a elevar volume de alertas e carga operacional.
- Top-k 20% e contingencia de capacidade para operacao em lote; nao e politica padrao por request.

## Riscos e limitacoes

Existe shift temporal relevante entre treino e holdout (prevalencia aproximada de 0.61 para 0.40), o que pode degradar estabilidade. A robustez depende da manutencao do contrato de dados e de monitoramento para categorias novas ou distribuicoes nao vistas.

## Notas

- Fallback threshold 0.5 used because holdout@0.30 is unavailable on: ['baseline_logreg/balanced', 'nonlinear_hgb/tuned']
