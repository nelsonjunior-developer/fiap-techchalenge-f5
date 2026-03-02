# Evidências para Banca (Mapeamento Requisito -> Screenshot)

Este documento mapeia os requisitos de demonstração para a banca às capturas de tela esperadas, sem embutir imagens no Markdown.

## Escopo e fonte dos arquivos

- Diretório base das capturas: `artifacts/evidence_pack/screenshots/`
- Manifesto técnico da coleta: `artifacts/evidence_pack/screenshots/capture_manifest.json`
- Observação: os arquivos são locais e voltados para apresentação.

## Mapeamento

| Requisito de demonstração | Screenshot esperado | O que validar na imagem | Evidência complementar |
|---|---|---|---|
| API disponível e íntegra (`/health`, `/version`, validação de entrada) | `artifacts/evidence_pack/screenshots/api_snapshot.png` | `GET /health` com `200`, `GET /version` com `200`, `POST /predict` inválido com `422`, `POST /predict` de probe com `400` | `artifacts/evidence_pack/screenshots/api_snapshot.html`, `artifacts/evidence_pack/screenshots/capture_manifest.json` |
| Relatório de drift Evidently gerado | `artifacts/evidence_pack/screenshots/drift_report_html.png` | painel com detecção de drift e resumo visual de colunas com drift | `artifacts/evidence_pack/drift/drift_report.html`, `artifacts/evidence_pack/drift/drift_report_summary.json` |
| Dashboard de drift (Streamlit) carregando resumo + HTML | `artifacts/evidence_pack/screenshots/streamlit_drift_dashboard.png` | seção "Resumo (agregado)" preenchida e embed do relatório HTML visível | `artifacts/evidence_pack/drift/drift_report_summary.json` |
| Dashboard operacional consolidado (online + drift + offline) | `artifacts/evidence_pack/screenshots/streamlit_ops_dashboard.png` | cards de métricas online agregadas (eventos, error rate, validation rate, positive rate) e navegação por abas | `logs/online_metrics.jsonl`, `artifacts/offline_metrics_2023_2024.json` |

## Checklist rápido de uso na apresentação

1. Confirmar que os quatro screenshots existem no diretório base.
2. Conferir `capture_manifest.json` para status HTTP capturados.
3. Usar os PNGs nos slides/anexo da banca.
4. Evitar uso de payloads com PII nos materiais (manter somente evidências agregadas).
