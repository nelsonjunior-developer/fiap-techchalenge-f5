#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
ROUNDS="${ROUNDS:-120}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-20}"
SLEEP_S="${SLEEP_S:-0.05}"
SHOW_EVERY="${SHOW_EVERY:-20}"
PAYLOAD_PATH="${PAYLOAD_PATH:-/tmp/predict_observability_payload.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v curl >/dev/null 2>&1; then
  echo "Erro: curl nao encontrado no PATH." >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Erro: interpretador Python nao encontrado: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! "${ROUNDS}" =~ ^[0-9]+$ || ! "${PREDICT_BATCH_SIZE}" =~ ^[0-9]+$ ]]; then
  echo "Erro: ROUNDS e PREDICT_BATCH_SIZE devem ser inteiros." >&2
  exit 1
fi

health_code="$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/health" || true)"
if [[ "${health_code}" != "200" ]]; then
  echo "Erro: API indisponivel em ${BASE_URL} (GET /health -> ${health_code})." >&2
  echo "Suba a API antes: bash scripts/run_api_local.sh" >&2
  exit 1
fi

echo "Gerando payload de /predict a partir de app/model/metadata.json ..."
cd "${REPO_ROOT}"
"${PYTHON_BIN}" - <<'PY' "${PAYLOAD_PATH}" "${PREDICT_BATCH_SIZE}"
import json
import sys
from pathlib import Path

payload_path = Path(sys.argv[1])
batch_size = int(sys.argv[2])
meta_path = Path("app/model/metadata.json")
if not meta_path.exists():
    raise SystemExit("app/model/metadata.json nao encontrado.")

meta = json.loads(meta_path.read_text(encoding="utf-8"))
expected = [c for c in meta.get("expected_raw_cols", []) if isinstance(c, str) and c.strip()]
if not expected:
    raise SystemExit("metadata.json sem expected_raw_cols valido.")

record = {col: None for col in expected}
payload = {"records": [record for _ in range(batch_size)]}
payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
print(f"payload_ok={payload_path} cols={len(expected)} batch_size={batch_size}")
PY

predict_ok_warmup="$(curl -s -o /dev/null -w "%{http_code}" \
  -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  --data-binary "@${PAYLOAD_PATH}" || true)"

use_success_predict=0
if [[ "${predict_ok_warmup}" == "200" ]]; then
  use_success_predict=1
  echo "Warmup /predict 200: paineis de throughput/positive_rate serao populados."
else
  echo "Aviso: warmup /predict retornou ${predict_ok_warmup}."
  echo "Paineis de throughput/positive_rate podem permanecer vazios se nao houver 200."
fi

ok_200=0
err_400=0
err_422=0
other_codes=0
health_hits=0
version_hits=0

echo "Gerando trafego (${ROUNDS} rounds) ..."
for ((i=1; i<=ROUNDS; i++)); do
  curl -s -o /dev/null "${BASE_URL}/health" || true
  ((health_hits+=1))
  curl -s -o /dev/null "${BASE_URL}/version" || true
  ((version_hits+=1))

  if [[ "${use_success_predict}" == "1" ]]; then
    code="$(curl -s -o /dev/null -w "%{http_code}" \
      -X POST "${BASE_URL}/predict" \
      -H "Content-Type: application/json" \
      --data-binary "@${PAYLOAD_PATH}" || true)"
    case "${code}" in
      200) ((ok_200+=1)) ;;
      400) ((err_400+=1)) ;;
      422) ((err_422+=1)) ;;
      *) ((other_codes+=1)) ;;
    esac
  fi

  if (( i % 3 == 0 )); then
    code="$(curl -s -o /dev/null -w "%{http_code}" \
      -X POST "${BASE_URL}/predict" \
      -H "Content-Type: application/json" \
      -d '{"records":"x"}' || true)"
    case "${code}" in
      422) ((err_422+=1)) ;;
      200) ((ok_200+=1)) ;;
      400) ((err_400+=1)) ;;
      *) ((other_codes+=1)) ;;
    esac
  fi

  if (( i % 4 == 0 )); then
    code="$(curl -s -o /dev/null -w "%{http_code}" \
      -X POST "${BASE_URL}/predict" \
      -H "Content-Type: application/json" \
      -d '{"target":1}' || true)"
    case "${code}" in
      400) ((err_400+=1)) ;;
      422) ((err_422+=1)) ;;
      200) ((ok_200+=1)) ;;
      *) ((other_codes+=1)) ;;
    esac
  fi

  if [[ "${SLEEP_S}" != "0" ]]; then
    sleep "${SLEEP_S}"
  fi

  if (( SHOW_EVERY > 0 && i % SHOW_EVERY == 0 )); then
    echo "Progresso: ${i}/${ROUNDS} rounds"
  fi
done

echo
echo "Resumo de trafego gerado:"
echo "  health hits:   ${health_hits}"
echo "  version hits:  ${version_hits}"
echo "  /predict 200:  ${ok_200}"
echo "  /predict 400:  ${err_400}"
echo "  /predict 422:  ${err_422}"
echo "  outros status: ${other_codes}"
echo

echo "Metricas-chave atuais em /metrics:"
curl -s "${BASE_URL}/metrics" | egrep "model_loaded|metadata_loaded|inference_records_total|inference_positive_total" || true
echo
echo "Dica: aguarde 30-90s e atualize o Grafana (range: Last 15 minutes)."

