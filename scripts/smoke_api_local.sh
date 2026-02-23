#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-5}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIRE_PREDICT_200="${REQUIRE_PREDICT_200:-0}"

if ! command -v curl >/dev/null 2>&1; then
  echo "Erro: curl nao encontrado no PATH." >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Erro: interpretador Python nao encontrado: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! "${TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "Erro: TIMEOUT_SECONDS invalido (${TIMEOUT_SECONDS})." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

_http_request() {
  local method="$1"
  local url="$2"
  local body_file="$3"
  local headers_file="$4"
  local payload_file="${5:-}"

  local -a curl_args=(
    -sS
    --max-time "${TIMEOUT_SECONDS}"
    -X "${method}"
    -D "${headers_file}"
    -o "${body_file}"
    -w "%{http_code}"
  )

  if [[ -n "${payload_file}" ]]; then
    curl_args+=(-H "Content-Type: application/json" --data-binary "@${payload_file}")
  fi

  curl "${curl_args[@]}" "${url}"
}

_require_status() {
  local got="$1"
  local expected="$2"
  local label="$3"
  if [[ "${got}" != "${expected}" ]]; then
    echo "Falha em ${label}: esperado ${expected}, recebido ${got}" >&2
    return 1
  fi
}

_extract_header() {
  local headers_file="$1"
  local header_name="$2"
  awk -v name="${header_name}" '
    BEGIN { IGNORECASE = 1 }
    $0 ~ ("^" name ":") {
      gsub("\r", "", $0)
      sub("^[^:]+:[[:space:]]*", "", $0)
      print $0
      exit
    }
  ' "${headers_file}"
}

echo "Smoke test da API local"
echo "  BASE_URL=${BASE_URL}"
echo "  TIMEOUT_SECONDS=${TIMEOUT_SECONDS}"
echo

HEALTH_BODY="${TMP_DIR}/health.json"
HEALTH_HEADERS="${TMP_DIR}/health.headers"
HEALTH_CODE="$(_http_request "GET" "${BASE_URL}/health" "${HEALTH_BODY}" "${HEALTH_HEADERS}")"
_require_status "${HEALTH_CODE}" "200" "GET /health"
"${PYTHON_BIN}" - <<'PY' "${HEALTH_BODY}"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)
if payload.get("status") != "ok":
    raise SystemExit("GET /health retornou payload inesperado")
print("GET /health -> 200 OK")
PY

VERSION_BODY="${TMP_DIR}/version.json"
VERSION_HEADERS="${TMP_DIR}/version.headers"
VERSION_CODE="$(_http_request "GET" "${BASE_URL}/version" "${VERSION_BODY}" "${VERSION_HEADERS}")"
_require_status "${VERSION_CODE}" "200" "GET /version"
VERSION_PARSED="$("${PYTHON_BIN}" - <<'PY' "${VERSION_BODY}"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)
model_loaded = 1 if bool(payload.get("model_loaded")) else 0
metadata_loaded = 1 if bool(payload.get("metadata_loaded")) else 0
model_version = str(payload.get("model_version", "unknown"))
threshold = payload.get("threshold_operational", "")
print(f"{model_loaded}\t{metadata_loaded}\t{model_version}\t{threshold}")
PY
)"
IFS=$'\t' read -r VERSION_MODEL_LOADED VERSION_METADATA_LOADED VERSION_MODEL_VERSION VERSION_THRESHOLD <<<"${VERSION_PARSED}"
echo "GET /version -> 200 OK (model_loaded=${VERSION_MODEL_LOADED}, metadata_loaded=${VERSION_METADATA_LOADED}, model_version=${VERSION_MODEL_VERSION}, threshold=${VERSION_THRESHOLD})"

# Probe de rota/validacao HTTP independente do modelo.
INVALID_422_PAYLOAD="${TMP_DIR}/invalid_422.json"
printf '%s' '{"records":"x"}' >"${INVALID_422_PAYLOAD}"
P422_BODY="${TMP_DIR}/predict_422.json"
P422_HEADERS="${TMP_DIR}/predict_422.headers"
P422_CODE="$(_http_request "POST" "${BASE_URL}/predict" "${P422_BODY}" "${P422_HEADERS}" "${INVALID_422_PAYLOAD}")"
_require_status "${P422_CODE}" "422" "POST /predict (body invalido)"
P422_REQUEST_ID="$(_extract_header "${P422_HEADERS}" "X-Request-ID" || true)"
echo "POST /predict (body invalido) -> 422 OK${P422_REQUEST_ID:+ | X-Request-ID=${P422_REQUEST_ID}}"

# Payload sintetico para probe funcional do /predict.
PREDICT_PAYLOAD="${TMP_DIR}/predict_payload.json"
LOCAL_METADATA="${REPO_ROOT}/app/model/metadata.json"
if [[ -f "${LOCAL_METADATA}" ]]; then
  "${PYTHON_BIN}" - <<'PY' "${LOCAL_METADATA}" "${PREDICT_PAYLOAD}"
import json, sys
meta_path, out_path = sys.argv[1], sys.argv[2]
with open(meta_path, "r", encoding="utf-8") as fh:
    meta = json.load(fh)
cols = meta.get("expected_raw_cols")
payload = {}
if isinstance(cols, list):
    for col in cols:
        if isinstance(col, str) and col.strip():
            payload[col] = None
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(payload if payload else {"coluna_1": 1}, fh, ensure_ascii=False)
PY
else
  printf '%s' '{"coluna_1": 1}' >"${PREDICT_PAYLOAD}"
fi

PREDICT_BODY="${TMP_DIR}/predict.json"
PREDICT_HEADERS="${TMP_DIR}/predict.headers"
PREDICT_CODE="$(_http_request "POST" "${BASE_URL}/predict" "${PREDICT_BODY}" "${PREDICT_HEADERS}" "${PREDICT_PAYLOAD}")"
PREDICT_REQUEST_ID="$(_extract_header "${PREDICT_HEADERS}" "X-Request-ID" || true)"

if [[ "${VERSION_MODEL_LOADED}" == "0" || "${VERSION_METADATA_LOADED}" == "0" ]]; then
  _require_status "${PREDICT_CODE}" "503" "POST /predict (sem modelo/metadata)"
  echo "POST /predict -> 503 OK (esperado sem modelo/metadata)${PREDICT_REQUEST_ID:+ | X-Request-ID=${PREDICT_REQUEST_ID}}"
else
  if [[ "${REQUIRE_PREDICT_200}" == "1" ]]; then
    _require_status "${PREDICT_CODE}" "200" "POST /predict (modo strict REQUIRE_PREDICT_200=1)"
    echo "POST /predict -> 200 OK${PREDICT_REQUEST_ID:+ | X-Request-ID=${PREDICT_REQUEST_ID}}"
  else
    case "${PREDICT_CODE}" in
      200)
        echo "POST /predict -> 200 OK${PREDICT_REQUEST_ID:+ | X-Request-ID=${PREDICT_REQUEST_ID}}"
        ;;
      400)
        echo "POST /predict -> 400 (rota acessivel; payload sintetico nao aceito pelo contrato/politica atual)${PREDICT_REQUEST_ID:+ | X-Request-ID=${PREDICT_REQUEST_ID}}"
        ;;
      *)
        echo "Falha em POST /predict: esperado 200 ou 400 com modelo/metadata disponiveis; recebido ${PREDICT_CODE}" >&2
        echo "Resumo do body:" >&2
        "${PYTHON_BIN}" - <<'PY' "${PREDICT_BODY}" >&2 || true
import json, sys
from pathlib import Path
raw = Path(sys.argv[1]).read_text(encoding="utf-8")
try:
    payload = json.loads(raw)
except Exception:
    print(raw[:500])
else:
    print(json.dumps(payload, ensure_ascii=False)[:1000])
PY
        exit 1
        ;;
    esac
  fi
fi

echo
echo "Smoke test concluido com sucesso."
echo "Resultado esperado minimo para 'API acessivel localmente': GET /health=200 e GET /version=200."
