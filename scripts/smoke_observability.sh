#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DOCKER_BIN="${DOCKER_BIN:-docker}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.observability.yml}"
TIMEOUT_S="${TIMEOUT_S:-180}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-2}"
NO_BUILD="${NO_BUILD:-0}"
KEEP_UP="${KEEP_UP:-0}"
API_PORT="${API_PORT:-8000}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"

compose() {
  "${DOCKER_BIN}" compose -f "${COMPOSE_FILE}" "$@"
}

cleanup() {
  if [[ "${KEEP_UP}" == "1" ]]; then
    echo "KEEP_UP=1 -> mantendo stack em execucao."
    return 0
  fi
  echo "Encerrando stack de observabilidade..."
  compose down --remove-orphans >/dev/null 2>&1 || true
}

wait_http_ok() {
  local name="$1"
  local url="$2"
  local deadline=$((SECONDS + TIMEOUT_S))

  while (( SECONDS < deadline )); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "OK: ${name} (${url})"
      return 0
    fi
    sleep "${POLL_INTERVAL_S}"
  done

  echo "Erro: timeout aguardando ${name} (${url})." >&2
  return 1
}

wait_prometheus_target_up() {
  local deadline=$((SECONDS + TIMEOUT_S))
  local query_url="http://127.0.0.1:${PROMETHEUS_PORT}/api/v1/query?query=up%7Bjob%3D%22ml-api%22%7D"

  while (( SECONDS < deadline )); do
    if payload="$(curl -fsS "${query_url}" 2>/dev/null)"; then
      if python - "${payload}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
if payload.get("status") != "success":
    raise SystemExit(1)
for item in payload.get("data", {}).get("result", []):
    value = item.get("value")
    if isinstance(value, list) and len(value) >= 2:
        if str(value[1]) == "1":
            raise SystemExit(0)
raise SystemExit(1)
PY
      then
        echo "OK: Prometheus scrape job ml-api com target UP."
        return 0
      fi
    fi
    sleep "${POLL_INTERVAL_S}"
  done

  echo "Erro: Prometheus nao confirmou target ml-api UP." >&2
  return 1
}

wait_grafana_ready() {
  local deadline=$((SECONDS + TIMEOUT_S))
  local health_url="http://127.0.0.1:${GRAFANA_PORT}/api/health"

  while (( SECONDS < deadline )); do
    if payload="$(curl -fsS "${health_url}" 2>/dev/null)"; then
      if python - "${payload}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
database = str(payload.get("database", "")).lower()
raise SystemExit(0 if database == "ok" else 1)
PY
      then
        echo "OK: Grafana health database=ok."
        return 0
      fi
    fi
    sleep "${POLL_INTERVAL_S}"
  done

  echo "Erro: Grafana nao ficou saudavel no tempo esperado." >&2
  return 1
}

wait_grafana_dashboard() {
  local deadline=$((SECONDS + TIMEOUT_S))
  local search_url="http://127.0.0.1:${GRAFANA_PORT}/api/search?query=ML%20API%20Observability"

  while (( SECONDS < deadline )); do
    if payload="$(curl -fsS -u admin:admin "${search_url}" 2>/dev/null)"; then
      if python - "${payload}" <<'PY'
import json
import sys

items = json.loads(sys.argv[1])
if not isinstance(items, list):
    raise SystemExit(1)
for item in items:
    title = str(item.get("title", ""))
    if "ML API Observability (Local)" in title:
        raise SystemExit(0)
raise SystemExit(1)
PY
      then
        echo "OK: Dashboard provisionado encontrado no Grafana."
        return 0
      fi
    fi
    sleep "${POLL_INTERVAL_S}"
  done

  echo "Erro: dashboard provisionado nao encontrado no Grafana." >&2
  return 1
}

if ! command -v "${DOCKER_BIN}" >/dev/null 2>&1; then
  echo "Erro: docker nao encontrado no PATH." >&2
  exit 1
fi

if ! "${DOCKER_BIN}" info >/dev/null 2>&1; then
  echo "Erro: daemon Docker indisponivel. Inicie o Docker e tente novamente." >&2
  exit 1
fi

trap cleanup EXIT

cd "${REPO_ROOT}"

up_args=(up -d)
if [[ "${NO_BUILD}" != "1" ]]; then
  up_args+=(--build)
else
  up_args+=(--no-build)
fi

echo "Subindo stack de observabilidade..."
compose "${up_args[@]}"

wait_http_ok "api health" "http://127.0.0.1:${API_PORT}/health"
wait_http_ok "api metrics" "http://127.0.0.1:${API_PORT}/metrics"
wait_http_ok "prometheus ui" "http://127.0.0.1:${PROMETHEUS_PORT}/-/ready"
wait_http_ok "grafana login" "http://127.0.0.1:${GRAFANA_PORT}/login"

echo "Gerando trafego basico na API..."
curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null
curl -fsS "http://127.0.0.1:${API_PORT}/version" >/dev/null
predict_status="$(curl -sS -o /tmp/observability-predict-response.json -w "%{http_code}" \
  -X POST "http://127.0.0.1:${API_PORT}/predict" \
  -H "content-type: application/json" \
  -d '{"records":[{"a":1}]}' || true)"
case "${predict_status}" in
  200|400|422|503)
    echo "OK: /predict respondeu com status esperado (${predict_status})."
    ;;
  *)
    echo "Erro: /predict respondeu com status inesperado (${predict_status})." >&2
    exit 1
    ;;
esac

wait_prometheus_target_up
wait_grafana_ready
wait_grafana_dashboard

echo "Smoke de observabilidade concluido com sucesso."
