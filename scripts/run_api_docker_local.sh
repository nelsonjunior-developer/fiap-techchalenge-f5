#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DOCKER_BIN="${DOCKER_BIN:-docker}"
IMAGE_NAME="${IMAGE_NAME:-fiap-ml-api}"
HOST_PORT="${HOST_PORT:-8000}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"
MODEL_MOUNT="${MODEL_MOUNT:-1}"
AUTO_BUILD="${AUTO_BUILD:-0}"

if ! command -v "${DOCKER_BIN}" >/dev/null 2>&1; then
  echo "Erro: docker nao encontrado no PATH." >&2
  exit 1
fi

if ! "${DOCKER_BIN}" info >/dev/null 2>&1; then
  echo "Erro: daemon do Docker indisponivel. Inicie o Docker e tente novamente." >&2
  exit 1
fi

if [[ ! "${HOST_PORT}" =~ ^[0-9]+$ || ! "${CONTAINER_PORT}" =~ ^[0-9]+$ ]]; then
  echo "Erro: HOST_PORT/CONTAINER_PORT devem ser inteiros." >&2
  exit 1
fi

cd "${REPO_ROOT}"

if [[ "${AUTO_BUILD}" == "1" ]]; then
  echo "AUTO_BUILD=1 -> executando docker build -t ${IMAGE_NAME} ."
  "${DOCKER_BIN}" build -t "${IMAGE_NAME}" .
fi

args=(
  run
  --rm
  -p "${HOST_PORT}:${CONTAINER_PORT}"
)

if [[ "${MODEL_MOUNT}" == "1" ]]; then
  args+=(-v "${REPO_ROOT}/app/model:/app/app/model")
fi

if [[ -n "${LOG_LEVEL:-}" ]]; then
  args+=(-e "LOG_LEVEL=${LOG_LEVEL}")
fi
if [[ -n "${ALLOW_PARTIAL_PAYLOAD:-}" ]]; then
  args+=(-e "ALLOW_PARTIAL_PAYLOAD=${ALLOW_PARTIAL_PAYLOAD}")
fi
if [[ -n "${LOG_FORMAT:-}" ]]; then
  args+=(-e "LOG_FORMAT=${LOG_FORMAT}")
fi

echo "Subindo API via Docker..."
echo "  image: ${IMAGE_NAME}"
echo "  ports: ${HOST_PORT}:${CONTAINER_PORT}"
echo "  model_mount: ${MODEL_MOUNT}"
if [[ "${MODEL_MOUNT}" == "1" ]]; then
  echo "  volume: ${REPO_ROOT}/app/model:/app/app/model"
fi
echo
echo "Exemplos de verificacao rapida:"
echo "  curl -s http://127.0.0.1:${HOST_PORT}/health | jq"
echo "  BASE_URL=http://127.0.0.1:${HOST_PORT} scripts/smoke_api_local.sh"

exec "${DOCKER_BIN}" "${args[@]}" "${IMAGE_NAME}"
