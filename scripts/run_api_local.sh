#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${VENV_DIR:-.venv}"
DEFAULT_PYTHON_BIN="${REPO_ROOT}/${VENV_DIR}/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-1}"
APP_MODULE="${APP_MODULE:-app.main:app}"

if [[ -x "${PYTHON_BIN}" ]]; then
  :
elif command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
else
  echo "Erro: Python nao encontrado para subir a API." >&2
  echo "Tentado: ${PYTHON_BIN}" >&2
  echo "Sugestao: crie a venv e instale dependencias:" >&2
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import uvicorn" >/dev/null 2>&1; then
  echo "Erro: uvicorn nao esta disponivel no Python selecionado: ${PYTHON_BIN}" >&2
  echo "Instale as dependencias no ambiente alvo (ex.: requirements-dev.txt)." >&2
  exit 1
fi

if [[ ! "${PORT}" =~ ^[0-9]+$ ]]; then
  echo "Erro: PORT invalido (${PORT}). Use um inteiro, ex.: 8000" >&2
  exit 1
fi

cd "${REPO_ROOT}"

cmd=("${PYTHON_BIN}" -m uvicorn "${APP_MODULE}" --host "${HOST}" --port "${PORT}")
if [[ "${RELOAD}" == "1" ]]; then
  cmd+=(--reload)
fi

echo "Subindo API local..."
echo "  app: ${APP_MODULE}"
echo "  host: ${HOST}"
echo "  port: ${PORT}"
echo "  reload: ${RELOAD}"
echo "  python: ${PYTHON_BIN}"
echo "  cwd: ${REPO_ROOT}"
if [[ -n "${LOG_LEVEL:-}" ]]; then
  echo "  LOG_LEVEL: ${LOG_LEVEL}"
fi
if [[ -n "${ALLOW_PARTIAL_PAYLOAD:-}" ]]; then
  echo "  ALLOW_PARTIAL_PAYLOAD: ${ALLOW_PARTIAL_PAYLOAD}"
fi
echo
echo "Exemplos de verificacao rapida:"
echo "  curl -s http://${HOST}:${PORT}/health | jq"
echo "  curl -s http://${HOST}:${PORT}/version | jq"

exec "${cmd[@]}"
