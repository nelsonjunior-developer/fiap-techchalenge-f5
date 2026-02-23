#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv-dashboard}"
PYTHON_BIN="${PYTHON_BIN:-${2:-python3}}"

_version_ge() {
  local current="$1"
  local minimum="$2"
  local IFS=.
  local c1 c2 c3 m1 m2 m3
  read -r c1 c2 c3 <<<"${current}"
  read -r m1 m2 m3 <<<"${minimum}"
  c1="${c1:-0}"; c2="${c2:-0}"; c3="${c3:-0}"
  m1="${m1:-0}"; m2="${m2:-0}"; m3="${m3:-0}"
  if (( c1 > m1 )); then return 0; fi
  if (( c1 < m1 )); then return 1; fi
  if (( c2 > m2 )); then return 0; fi
  if (( c2 < m2 )); then return 1; fi
  if (( c3 >= m3 )); then return 0; fi
  return 1
}

if [[ -z "${VENV_DIR}" ]]; then
  echo "Uso: scripts/bootstrap_dashboard_env.sh [venv_dir] [python_bin]" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Erro: interpretador Python nao encontrado: ${PYTHON_BIN}" >&2
  echo "Use um Python compativel, ex.: PYTHON_BIN=python3.11 scripts/bootstrap_dashboard_env.sh" >&2
  exit 1
fi

PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
if [[ -z "${PYTHON_VERSION}" ]]; then
  echo "Erro: nao foi possivel detectar a versao do Python em ${PYTHON_BIN}" >&2
  exit 1
fi

# Streamlit 1.39.x (pin atual) tem restricoes por versao de Python.
if [[ "${PYTHON_VERSION}" == "3.9.7" ]]; then
  echo "Erro: Python ${PYTHON_VERSION} nao e compativel com streamlit==1.39.0 (pin atual do dashboard)." >&2
  echo "Use Python 3.11 (recomendado)." >&2
  echo "Exemplos:" >&2
  echo "  PYTHON_BIN=python3.11 scripts/bootstrap_dashboard_env.sh" >&2
  echo "  scripts/bootstrap_dashboard_env.sh .venv-dashboard python3.11" >&2
  echo "  PYENV_VERSION=3.11.9 scripts/bootstrap_dashboard_env.sh .venv-dashboard python3.11" >&2
  exit 1
fi

if ! _version_ge "${PYTHON_VERSION}" "3.8.0"; then
  echo "Erro: Python ${PYTHON_VERSION} nao e compativel com a stack de dashboard atual." >&2
  echo "Minimo suportado pelo pin atual de Streamlit: 3.8+ (com excecoes)." >&2
  echo "Use Python 3.11 (recomendado)." >&2
  exit 1
fi

echo "Usando Python ${PYTHON_VERSION} (${PYTHON_BIN}) para criar ${VENV_DIR}"
echo "Recomendacao: Python 3.11 para a stack de dashboard (Streamlit + Evidently)."

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements-dashboard.txt

echo
echo "Ambiente de dashboard pronto: ${VENV_DIR}"
echo "Ative com: source ${VENV_DIR}/bin/activate"
echo "Depois rode: streamlit run dashboards/streamlit_app.py"
echo "Motivo: isola Streamlit/Evidently do ambiente principal da API/treino."
