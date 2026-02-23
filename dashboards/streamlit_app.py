"""Local Streamlit app to visualize Evidently drift HTML + aggregated summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import streamlit as st
import streamlit.components.v1 as components

from src.privacy import is_safe_json_payload

DEFAULT_REPORT_PATH = "artifacts/drift_report.html"
DEFAULT_SUMMARY_PATH = "artifacts/drift_report_summary.json"
MAX_EMBED_HTML_WARN_BYTES = 20 * 1024 * 1024  # 20 MiB


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _safe_read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(_safe_read_text(path))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload (expected object): {path}")
    return payload


def _get_nested(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _extract_summary_view(summary: Mapping[str, Any]) -> dict[str, Any]:
    status = summary.get("status")
    generated_at = summary.get("generated_at")

    model_version = summary.get("model_version")
    if model_version is None:
        model_version = _get_nested(summary, "reference", "model_version")

    model_family = summary.get("model_family")
    if model_family is None:
        model_family = _get_nested(summary, "reference", "model_family")

    variant = summary.get("variant")
    if variant is None:
        variant = _get_nested(summary, "reference", "variant")

    n_reference = summary.get("n_reference")
    if n_reference is None:
        n_reference = _get_nested(summary, "reference", "n_rows")

    n_current = summary.get("n_current")
    if n_current is None:
        n_current = _get_nested(summary, "current", "n_rows")

    drifted_features_count = summary.get("drifted_features_count")
    if drifted_features_count is None:
        drifted_features_count = _get_nested(summary, "drift", "drifted_features_count")

    share_drifted_features = summary.get("share_drifted_features")
    if share_drifted_features is None:
        share_drifted_features = _get_nested(summary, "drift", "share_drifted_features")

    dataset_drift = summary.get("dataset_drift")
    if dataset_drift is None:
        dataset_drift = _get_nested(summary, "drift", "dataset_drift")

    extra_cols_dropped = _get_nested(summary, "contract", "extra_cols_dropped_count")
    n_features = _get_nested(summary, "contract", "n_features")
    notes = summary.get("notes") if isinstance(summary.get("notes"), list) else []
    errors = summary.get("errors") if isinstance(summary.get("errors"), list) else []

    return {
        "status": status,
        "generated_at": generated_at,
        "model_version": model_version,
        "model_family": model_family,
        "variant": variant,
        "n_reference": n_reference,
        "n_current": n_current,
        "drifted_features_count": drifted_features_count,
        "share_drifted_features": share_drifted_features,
        "dataset_drift": dataset_drift,
        "extra_cols_dropped_count": extra_cols_dropped,
        "n_features": n_features,
        "notes": notes,
        "errors": errors,
    }


def _render_summary(summary_payload: Mapping[str, Any], *, summary_source_label: str) -> None:
    st.subheader("Resumo (agregado)")
    if not is_safe_json_payload(summary_payload):
        st.error("Resumo JSON reprovado na checagem de privacidade. Revise o artefato.")
        return

    view = _extract_summary_view(summary_payload)
    status = str(view.get("status") or "UNKNOWN")
    st.caption(f"Fonte do resumo: `{summary_source_label}`")

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", status)
    col2.metric("Features com drift", str(view.get("drifted_features_count") or "-"))
    share = view.get("share_drifted_features")
    col3.metric(
        "Share drift",
        f"{float(share):.1%}" if isinstance(share, (int, float)) else "-",
    )

    col4, col5, col6 = st.columns(3)
    col4.metric("Linhas referência", str(view.get("n_reference") or "-"))
    col5.metric("Linhas atuais", str(view.get("n_current") or "-"))
    dataset_drift = view.get("dataset_drift")
    col6.metric(
        "Dataset drift",
        "-" if dataset_drift is None else ("Sim" if bool(dataset_drift) else "Não"),
    )

    meta_cols = st.columns(2)
    with meta_cols[0]:
        st.write("**Modelo**")
        st.write(f"- version: `{view.get('model_version') or '-'}`")
        st.write(f"- family: `{view.get('model_family') or '-'}`")
        st.write(f"- variant: `{view.get('variant') or '-'}`")
    with meta_cols[1]:
        st.write("**Contexto**")
        st.write(f"- generated_at: `{view.get('generated_at') or '-'}`")
        st.write(f"- n_features: `{view.get('n_features') or '-'}`")
        st.write(f"- extras dropped: `{view.get('extra_cols_dropped_count') or 0}`")

    notes = [str(item) for item in view.get("notes", []) if str(item).strip()]
    if notes:
        with st.expander("Notas do resumo"):
            for note in notes[:20]:
                st.write(f"- {note}")
            if len(notes) > 20:
                st.caption(f"Mostrando 20 de {len(notes)} notas.")

    errors = [str(item) for item in view.get("errors", []) if str(item).strip()]
    if errors:
        with st.expander("Erros reportados no resumo"):
            for err in errors[:20]:
                st.write(f"- {err}")
            if len(errors) > 20:
                st.caption(f"Mostrando 20 de {len(errors)} erros.")


def _render_html_report(*, html_text: str, source_label: str) -> None:
    st.subheader("Relatório HTML (Evidently)")
    html_size_bytes = len(html_text.encode("utf-8", errors="ignore"))
    st.caption(f"Fonte do HTML: `{source_label}` | tamanho: {html_size_bytes/1024:.1f} KB")
    if html_size_bytes > MAX_EMBED_HTML_WARN_BYTES:
        st.warning(
            "O HTML do relatório é grande e pode ficar pesado no navegador. "
            "Considere gerar um relatório com amostra menor (`--max-rows`)."
        )
    components.html(html_text, height=900, scrolling=True)


def _load_summary_from_path(summary_path_raw: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
    summary_path = Path(summary_path_raw).expanduser()
    if not summary_path.exists():
        return None, None, f"Resumo não encontrado em `{summary_path}` (opcional)."
    try:
        payload = _safe_read_json(summary_path)
        return payload, str(summary_path), None
    except Exception as exc:
        return None, str(summary_path), f"Falha ao ler resumo JSON: {exc}"


def _load_html_from_path(report_path_raw: str) -> tuple[str | None, str | None, str | None]:
    report_path = Path(report_path_raw).expanduser()
    if not report_path.exists():
        return None, None, (
            "Relatório HTML não encontrado. Gere primeiro com:\n\n"
            "`python -m src.drift --reference-dir app/model/reference --current-csv <...> "
            "--out-html artifacts/drift_report.html --out-json artifacts/drift_report_summary.json`"
        )
    try:
        return _safe_read_text(report_path), str(report_path), None
    except Exception as exc:
        return None, str(report_path), f"Falha ao ler HTML: {exc}"


def main() -> None:
    st.set_page_config(page_title="Drift Report (Evidently)", layout="wide")
    st.title("Drift Report (Evidently)")
    st.caption("Visualização local (read-only) do relatório HTML e resumo agregado de drift.")

    with st.sidebar:
        st.header("Configuração")
        report_path = st.text_input("Path do relatório HTML", value=DEFAULT_REPORT_PATH)
        summary_path = st.text_input("Path do resumo JSON", value=DEFAULT_SUMMARY_PATH)
        uploaded_html = st.file_uploader(
            "Upload alternativo (.html)",
            type=["html", "htm"],
            help="Se enviado, este HTML tem prioridade sobre o path local.",
        )
        _ = st.button("Recarregar", help="O clique já força reexecução do app.")
        st.caption("Dashboard local, somente leitura, sem cloud/autenticação.")

    summary_payload, summary_source, summary_error = _load_summary_from_path(summary_path)
    if summary_error:
        st.info(summary_error)
    if summary_payload is not None and summary_source is not None:
        _render_summary(summary_payload, summary_source_label=summary_source)

    st.divider()

    if uploaded_html is not None:
        try:
            html_text = uploaded_html.getvalue().decode("utf-8", errors="replace")
            source_label = f"upload:{uploaded_html.name}"
            _render_html_report(html_text=html_text, source_label=source_label)
            return
        except Exception as exc:
            st.error(f"Falha ao processar HTML enviado: {exc}")

    html_text, html_source, html_error = _load_html_from_path(report_path)
    if html_error:
        st.warning(html_error)
        return
    if html_text is None or html_source is None:
        st.warning("Nenhum relatório HTML disponível para exibição.")
        return
    _render_html_report(html_text=html_text, source_label=html_source)


if __name__ == "__main__":
    main()

