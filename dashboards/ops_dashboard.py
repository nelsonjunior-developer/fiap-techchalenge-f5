"""Operational dashboard (local, read-only) for online, drift, and offline metrics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.privacy import is_safe_json_payload

DEFAULT_ONLINE_METRICS_PATH = "logs/online_metrics.jsonl"
DEFAULT_DRIFT_HTML_PATH = "artifacts/drift_report.html"
DEFAULT_DRIFT_SUMMARY_PATH = "artifacts/drift_report_summary.json"
DEFAULT_OFFLINE_METRICS_GLOB = "artifacts/offline_metrics_*.json"
MAX_EMBED_HTML_WARN_BYTES = 20 * 1024 * 1024  # 20 MiB
MAX_POINTS_CHART = 1500


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _safe_read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(_safe_read_text(path))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload (expected object): {path}")
    return payload


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def _fmt_num(value: float | int | None, decimals: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{decimals}f}"


def _parse_ts(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_online_events(path_raw: str) -> tuple[list[dict[str, Any]], dict[str, int], str | None]:
    path = Path(path_raw).expanduser()
    if not path.exists():
        return [], {"invalid_lines": 0, "unsafe_events": 0}, (
            "Arquivo de métricas online não encontrado. Gere chamadas em `/predict` "
            "para alimentar `logs/online_metrics.jsonl`."
        )

    events: list[dict[str, Any]] = []
    invalid_lines = 0
    unsafe_events = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            invalid_lines += 1
            continue
        if not isinstance(event, dict):
            invalid_lines += 1
            continue
        if not is_safe_json_payload(event):
            unsafe_events += 1
            continue
        events.append(event)

    return events, {"invalid_lines": invalid_lines, "unsafe_events": unsafe_events}, None


def _aggregate_online(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    event_list = list(events)
    n_events = len(event_list)
    n_records_total = 0
    n_non_2xx = 0
    n_validation_errors = 0
    n_model_unavailable = 0
    n_with_positive = 0
    weighted_positive_numerator = 0.0
    weighted_positive_denominator = 0
    weighted_missing_cols_numerator = 0.0
    weighted_missing_cols_denominator = 0
    weighted_missing_values_numerator = 0.0
    weighted_missing_values_denominator = 0

    histogram_edges: list[float] | None = None
    histogram_counts: list[int] | None = None
    histogram_events = 0
    histogram_events_mismatch = 0

    positive_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for event in event_list:
        n_records = _safe_int(event.get("n_records")) or 0
        n_records_total += max(n_records, 0)

        status_code = _safe_int(event.get("status_code")) or 0
        status_family = str(event.get("status_family") or "")
        is_error = status_family != "2xx" if status_family else (status_code >= 400)
        if is_error:
            n_non_2xx += 1
        if status_code in (400, 422):
            n_validation_errors += 1
        if status_code == 503:
            n_model_unavailable += 1

        ts = _parse_ts(event.get("generated_at"))
        if ts is not None:
            error_rows.append({"ts": ts, "error_rate": 1.0 if is_error else 0.0})

        positive_rate = _safe_float(event.get("positive_rate_at_threshold"))
        if positive_rate is not None and n_records > 0:
            n_with_positive += 1
            weighted_positive_numerator += float(positive_rate) * int(n_records)
            weighted_positive_denominator += int(n_records)
            if ts is not None:
                positive_rows.append(
                    {"ts": ts, "positive_rate_at_threshold": float(positive_rate)}
                )

        missing_cols_rate = _safe_float(event.get("missing_cols_rate"))
        if missing_cols_rate is not None and n_records > 0:
            weighted_missing_cols_numerator += float(missing_cols_rate) * int(n_records)
            weighted_missing_cols_denominator += int(n_records)

        missing_values_rate = _safe_float(event.get("missing_values_rate"))
        if missing_values_rate is not None and n_records > 0:
            weighted_missing_values_numerator += float(missing_values_rate) * int(n_records)
            weighted_missing_values_denominator += int(n_records)

        histogram = event.get("score_histogram")
        if not isinstance(histogram, Mapping):
            continue
        edges = histogram.get("bin_edges")
        counts = histogram.get("bin_counts")
        if not (
            isinstance(edges, list)
            and isinstance(counts, list)
            and len(edges) >= 2
            and len(counts) == len(edges) - 1
        ):
            continue
        try:
            edges_cast = [float(value) for value in edges]
            counts_cast = [int(value) for value in counts]
        except (TypeError, ValueError):
            continue
        if histogram_edges is None:
            histogram_edges = edges_cast
            histogram_counts = [0 for _ in counts_cast]
        if edges_cast != histogram_edges or histogram_counts is None:
            histogram_events_mismatch += 1
            continue
        histogram_counts = [a + b for a, b in zip(histogram_counts, counts_cast)]
        histogram_events += 1

    positive_df = (
        pd.DataFrame(positive_rows).sort_values("ts").set_index("ts")
        if positive_rows
        else pd.DataFrame()
    )
    if len(positive_df) > MAX_POINTS_CHART:
        positive_df = positive_df.tail(MAX_POINTS_CHART)

    error_df = (
        pd.DataFrame(error_rows).sort_values("ts").set_index("ts")
        if error_rows
        else pd.DataFrame()
    )
    if len(error_df) > MAX_POINTS_CHART:
        error_df = error_df.tail(MAX_POINTS_CHART)

    histogram_total = int(sum(histogram_counts or []))
    histogram_payload = None
    if histogram_edges is not None and histogram_counts is not None:
        bins = []
        for idx, count in enumerate(histogram_counts):
            bins.append(
                {
                    "bin": f"[{histogram_edges[idx]:.1f}, {histogram_edges[idx + 1]:.1f})",
                    "count": int(count),
                }
            )
        histogram_payload = {
            "edges": histogram_edges,
            "counts": histogram_counts,
            "bins_df": pd.DataFrame(bins),
            "total": histogram_total,
            "events_with_histogram": histogram_events,
            "events_histogram_mismatch": histogram_events_mismatch,
        }

    return {
        "n_events": n_events,
        "n_records_total": int(n_records_total),
        "positive_rate_avg_weighted": _safe_ratio(
            weighted_positive_numerator, float(weighted_positive_denominator)
        ),
        "error_rate": _safe_ratio(float(n_non_2xx), float(n_events)),
        "validation_error_rate": _safe_ratio(float(n_validation_errors), float(n_events)),
        "model_unavailable_rate": _safe_ratio(float(n_model_unavailable), float(n_events)),
        "missing_cols_rate_avg_weighted": _safe_ratio(
            weighted_missing_cols_numerator,
            float(weighted_missing_cols_denominator),
        ),
        "missing_values_rate_avg_weighted": _safe_ratio(
            weighted_missing_values_numerator,
            float(weighted_missing_values_denominator),
        ),
        "events_with_positive_rate": n_with_positive,
        "positive_series_df": positive_df,
        "error_series_df": error_df,
        "histogram": histogram_payload,
    }


def _render_online_tab(online_path: str) -> None:
    events, ingest_stats, ingest_error = _load_online_events(online_path)
    if ingest_error:
        st.warning(ingest_error)
        return

    if ingest_stats["invalid_lines"] > 0:
        st.info(
            f"Linhas inválidas ignoradas no JSONL: {ingest_stats['invalid_lines']}."
        )
    if ingest_stats["unsafe_events"] > 0:
        st.warning(
            f"Eventos inseguros (privacy check) ignorados: {ingest_stats['unsafe_events']}."
        )

    if not events:
        st.warning("Nenhum evento online válido disponível.")
        return

    agg = _aggregate_online(events)
    c1, c2, c3 = st.columns(3)
    c1.metric("Eventos (requests)", str(agg["n_events"]))
    c2.metric("Registros totais", str(agg["n_records_total"]))
    c3.metric(
        "Positive rate média ponderada",
        _fmt_pct(agg["positive_rate_avg_weighted"]),
    )

    c4, c5, c6 = st.columns(3)
    c4.metric("Error rate (4xx/5xx)", _fmt_pct(agg["error_rate"]))
    c5.metric("Validation error rate (400/422)", _fmt_pct(agg["validation_error_rate"]))
    c6.metric("Model unavailable rate (503)", _fmt_pct(agg["model_unavailable_rate"]))

    c7, c8 = st.columns(2)
    c7.metric(
        "Missing cols rate média ponderada",
        _fmt_pct(agg["missing_cols_rate_avg_weighted"]),
    )
    c8.metric(
        "Missing values rate média ponderada",
        _fmt_pct(agg["missing_values_rate_avg_weighted"]),
    )

    st.divider()
    st.subheader("Séries temporais")
    positive_df = agg["positive_series_df"]
    if isinstance(positive_df, pd.DataFrame) and not positive_df.empty:
        st.caption("Positive rate por request ao longo do tempo (últimos pontos).")
        st.line_chart(positive_df[["positive_rate_at_threshold"]], use_container_width=True)
    else:
        st.info("Sem eventos com `positive_rate_at_threshold` para plotar.")

    error_df = agg["error_series_df"]
    if isinstance(error_df, pd.DataFrame) and not error_df.empty:
        st.caption("Error rate por request (0=2xx, 1=erro) ao longo do tempo.")
        st.line_chart(error_df[["error_rate"]], use_container_width=True)
    else:
        st.info("Sem timestamps válidos para série de erro.")

    st.divider()
    st.subheader("Distribuição de scores (histograma agregado)")
    histogram = agg["histogram"]
    if isinstance(histogram, Mapping):
        bins_df = histogram.get("bins_df")
        if isinstance(bins_df, pd.DataFrame) and not bins_df.empty:
            st.bar_chart(
                bins_df.set_index("bin")[["count"]],
                use_container_width=True,
            )
            st.caption(
                "Total de scores no histograma: "
                f"{histogram.get('total', 0)} | eventos com histograma: "
                f"{histogram.get('events_with_histogram', 0)}"
            )
            mismatch_count = int(histogram.get("events_histogram_mismatch", 0))
            if mismatch_count > 0:
                st.info(
                    "Eventos com binagem incompatível ignorados na soma: "
                    f"{mismatch_count}."
                )
        else:
            st.info("Histograma agregado vazio.")
    else:
        st.info("Nenhum evento com `score_histogram` disponível.")


def _render_drift_tab(summary_path_raw: str, html_path_raw: str) -> None:
    summary_path = Path(summary_path_raw).expanduser()
    html_path = Path(html_path_raw).expanduser()

    if summary_path.exists():
        try:
            summary = _safe_read_json(summary_path)
            if not is_safe_json_payload(summary):
                st.error("Resumo de drift reprovado na checagem de privacidade.")
            else:
                st.subheader("Resumo de Drift")
                c1, c2, c3 = st.columns(3)
                c1.metric("Status", str(summary.get("status") or "-"))
                c2.metric(
                    "Share drifted features",
                    _fmt_pct(_safe_float(summary.get("share_drifted_features"))),
                )
                c3.metric(
                    "Drifted features",
                    _fmt_num(_safe_int(summary.get("drifted_features_count")), decimals=0),
                )

                c4, c5, c6 = st.columns(3)
                c4.metric("Model version", str(summary.get("model_version") or "-"))
                c5.metric("N referência", _fmt_num(_safe_int(summary.get("n_reference")), 0))
                c6.metric("N atual", _fmt_num(_safe_int(summary.get("n_current")), 0))

                st.caption(f"generated_at: `{summary.get('generated_at') or '-'}`")
        except Exception as exc:
            st.warning(f"Falha ao ler resumo de drift: {exc}")
    else:
        st.info(f"Resumo de drift não encontrado em `{summary_path}` (opcional).")

    st.divider()
    st.subheader("Relatório HTML (Evidently)")
    if not html_path.exists():
        st.warning(
            "Relatório de drift HTML não encontrado. Gere com:\n\n"
            "`python -m src.drift --reference-dir app/model/reference "
            "--current-csv <...> --out-html artifacts/drift_report.html "
            "--out-json artifacts/drift_report_summary.json`"
        )
        return

    html_text = _safe_read_text(html_path)
    html_size_bytes = len(html_text.encode("utf-8", errors="ignore"))
    st.caption(f"Fonte: `{html_path}` | tamanho: {html_size_bytes/1024:.1f} KB")
    if html_size_bytes > MAX_EMBED_HTML_WARN_BYTES:
        st.warning(
            "HTML grande; pode ficar pesado no navegador. "
            "Considere reduzir `--max-rows` na geração."
        )
    components.html(html_text, height=900, scrolling=True)


def _discover_offline_metrics(glob_pattern: str) -> list[Path]:
    paths = sorted(
        [Path(p) for p in Path(".").glob(glob_pattern)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    fallback = Path("artifacts/offline_metrics.json")
    if fallback.exists() and fallback not in paths:
        paths.append(fallback)
    return paths


def _render_offline_tab(glob_pattern: str) -> None:
    candidates = _discover_offline_metrics(glob_pattern)
    if not candidates:
        st.warning(
            "Nenhum arquivo de métricas offline encontrado. Gere com:\n\n"
            "`python -m src.offline_evaluation --dataset-path <xlsx> "
            "--year-t 2023 --year-t1 2024 "
            "--out-json artifacts/offline_metrics_2023_2024.json`"
        )
        return

    choice = st.selectbox(
        "Arquivo de métricas offline",
        options=[str(path) for path in candidates],
        index=0,
    )
    chosen_path = Path(choice)
    try:
        payload = _safe_read_json(chosen_path)
    except Exception as exc:
        st.error(f"Falha ao ler arquivo offline: {exc}")
        return
    if not is_safe_json_payload(payload):
        st.error("Arquivo offline reprovado na checagem de privacidade.")
        return

    pair = payload.get("pair")
    pair_label = "-"
    if isinstance(pair, Mapping):
        year_t = pair.get("year_t")
        year_t1 = pair.get("year_t1")
        if year_t is not None and year_t1 is not None:
            pair_label = f"{year_t}->{year_t1}"

    metrics = payload.get("metrics_at_operational_threshold")
    metrics_dict = metrics if isinstance(metrics, Mapping) else {}
    confusion = metrics_dict.get("confusion_matrix")
    confusion_dict = confusion if isinstance(confusion, Mapping) else {}
    model = payload.get("model")
    model_dict = model if isinstance(model, Mapping) else {}

    c1, c2, c3 = st.columns(3)
    c1.metric("Pair", pair_label)
    c2.metric("Status", str(payload.get("status") or "-"))
    c3.metric(
        "Threshold operacional",
        _fmt_num(
            _safe_float(payload.get("threshold_operational"))
            or _safe_float(metrics_dict.get("threshold")),
            decimals=3,
        ),
    )

    c4, c5, c6 = st.columns(3)
    c4.metric("Recall", _fmt_num(_safe_float(metrics_dict.get("recall"))))
    c5.metric("PR-AUC", _fmt_num(_safe_float(metrics_dict.get("pr_auc"))))
    c6.metric("ROC-AUC", _fmt_num(_safe_float(metrics_dict.get("roc_auc"))))

    c7, c8, c9 = st.columns(3)
    c7.metric("Precision", _fmt_num(_safe_float(metrics_dict.get("precision"))))
    c8.metric("F1", _fmt_num(_safe_float(metrics_dict.get("f1"))))
    c9.metric("Prevalence", _fmt_pct(_safe_float(metrics_dict.get("prevalence"))))

    c10, c11 = st.columns(2)
    c10.metric("N avaliado", _fmt_num(_safe_int(metrics_dict.get("n")), 0))
    c11.metric("Model version", str(model_dict.get("model_version") or "-"))

    st.subheader("Matriz de confusão")
    if confusion_dict:
        matrix_df = pd.DataFrame(
            [
                {"term": "TN", "value": _safe_int(confusion_dict.get("tn")) or 0},
                {"term": "FP", "value": _safe_int(confusion_dict.get("fp")) or 0},
                {"term": "FN", "value": _safe_int(confusion_dict.get("fn")) or 0},
                {"term": "TP", "value": _safe_int(confusion_dict.get("tp")) or 0},
            ]
        )
        st.table(matrix_df)
    else:
        st.info("Matriz de confusão não encontrada neste arquivo.")

    notes = payload.get("notes")
    if isinstance(notes, list) and notes:
        with st.expander("Notas"):
            for note in notes[:20]:
                st.write(f"- {note}")


def _render_runbook_tab() -> None:
    st.markdown(
        """
### Comandos úteis (local)

Gerar drift:
```bash
python -m src.drift --reference-dir app/model/reference --current-csv <...> --out-html artifacts/drift_report.html --out-json artifacts/drift_report_summary.json
```

Gerar métricas pós-fato:
```bash
python -m src.offline_evaluation --dataset-path <xlsx> --year-t 2023 --year-t1 2024 --out-json artifacts/offline_metrics_2023_2024.json --out-md artifacts/offline_metrics_2023_2024.md
```

Rodar retenção:
```bash
python -m src.retention --dry-run 1
```

Dry-run de retreino automatizado:
```bash
python -m src.retrain_orchestrator --execute 0
```
"""
    )


def main() -> None:
    st.set_page_config(page_title="Ops Dashboard (Local)", layout="wide")
    st.title("Dashboard Operacional Consolidado")
    st.caption(
        "Local-only, read-only, sem cloud. Exibe apenas agregados de inferência, drift e métricas pós-fato."
    )

    with st.sidebar:
        st.header("Entradas")
        online_metrics_path = st.text_input(
            "Online metrics JSONL",
            value=DEFAULT_ONLINE_METRICS_PATH,
        )
        drift_html_path = st.text_input(
            "Drift report HTML",
            value=DEFAULT_DRIFT_HTML_PATH,
        )
        drift_summary_path = st.text_input(
            "Drift summary JSON",
            value=DEFAULT_DRIFT_SUMMARY_PATH,
        )
        offline_metrics_glob = st.text_input(
            "Offline metrics glob",
            value=DEFAULT_OFFLINE_METRICS_GLOB,
        )
        _ = st.button("Recarregar", help="A interação já força reexecução do app.")

    tab_online, tab_drift, tab_offline, tab_runbook = st.tabs(
        [
            "Online Inference",
            "Drift",
            "Métricas Pós-Fato (Offline)",
            "Runbook",
        ]
    )
    with tab_online:
        _render_online_tab(online_metrics_path)
    with tab_drift:
        _render_drift_tab(drift_summary_path, drift_html_path)
    with tab_offline:
        _render_offline_tab(offline_metrics_glob)
    with tab_runbook:
        _render_runbook_tab()


if __name__ == "__main__":
    main()
