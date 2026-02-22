"""Generate an auditable champion-model justification from model_selection.json."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.privacy import find_forbidden_json_keys
from src.utils import get_logger, setup_logging

_logger = get_logger(__name__)


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _threshold_label(value: float | None) -> str:
    if value is None:
        return "-"
    threshold = float(value)
    if abs(threshold - 0.30) < 1e-9:
        return "0.30"
    if abs(threshold - 0.50) < 1e-9:
        return "0.5"
    return f"{threshold:.2f}"


def _winner_key_from_selection(selection: dict[str, Any]) -> tuple[str, str] | None:
    winner = selection.get("winner")
    if not isinstance(winner, dict):
        return None
    family = str(winner.get("model_family") or "").strip()
    variant = str(winner.get("variant") or "").strip()
    if not family or not variant:
        return None
    return family, variant


def _find_winner_row(selection: dict[str, Any]) -> dict[str, Any] | None:
    ranked = selection.get("ranked_candidates")
    if not isinstance(ranked, list):
        return None

    winner_key = _winner_key_from_selection(selection)
    if winner_key is None:
        return None
    winner_family, winner_variant = winner_key
    for row in ranked:
        if not isinstance(row, dict):
            continue
        if (
            str(row.get("model_family", "")) == winner_family
            and str(row.get("variant", "")) == winner_variant
        ):
            return row
    return None


def _find_runner_up_row(selection: dict[str, Any], winner_row: dict[str, Any] | None) -> dict[str, Any] | None:
    ranked = selection.get("ranked_candidates")
    if not isinstance(ranked, list):
        return None
    winner_key = None
    if isinstance(winner_row, dict):
        winner_key = (
            str(winner_row.get("model_family", "")),
            str(winner_row.get("variant", "")),
        )

    normalized_rows: list[dict[str, Any]] = []
    for row in ranked:
        if not isinstance(row, dict):
            continue
        normalized_rows.append(row)
    normalized_rows.sort(
        key=lambda row: (
            int(row.get("rank", 999999) or 999999),
            str(row.get("model_family", "")),
            str(row.get("variant", "")),
        )
    )

    for row in normalized_rows:
        key = (str(row.get("model_family", "")), str(row.get("variant", "")))
        if winner_key is not None and key == winner_key:
            continue
        if row.get("eligible") is True:
            return row
    return None


def _build_ranking_explanation(
    winner_row: dict[str, Any] | None,
    runner_up_row: dict[str, Any] | None,
) -> str:
    if not isinstance(winner_row, dict):
        return "Nao foi possivel identificar um vencedor elegivel no ranking."

    winner_name = f"{winner_row.get('model_family')}/{winner_row.get('variant')}"
    if not isinstance(runner_up_row, dict):
        return f"{winner_name} foi o unico candidato elegivel com metricas completas no holdout."

    runner_name = f"{runner_up_row.get('model_family')}/{runner_up_row.get('variant')}"
    winner_metrics = winner_row.get("metrics_holdout", {})
    runner_metrics = runner_up_row.get("metrics_holdout", {})
    if not isinstance(winner_metrics, dict) or not isinstance(runner_metrics, dict):
        return f"{winner_name} venceu no ranking deterministico; runner-up: {runner_name}."

    winner_recall = _to_float_or_none(winner_metrics.get("recall"))
    runner_recall = _to_float_or_none(runner_metrics.get("recall"))
    winner_pr_auc = _to_float_or_none(winner_metrics.get("pr_auc"))
    runner_pr_auc = _to_float_or_none(runner_metrics.get("pr_auc"))
    winner_positive_rate = _to_float_or_none(winner_metrics.get("positive_rate"))
    runner_positive_rate = _to_float_or_none(runner_metrics.get("positive_rate"))
    eps = 1e-9

    if winner_recall is not None and runner_recall is not None and winner_recall > runner_recall + eps:
        return (
            f"{winner_name} ganhou por maior Recall no holdout "
            f"({winner_recall:.4f} vs {runner_recall:.4f}). "
            f"Runner-up: {runner_name}."
        )
    if (
        winner_recall is not None
        and runner_recall is not None
        and abs(winner_recall - runner_recall) <= eps
        and winner_pr_auc is not None
        and runner_pr_auc is not None
        and winner_pr_auc > runner_pr_auc + eps
    ):
        return (
            f"Recall empatado; {winner_name} ganhou por maior PR-AUC "
            f"({winner_pr_auc:.4f} vs {runner_pr_auc:.4f}). "
            f"Runner-up: {runner_name}."
        )
    if (
        winner_recall is not None
        and runner_recall is not None
        and abs(winner_recall - runner_recall) <= eps
        and winner_pr_auc is not None
        and runner_pr_auc is not None
        and abs(winner_pr_auc - runner_pr_auc) <= eps
        and winner_positive_rate is not None
        and runner_positive_rate is not None
        and winner_positive_rate < runner_positive_rate - eps
    ):
        return (
            f"Recall e PR-AUC empatados; {winner_name} ganhou por menor positive_rate "
            f"({winner_positive_rate:.4f} vs {runner_positive_rate:.4f}). "
            f"Runner-up: {runner_name}."
        )
    return (
        f"{winner_name} venceu pelo ranking deterministico configurado "
        "(Recall, PR-AUC, positive_rate e desempate lexicografico). "
        f"Runner-up: {runner_name}."
    )


def _coalesced_notes(selection: dict[str, Any]) -> list[str]:
    merged: list[str] = []
    for key in ("notes", "warnings", "errors"):
        raw = selection.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            text = str(item).strip()
            if text and text not in merged:
                merged.append(text)
    return merged


def _build_warning_reason_summary(selection: dict[str, Any]) -> str:
    warnings = selection.get("warnings")
    warning_text = " ".join(str(item) for item in warnings) if isinstance(warnings, list) else ""
    lowered = warning_text.lower()

    reasons: list[str] = []
    if "fallback threshold 0.5 used" in lowered or "fallback_to_threshold_0.5_used" in lowered:
        reasons.append("parte dos candidatos nao tinha avaliacao em threshold=0.30 (fallback para 0.5)")
    if "no candidate passed holdout gates" in lowered:
        reasons.append("nenhum candidato passou todos os gates minimos")
    if "missing holdout metrics" in lowered:
        reasons.append("alguns candidatos foram excluidos por ausencia de metricas de holdout")

    if reasons:
        return " e/ou ".join(reasons)
    if warning_text.strip():
        return warning_text.strip()
    return "houve condicoes de warning na selecao formal; consultar notas para detalhes."


def _build_decision_applied_text(status: str) -> str:
    if str(status).upper() == "WARNING":
        return (
            "selecionado o melhor por ranking (Recall principal, PR-AUC desempate), "
            "com transparencia de fallback e trade-offs."
        )
    if str(status).upper() == "PASS":
        return "selecionado o melhor candidato elegivel pelo ranking oficial e gates minimos."
    return "nao foi possivel confirmar uma decisao valida; revisar erros e rerodar a selecao."


def load_model_selection(path: str | Path = "artifacts/model_selection.json") -> dict[str, Any]:
    """Load model selection artifact used as the single source for justification."""
    selection_path = Path(path)
    if not selection_path.exists():
        raise FileNotFoundError(f"Selection artifact not found: {selection_path}")
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid selection payload: {selection_path}")
    return payload


def build_model_justification(selection: dict[str, Any]) -> dict[str, Any]:
    """Build compact, auditable model justification data from selection payload."""
    status = str(selection.get("status", "FAIL")).upper()
    criteria = selection.get("selection_criteria", {})
    if not isinstance(criteria, dict):
        criteria = {}

    winner = selection.get("winner", {})
    if not isinstance(winner, dict):
        winner = {}
    winner_key = _winner_key_from_selection(selection)
    winner_row = _find_winner_row(selection)
    runner_up_row = _find_runner_up_row(selection, winner_row)
    notes = _coalesced_notes(selection)
    errors: list[str] = []

    if winner_key is None:
        errors.append("winner key missing in selection")
    if winner_key is not None and winner_row is None:
        errors.append("winner not found in ranked_candidates")

    threshold_used = None
    if isinstance(winner_row, dict):
        threshold_used = _to_float_or_none(winner_row.get("threshold_used"))

    threshold_reason = "preferred_operational_threshold_0.30"
    if threshold_used is not None and abs(threshold_used - 0.30) > 1e-9:
        threshold_reason = "fallback_to_threshold_0.5_due_to_missing_or_incomplete_holdout_at_0.30"
    if threshold_used is None and errors:
        threshold_reason = "threshold_unavailable_due_to_inconsistent_selection_artifact"

    winner_metrics = winner.get("metrics_holdout", {})
    if not isinstance(winner_metrics, dict):
        winner_metrics = {}
    if isinstance(winner_row, dict):
        row_metrics = winner_row.get("metrics_holdout")
        if isinstance(row_metrics, dict):
            winner_metrics = row_metrics

    min_recall = _to_float_or_none(criteria.get("min_recall_holdout"))
    min_pr_auc = _to_float_or_none(criteria.get("min_pr_auc_holdout"))
    winner_recall = _to_float_or_none(winner_metrics.get("recall"))
    winner_pr_auc = _to_float_or_none(winner_metrics.get("pr_auc"))

    passed_gates = bool(winner_row.get("passed_gates")) if isinstance(winner_row, dict) else False
    if (
        min_recall is not None
        and min_pr_auc is not None
        and winner_recall is not None
        and winner_pr_auc is not None
    ):
        passed_gates = winner_recall >= min_recall and winner_pr_auc >= min_pr_auc

    if errors:
        status = "FAIL"
        for error in errors:
            if error not in notes:
                notes.append(error)

    justification = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_generated_at": selection.get("generated_at"),
        "status": status,
        "winner": {
            "family": str(winner.get("model_family") or ""),
            "variant": str(winner.get("variant") or ""),
            "path_model": str(winner.get("path_model") or ""),
            "path_metadata": str(winner.get("path_metadata") or ""),
        },
        "chosen_threshold": {
            "value": threshold_used,
            "label": _threshold_label(threshold_used),
            "reason": threshold_reason,
            "is_fallback": bool(threshold_used is not None and abs(threshold_used - 0.30) > 1e-9),
        },
        "key_metrics_holdout": {
            "recall": winner_recall,
            "pr_auc": winner_pr_auc,
            "positive_rate": _to_float_or_none(winner_metrics.get("positive_rate")),
            "precision": _to_float_or_none(winner_metrics.get("precision")),
            "f1": _to_float_or_none(winner_metrics.get("f1")),
            "roc_auc": _to_float_or_none(winner_metrics.get("roc_auc")),
            "confusion_matrix": (
                winner_metrics.get("confusion_matrix")
                if isinstance(winner_metrics.get("confusion_matrix"), dict)
                else None
            ),
        },
        "gates": {
            "min_recall": min_recall,
            "min_pr_auc": min_pr_auc,
            "passed_gates": bool(passed_gates),
        },
        "ranking_explanation": _build_ranking_explanation(winner_row, runner_up_row),
        "operational_note": (
            "Operacao padrao: threshold fixo 0.30 (alerta se proba >= 0.30). "
            "Contingencia de capacidade: top-k 20% em processamento em lote (ranking de score)."
        ),
        "selection_criteria": {
            "primary_metric": "recall_holdout_at_operational_threshold",
            "secondary_metric": "pr_auc_holdout",
            "tie_breaker": "lowest_positive_rate_then_lexicographic_name",
            "threshold_preference": _threshold_label(
                _to_float_or_none(criteria.get("threshold_used"))
            ),
            "ranking_order": (
                list(criteria.get("ranking_order", []))
                if isinstance(criteria.get("ranking_order"), list)
                else []
            ),
        },
        "status_context": {
            "status": status,
            "reason": _build_warning_reason_summary(selection) if status == "WARNING" else "n/a",
            "decision_applied": _build_decision_applied_text(status),
        },
        "notes": notes,
        "errors": errors,
    }

    forbidden = find_forbidden_json_keys(justification)
    if forbidden:
        justification["status"] = "FAIL"
        justification["notes"].append(
            f"privacy_violation_detected_in_keys: {forbidden}"
        )
        justification["errors"].append(
            f"privacy_violation_detected_in_keys: {forbidden}"
        )
        justification["status_context"]["status"] = "FAIL"
        justification["status_context"]["reason"] = "privacy check failed"
        justification["status_context"]["decision_applied"] = _build_decision_applied_text("FAIL")
    return justification


def render_justification_md(justif: dict[str, Any]) -> str:
    """Render a short one-page markdown justification for reviewers."""
    status = str(justif.get("status", "UNKNOWN"))
    winner = justif.get("winner", {})
    if not isinstance(winner, dict):
        winner = {}
    metrics = justif.get("key_metrics_holdout", {})
    if not isinstance(metrics, dict):
        metrics = {}
    gates = justif.get("gates", {})
    if not isinstance(gates, dict):
        gates = {}
    chosen_threshold = justif.get("chosen_threshold", {})
    if not isinstance(chosen_threshold, dict):
        chosen_threshold = {}
    status_context = justif.get("status_context", {})
    if not isinstance(status_context, dict):
        status_context = {}

    def _fmt_metric(value: Any) -> str:
        parsed = _to_float_or_none(value)
        if parsed is None:
            return "-"
        return f"{parsed:.4f}"

    lines: list[str] = []
    lines.append("# Justificativa do Modelo Final")
    lines.append("")
    lines.append(f"- generated_at: `{justif.get('generated_at')}`")
    lines.append(f"- status: **{status}**")
    if justif.get("selection_generated_at"):
        lines.append(f"- selection_generated_at: `{justif.get('selection_generated_at')}`")
    lines.append("")
    lines.append("## Status da selecao")
    lines.append("")
    lines.append(f"- Status da selecao: **{status_context.get('status', status)}**")
    if str(status).upper() == "WARNING":
        lines.append(f"- Motivo: {status_context.get('reason', '-')}")
    lines.append(f"- Decisao aplicada: {status_context.get('decision_applied', '-')}")
    lines.append("")
    lines.append("## Decisao")
    lines.append("")
    lines.append(
        f"- Modelo campeao: `{winner.get('family', '-')}/{winner.get('variant', '-')}`"
    )
    lines.append(f"- path_model: `{winner.get('path_model', '-')}`")
    lines.append(f"- path_metadata: `{winner.get('path_metadata', '-')}`")
    lines.append("")
    lines.append("## Por que este modelo")
    lines.append("")
    lines.append(f"- {justif.get('ranking_explanation', '-')}")
    lines.append(
        "- Threshold aplicado na selecao: `{}` ({})".format(
            chosen_threshold.get("label", "-"),
            chosen_threshold.get("reason", "-"),
        )
    )
    lines.append(
        "- Gates minimos: Recall >= `{}` e PR-AUC >= `{}` | passed_gates=`{}`".format(
            gates.get("min_recall", "-"),
            gates.get("min_pr_auc", "-"),
            gates.get("passed_gates", False),
        )
    )
    if status == "WARNING":
        lines.append(
            "- ALERTA: a selecao formal retornou `WARNING`; ver secoes de trade-offs e riscos."
        )
    lines.append("")
    lines.append("## Metricas no holdout (2023->2024)")
    lines.append("")
    lines.append("| Metrica | Valor |")
    lines.append("|---|---:|")
    lines.append(f"| Recall | {_fmt_metric(metrics.get('recall'))} |")
    lines.append(f"| PR-AUC | {_fmt_metric(metrics.get('pr_auc'))} |")
    lines.append(f"| Positive rate | {_fmt_metric(metrics.get('positive_rate'))} |")
    lines.append(f"| Precision | {_fmt_metric(metrics.get('precision'))} |")
    lines.append(f"| F1 | {_fmt_metric(metrics.get('f1'))} |")
    lines.append(f"| ROC-AUC | {_fmt_metric(metrics.get('roc_auc'))} |")
    confusion = metrics.get("confusion_matrix")
    if isinstance(confusion, dict):
        lines.append(
            "| Confusion matrix (tn/fp/fn/tp) | `{}/{}/{}/{}` |".format(
                confusion.get("tn", "-"),
                confusion.get("fp", "-"),
                confusion.get("fn", "-"),
                confusion.get("tp", "-"),
            )
        )
    lines.append("")
    lines.append("## Criterio de selecao")
    lines.append("")
    lines.append("- Primaria: maior Recall no holdout no threshold operacional.")
    lines.append("- Secundaria: maior PR-AUC no holdout.")
    lines.append("- Desempate: menor positive_rate; empate final lexicografico por modelo/variante.")
    lines.append(
        "- Threshold preferencial da politica: `0.30` (fallback para `0.5` com warning quando necessario)."
    )
    lines.append("")
    lines.append("## Trade-offs operacionais")
    lines.append("")
    lines.append(f"- {justif.get('operational_note', '-')}")
    lines.append(
        "- Positive_rate do campeao: `{}`. Maior recall tende a elevar volume de alertas e carga operacional.".format(
            _fmt_metric(metrics.get("positive_rate"))
        )
    )
    lines.append(
        "- Top-k 20% e contingencia de capacidade para operacao em lote; nao e politica padrao por request."
    )
    lines.append("")
    lines.append("## Riscos e limitacoes")
    lines.append("")
    lines.append(
        "Existe shift temporal relevante entre treino e holdout (prevalencia aproximada de 0.61 para 0.40), "
        "o que pode degradar estabilidade. A robustez depende da manutencao do contrato de dados e de "
        "monitoramento para categorias novas ou distribuicoes nao vistas."
    )
    notes = justif.get("notes", [])
    if isinstance(notes, list) and notes:
        lines.append("")
        lines.append("## Notas")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines).strip() + "\n"


def persist_justification(
    justif: dict[str, Any],
    *,
    output_md: str | Path = "docs/model_final_justification.md",
    output_json: str | Path = "artifacts/model_final_justification.json",
    write_json: bool = True,
) -> dict[str, str | None]:
    """Persist markdown (versioned) and optional JSON artifact."""
    md_path = Path(output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_justification_md(justif), encoding="utf-8")

    json_path: Path | None = None
    if write_json:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(justif, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return {
        "output_md": str(md_path),
        "output_json": str(json_path) if json_path is not None else None,
    }


def run_model_justification(
    *,
    selection_path: str | Path = "artifacts/model_selection.json",
    output_md: str | Path = "docs/model_final_justification.md",
    output_json: str | Path = "artifacts/model_final_justification.json",
    write_json: bool = True,
) -> dict[str, Any]:
    selection = load_model_selection(selection_path)
    justification = build_model_justification(selection)
    persist_justification(
        justification,
        output_md=output_md,
        output_json=output_json,
        write_json=write_json,
    )
    return justification


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final-model justification from artifacts/model_selection.json."
    )
    parser.add_argument(
        "--selection-path",
        type=str,
        default="artifacts/model_selection.json",
        help="Path to model_selection.json.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="docs/model_final_justification.md",
        help="Output markdown path.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="artifacts/model_final_justification.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable JSON output generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()

    try:
        report = run_model_justification(
            selection_path=args.selection_path,
            output_md=args.output_md,
            output_json=args.output_json,
            write_json=not bool(args.no_json),
        )
    except FileNotFoundError:
        print(
            (
                f"Selection artifact not found at '{args.selection_path}'. "
                "Run `python -m src.model_selection` before generating justification."
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)
    except ValueError as exc:
        print(f"Invalid selection artifact: {exc}", file=sys.stderr)
        raise SystemExit(1)

    _logger.info(
        "Model justification generated | status=%s winner=%s",
        report.get("status"),
        report.get("winner"),
    )
    if str(report.get("status", "")).upper() == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
