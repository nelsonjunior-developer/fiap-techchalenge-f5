"""Automatic validation of yearly dataframes against exported data contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import get_logger

_logger = get_logger(__name__)

_VALID_ENFORCEMENTS = {"error", "warning", "info"}
_VALID_PRESENCE = {"original", "structural_optional"}


def load_year_contract(year: int, contracts_dir: str | Path = "docs/contracts") -> dict[str, Any]:
    """Load one exported yearly contract from docs/contracts."""
    path = Path(contracts_dir) / f"data_contract_{year}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Contrato não encontrado para year={year}: '{path}'. "
            "Verifique docs/contracts e execute export dos contratos."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_enforcement(value: Any) -> str:
    text = str(value).strip().lower()
    if text in _VALID_ENFORCEMENTS:
        return text
    return "info"


def _normalize_presence(value: Any) -> str:
    text = str(value).strip().lower()
    if text in _VALID_PRESENCE:
        return text
    return "original"


def _normalize_dtype_name(dtype: Any) -> str:
    text = str(dtype).strip()
    low = text.lower()
    if low.startswith("string"):
        return "string"
    if low.startswith("datetime64"):
        return "datetime64[ns]"
    if text in {"Int64", "Float64"}:
        return text
    return text


def _rule_violation(
    *,
    year: int,
    column: str,
    rule_type: str,
    kind: str,
    enforcement: str,
    message: str,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "year": year,
        "column": column,
        "rule_type": rule_type,
        "kind": kind,
        "enforcement": enforcement,
        "message": message,
        "metrics": metrics or {},
    }


def _domain_range_metrics(series: pd.Series, spec: dict[str, Any]) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = int(series.notna().sum())
    cast_invalid = int((series.notna() & numeric.isna()).sum())

    out_mask = pd.Series(False, index=series.index, dtype=bool)
    if spec.get("min") is not None:
        out_mask = out_mask | (numeric < spec["min"])
    if spec.get("max") is not None:
        out_mask = out_mask | (numeric > spec["max"])
    out_of_range = int((numeric.notna() & out_mask).sum())

    invalid_total = cast_invalid + out_of_range
    invalid_rate = float(invalid_total / non_null) if non_null > 0 else 0.0
    return {
        "n_non_null": non_null,
        "n_cast_invalid": cast_invalid,
        "n_out_of_range": out_of_range,
        "n_invalid": invalid_total,
        "invalid_rate": round(invalid_rate, 6),
    }


def _domain_set_metrics(series: pd.Series, allowed: set[str]) -> dict[str, Any]:
    as_text = series.astype("string")
    non_null_mask = as_text.notna()
    non_null = int(non_null_mask.sum())
    invalid_mask = non_null_mask & ~as_text.isin(list(allowed))
    not_allowed = int(invalid_mask.sum())
    invalid_rate = float(not_allowed / non_null) if non_null > 0 else 0.0
    return {
        "n_non_null": non_null,
        "n_not_allowed": not_allowed,
        "invalid_rate": round(invalid_rate, 6),
    }


def _domain_regex_metrics(series: pd.Series, pattern: str) -> dict[str, Any]:
    as_text = series.astype("string")
    non_null_mask = as_text.notna()
    non_null = int(non_null_mask.sum())
    invalid_mask = non_null_mask & ~as_text.str.fullmatch(pattern, na=False)
    invalid = int(invalid_mask.sum())
    invalid_rate = float(invalid / non_null) if non_null > 0 else 0.0
    return {
        "n_non_null": non_null,
        "n_not_matching": invalid,
        "invalid_rate": round(invalid_rate, 6),
    }


def _domain_date_range_metrics(
    series: pd.Series,
    start: str | None,
    end: str | None,
) -> dict[str, Any]:
    parsed = pd.to_datetime(series, errors="coerce")
    non_null_mask = series.notna()
    non_null = int(non_null_mask.sum())
    parse_invalid = int((non_null_mask & parsed.isna()).sum())

    out_mask = pd.Series(False, index=series.index, dtype=bool)
    if start is not None:
        start_ts = pd.Timestamp(start)
        out_mask = out_mask | (parsed < start_ts)
    if end is not None:
        end_ts = pd.Timestamp(end)
        out_mask = out_mask | (parsed > end_ts)
    out_of_range = int((parsed.notna() & out_mask).sum())

    invalid_total = parse_invalid + out_of_range
    invalid_rate = float(invalid_total / non_null) if non_null > 0 else 0.0
    return {
        "n_non_null": non_null,
        "n_parse_invalid": parse_invalid,
        "n_out_of_range": out_of_range,
        "n_invalid": invalid_total,
        "invalid_rate": round(invalid_rate, 6),
    }


def validate_frame_against_contract(
    df: pd.DataFrame,
    year: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate one dataframe against one yearly contract definition."""
    contract_columns = set(contract.get("columns", {}).keys())
    df_columns = set(df.columns)
    findings: list[dict[str, Any]] = []

    missing_cols = sorted(contract_columns - df_columns)
    extra_cols = sorted(df_columns - contract_columns)

    for col in missing_cols:
        findings.append(
            _rule_violation(
                year=year,
                column=col,
                rule_type="schema",
                kind="missing_column",
                enforcement="error",
                message=(
                    f"[year={year}] coluna esperada ausente no DataFrame: '{col}'."
                ),
            )
        )

    for col in extra_cols:
        findings.append(
            _rule_violation(
                year=year,
                column=col,
                rule_type="schema",
                kind="extra_column",
                enforcement="warning",
                message=f"[year={year}] coluna extra não prevista no contrato: '{col}'.",
            )
        )

    shared_columns = sorted(contract_columns & df_columns)
    contract_specs = contract.get("columns", {})

    for col in shared_columns:
        col_spec = contract_specs[col]
        rules = col_spec.get("rules", [])
        presence = _normalize_presence(col_spec.get("presence"))

        for rule in rules:
            rule_type = str(rule.get("rule_type", "")).strip()
            enforcement = _normalize_enforcement(rule.get("enforcement"))
            spec = rule.get("spec", {}) or {}
            series = df[col]

            if rule_type == "dtype":
                expected = spec.get("expected_dtype") or spec.get("expected")
                expected_norm = _normalize_dtype_name(expected)
                observed = str(series.dtype)
                observed_norm = _normalize_dtype_name(observed)
                if expected_norm != observed_norm:
                    findings.append(
                        _rule_violation(
                            year=year,
                            column=col,
                            rule_type="dtype",
                            kind="dtype",
                            enforcement=enforcement,
                            message=(
                                f"[year={year}] dtype inválido em '{col}': "
                                f"expected={expected_norm}, observed={observed_norm}."
                            ),
                            metrics={
                                "expected_dtype": expected_norm,
                                "observed_dtype": observed_norm,
                            },
                        )
                    )
                continue

            if rule_type == "missing":
                allow_missing = bool(spec.get("allow_missing", True))
                missing_count = int(series.isna().sum())
                missing_rate = float(missing_count / len(series)) if len(series) else 0.0
                if (not allow_missing) and missing_count > 0:
                    effective_enforcement = (
                        "info" if presence == "structural_optional" else enforcement
                    )
                    findings.append(
                        _rule_violation(
                            year=year,
                            column=col,
                            rule_type="missing",
                            kind="missing",
                            enforcement=effective_enforcement,
                            message=(
                                f"[year={year}] missing não permitido em '{col}' "
                                f"(missing_rate={missing_rate:.2%})."
                            ),
                            metrics={
                                "allow_missing": allow_missing,
                                "missing_count": missing_count,
                                "missing_rate": round(missing_rate, 6),
                            },
                        )
                    )
                continue

            if rule_type != "domain":
                continue

            kind = str(spec.get("kind", "none")).strip().lower()
            if kind == "none":
                continue

            effective_enforcement = "info" if presence == "structural_optional" else enforcement

            if kind == "range":
                metrics = _domain_range_metrics(series, spec)
                if metrics["n_invalid"] > 0:
                    findings.append(
                        _rule_violation(
                            year=year,
                            column=col,
                            rule_type="domain",
                            kind="range",
                            enforcement=effective_enforcement,
                            message=(
                                f"[year={year}] domínio range inválido em '{col}' "
                                f"(n_invalid={metrics['n_invalid']})."
                            ),
                            metrics=metrics,
                        )
                    )
                continue

            if kind == "set":
                allowed = {str(item) for item in (spec.get("allowed") or [])}
                metrics = _domain_set_metrics(series, allowed)
                if metrics["n_not_allowed"] > 0:
                    findings.append(
                        _rule_violation(
                            year=year,
                            column=col,
                            rule_type="domain",
                            kind="set",
                            enforcement=effective_enforcement,
                            message=(
                                f"[year={year}] domínio set inválido em '{col}' "
                                f"(n_not_allowed={metrics['n_not_allowed']})."
                            ),
                            metrics=metrics,
                        )
                    )
                continue

            if kind == "regex":
                pattern = str(spec.get("pattern") or "")
                if pattern == "":
                    continue
                # Fail-fast for malformed regex in contract configuration.
                re.compile(pattern)
                metrics = _domain_regex_metrics(series, pattern)
                if metrics["n_not_matching"] > 0:
                    findings.append(
                        _rule_violation(
                            year=year,
                            column=col,
                            rule_type="domain",
                            kind="regex",
                            enforcement=effective_enforcement,
                            message=(
                                f"[year={year}] domínio regex inválido em '{col}' "
                                f"(n_not_matching={metrics['n_not_matching']})."
                            ),
                            metrics=metrics,
                        )
                    )
                continue

            if kind == "date_range":
                metrics = _domain_date_range_metrics(
                    series,
                    spec.get("start"),
                    spec.get("end"),
                )
                if metrics["n_invalid"] > 0:
                    findings.append(
                        _rule_violation(
                            year=year,
                            column=col,
                            rule_type="domain",
                            kind="date_range",
                            enforcement=effective_enforcement,
                            message=(
                                f"[year={year}] domínio date_range inválido em '{col}' "
                                f"(n_invalid={metrics['n_invalid']})."
                            ),
                            metrics=metrics,
                        )
                    )
                continue

            findings.append(
                _rule_violation(
                    year=year,
                    column=col,
                    rule_type="domain",
                    kind=kind,
                    enforcement="info",
                    message=(
                        f"[year={year}] kind de domínio desconhecido para '{col}': '{kind}'."
                    ),
                )
            )

    errors_count = sum(item["enforcement"] == "error" for item in findings)
    warnings_count = sum(item["enforcement"] == "warning" for item in findings)
    infos_count = sum(item["enforcement"] == "info" for item in findings)
    status = "FAIL" if errors_count > 0 else "PASS"

    result = {
        "year": year,
        "status": status,
        "errors_count": int(errors_count),
        "warnings_count": int(warnings_count),
        "infos_count": int(infos_count),
        "schema": {
            "contract_columns_count": len(contract_columns),
            "df_columns_count": len(df_columns),
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
        },
        "findings": findings,
    }

    _logger.info(
        "Contract validation year=%d | status=%s errors=%d warnings=%d infos=%d "
        "missing_cols=%d extra_cols=%d",
        year,
        status,
        errors_count,
        warnings_count,
        infos_count,
        len(missing_cols),
        len(extra_cols),
    )
    return result
