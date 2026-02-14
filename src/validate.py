"""Data consistency validation for yearly PEDE datasets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.contract_validate import load_year_contract, validate_frame_against_contract
from src.data import get_default_dataset_path, load_pede_workbook_with_metadata
from src.utils import get_logger, setup_logging

_logger = get_logger(__name__)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "warning_missing_rate": 0.60,
    "warning_coerced_rate": 0.10,
    "error_critical_coerced_rate": 0.30,
    "error_critical_missing_rate": 1.00,
}

CRITICAL_COLUMNS: tuple[str, ...] = (
    "Defasagem",
    "INDE",
    "Idade",
    "Mat",
    "Por",
    "Ing",
)

NON_NUMERIC_COERCION_COLUMNS: set[str] = {"Fase", "Fase_Ideal"}


def _resolved_thresholds(overrides: dict[str, float] | None) -> dict[str, float]:
    resolved = dict(DEFAULT_THRESHOLDS)
    if overrides:
        for key, value in overrides.items():
            if key in resolved:
                resolved[key] = float(value)
    return resolved


def _resolve_dataset_basename(dataset_path: str | Path | None) -> str | None:
    if dataset_path is None:
        return None
    return Path(dataset_path).name


def _build_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Relatorio de Qualidade de Dados")
    lines.append("")
    lines.append(f"- Timestamp: `{report['timestamp']}`")
    lines.append(f"- Dataset: `{report.get('dataset_path')}`")
    lines.append(f"- Status: `{report['overall']['status']}`")
    lines.append(f"- Errors: `{report['overall']['n_errors']}`")
    lines.append(f"- Warnings: `{report['overall']['n_warnings']}`")
    lines.append("")

    for year_str, year_report in report["years"].items():
        lines.append(f"## Ano {year_str}")
        lines.append(f"- Linhas: `{year_report['n_rows']}`")
        lines.append(f"- RA nulo: `{year_report['ra_null']}`")
        lines.append(f"- RA em branco: `{year_report['ra_blank']}`")
        lines.append(f"- RA duplicado: `{year_report['ra_duplicates']}`")
        lines.append(f"- Errors: `{year_report['n_errors']}`")
        lines.append(f"- Warnings: `{year_report['n_warnings']}`")
        contract_year = report.get("contract_validation", {}).get("years", {}).get(year_str)
        if contract_year is not None:
            lines.append(
                f"- Contract: `{contract_year['status']}` "
                f"(errors={contract_year['errors_count']}, "
                f"warnings={contract_year['warnings_count']}, "
                f"infos={contract_year['infos_count']})"
            )
        lines.append("")

        top_missing = year_report.get("top_real_missing", [])[:5]
        if top_missing:
            lines.append("Top missing real:")
            for item in top_missing:
                lines.append(
                    f"- `{item['column']}`: {item['missing_rate']:.2%} ({item['missing_type']})"
                )
            lines.append("")

        top_coercion = year_report.get("top_coercion", [])[:5]
        if top_coercion:
            lines.append("Top coercao:")
            for item in top_coercion:
                lines.append(
                    f"- `{item['column']}`: coerced_rate={item['coerced_rate']:.2%}, "
                    f"n_coerced={item['n_coerced_to_na']}"
                )
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def _validate_ra(year: int, df: pd.DataFrame, errors: list[str]) -> dict[str, int]:
    if "RA" not in df.columns:
        errors.append(f"[year={year}] ERROR: coluna RA ausente.")
        return {
            "ra_null": 0,
            "ra_blank": 0,
            "ra_duplicates": 0,
        }

    ra_series = df["RA"].astype("string")
    ra_null = int(ra_series.isna().sum())
    ra_blank = int((ra_series.str.strip() == "").fillna(False).sum())
    ra_duplicates = int(ra_series.duplicated(keep=False).sum())

    if ra_null > 0:
        errors.append(f"[year={year}] ERROR: RA possui valores nulos ({ra_null}).")
    if ra_blank > 0:
        errors.append(f"[year={year}] ERROR: RA possui valores em branco ({ra_blank}).")
    if ra_duplicates > 0:
        errors.append(f"[year={year}] ERROR: RA possui duplicados ({ra_duplicates}).")

    return {
        "ra_null": ra_null,
        "ra_blank": ra_blank,
        "ra_duplicates": ra_duplicates,
    }


def _validate_missingness(
    *,
    year: int,
    df: pd.DataFrame,
    original_columns: set[str],
    thresholds: dict[str, float],
    critical_columns: set[str],
    errors: list[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    n_rows = len(df)

    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        missing_rate = float(missing_count / n_rows) if n_rows > 0 else 0.0
        existed_in_original = column in original_columns
        is_critical = column in critical_columns
        missing_type = "none"

        if n_rows > 0 and missing_count == n_rows:
            if existed_in_original:
                missing_type = "real_100_missing"
                if is_critical and missing_rate >= thresholds["error_critical_missing_rate"]:
                    errors.append(
                        f"[year={year}] ERROR: coluna critica '{column}' com 100% missing real."
                    )
                else:
                    warnings.append(
                        f"[year={year}] WARNING: coluna '{column}' com 100% missing real."
                    )
            else:
                missing_type = "structural_100_missing"
        elif existed_in_original and missing_rate > thresholds["warning_missing_rate"]:
            missing_type = "real_high_missing"
            warnings.append(
                f"[year={year}] WARNING: coluna '{column}' com missing real alto ({missing_rate:.2%})."
            )

        summary.append(
            {
                "column": column,
                "missing_count": missing_count,
                "missing_rate": round(missing_rate, 6),
                "existed_in_original": existed_in_original,
                "missing_type": missing_type,
                "is_critical": is_critical,
            }
        )

    summary.sort(key=lambda item: item["missing_rate"], reverse=True)
    return summary


def _validate_coercion(
    *,
    year: int,
    coercion_year_report: dict[str, Any] | None,
    thresholds: dict[str, float],
    critical_columns: set[str],
    errors: list[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    if not coercion_year_report:
        return summary

    numeric_columns = coercion_year_report.get("numeric_columns", {})
    for column, stats in numeric_columns.items():
        if column in NON_NUMERIC_COERCION_COLUMNS:
            continue

        n_original_non_null = int(stats.get("n_original_non_null", 0) or 0)
        n_coerced_to_na = int(stats.get("n_coerced_to_na", 0) or 0)
        n_invalid_tokens_replaced = int(stats.get("n_invalid_tokens_replaced", 0) or 0)
        dtype_final = str(stats.get("dtype_final", "unknown"))

        no_signal = n_original_non_null == 0
        coerced_rate = (
            float(n_coerced_to_na / n_original_non_null)
            if n_original_non_null > 0
            else 0.0
        )
        is_critical = column in critical_columns

        if not no_signal:
            if is_critical and coerced_rate > thresholds["error_critical_coerced_rate"]:
                errors.append(
                    f"[year={year}] ERROR: coluna critica '{column}' com coerced_rate alto ({coerced_rate:.2%})."
                )
            elif coerced_rate > thresholds["warning_coerced_rate"]:
                warnings.append(
                    f"[year={year}] WARNING: coluna '{column}' com coerced_rate alto ({coerced_rate:.2%})."
                )

        summary.append(
            {
                "column": column,
                "n_original_non_null": n_original_non_null,
                "n_coerced_to_na": n_coerced_to_na,
                "n_invalid_tokens_replaced": n_invalid_tokens_replaced,
                "coerced_rate": round(coerced_rate, 6),
                "no_signal": no_signal,
                "is_critical": is_critical,
                "dtype_final": dtype_final,
            }
        )

    summary.sort(key=lambda item: item["coerced_rate"], reverse=True)
    return summary


def validate_yearly_frames(
    dfs: dict[int, pd.DataFrame],
    original_columns: dict[int, set[str]],
    coercion_report: dict | None = None,
    thresholds: dict | None = None,
    strict: bool = False,
    output_dir: str | Path = "artifacts",
    write_markdown: bool = True,
    dataset_path: str | Path | None = None,
    contracts_dir: str | Path | None = "docs/contracts",
) -> dict:
    """Validate yearly datasets and persist data quality artifacts."""
    resolved_thresholds = _resolved_thresholds(thresholds)
    critical_columns = set(CRITICAL_COLUMNS)

    years_sorted = sorted(dfs)
    columns_by_year = {year: list(dfs[year].columns) for year in years_sorted}
    schema_identical = len({tuple(cols) for cols in columns_by_year.values()}) <= 1

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_path": _resolve_dataset_basename(dataset_path),
        "thresholds": resolved_thresholds,
        "critical_columns": sorted(critical_columns),
        "schema_identical": schema_identical,
        "years": {},
        "contract_validation": {
            "enabled": contracts_dir is not None,
            "contracts_dir": str(contracts_dir) if contracts_dir is not None else None,
            "years": {},
            "overall": {
                "status": "SKIPPED" if contracts_dir is None else "PASS",
                "errors_count": 0,
                "warnings_count": 0,
                "infos_count": 0,
            },
        },
        "overall": {
            "status": "PASS",
            "n_errors": 0,
            "n_warnings": 0,
        },
    }

    if not schema_identical:
        _logger.warning("Data quality | schema_identical=False")

    total_errors = 0
    total_warnings = 0
    contract_total_errors = 0
    contract_total_warnings = 0
    contract_total_infos = 0

    for year in years_sorted:
        df = dfs[year]
        year_errors: list[str] = []
        year_warnings: list[str] = []

        ra_metrics = _validate_ra(year=year, df=df, errors=year_errors)

        year_original_columns = original_columns.get(year, set(df.columns))
        missing_summary = _validate_missingness(
            year=year,
            df=df,
            original_columns=year_original_columns,
            thresholds=resolved_thresholds,
            critical_columns=critical_columns,
            errors=year_errors,
            warnings=year_warnings,
        )

        year_coercion_report = (coercion_report or {}).get(year, {})
        coercion_summary = _validate_coercion(
            year=year,
            coercion_year_report=year_coercion_report,
            thresholds=resolved_thresholds,
            critical_columns=critical_columns,
            errors=year_errors,
            warnings=year_warnings,
        )

        top_real_missing = [
            item
            for item in missing_summary
            if item["missing_type"] in {"real_100_missing", "real_high_missing"}
        ][:5]
        top_coercion = [item for item in coercion_summary if not item["no_signal"]][:5]

        year_report = {
            "n_rows": int(len(df)),
            **ra_metrics,
            "missing_summary": missing_summary,
            "coercion_summary": coercion_summary,
            "errors": year_errors,
            "warnings": year_warnings,
            "n_errors": len(year_errors),
            "n_warnings": len(year_warnings),
            "top_real_missing": top_real_missing,
            "top_coercion": top_coercion,
        }

        report["years"][str(year)] = year_report
        total_errors += len(year_errors)
        total_warnings += len(year_warnings)

        if contracts_dir is not None:
            try:
                contract = load_year_contract(year=year, contracts_dir=contracts_dir)
                contract_year_result = validate_frame_against_contract(
                    df=df,
                    year=year,
                    contract=contract,
                )
            except FileNotFoundError as exc:
                contract_year_result = {
                    "year": year,
                    "status": "FAIL",
                    "errors_count": 1,
                    "warnings_count": 0,
                    "infos_count": 0,
                    "schema": {
                        "contract_columns_count": 0,
                        "df_columns_count": len(df.columns),
                        "missing_columns": [],
                        "extra_columns": [],
                    },
                    "findings": [
                        {
                            "year": year,
                            "column": None,
                            "rule_type": "contract",
                            "kind": "missing_contract_file",
                            "enforcement": "error",
                            "message": str(exc),
                            "metrics": {},
                        }
                    ],
                }
            report["contract_validation"]["years"][str(year)] = contract_year_result
            contract_total_errors += int(contract_year_result["errors_count"])
            contract_total_warnings += int(contract_year_result["warnings_count"])
            contract_total_infos += int(contract_year_result["infos_count"])

        _logger.info(
            "Data quality year=%d | rows=%d errors=%d warnings=%d top_real_missing=%s top_coercion=%s",
            year,
            len(df),
            len(year_errors),
            len(year_warnings),
            [(item["column"], item["missing_rate"]) for item in top_real_missing],
            [(item["column"], item["coerced_rate"]) for item in top_coercion],
        )

    if contracts_dir is not None:
        contract_status = "PASS" if contract_total_errors == 0 else "FAIL"
        report["contract_validation"]["overall"] = {
            "status": contract_status,
            "errors_count": contract_total_errors,
            "warnings_count": contract_total_warnings,
            "infos_count": contract_total_infos,
        }

    overall_errors = total_errors + contract_total_errors
    overall_warnings = total_warnings + contract_total_warnings
    status = "PASS" if overall_errors == 0 else "FAIL"
    report["overall"] = {
        "status": status,
        "n_errors": overall_errors,
        "n_warnings": overall_warnings,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "data_quality_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if write_markdown:
        md_path = output_path / "data_quality_report.md"
        md_path.write_text(_build_markdown_report(report), encoding="utf-8")

    _logger.info(
        "Data quality validation completed | status=%s errors=%d warnings=%d "
        "contract_errors=%d report=%s",
        status,
        overall_errors,
        overall_warnings,
        contract_total_errors,
        json_path,
    )

    if strict and status == "FAIL":
        raise RuntimeError(
            f"Data consistency validation failed with {overall_errors} errors. "
            f"See report: {json_path}"
        )

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate yearly PEDE dataframe consistency.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Caminho do XLSX. Se omitido, usa DATASET_PATH ou caminho padrão do projeto.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Falha a execução quando houver erros de consistência.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Diretório de saída para relatórios de qualidade.",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Não gerar o relatório markdown (apenas JSON).",
    )
    parser.add_argument(
        "--contracts-dir",
        type=str,
        default="docs/contracts",
        help="Diretório de contratos versionados (data_contract_YYYY.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()

    dataset_path = Path(args.dataset_path) if args.dataset_path else get_default_dataset_path()
    dfs, align_metadata, coercion = load_pede_workbook_with_metadata(dataset_path)
    original_columns = align_metadata.get("original_columns", {})

    report = validate_yearly_frames(
        dfs=dfs,
        original_columns=original_columns,
        coercion_report=coercion,
        strict=args.strict,
        output_dir=args.output_dir,
        write_markdown=not args.no_markdown,
        dataset_path=dataset_path,
        contracts_dir=args.contracts_dir,
    )

    _logger.info(
        "Data quality CLI summary | status=%s errors=%d warnings=%d",
        report["overall"]["status"],
        report["overall"]["n_errors"],
        report["overall"]["n_warnings"],
    )


if __name__ == "__main__":
    main()
