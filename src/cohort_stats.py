"""RA intersection statistics for yearly PEDE cohorts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import get_logger, setup_logging

_logger = get_logger(__name__)

DEFAULT_PAIRS: tuple[tuple[int, int], ...] = (
    (2022, 2023),
    (2023, 2024),
    (2022, 2024),
)


def compute_ra_sets(dfs: dict[int, pd.DataFrame]) -> tuple[dict[int, set[str]], dict[int, int]]:
    """Build yearly RA sets and invalid RA discard counters."""
    ra_sets: dict[int, set[str]] = {}
    invalid_counts: dict[int, int] = {}

    for year in sorted(dfs):
        df = dfs[year]
        if "RA" not in df.columns:
            raise ValueError(
                f"Coluna RA ausente para year={year}. Colunas disponíveis: {list(df.columns)}"
            )

        ra = df["RA"].astype("string").str.strip()
        invalid_mask = ra.isna() | (ra == "")
        invalid_counts[year] = int(invalid_mask.sum())
        valid = ra[~invalid_mask]
        ra_sets[year] = set(valid.tolist())

    return ra_sets, invalid_counts


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_intersections(
    ra_sets: dict[int, set[str]],
    invalid_counts: dict[int, int],
    pairs: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Compute aggregate intersection metrics for fixed year pairs."""
    selected_pairs = list(DEFAULT_PAIRS if pairs is None else pairs)
    counts = {str(year): len(values) for year, values in sorted(ra_sets.items())}
    invalid = {str(year): int(invalid_counts.get(year, 0)) for year in sorted(ra_sets)}

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "years": sorted(ra_sets),
        "counts": counts,
        "ra_invalid_discarded_count": invalid,
        "pairs": {},
    }

    for year_a, year_b in selected_pairs:
        if year_a not in ra_sets or year_b not in ra_sets:
            raise ValueError(
                f"Par inválido ({year_a}, {year_b}). Anos disponíveis: {sorted(ra_sets)}"
            )

        set_a = ra_sets[year_a]
        set_b = ra_sets[year_b]
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        pair_key = f"{year_a}_{year_b}"

        report["pairs"][pair_key] = {
            "intersection": intersection,
            f"pct_of_{year_a}": _safe_ratio(intersection, len(set_a)),
            f"pct_of_{year_b}": _safe_ratio(intersection, len(set_b)),
            "union": union,
            "jaccard": _safe_ratio(intersection, union),
        }

    return report


def _build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Estatísticas de Interseção por RA")
    lines.append("")
    lines.append(f"- Gerado em: `{report['generated_at']}`")
    lines.append("")
    lines.append("| Par | Interseção | % ano A | % ano B | União | Jaccard |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for pair_key, pair_stats in report["pairs"].items():
        year_a, year_b = pair_key.split("_")
        lines.append(
            "| "
            f"{pair_key} | {pair_stats['intersection']} | "
            f"{pair_stats[f'pct_of_{year_a}']:.2%} | {pair_stats[f'pct_of_{year_b}']:.2%} | "
            f"{pair_stats['union']} | {pair_stats['jaccard']:.4f} |"
        )

    lines.append("")
    lines.append("| Ano | RAs válidos | RAs inválidos descartados |")
    lines.append("|---|---:|---:|")
    for year in report["years"]:
        year_key = str(year)
        lines.append(
            f"| {year} | {report['counts'][year_key]} | {report['ra_invalid_discarded_count'][year_key]} |"
        )

    return "\n".join(lines).strip() + "\n"


def persist_ra_intersections(
    report: dict[str, Any],
    output_dir: str | Path = "artifacts",
    write_markdown: bool = True,
) -> tuple[Path, Path | None]:
    """Persist RA intersection report artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "ra_intersections.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path: Path | None = None
    if write_markdown:
        md_path = output_path / "ra_intersections.md"
        md_path.write_text(_build_markdown(report), encoding="utf-8")

    return json_path, md_path


def run_from_loaded_data(
    output_dir: str | Path = "artifacts",
    write_markdown: bool = True,
) -> dict[str, Any]:
    """Load dataset through pipeline, compute intersections and persist report."""
    # Local import avoids circular dependency with src.data integration.
    from src.data import get_default_dataset_path, load_pede_workbook_with_metadata

    dataset_path = get_default_dataset_path()
    dfs, _, _ = load_pede_workbook_with_metadata(dataset_path)
    ra_sets, invalid_counts = compute_ra_sets(dfs)
    report = compute_intersections(ra_sets, invalid_counts)
    json_path, _ = persist_ra_intersections(
        report,
        output_dir=output_dir,
        write_markdown=write_markdown,
    )

    _logger.info(
        "RA intersections generated | file=%s years=%s pairs=%s",
        json_path,
        report["years"],
        list(report["pairs"].keys()),
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RA intersection cohort statistics.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Output directory for report artifacts.",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Generate JSON only (skip markdown report).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()

    report = run_from_loaded_data(
        output_dir=args.output_dir,
        write_markdown=not args.no_markdown,
    )

    for pair_key, pair_stats in report["pairs"].items():
        year_a, year_b = pair_key.split("_")
        _logger.info(
            "RA pair=%s | intersection=%d pct_%s=%.4f pct_%s=%.4f union=%d jaccard=%.4f",
            pair_key,
            pair_stats["intersection"],
            year_a,
            pair_stats[f"pct_of_{year_a}"],
            year_b,
            pair_stats[f"pct_of_{year_b}"],
            pair_stats["union"],
            pair_stats["jaccard"],
        )


if __name__ == "__main__":
    main()
