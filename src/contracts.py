"""Data contract definitions for yearly PEDE datasets."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class Presence(str, Enum):
    """Column presence origin in a specific yearly contract."""

    ORIGINAL = "original"
    STRUCTURAL_OPTIONAL = "structural_optional"


class Enforcement(str, Enum):
    """Severity level for contract rules."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class DomainSpec:
    """Domain and plausibility constraints for a column."""

    kind: str = "none"
    min: float | int | None = None
    max: float | int | None = None
    allowed: list[str] | None = None
    pattern: str | None = None
    start: str | None = None
    end: str | None = None
    notes: str | None = None


@dataclass
class ColumnRule:
    """Single validation rule declaration for a column."""

    rule_type: str
    enforcement: Enforcement
    spec: dict[str, Any]
    notes: str | None = None


@dataclass
class ColumnSpec:
    """Column contract including type, presence and rules."""

    name: str
    dtype: str
    presence: Presence
    pii: bool
    rules: list[ColumnRule] = field(default_factory=list)
    description: str | None = None


@dataclass
class YearContract:
    """Contract container for one reference year."""

    year: int
    columns: dict[str, ColumnSpec]
    metadata: dict[str, Any]


CONTRACT_VERSION = "1.0.0"
SUPPORTED_YEARS: tuple[int, ...] = (2022, 2023, 2024)

ROWS_EXPECTED_BY_YEAR: dict[int, int] = {
    2022: 860,
    2023: 1014,
    2024: 1156,
}

ORIGINAL_COLUMNS_BY_YEAR: dict[int, set[str]] = {
    2022: {
        "Ano ingresso",
        "Atingiu PV",
        "Avaliador1",
        "Avaliador2",
        "Avaliador3",
        "Avaliador4",
        "Cf",
        "Cg",
        "Ct",
        "Data_Nasc",
        "Defasagem",
        "Destaque IDA",
        "Destaque IEG",
        "Destaque IPV",
        "Fase",
        "Fase_Ideal",
        "Gênero",
        "IAA",
        "IAN",
        "IDA",
        "IEG",
        "INDE",
        "INDE 22",
        "IPS",
        "IPV",
        "Idade",
        "Indicado",
        "Ing",
        "Instituição de ensino",
        "Mat",
        "Nome_Anon",
        "Nº Av",
        "Pedra 20",
        "Pedra 21",
        "Pedra 22",
        "Pedra_Ano",
        "Por",
        "RA",
        "Rec Av1",
        "Rec Av2",
        "Rec Av3",
        "Rec Av4",
        "Rec Psicologia",
        "Turma",
    },
    2023: {
        "Ano ingresso",
        "Atingiu PV",
        "Avaliador1",
        "Avaliador2",
        "Avaliador3",
        "Avaliador4",
        "Cf",
        "Cg",
        "Ct",
        "Data_Nasc",
        "Defasagem",
        "Destaque IDA",
        "Destaque IEG",
        "Destaque IPV",
        "Destaque IPV__dup1",
        "Fase",
        "Fase_Ideal",
        "Gênero",
        "IAA",
        "IAN",
        "IDA",
        "IEG",
        "INDE",
        "INDE 2023",
        "INDE 22",
        "INDE 23",
        "IPP",
        "IPS",
        "IPV",
        "Idade",
        "Indicado",
        "Ing",
        "Instituição de ensino",
        "Mat",
        "Nome_Anon",
        "Nº Av",
        "Pedra 20",
        "Pedra 2023",
        "Pedra 21",
        "Pedra 22",
        "Pedra 23",
        "Pedra_Ano",
        "Por",
        "RA",
        "Rec Av1",
        "Rec Av2",
        "Rec Av3",
        "Rec Av4",
        "Rec Psicologia",
        "Turma",
    },
    2024: {
        "Ano ingresso",
        "Atingiu PV",
        "Ativo/ Inativo",
        "Ativo/ Inativo__dup1",
        "Avaliador1",
        "Avaliador2",
        "Avaliador3",
        "Avaliador4",
        "Avaliador5",
        "Avaliador6",
        "Cf",
        "Cg",
        "Ct",
        "Data_Nasc",
        "Defasagem",
        "Destaque IDA",
        "Destaque IEG",
        "Destaque IPV",
        "Escola",
        "Fase",
        "Fase_Ideal",
        "Gênero",
        "IAA",
        "IAN",
        "IDA",
        "IEG",
        "INDE",
        "INDE 2024",
        "INDE 22",
        "INDE 23",
        "IPP",
        "IPS",
        "IPV",
        "Idade",
        "Indicado",
        "Ing",
        "Instituição de ensino",
        "Mat",
        "Nome_Anon",
        "Nº Av",
        "Pedra 20",
        "Pedra 2024",
        "Pedra 21",
        "Pedra 22",
        "Pedra 23",
        "Pedra_Ano",
        "Por",
        "RA",
        "Rec Av1",
        "Rec Av2",
        "Rec Psicologia",
        "Turma",
    },
}

FINAL_DTYPES: dict[str, str] = {
    "RA": "string",
    "Ano ingresso": "Int64",
    "Atingiu PV": "string",
    "Ativo/ Inativo": "string",
    "Ativo/ Inativo__dup1": "string",
    "Avaliador1": "string",
    "Avaliador2": "string",
    "Avaliador3": "string",
    "Avaliador4": "string",
    "Avaliador5": "string",
    "Avaliador6": "string",
    "Cf": "Float64",
    "Cg": "Float64",
    "Ct": "Float64",
    "Data_Nasc": "datetime64[ns]",
    "Defasagem": "Int64",
    "Destaque IDA": "string",
    "Destaque IEG": "string",
    "Destaque IPV": "string",
    "Destaque IPV__dup1": "string",
    "Escola": "string",
    "Fase": "string",
    "Fase_Ideal": "string",
    "Gênero": "string",
    "IAA": "Float64",
    "IAN": "Float64",
    "IDA": "Float64",
    "IEG": "Float64",
    "INDE": "Float64",
    "INDE 2023": "Float64",
    "INDE 2024": "Float64",
    "INDE 22": "Float64",
    "INDE 23": "Float64",
    "IPP": "Float64",
    "IPS": "Float64",
    "IPV": "Float64",
    "Idade": "Int64",
    "Indicado": "string",
    "Ing": "Float64",
    "Instituição de ensino": "string",
    "Mat": "Float64",
    "Nome_Anon": "string",
    "Nº Av": "Int64",
    "Pedra 20": "string",
    "Pedra 2023": "string",
    "Pedra 2024": "string",
    "Pedra 21": "string",
    "Pedra 22": "string",
    "Pedra 23": "string",
    "Pedra_Ano": "string",
    "Por": "Float64",
    "Rec Av1": "string",
    "Rec Av2": "string",
    "Rec Av3": "string",
    "Rec Av4": "string",
    "Rec Psicologia": "Float64",
    "Turma": "string",
}

PII_COLUMNS: set[str] = {
    "RA",
    "Nome_Anon",
    "Avaliador1",
    "Avaliador2",
    "Avaliador3",
    "Avaliador4",
    "Avaliador5",
    "Avaliador6",
}

OPEN_DOMAIN_COLUMNS: set[str] = {
    "Escola",
    "Turma",
    "Instituição de ensino",
    "Fase",
    "Fase_Ideal",
}

NUMERIC_RANGE_0_10_5: set[str] = {
    "INDE",
    "IAA",
    "IAN",
    "IDA",
    "IEG",
    "IPS",
    "IPP",
    "IPV",
    "Mat",
    "Por",
    "Ing",
    "INDE 22",
    "INDE 23",
    "INDE 2023",
    "INDE 2024",
}

PEDRA_COLUMNS: set[str] = {
    "Pedra_Ano",
    "Pedra 20",
    "Pedra 21",
    "Pedra 22",
    "Pedra 23",
    "Pedra 2023",
    "Pedra 2024",
}


def _presence_for(year: int, column: str) -> Presence:
    if column in ORIGINAL_COLUMNS_BY_YEAR[year]:
        return Presence.ORIGINAL
    return Presence.STRUCTURAL_OPTIONAL


def _dtype_rule(dtype: str) -> ColumnRule:
    return ColumnRule(
        rule_type="dtype",
        enforcement=Enforcement.ERROR,
        spec={"expected_dtype": dtype},
    )


def _missing_rule(
    *,
    year: int,
    column: str,
    presence: Presence,
) -> ColumnRule:
    if presence == Presence.STRUCTURAL_OPTIONAL:
        return ColumnRule(
            rule_type="missing",
            enforcement=Enforcement.INFO,
            spec={"allow_missing": True},
            notes="Coluna estrutural do alinhamento entre anos.",
        )

    if column in {"RA", "Idade", "Defasagem", "Gênero", "Ano ingresso"}:
        return ColumnRule(
            rule_type="missing",
            enforcement=Enforcement.ERROR,
            spec={"allow_missing": False},
        )

    if column == "Data_Nasc":
        return ColumnRule(
            rule_type="missing",
            enforcement=Enforcement.WARNING,
            spec={"allow_missing": False},
        )

    if column in {"INDE", "IAA", "IAN", "IDA", "IEG", "IPS", "IPP", "IPV", "Mat", "Por"}:
        return ColumnRule(
            rule_type="missing",
            enforcement=Enforcement.WARNING,
            spec={"allow_missing": True},
        )

    if column == "Ing":
        return ColumnRule(
            rule_type="missing",
            enforcement=Enforcement.INFO,
            spec={"allow_missing": True},
            notes="Missing historicamente alto nesta variável.",
        )

    if column == "Nº Av":
        if year == 2023:
            return ColumnRule(
                rule_type="missing",
                enforcement=Enforcement.WARNING,
                spec={"allow_missing": True},
            )
        return ColumnRule(
            rule_type="missing",
            enforcement=Enforcement.ERROR,
            spec={"allow_missing": False},
        )

    if column in {"Cg", "Cf", "Ct"}:
        if year == 2022:
            return ColumnRule(
                rule_type="missing",
                enforcement=Enforcement.ERROR,
                spec={"allow_missing": False},
            )
        return ColumnRule(
            rule_type="missing",
            enforcement=Enforcement.INFO,
            spec={"allow_missing": True},
            notes="Variável estruturalmente ausente neste ano.",
        )

    if column in {"Indicado", "Atingiu PV"}:
        if year == 2022:
            return ColumnRule(
                rule_type="missing",
                enforcement=Enforcement.ERROR,
                spec={"allow_missing": False},
            )
        return ColumnRule(
            rule_type="missing",
            enforcement=Enforcement.INFO,
            spec={"allow_missing": True},
            notes="Coluna presente mas sem preenchimento neste ano.",
        )

    if column in OPEN_DOMAIN_COLUMNS or column in {"Ativo/ Inativo", "Ativo/ Inativo__dup1"}:
        return ColumnRule(
            rule_type="missing",
            enforcement=Enforcement.WARNING,
            spec={"allow_missing": True},
        )

    return ColumnRule(
        rule_type="missing",
        enforcement=Enforcement.INFO,
        spec={"allow_missing": True},
    )


def _domain_rule(
    *,
    year: int,
    column: str,
    presence: Presence,
) -> ColumnRule:
    if column == "Data_Nasc":
        domain = DomainSpec(
            kind="date_range",
            start="1990-01-01",
            end="2030-12-31",
            notes="Faixa plausível para data de nascimento após padronização.",
        )
        return ColumnRule(
            rule_type="domain",
            enforcement=Enforcement.WARNING,
            spec=asdict(domain),
        )

    if column == "Idade":
        return ColumnRule(
            rule_type="domain",
            enforcement=Enforcement.ERROR,
            spec=asdict(DomainSpec(kind="range", min=3, max=30)),
        )

    if column == "Defasagem":
        return ColumnRule(
            rule_type="domain",
            enforcement=Enforcement.ERROR,
            spec=asdict(DomainSpec(kind="range", min=-10, max=10)),
        )

    if column in NUMERIC_RANGE_0_10_5:
        return ColumnRule(
            rule_type="domain",
            enforcement=Enforcement.ERROR,
            spec=asdict(DomainSpec(kind="range", min=0, max=10.5)),
        )

    if column == "Nº Av":
        return ColumnRule(
            rule_type="domain",
            enforcement=Enforcement.ERROR,
            spec=asdict(DomainSpec(kind="range", min=0, max=10)),
        )

    if column == "Ano ingresso":
        return ColumnRule(
            rule_type="domain",
            enforcement=Enforcement.ERROR,
            spec=asdict(DomainSpec(kind="range", min=2010, max=2030)),
        )

    if column == "Cg":
        enforcement = Enforcement.WARNING if year == 2022 else Enforcement.INFO
        return ColumnRule(
            rule_type="domain",
            enforcement=enforcement,
            spec=asdict(DomainSpec(kind="range", min=0, max=1000)),
            notes="Semântica muda por ano; ajustar com evidência de negócio.",
        )

    if column == "Cf":
        enforcement = Enforcement.WARNING if year == 2022 else Enforcement.INFO
        return ColumnRule(
            rule_type="domain",
            enforcement=enforcement,
            spec=asdict(DomainSpec(kind="range", min=0, max=300)),
            notes="Semântica muda por ano; ajustar com evidência de negócio.",
        )

    if column == "Ct":
        enforcement = Enforcement.WARNING if year == 2022 else Enforcement.INFO
        return ColumnRule(
            rule_type="domain",
            enforcement=enforcement,
            spec=asdict(DomainSpec(kind="range", min=0, max=50)),
            notes="Semântica muda por ano; ajustar com evidência de negócio.",
        )

    if column == "Gênero":
        return ColumnRule(
            rule_type="domain",
            enforcement=Enforcement.ERROR,
            spec=asdict(DomainSpec(kind="set", allowed=["Feminino", "Masculino"])),
        )

    if column in PEDRA_COLUMNS:
        return ColumnRule(
            rule_type="domain",
            enforcement=Enforcement.WARNING,
            spec=asdict(
                DomainSpec(
                    kind="set",
                    allowed=["Ametista", "Ágata", "Quartzo", "Topázio"],
                    notes="Missing permitido; tokens inválidos devem virar NA.",
                )
            ),
        )

    if column in {"Indicado", "Atingiu PV"}:
        return ColumnRule(
            rule_type="domain",
            enforcement=Enforcement.WARNING,
            spec=asdict(DomainSpec(kind="set", allowed=["Sim", "Não"])),
        )

    if column == "Ativo/ Inativo" and year == 2024 and presence == Presence.ORIGINAL:
        return ColumnRule(
            rule_type="domain",
            enforcement=Enforcement.WARNING,
            spec=asdict(DomainSpec(kind="set", allowed=["Cursando"])),
        )

    if column in OPEN_DOMAIN_COLUMNS:
        return ColumnRule(
            rule_type="domain",
            enforcement=Enforcement.INFO,
            spec=asdict(
                DomainSpec(
                    kind="none",
                    notes="Domínio aberto/alta cardinalidade; sem enumeração estrita.",
                )
            ),
        )

    if column == "RA":
        return ColumnRule(
            rule_type="domain",
            enforcement=Enforcement.INFO,
            spec=asdict(
                DomainSpec(kind="none", notes="Identificador operacional; não usar como feature.")
            ),
        )

    if presence == Presence.STRUCTURAL_OPTIONAL:
        return ColumnRule(
            rule_type="domain",
            enforcement=Enforcement.INFO,
            spec=asdict(DomainSpec(kind="none", notes="Coluna estrutural opcional no ano.")),
        )

    return ColumnRule(
        rule_type="domain",
        enforcement=Enforcement.INFO,
        spec=asdict(DomainSpec(kind="none")),
    )


def _description_for(column: str) -> str | None:
    if column == "RA":
        return "Identificador do estudante (somente chave/auditoria)."
    if column == "Defasagem":
        return "Indicador de defasagem escolar."
    if column == "Data_Nasc":
        return "Data de nascimento padronizada."
    if column == "Nome_Anon":
        return "Campo sensível; em 2022 pode não estar totalmente anonimizado."
    return None


def _build_column_spec(year: int, column: str) -> ColumnSpec:
    dtype = FINAL_DTYPES[column]
    presence = _presence_for(year, column)
    rules = [
        _dtype_rule(dtype),
        _missing_rule(year=year, column=column, presence=presence),
        _domain_rule(year=year, column=column, presence=presence),
    ]
    return ColumnSpec(
        name=column,
        dtype=dtype,
        presence=presence,
        pii=column in PII_COLUMNS,
        rules=rules,
        description=_description_for(column),
    )


def _build_year_contract(year: int) -> YearContract:
    columns = {
        column: _build_column_spec(year, column) for column in sorted(FINAL_DTYPES)
    }
    metadata = {
        "contract_version": CONTRACT_VERSION,
        "rows_expected": ROWS_EXPECTED_BY_YEAR[year],
        "dataset_basename": None,
        "dataset_sha256": None,
        "generated_at": None,
        "notes": (
            "Presence diferencia colunas originais do ano e colunas estruturais do alinhamento."
        ),
    }
    return YearContract(year=year, columns=columns, metadata=metadata)


def _build_contracts() -> dict[int, YearContract]:
    return {year: _build_year_contract(year) for year in SUPPORTED_YEARS}


CONTRACTS_BY_YEAR: dict[int, YearContract] = _build_contracts()


def get_year_contract(year: int) -> YearContract:
    """Return a deep copy of the year contract."""
    if year not in CONTRACTS_BY_YEAR:
        raise ValueError(f"Ano inválido: {year}. Anos suportados: {list(SUPPORTED_YEARS)}")
    return copy.deepcopy(CONTRACTS_BY_YEAR[year])


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _build_markdown(contract: YearContract) -> str:
    lines: list[str] = []
    lines.append(f"# Data Contract {contract.year}")
    lines.append("")
    lines.append("| Coluna | DType | Presence | PII | Regras |")
    lines.append("|---|---|---|---|---|")

    for column in sorted(contract.columns):
        spec = contract.columns[column]
        rules = ", ".join(
            f"{rule.rule_type}:{rule.enforcement.value}" for rule in spec.rules
        )
        pii = "yes" if spec.pii else "no"
        lines.append(
            f"| {column} | {spec.dtype} | {spec.presence.value} | {pii} | {rules} |"
        )

    return "\n".join(lines).strip() + "\n"


def export_contracts(
    output_dir: str | Path = "docs/contracts",
    dataset_basename: str | None = None,
    dataset_sha256: str | None = None,
    write_markdown: bool = True,
) -> None:
    """Export yearly data contracts into versioned JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    for year in SUPPORTED_YEARS:
        contract = get_year_contract(year)
        contract.metadata["generated_at"] = generated_at
        contract.metadata["dataset_basename"] = dataset_basename
        contract.metadata["dataset_sha256"] = dataset_sha256

        json_payload = _to_jsonable(contract)
        json_file = output_path / f"data_contract_{year}.json"
        json_file.write_text(
            json.dumps(json_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if write_markdown:
            md_file = output_path / f"data_contract_{year}.md"
            md_file.write_text(_build_markdown(contract), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export static yearly data contracts.")
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export contracts to docs/contracts (or custom output-dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/contracts",
        help="Output directory for contract files.",
    )
    parser.add_argument(
        "--dataset-basename",
        type=str,
        default=None,
        help="Optional dataset filename for metadata lineage.",
    )
    parser.add_argument(
        "--dataset-sha256",
        type=str,
        default=None,
        help="Optional dataset sha256 for metadata lineage.",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Export JSON only.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.export:
        export_contracts(
            output_dir=args.output_dir,
            dataset_basename=args.dataset_basename,
            dataset_sha256=args.dataset_sha256,
            write_markdown=not args.no_markdown,
        )


if __name__ == "__main__":
    main()
