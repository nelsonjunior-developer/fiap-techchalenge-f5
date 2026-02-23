"""Data contract definitions for yearly PEDE datasets."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
CONTRACT_CHANGELOG_SCHEMA_VERSION = "1.0.0"
CONTRACT_CHANGELOG_JSON_FILENAME = "contracts_changelog.json"
CONTRACT_CHANGELOG_MD_FILENAME = "CHANGELOG.md"

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


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _deepcopy_jsonable(value: Any) -> Any:
    return copy.deepcopy(value)


def _normalize_contract_payload_for_schema_diff(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _deepcopy_jsonable(payload)
    metadata = normalized.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("generated_at", None)
        metadata.pop("dataset_basename", None)
        metadata.pop("dataset_sha256", None)
    return normalized


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _schema_sha256(payload: dict[str, Any]) -> str:
    normalized = _normalize_contract_payload_for_schema_diff(payload)
    return hashlib.sha256(_stable_json_dumps(normalized).encode("utf-8")).hexdigest()


def _lineage_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        return {"dataset_basename": None, "dataset_sha256": None}
    return {
        "dataset_basename": metadata.get("dataset_basename"),
        "dataset_sha256": metadata.get("dataset_sha256"),
    }


def _column_rules_signature(column_payload: dict[str, Any]) -> str:
    return _stable_json_dumps(column_payload.get("rules") or [])


def _diff_contract_payloads(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, Any]:
    current_columns = current.get("columns")
    if not isinstance(current_columns, dict):
        current_columns = {}

    current_schema_sha = _schema_sha256(current)
    current_lineage = _lineage_metadata(current)

    if previous is None:
        return {
            "change_kind": "created",
            "columns_count": len(current_columns),
            "columns_added": sorted(current_columns),
            "columns_removed": [],
            "dtype_changed": [],
            "presence_changed": [],
            "pii_changed": [],
            "rules_changed": [],
            "rules_changed_count": 0,
            "schema_changed": True,
            "schema_sha256": current_schema_sha,
            "previous_schema_sha256": None,
            "lineage_changed": True,
            "lineage": current_lineage,
            "previous_lineage": {"dataset_basename": None, "dataset_sha256": None},
        }

    previous_columns = previous.get("columns")
    if not isinstance(previous_columns, dict):
        previous_columns = {}

    previous_schema_sha = _schema_sha256(previous)
    previous_lineage = _lineage_metadata(previous)

    current_names = set(current_columns)
    previous_names = set(previous_columns)
    shared_names = sorted(current_names & previous_names)

    dtype_changed: list[str] = []
    presence_changed: list[str] = []
    pii_changed: list[str] = []
    rules_changed: list[str] = []

    for column in shared_names:
        prev_col = previous_columns.get(column)
        curr_col = current_columns.get(column)
        if not isinstance(prev_col, dict) or not isinstance(curr_col, dict):
            rules_changed.append(column)
            continue

        if prev_col.get("dtype") != curr_col.get("dtype"):
            dtype_changed.append(column)
        if prev_col.get("presence") != curr_col.get("presence"):
            presence_changed.append(column)
        if bool(prev_col.get("pii")) != bool(curr_col.get("pii")):
            pii_changed.append(column)
        if _column_rules_signature(prev_col) != _column_rules_signature(curr_col):
            rules_changed.append(column)

    schema_changed = previous_schema_sha != current_schema_sha
    lineage_changed = previous_lineage != current_lineage

    if schema_changed:
        change_kind = "updated"
    elif lineage_changed:
        change_kind = "lineage_only"
    else:
        change_kind = "unchanged"

    return {
        "change_kind": change_kind,
        "columns_count": len(current_columns),
        "columns_added": sorted(current_names - previous_names),
        "columns_removed": sorted(previous_names - current_names),
        "dtype_changed": dtype_changed,
        "presence_changed": presence_changed,
        "pii_changed": pii_changed,
        "rules_changed": rules_changed,
        "rules_changed_count": len(rules_changed),
        "schema_changed": schema_changed,
        "schema_sha256": current_schema_sha,
        "previous_schema_sha256": previous_schema_sha,
        "lineage_changed": lineage_changed,
        "lineage": current_lineage,
        "previous_lineage": previous_lineage,
    }


def _load_contracts_changelog(output_dir: Path) -> dict[str, Any]:
    path = output_dir / CONTRACT_CHANGELOG_JSON_FILENAME
    payload = _read_json_if_exists(path)
    if not isinstance(payload, dict):
        return {
            "schema_version": CONTRACT_CHANGELOG_SCHEMA_VERSION,
            "entries": [],
        }
    entries = payload.get("entries")
    if not isinstance(entries, list):
        entries = []
    schema_version = payload.get("schema_version") or CONTRACT_CHANGELOG_SCHEMA_VERSION
    return {
        "schema_version": str(schema_version),
        "entries": entries,
    }


def _build_contracts_changelog_entry(
    *,
    generated_at: str,
    previous_by_year: dict[int, dict[str, Any] | None],
    current_by_year: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    changes_by_year: dict[str, Any] = {}
    created_years: list[int] = []
    updated_years: list[int] = []
    lineage_only_years: list[int] = []
    unchanged_years: list[int] = []

    for year in SUPPORTED_YEARS:
        diff = _diff_contract_payloads(previous_by_year.get(year), current_by_year[year])
        changes_by_year[str(year)] = diff
        kind = diff["change_kind"]
        if kind == "created":
            created_years.append(year)
        elif kind == "updated":
            updated_years.append(year)
        elif kind == "lineage_only":
            lineage_only_years.append(year)
        else:
            unchanged_years.append(year)

    changed_years = created_years + updated_years + lineage_only_years
    total_structural_changes = sum(
        1
        for year in SUPPORTED_YEARS
        if changes_by_year[str(year)]["schema_changed"]
    )

    sample_year = current_by_year[SUPPORTED_YEARS[0]]
    sample_metadata = sample_year.get("metadata") if isinstance(sample_year, dict) else {}
    if not isinstance(sample_metadata, dict):
        sample_metadata = {}

    notes: list[str] = []
    if all(previous_by_year.get(year) is not None for year in SUPPORTED_YEARS):
        notes.append("comparison_against_existing_contract_files")
    else:
        notes.append("initial_contract_export_or_missing_baseline")

    return {
        "generated_at": generated_at,
        "contract_version_declared": sample_metadata.get("contract_version"),
        "dataset_basename": sample_metadata.get("dataset_basename"),
        "dataset_sha256": sample_metadata.get("dataset_sha256"),
        "changes_by_year": changes_by_year,
        "summary": {
            "created_years": created_years,
            "updated_years": updated_years,
            "lineage_only_years": lineage_only_years,
            "unchanged_years": unchanged_years,
            "changed_years": changed_years,
            "total_structural_changes": total_structural_changes,
        },
        "notes": notes,
    }


def _changelog_entry_has_meaningful_change(entry: dict[str, Any]) -> bool:
    summary = entry.get("summary")
    if not isinstance(summary, dict):
        return True
    for key in ("created_years", "updated_years", "lineage_only_years"):
        value = summary.get(key)
        if isinstance(value, list) and len(value) > 0:
            return True
    return False


def _build_contracts_changelog_markdown(changelog: dict[str, Any]) -> str:
    entries = changelog.get("entries")
    entries_list = entries if isinstance(entries, list) else []

    lines: list[str] = []
    lines.append("# Changelog dos Data Contracts")
    lines.append("")
    lines.append(
        "Histórico agregado das mudanças dos contratos exportados em `docs/contracts/`."
    )
    lines.append(
        "Diferenças estruturais ignoram metadados voláteis como `generated_at`."
    )
    lines.append("")

    if not entries_list:
        lines.append("_Sem entradas registradas ainda._")
        return "\n".join(lines).strip() + "\n"

    for idx, entry_raw in enumerate(reversed(entries_list), start=1):
        entry = entry_raw if isinstance(entry_raw, dict) else {}
        generated_at = entry.get("generated_at", "unknown")
        declared_version = entry.get("contract_version_declared", "unknown")
        summary = entry.get("summary")
        summary = summary if isinstance(summary, dict) else {}
        lines.append(f"## Entrada {idx} - {generated_at}")
        lines.append("")
        lines.append(f"- `contract_version_declared`: `{declared_version}`")
        if entry.get("dataset_basename") is not None:
            lines.append(f"- `dataset_basename`: `{entry.get('dataset_basename')}`")
        if entry.get("dataset_sha256") is not None:
            lines.append(f"- `dataset_sha256`: `{entry.get('dataset_sha256')}`")
        lines.append(
            f"- `changed_years`: `{summary.get('changed_years', [])}` | "
            f"`structural_changes`: `{summary.get('total_structural_changes', 0)}`"
        )
        lines.append("")
        lines.append("| Ano | Tipo | Colunas | Adds | Removes | dtype | presence | pii | rules |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")

        changes_by_year = entry.get("changes_by_year")
        changes_by_year = changes_by_year if isinstance(changes_by_year, dict) else {}
        for year in SUPPORTED_YEARS:
            diff = changes_by_year.get(str(year))
            diff = diff if isinstance(diff, dict) else {}
            lines.append(
                "| "
                f"{year} | {diff.get('change_kind', 'unknown')} | "
                f"{int(diff.get('columns_count', 0) or 0)} | "
                f"{len(diff.get('columns_added', []) or [])} | "
                f"{len(diff.get('columns_removed', []) or [])} | "
                f"{len(diff.get('dtype_changed', []) or [])} | "
                f"{len(diff.get('presence_changed', []) or [])} | "
                f"{len(diff.get('pii_changed', []) or [])} | "
                f"{int(diff.get('rules_changed_count', 0) or 0)} |"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _write_contracts_changelog(
    *,
    output_dir: Path,
    generated_at: str,
    previous_by_year: dict[int, dict[str, Any] | None],
    current_by_year: dict[int, dict[str, Any]],
) -> None:
    changelog = _load_contracts_changelog(output_dir)
    entries = changelog.get("entries")
    entries_list = entries if isinstance(entries, list) else []

    entry = _build_contracts_changelog_entry(
        generated_at=generated_at,
        previous_by_year=previous_by_year,
        current_by_year=current_by_year,
    )

    changelog_path = output_dir / CONTRACT_CHANGELOG_JSON_FILENAME
    markdown_path = output_dir / CONTRACT_CHANGELOG_MD_FILENAME

    if not entries_list or _changelog_entry_has_meaningful_change(entry):
        entries_list.append(entry)
        changelog["entries"] = entries_list
        changelog["schema_version"] = CONTRACT_CHANGELOG_SCHEMA_VERSION
        changelog_path.write_text(
            json.dumps(changelog, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    elif not changelog_path.exists():
        changelog["entries"] = entries_list
        changelog["schema_version"] = CONTRACT_CHANGELOG_SCHEMA_VERSION
        changelog_path.write_text(
            json.dumps(changelog, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    markdown_path.write_text(
        _build_contracts_changelog_markdown(changelog),
        encoding="utf-8",
    )


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
    write_changelog: bool = True,
) -> None:
    """Export yearly data contracts into versioned JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    previous_by_year: dict[int, dict[str, Any] | None] = {}
    current_by_year: dict[int, dict[str, Any]] = {}

    for year in SUPPORTED_YEARS:
        json_file = output_path / f"data_contract_{year}.json"
        previous_by_year[year] = _read_json_if_exists(json_file)

    for year in SUPPORTED_YEARS:
        contract = get_year_contract(year)
        contract.metadata["generated_at"] = generated_at
        contract.metadata["dataset_basename"] = dataset_basename
        contract.metadata["dataset_sha256"] = dataset_sha256

        json_payload = _to_jsonable(contract)
        current_by_year[year] = json_payload
        json_file = output_path / f"data_contract_{year}.json"
        json_file.write_text(
            json.dumps(json_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if write_markdown:
            md_file = output_path / f"data_contract_{year}.md"
            md_file.write_text(_build_markdown(contract), encoding="utf-8")

    if write_changelog:
        _write_contracts_changelog(
            output_dir=output_path,
            generated_at=generated_at,
            previous_by_year=previous_by_year,
            current_by_year=current_by_year,
        )


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
    parser.add_argument(
        "--no-changelog",
        action="store_true",
        help="Disable contracts changelog generation in docs/contracts.",
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
            write_changelog=not args.no_changelog,
        )


if __name__ == "__main__":
    main()
