import json
from pathlib import Path

import pytest

import src.contracts as contracts_mod
from src.contracts import (
    Enforcement,
    Presence,
    export_contracts,
    get_year_contract,
)


def _get_rule(column_spec, rule_type: str):
    for rule in column_spec.rules:
        if rule.rule_type == rule_type:
            return rule
    raise AssertionError(f"Rule {rule_type} not found for {column_spec.name}")


def test_get_year_contract_contains_expected_columns() -> None:
    for year in (2022, 2023, 2024):
        contract = get_year_contract(year)
        assert "RA" in contract.columns
        assert "Defasagem" in contract.columns
        assert "Data_Nasc" in contract.columns


def test_presence_original_vs_structural_optional() -> None:
    c2022 = get_year_contract(2022)
    c2024 = get_year_contract(2024)

    assert c2022.columns["Pedra 2024"].presence == Presence.STRUCTURAL_OPTIONAL
    assert c2024.columns["Escola"].presence == Presence.ORIGINAL


def test_pii_flags_for_sensitive_columns() -> None:
    contract = get_year_contract(2024)

    assert contract.columns["RA"].pii is True
    assert contract.columns["Nome_Anon"].pii is True
    for col in ("Avaliador1", "Avaliador2", "Avaliador3", "Avaliador4", "Avaliador5", "Avaliador6"):
        assert contract.columns[col].pii is True


def test_data_nasc_has_date_range_rule() -> None:
    contract = get_year_contract(2023)
    domain_rule = _get_rule(contract.columns["Data_Nasc"], "domain")

    assert domain_rule.spec["kind"] == "date_range"
    assert domain_rule.spec["start"] == "1990-01-01"
    assert domain_rule.spec["end"] == "2030-12-31"
    assert domain_rule.enforcement in {Enforcement.WARNING, Enforcement.ERROR}


def test_numeric_domains_for_idade_and_inde() -> None:
    contract = get_year_contract(2024)

    idade_rule = _get_rule(contract.columns["Idade"], "domain")
    assert idade_rule.enforcement == Enforcement.ERROR
    assert idade_rule.spec["kind"] == "range"
    assert idade_rule.spec["min"] == 3
    assert idade_rule.spec["max"] == 30

    inde_rule = _get_rule(contract.columns["INDE"], "domain")
    assert inde_rule.spec["kind"] == "range"
    assert inde_rule.spec["min"] == 0
    assert inde_rule.spec["max"] == 10.5


def test_export_contracts_writes_json_with_metadata(tmp_path: Path) -> None:
    export_contracts(
        output_dir=tmp_path,
        dataset_basename="BASE DE DADOS PEDE 2024 - DATATHON.xlsx",
        dataset_sha256="abc123",
    )

    for year in (2022, 2023, 2024):
        json_path = tmp_path / f"data_contract_{year}.json"
        md_path = tmp_path / f"data_contract_{year}.md"
        assert json_path.exists()
        assert md_path.exists()

        content = json_path.read_text(encoding="utf-8")
        assert '"contract_version": "1.0.0"' in content
        assert '"generated_at"' in content
        assert '"dataset_basename": "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"' in content
        assert '"dataset_sha256": "abc123"' in content

    changelog_json = tmp_path / "contracts_changelog.json"
    changelog_md = tmp_path / "CHANGELOG.md"
    assert changelog_json.exists()
    assert changelog_md.exists()

    changelog_payload = json.loads(changelog_json.read_text(encoding="utf-8"))
    assert changelog_payload["schema_version"] == "1.0.0"
    assert len(changelog_payload["entries"]) == 1
    entry = changelog_payload["entries"][0]
    assert entry["summary"]["created_years"] == [2022, 2023, 2024]
    assert entry["summary"]["updated_years"] == []
    assert entry["changes_by_year"]["2024"]["columns_count"] > 0
    assert entry["changes_by_year"]["2024"]["schema_sha256"]


def test_export_contracts_changelog_is_idempotent_when_only_generated_at_changes(
    tmp_path: Path,
) -> None:
    export_contracts(output_dir=tmp_path)
    export_contracts(output_dir=tmp_path)

    changelog_payload = json.loads(
        (tmp_path / "contracts_changelog.json").read_text(encoding="utf-8")
    )
    assert len(changelog_payload["entries"]) == 1
    entry = changelog_payload["entries"][0]
    assert entry["summary"]["created_years"] == [2022, 2023, 2024]


def test_export_contracts_changelog_appends_entry_on_structural_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_contracts(output_dir=tmp_path)

    original_get_year_contract = contracts_mod.get_year_contract

    def _patched_get_year_contract(year: int):
        contract = original_get_year_contract(year)
        if year == 2024:
            contract.columns["Idade"].dtype = "Float64"
        return contract

    monkeypatch.setattr(contracts_mod, "get_year_contract", _patched_get_year_contract)

    export_contracts(output_dir=tmp_path)

    changelog_payload = json.loads(
        (tmp_path / "contracts_changelog.json").read_text(encoding="utf-8")
    )
    assert len(changelog_payload["entries"]) == 2
    latest = changelog_payload["entries"][-1]
    assert latest["summary"]["updated_years"] == [2024]
    assert latest["summary"]["created_years"] == []
    assert "Idade" in latest["changes_by_year"]["2024"]["dtype_changed"]
    assert latest["changes_by_year"]["2024"]["change_kind"] == "updated"


def test_get_year_contract_rejects_invalid_year() -> None:
    with pytest.raises(ValueError, match="Ano inválido"):
        get_year_contract(2025)
