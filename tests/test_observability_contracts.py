from __future__ import annotations

import json
from pathlib import Path


def test_observability_compose_declares_required_services() -> None:
    payload = Path("docker-compose.observability.yml").read_text(encoding="utf-8")
    assert "services:" in payload
    assert "api:" in payload
    assert "prometheus:" in payload
    assert "grafana:" in payload
    assert "prom/prometheus:v2.53.1" in payload
    assert "grafana/grafana:11.2.2" in payload


def test_prometheus_scrape_config_targets_api_metrics() -> None:
    payload = Path("observability/prometheus/prometheus.yml").read_text(encoding="utf-8")
    assert 'job_name: "ml-api"' in payload
    assert "metrics_path: /metrics" in payload
    assert "api:8000" in payload


def test_grafana_provisioning_contract() -> None:
    datasource = Path(
        "observability/grafana/provisioning/datasources/datasource.yml"
    ).read_text(encoding="utf-8")
    assert "name: Prometheus" in datasource
    assert "url: http://prometheus:9090" in datasource

    dashboard_provider = Path(
        "observability/grafana/provisioning/dashboards/dashboard.yml"
    ).read_text(encoding="utf-8")
    assert "type: file" in dashboard_provider
    assert "path: /var/lib/grafana/dashboards" in dashboard_provider


def test_grafana_dashboard_queries_include_guardrails() -> None:
    dashboard = json.loads(
        Path("observability/grafana/dashboards/api_observability.json").read_text(
            encoding="utf-8"
        )
    )
    assert dashboard.get("title") == "ML API Observability (Local)"
    panels = dashboard.get("panels", [])
    assert isinstance(panels, list) and panels

    expressions: list[str] = []
    for panel in panels:
        for target in panel.get("targets", []):
            expr = target.get("expr")
            if isinstance(expr, str):
                expressions.append(expr)

    assert any("http_requests_total" in expr for expr in expressions)
    assert any("http_request_duration_seconds_bucket" in expr for expr in expressions)
    assert any("inference_records_total" in expr for expr in expressions)
    assert any("inference_positive_total" in expr for expr in expressions)
    assert any("model_loaded" in expr for expr in expressions)
    assert any("metadata_loaded" in expr for expr in expressions)
    assert any('path!="/metrics"' in expr for expr in expressions)
