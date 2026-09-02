"""Static security contract for the authenticated dashboard runtime."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_dashboard_auth_uses_an_injected_file_secret() -> None:
    compose = _read("compose.yaml")

    assert "dashboard_htpasswd:" in compose
    assert "NFL_ATS_HTPASSWD_FILE:-.secrets/dashboard.htpasswd" in compose
    assert "password:" not in compose.lower()
    assert "environment:" not in compose.lower()
    assert ".secrets/" in _read(".gitignore")


def test_nginx_requires_auth_except_for_content_free_health_check() -> None:
    config = _read("deploy/nginx.conf")

    assert 'auth_basic "ATS Terminal";' in config
    assert "auth_basic_user_file /run/secrets/dashboard_htpasswd;" in config
    health = config.split("location = /healthz", maxsplit=1)[1].split("location = / {", maxsplit=1)[
        0
    ]
    assert "auth_basic off;" in health
    assert 'return 200 "ok\\n";' in health


def test_secret_and_research_state_cannot_enter_the_image() -> None:
    dockerfile = _read("Dockerfile")
    dockerignore = _read(".dockerignore")
    compose = _read("compose.yaml")

    assert "COPY ." not in dockerfile
    assert dockerignore.splitlines()[3] == "**"
    assert "!.secrets" not in dockerignore
    assert "!data" not in dockerignore
    assert "!artifacts" not in dockerignore
    assert "volumes:" not in compose
    assert "/run/secrets/dashboard_htpasswd" not in dockerfile
    for page in ("index.html", "model.html", "history.html", "findings.html"):
        public_html = _read(f"docs/{page}")
        assert "dashboard_htpasswd" not in public_html
        assert "BEGIN PRIVATE KEY" not in public_html
        assert "Authorization:" not in public_html


def test_hosting_docs_require_tls_and_forbid_cleartext_secret_channels() -> None:
    docs = _read("docs/hosted_dashboard.md")

    assert "HTTPS is mandatory" in docs
    assert "no password is placed in an image layer or\nenvironment variable" in docs
    assert "Do not paste a cleartext password" in docs
    assert "does not provision a DNS name" in docs
