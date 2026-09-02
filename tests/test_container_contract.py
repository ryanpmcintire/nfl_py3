"""Static deployment contract for the public dashboard container.

These checks run on CI hosts without a Docker daemon. An actual image build is
still the strongest integration check when Docker is available.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PAGES = ("index.html", "model.html", "history.html", "findings.html")


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_dockerfile_is_pinned_non_root_and_copies_only_public_pages() -> None:
    dockerfile = _read("Dockerfile")

    base = re.search(r"^FROM (\S+)$", dockerfile, flags=re.MULTILINE)
    assert base is not None
    assert re.fullmatch(
        r"nginxinc/nginx-unprivileged:\d+\.\d+\.\d+-alpine\d+\.\d+"
        r"@sha256:[0-9a-f]{64}",
        base.group(1),
    )
    assert "USER 101:101" in dockerfile
    assert "EXPOSE 8080" in dockerfile
    assert "http://127.0.0.1:8080/healthz" in dockerfile
    assert "COPY ." not in dockerfile
    assert "COPY docs/" not in dockerfile

    for page in PUBLIC_PAGES:
        assert f"docs/{page}" in dockerfile
        assert (ROOT / "docs" / page).is_file()


def test_docker_build_context_is_an_explicit_public_allowlist() -> None:
    patterns = [
        line.strip()
        for line in _read(".dockerignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert patterns[0] == "**"
    assert set(patterns[1:]) == {
        "!Dockerfile",
        "!deploy/",
        "!deploy/nginx.conf",
        "!docs/",
        *(f"!docs/{page}" for page in PUBLIC_PAGES),
    }
    forbidden = ("data/", "artifacts/", ".env", ".creds", ".cookies", ".venv/", ".tools/")
    for path in forbidden:
        assert f"!{path}" not in patterns


def test_nginx_contract_is_static_read_only_and_browser_hardened() -> None:
    config = _read("deploy/nginx.conf")

    assert "listen 8080;" in config
    assert "location = /healthz" in config
    assert "location ~ ^/(?:index|model|history|findings)\\.html$" in config
    assert "try_files $uri =404;" in config
    assert "limit_except GET HEAD" in config
    assert "autoindex on" not in config
    assert 'auth_basic "ATS Terminal";' in config
    assert "auth_basic_user_file /run/secrets/dashboard_htpasswd;" in config
    for header in (
        "Content-Security-Policy",
        "Permissions-Policy",
        "Referrer-Policy",
        "X-Content-Type-Options",
        "X-Frame-Options",
    ):
        assert f"add_header {header}" in config


def test_compose_defaults_to_loopback_and_hardens_the_runtime() -> None:
    compose = _read("compose.yaml")

    assert "${NFL_ATS_BIND_ADDRESS:-127.0.0.1}" in compose
    assert 'user: "101:101"' in compose
    assert "read_only: true" in compose
    assert "/tmp:rw,noexec,nosuid,size=16m,mode=1777" in compose
    assert "no-new-privileges:true" in compose
    assert re.search(r"cap_drop:\s*\n\s*- ALL", compose)
    assert re.search(r"secrets:\s*\n\s*- dashboard_htpasswd", compose)
