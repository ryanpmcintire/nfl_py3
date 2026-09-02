# Dashboard container deployment

The container serves the four already-published ATS Terminal pages behind
HTTP Basic authentication. It does
not run the research pipeline, regenerate predictions, or include Python,
`uv`, raw data, model artifacts, registries, or credentials. The image is a
deployment snapshot of `docs/index.html`, `docs/model.html`,
`docs/history.html`, and `docs/findings.html` at build time.

## Build and run locally

Create an ignored htpasswd file before the first start. `htpasswd` prompts for
the password, so the cleartext value does not enter shell history:

```powershell
New-Item -ItemType Directory -Force .secrets
htpasswd -cB .secrets/dashboard.htpasswd dashboard
```

Publish the dashboard before building when local artifacts contain a newer
deliberate forecast:

```powershell
.\.tools\uv.exe run nfl-ats publish-board
docker compose up --build --detach
```

Open <http://127.0.0.1:8080> and authenticate with that account. Check
readiness with:

```powershell
docker compose ps
Invoke-WebRequest http://127.0.0.1:8080/healthz
```

Stop it with `docker compose down`. The default bind address is loopback, so
the site is not exposed to the local network.

## Server use

The same Compose file can bind another address and port:

```powershell
$env:NFL_ATS_BIND_ADDRESS = "0.0.0.0"
$env:NFL_ATS_PORT = "8080"
docker compose up --build --detach
```

Put a TLS-terminating reverse proxy in front of port 8080 for an internet
deployment. Basic-auth credentials are only transport-safe inside TLS. Keep
the default loopback bind unless the surrounding network policy or reverse
proxy is ready. The full hosted contract and rotation procedure are in
[`hosted_dashboard.md`](hosted_dashboard.md).

For a registry deployment, build the same Dockerfile for the target platform,
tag the immutable result, push it to the chosen registry, and deploy that
image digest. The only runtime secret is the injected htpasswd file. Set
`NFL_ATS_HTPASSWD_FILE` when it lives outside the default ignored `.secrets/`
directory.

## Reproducibility and security contract

- The unprivileged NGINX base is pinned by both release tag and OCI index
  digest. Updating it is an explicit Dockerfile change.
- `.dockerignore` is a deny-by-default allowlist. Only the Dockerfile, NGINX
  configuration, and four public HTML files enter the build context.
- NGINX and Compose both run as numeric user/group `101:101`; all Linux
  capabilities are dropped and privilege escalation is disabled.
- The root filesystem is read-only. Only a small, `noexec` `/tmp` tmpfs is
  writable for NGINX's PID and temporary files.
- Dashboard routes require credentials from the read-only Docker secret at
  `/run/secrets/dashboard_htpasswd`. The image and Compose environment contain
  no password value. `/healthz` is the only unauthenticated route and returns
  only `ok`.
- The server accepts only `GET` and `HEAD` for dashboard paths, exposes a
  dependency-free `/healthz` probe, returns 404 for unknown files, disables
  directory listing, and sets browser hardening headers.
- The image copies generated output instead of running `publish-board` during
  the build. A build therefore cannot silently publish ignored local artifacts
  or credentials, and it does not need the Python/`uv` dependency graph.

The generated pages use inline CSS/JavaScript and Google-hosted font files.
The Content Security Policy permits those exact requirements and blocks
frames, forms, objects, browser network connections, and all other origins.

## Verification

The repository test suite statically checks the base-image digest, four-page
allowlist, non-root/read-only contract, health check, method restriction, and
security headers. When a Docker daemon is available, also run:

```powershell
docker build --check .
docker compose config
docker build --tag nfl-ats-dashboard:local .
docker compose up --detach
```
