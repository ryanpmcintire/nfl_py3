# Dashboard container deployment

The container serves the four already-published ATS Terminal pages. It does
not run the research pipeline, regenerate predictions, or include Python,
`uv`, raw data, model artifacts, registries, or credentials. The image is a
deployment snapshot of `docs/index.html`, `docs/model.html`,
`docs/history.html`, and `docs/findings.html` at build time.

## Build and run locally

Publish the dashboard before building when local artifacts contain a newer
deliberate forecast:

```powershell
.\.tools\uv.exe run nfl-ats publish-board
docker compose up --build --detach
```

Open <http://127.0.0.1:8080>. Check readiness with:

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
deployment. This image intentionally has no hosted authentication; that is
OPS-04, not part of the static OPS-03 runtime. Keep the default loopback bind
unless the surrounding network policy or reverse proxy is ready.

For a registry deployment, build the same Dockerfile for the target platform,
tag the immutable result, push it to the chosen registry, and deploy that
image digest. No runtime volumes or environment secrets are required.

## Reproducibility and security contract

- The unprivileged NGINX base is pinned by both release tag and OCI index
  digest. Updating it is an explicit Dockerfile change.
- `.dockerignore` is a deny-by-default allowlist. Only the Dockerfile, NGINX
  configuration, and four public HTML files enter the build context.
- NGINX and Compose both run as numeric user/group `101:101`; all Linux
  capabilities are dropped and privilege escalation is disabled.
- The root filesystem is read-only. Only a small, `noexec` `/tmp` tmpfs is
  writable for NGINX's PID and temporary files.
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
docker run --rm --read-only --tmpfs /tmp:rw,noexec,nosuid,size=16m --cap-drop ALL --security-opt no-new-privileges -p 127.0.0.1:8080:8080 nfl-ats-dashboard:local
```
