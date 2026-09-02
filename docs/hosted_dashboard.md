# Read-only hosted dashboard contract

OPS-04 uses the same four-page static image as the local dashboard. It does
not place the research application, data tree, model artifacts, registries,
cookies, or credentials in the image. Authentication is enforced by NGINX
before any dashboard page is returned.

## Credential injection

Create a bcrypt htpasswd file outside version control. The command prompts for
the password rather than accepting it as a command-line argument:

```powershell
New-Item -ItemType Directory -Force .secrets
htpasswd -cB .secrets/dashboard.htpasswd dashboard
```

`.secrets/` is ignored by Git and denied by the Docker build-context
allowlist. Compose injects only that file at
`/run/secrets/dashboard_htpasswd`; no password is placed in an image layer or
environment variable. To keep the file elsewhere, set
`NFL_ATS_HTPASSWD_FILE` to its path before running Compose.

Rotate access by generating a replacement file, atomically replacing the
host-side file, and recreating the container with `docker compose up -d
--force-recreate`. Remove an account with `htpasswd -D <file> <username>` and
recreate the container. Do not paste a cleartext password into Compose, an
environment file, a reverse-proxy label, or a command argument.

## Hosting boundary

The container continues to bind to `127.0.0.1` by default. For a hosted
deployment, keep that private bind and put a managed TLS reverse proxy or
private access gateway on the same host. HTTPS is mandatory because HTTP Basic
credentials are not encrypted without TLS. The proxy should forward only the
dashboard origin and should not mount or receive the repository's data or
artifact directories.

`/healthz` is deliberately unauthenticated so an orchestrator can test process
readiness without storing dashboard credentials; it returns only `ok`. Every
dashboard page and unknown path remains behind authentication. NGINX accepts
only `GET` and `HEAD` on the four page routes.

## Least privilege and leakage controls

- Numeric user/group `101:101`; all capabilities dropped; no privilege
  escalation.
- Read-only root filesystem and a 16 MiB `noexec,nosuid` temporary filesystem.
- No runtime volumes for source, data, artifacts, Git metadata, or logs.
- Deny-by-default build context containing only the Dockerfiles/configuration
  and four deliberately public generated pages.
- Browser policy blocks forms, framing, objects, and browser network requests;
  the inline board scripts only sort, select, and animate already-rendered
  values.
- Access logs use NGINX's default combined format, which does not record the
  `Authorization` header. Keep reverse-proxy header logging disabled as well.

This repository contract does not provision a DNS name, certificate, cloud
account, registry, or external service. Those are deployment-environment
choices and require separate authorization.
