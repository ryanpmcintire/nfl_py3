# syntax=docker/dockerfile:1.7

# Keep the public dashboard runtime independent of the Python research stack.
# The tag documents the upstream release; the digest makes the base immutable.
FROM nginxinc/nginx-unprivileged:1.30.4-alpine3.24@sha256:45ce1e2e699234253d1def7baa96218a5d00b498d1ba0cbb1a17b6bdf73d1351

LABEL org.opencontainers.image.title="NFL ATS dashboard" \
      org.opencontainers.image.description="Read-only static server for the published ATS Terminal" \
      org.opencontainers.image.source="https://github.com/ryanpmcintire/nfl_py3"

COPY --chown=101:101 --chmod=0444 deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY --chown=101:101 --chmod=0444 \
    docs/index.html \
    docs/model.html \
    docs/history.html \
    docs/findings.html \
    /usr/share/nginx/html/

EXPOSE 8080
USER 101:101

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1
