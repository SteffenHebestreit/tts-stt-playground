#!/bin/sh
# Start the gateway.
#
# A script rather than an inline CMD so that FRONTEND_WORKERS is read when the
# container starts (a JSON-array CMD does no variable expansion, and a shell-form
# CMD leaves the shell as PID 1, so `docker stop` never reaches uvicorn), and so
# that a mistyped value degrades to the default instead of killing the container.
#
#   --workers         the gateway is I/O-bound with no per-process model state.
#                     Asset cache-busting is derived from a hash of static/, so
#                     every worker computes the same value without pinning it.
#                     Memory-constrained boards (RK3588) set FRONTEND_WORKERS=1.
#   --timeout-keep-alive  >= the httpx keepalive_expiry used for upstream calls,
#                     so the server does not FIN first and hand back dead sockets.

workers="${FRONTEND_WORKERS:-2}"
if ! [ "$workers" -ge 1 ] 2>/dev/null; then
    echo "FRONTEND_WORKERS='$workers' is not a positive integer; using 2" >&2
    workers=2
fi

# exec: uvicorn becomes PID 1 and receives SIGTERM directly.
exec python -m uvicorn app:app \
    --host 0.0.0.0 --port 3000 \
    --workers "$workers" \
    --proxy-headers \
    --timeout-keep-alive 120 \
    --no-access-log \
    "$@"
