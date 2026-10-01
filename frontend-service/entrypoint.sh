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

# Every upstream name lookup runs on libuv's thread pool. uvicorn[standard]'s
# --loop auto picks uvloop, which resolves names on that pool; libuv lets only
# half of the pool resolve at once (2 of the default 4 threads), and a lookup
# whose caller has timed out still holds its thread until it finishes. On Docker
# Desktop for Windows a provider that is not running takes 2.5 to 4 s to fail to
# resolve (LLMNR/NetBIOS), so one health round over the absent providers queued a
# running provider's lookup past the gateway's 3 s connect timeout: ConnectTimeout,
# and 503 "is unavailable" for a healthy service. 32 threads allow 16 lookups at
# once; 16 threads were measured as marginal (with 9 absent providers, Magpie's
# lookup waited 2577 of its 3000 ms). A value set in the environment wins.
export UV_THREADPOOL_SIZE="${UV_THREADPOOL_SIZE:-32}"

# exec: uvicorn becomes PID 1 and receives SIGTERM directly.
exec python -m uvicorn app:app \
    --host 0.0.0.0 --port 3000 \
    --workers "$workers" \
    --proxy-headers \
    --timeout-keep-alive 120 \
    --no-access-log \
    "$@"
