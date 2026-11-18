#!/system/bin/sh
# Launcher for the OpenAI-compatible ASR service. init runs this, this execs the
# binary, so SIGTERM from init reaches the server directly and the graceful
# drain in main() still runs.
#
# Everything here exists because init cannot express it: conditional flags,
# an environment, and a guard against a tight crash loop hammering the DSP.

set -u

CONF=${PF_CONF:-/vendor/etc/paraformer/paraformer-asr.conf}

log() {
    # init captures stdout to the log tag, but a rotation-friendly file is what
    # a support engineer actually reads after the car has been out all day.
    printf '%s pf-launcher: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"${LOG:-/dev/null}" 2>/dev/null
    printf 'pf-launcher: %s\n' "$*"
}

die() {
    log "$*"
    exit 78
}

[ -r "$CONF" ] || die "cannot read $CONF"
# The config is root-owned and only ever written by the image, so sourcing it is
# the intended contract: it carries paths and flags, not user input.
. "$CONF"

: "${PF_BIN:?PF_BIN is not set}"
: "${PF_MODEL:?PF_MODEL is not set}"
: "${PF_LIBDIR:?PF_LIBDIR is not set}"
# The config may relocate logs and state (the non-root layout under
# /data/local/tmp does), so these are resolved after it is sourced.
PF_WORKDIR=${PF_WORKDIR:-/data/vendor/paraformer}
LOG=${PF_LOG:-/data/vendor/paraformer/paraformer-asr.log}
STATE_DIR=${PF_STATE:-$PF_WORKDIR}
PF_PORT=${PF_PORT:-8080}
PF_BIND=${PF_BIND:-127.0.0.1}
PF_ENGINES=${PF_ENGINES:-2}
PF_THREADS=${PF_THREADS:-4}

[ -x "$PF_BIN" ] || die "$PF_BIN is not executable"
[ -r "$PF_MODEL" ] || die "cannot read model $PF_MODEL"
[ -d "$PF_LIBDIR" ] || die "cannot read library directory $PF_LIBDIR"

mkdir -p "$PF_WORKDIR" "$STATE_DIR" 2>/dev/null

# ---- crash-loop guard ------------------------------------------------------
# A failed HTP init exits 3, and init restarts the service immediately. Without
# a guard that becomes a tight loop that re-runs DSP discovery and graph
# compilation forever, which is much worse than a service that stays down and
# says so. Count consecutive starts that died quickly, back off, and reset the
# count once a start survives long enough to be called healthy.
FAIL_FILE="$STATE_DIR/consecutive-fast-failures"
START_FILE="$STATE_DIR/last-start-epoch"
FAST_FAIL_SECONDS=${PF_FAST_FAIL_SECONDS:-60}
MAX_BACKOFF=${PF_MAX_BACKOFF_SECONDS:-60}

fails=0
[ -r "$FAIL_FILE" ] && fails=$(cat "$FAIL_FILE" 2>/dev/null || echo 0)
case "$fails" in ''|*[!0-9]*) fails=0 ;; esac

if [ -r "$START_FILE" ]; then
    last=$(cat "$START_FILE" 2>/dev/null || echo 0)
    case "$last" in ''|*[!0-9]*) last=0 ;; esac
    now=$(date +%s)
    ran=$((now - last))
    if [ "$last" -gt 0 ] && [ "$ran" -ge "$FAST_FAIL_SECONDS" ]; then
        # The previous run was healthy for a while; this is a new incident.
        fails=0
        printf '0' >"$FAIL_FILE" 2>/dev/null
    fi
fi

if [ "$fails" -ge 2 ]; then
    backoff=$((1 << (fails - 2)))
    [ "$backoff" -gt "$MAX_BACKOFF" ] && backoff=$MAX_BACKOFF
    log "previous start failed $fails times in a row, waiting ${backoff}s before retry"
    sleep "$backoff"
fi

date +%s >"$START_FILE" 2>/dev/null

# ---- environment -----------------------------------------------------------
# ADSP_LIBRARY_PATH is what lets the adsp daemon find the DSP skeleton
# (libggml-htp-v73.so). It is not a linker path, so it cannot be replaced by
# putting the file in /vendor/lib64 alone.
# The image keeps every .so in PF_LIBDIR. A split layout (ggml in one directory,
# sherpa and onnxruntime in another) sets PF_LIBDIR_EXTRA, which is appended to
# both paths so the DSP skeleton stays findable as well.
export LD_LIBRARY_PATH="$PF_LIBDIR${PF_LIBDIR_EXTRA:+:$PF_LIBDIR_EXTRA}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ADSP_LIBRARY_PATH="$PF_LIBDIR${PF_LIBDIR_EXTRA:+:$PF_LIBDIR_EXTRA}${ADSP_LIBRARY_PATH:+:$ADSP_LIBRARY_PATH}"
# One session per physical device is all v73 supports; state it rather than
# relying on the backend's default.
export GGML_HEXAGON_DEVICES=${GGML_HEXAGON_DEVICES:-HTP0:0}
export GGML_HEXAGON_OPPOLL=${GGML_HEXAGON_OPPOLL:-1}

# ---- optional arguments ----------------------------------------------------
# The device is explicit by default. "auto" keeps the binary's probe, which is
# what a host wants; a device that forgets to pass anything would otherwise fall
# back to CPU silently, and the server refuses to start on CPU without
# --allow-cpu-only anyway.
PF_DEVICE=${PF_DEVICE:-HTP0:0}
set -- "$PF_MODEL" --port "$PF_PORT" --bind "$PF_BIND" \
    --engines "$PF_ENGINES" --threads "$PF_THREADS"
if [ "$PF_DEVICE" != "auto" ]; then
    set -- "$@" --device "$PF_DEVICE"
fi
if [ -n "${PF_MAX_WS_SESSIONS:-}" ]; then
    set -- "$@" --max-ws-sessions "$PF_MAX_WS_SESSIONS"
fi
if [ "${PF_ALLOW_CPU_FALLBACK:-0}" = "1" ]; then
    set -- "$@" --allow-cpu-fallback
fi

# A warmup file is optional on purpose. A missing one must not keep the service
# down; the server falls back to silence and logs it. Only pass the flag when
# the file is actually there, so the log stays clean on images that omit it.
if [ -n "${PF_WARMUP:-}" ] && [ -r "$PF_WARMUP" ]; then
    set -- "$@" --warmup "$PF_WARMUP"
elif [ -n "${PF_WARMUP:-}" ]; then
    log "warmup $PF_WARMUP is missing, starting without it"
fi

if [ -n "${PF_WEB_DIR:-}" ] && [ -d "$PF_WEB_DIR" ]; then
    set -- "$@" --web-dir "$PF_WEB_DIR"
fi
if [ -n "${PF_HOTWORDS:-}" ] && [ -r "$PF_HOTWORDS" ]; then
    # The engine reads these from the environment, not from argv; there is no
    # --hotwords on the API server (that flag exists on the CLI only).
    export PF_HOTWORDS
fi
if [ -n "${PF_TEXT_RULES:-}" ] && [ -r "$PF_TEXT_RULES" ]; then
    export PF_TEXT_RULES
fi
if [ -n "${PF_SPEAKER_MODEL:-}" ] && [ -r "$PF_SPEAKER_MODEL" ] && [ -n "${PF_ENROLL_DIR:-}" ] && [ -d "$PF_ENROLL_DIR" ]; then
    set -- "$@" --speaker-model "$PF_SPEAKER_MODEL" --enroll-dir "$PF_ENROLL_DIR"
fi
if [ -n "${PF_VAD_MODEL:-}" ] && [ -r "$PF_VAD_MODEL" ]; then
    set -- "$@" --vad-model "$PF_VAD_MODEL"
fi
# Prefer the HTP worker while it is free. Without this a single stream alternates
# between HTP and the CPU worker, so every other result comes from the slower
# backend (measured 265-283 ms on HTP against 192-512 ms on CPU).
if [ "${PF_PREFER_IDLE_HTP:-0}" = "1" ]; then
    set -- "$@" --prefer-idle-htp
fi
if [ -n "${PF_MODEL_ID:-}" ]; then
    set -- "$@" --model-id "$PF_MODEL_ID"
fi
if [ -n "${PF_EXTRA_ARGS:-}" ]; then
    # Deliberate word splitting: this is a flag list from a root-owned file.
    set -- "$@" $PF_EXTRA_ARGS
fi

# ---- credentials -----------------------------------------------------------
# Only the token goes in the config, never on the command line: argv is world
# readable through /proc, and this runs for the life of the car.
if [ -n "${PF_API_KEY_FILE:-}" ] && [ -r "$PF_API_KEY_FILE" ]; then
    PF_API_KEY=$(cat "$PF_API_KEY_FILE" 2>/dev/null)
fi
if [ -n "${PF_API_KEY:-}" ]; then
    export PF_API_KEY
fi

log "starting $PF_BIN $PF_MODEL on $PF_BIND:$PF_PORT engines=$PF_ENGINES"
# exec, not a child: init's SIGTERM has to reach the server so it drains
# in-flight requests instead of dropping them on every restart.
exec "$PF_BIN" "$@" >>"$LOG" 2>&1