#!/bin/sh
# Run on the device with the backend library paths already set.
set -eu

if [ "$#" -lt 3 ]; then
    printf 'usage: %s API MODEL HTP_DEVICE [API OPTIONS...]\n' "$0" >&2
    exit 1
fi
api=$1
model=$2
device=$3
shift 3
case "$device" in
    HTP*) ;;
    *) printf 'expected an HTP device\n' >&2; exit 1 ;;
esac
for opt in "$@"; do
    case "$opt" in
        --device|--allow-cpu-fallback|--recover-on-cpu)
            printf '%s is owned by the supervisor\n' "$opt" >&2
            exit 1 ;;
    esac
done

child=
stop() {
    if [ -n "$child" ]; then
        kill "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    exit 0
}
trap stop INT TERM

launch() {
    mode=$1
    shift
    "$api" "$model" --device "$device" "$@" &
    child=$!
    printf 'pf-supervisor: started mode=%s pid=%s\n' "$mode" "$child" >&2
    status=0
    wait "$child" || status=$?
    child=
    printf 'pf-supervisor: exited mode=%s status=%s\n' "$mode" "$status" >&2
}

launch htp "$@"
if [ "$status" -eq 0 ]; then
    exit 0
fi
launch cpu --recover-on-cpu "$@"
exit "$status"
