#!/usr/bin/env python3
"""Check that /metrics is valid Prometheus exposition and agrees with /metrics.json.

Point it at a running server:

    python3 tools/check_metrics.py --port 18810

Why this exists rather than trusting promtool: promtool validates a *file*, and
what has to be validated is a running process whose counters move between the
two requests. The failure this catches is specific - /metrics and /metrics.json
are built from one snapshot, and if someone later adds a counter to one renderer
and forgets the other, the dashboard and the deploy gate disagree while both
keep returning 200.

Asserts:

1. Content type is an exposition text type, not application/json. A scraper
   keys off this; serving JSON here is the original bug.
2. Every sample line is `name value` with a numeric value, and every sample has
   a preceding `# TYPE`. Prometheus rejects a metric whose type it never saw.
3. No name ends in `_total` unless it is declared a counter. A `_total` on a
   gauge makes rate() and increase() produce nonsense, and the suffix is the
   only signal a reader gets.
4. Gauges are not declared as counters. queue, inflight, accepted, engines,
   ws_sessions, queue_bytes* and cpu_fallback all go up and down; declaring one
   a counter is silently wrong forever after.
5. Every series in the text is present in the JSON and carries the same value.
   One request apart, so this compares the *last* reading of each: a burst of
   traffic between the two calls would show up as a counter mismatch, which is
   why the JSON is fetched twice and the check is only enforced when the two
   JSON readings agree with each other.
6. Sizes are exact integers. 8388608 printed as 8.38861e+06 parses fine, so an
   alert built on the cap is off by 8 bytes and nothing complains.
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("%s %-58s %s" % ("ok  " if ok else "FAIL", name, detail))


def get(port, path, timeout=15):
    url = "http://127.0.0.1:%d%s" % (port, path)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


# Metrics that are expected to move up and down. Declaring one as a counter
# cannot be caught by format validation, only by knowing what it means.
GAUGES = {
    "pf_queue_depth", "pf_queue_bytes", "pf_queue_bytes_limit", "pf_inflight",
    "pf_accepted", "pf_ws_sessions", "pf_engines", "pf_cpu_fallback",
}


def parse_exposition(text):
    """Return ({name: value}, {name: type}, [problems]).

    Strict on purpose: a lenient parser here would let the server ship output
    that a real Prometheus server then silently drops the metric from.
    """
    values, types, problems = {}, {}, []
    for i, line in enumerate(text.splitlines(), 1):
        if not line or line.startswith("#"):
            if line.startswith("# TYPE "):
                parts = line.split()
                if len(parts) != 4:
                    problems.append("line %d: TYPE arity %r" % (i, line))
                else:
                    types[parts[2]] = parts[3]
            elif line.startswith("# HELP ") and len(line.split(None, 3)) < 4:
                problems.append("line %d: HELP without a docstring" % i)
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            problems.append("line %d: sample is not 'name value': %r" % (i, line))
            continue
        name, raw = parts
        if re.search(r"[{}\"']", name):
            # Labels are legal in the format; this server has no labelled
            # series, so a label appearing means someone built one by accident.
            problems.append("line %d: unexpected label syntax in %r" % (i, name))
        try:
            values[name] = float(raw)
        except ValueError:
            problems.append("line %d: value is not numeric: %r" % (i, line))
    for name in values:
        if name not in types:
            problems.append("%s has a sample but no # TYPE" % name)
    for name in types:
        if name not in values:
            problems.append("%s is declared but never sampled" % name)
    return values, types, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--verbose", action="store_true", help="print the exposition body")
    args = ap.parse_args()

    # --- format
    st, ctype, body = get(args.port, "/metrics")
    check("/metrics answers", st == 200, "http=%d" % st)
    text = body.decode("utf-8", "replace")
    check("/metrics content type is exposition text",
          ctype.startswith("text/plain") and "version=0.0.4" in ctype, ctype)
    check("/metrics is not JSON", not ctype.startswith("application/json"), ctype)

    values, types, problems = parse_exposition(text)
    check("exposition parses with no problems", not problems, "; ".join(problems[:3]))
    check("exposition has series", len(values) > 0, "%d series" % len(values))

    # --- naming and type conventions
    bad_total = [n for n, t in types.items() if n.endswith("_total") and t != "counter"]
    check("no gauge is named *_total", not bad_total, ",".join(bad_total))
    bad_gauge = [n for n in GAUGES if types.get(n) == "counter"]
    check("gauges are not declared as counters", not bad_gauge, ",".join(bad_gauge))
    undeclared_gauge = [n for n in GAUGES if n not in types]
    check("every known gauge is exported", not undeclared_gauge, ",".join(undeclared_gauge))

    # --- sizes must be exact, not 8.38861e+06
    for name in ("pf_queue_bytes_limit", "pf_queue_bytes", "pf_queue_depth"):
        if name in values:
            raw = [l.rsplit(None, 1)[1] for l in text.splitlines()
                   if not l.startswith("#") and l.rsplit(None, 1)[0] == name]
            exact = raw and re.fullmatch(r"-?\d+", raw[0])
            check("%s is printed as an integer" % name, exact, raw[0] if raw else "missing")

    # --- agreement with the JSON twin
    st, jtype, jbody = get(args.port, "/metrics.json")
    check("/metrics.json answers", st == 200, "http=%d" % st)
    check("/metrics.json is JSON", jtype.startswith("application/json"), jtype)
    try:
        j = json.loads(jbody.decode())
    except ValueError as e:
        check("/metrics.json parses", False, str(e))
        return finish()

    st2, _, jbody2 = get(args.port, "/metrics.json")
    try:
        j2 = json.loads(jbody2.decode())
    except ValueError:
        j2 = {}
    stable = j == j2

    # Name mapping is explicit rather than inferred: `queue` -> `pf_queue_depth`
    # is not a rule, it is a decision, and a silent fallback to `pf_` + name
    # would agree with itself while both sides were wrong.
    pairs = {
        "requests": "pf_requests", "rejected": "pf_rejected", "failed": "pf_failed",
        "deadline_expired": "pf_deadline_expired", "cancelled_queued": "pf_cancelled_queued",
        "auth_rejected": "pf_auth_rejected", "partials": "pf_partials",
        "partials_skipped": "pf_partials_skipped", "speaker_started": "pf_speaker_started",
        "speaker_assessed": "pf_speaker_assessed", "speaker_deferred": "pf_speaker_deferred",
        "queue": "pf_queue_depth", "queue_bytes": "pf_queue_bytes",
        "queue_bytes_limit": "pf_queue_bytes_limit", "inflight": "pf_inflight",
        "accepted": "pf_accepted", "ws_sessions": "pf_ws_sessions", "engines": "pf_engines",
    }
    missing = [k for k in pairs if k not in j]
    check("JSON carries every field the text exports", not missing, ",".join(missing))

    if not stable:
        # Counters moved between the two JSON reads, so a value comparison would
        # report a mismatch that is really traffic. The set comparison still
        # holds, and it is the part that matters.
        print("note: counters moved between the JSON reads, comparing names only")
        only_text = sorted(set(values) - set(pairs.values()))
        only_json = sorted(pairs[v] for v in pairs if v not in j)
        check("text exports no series the JSON lacks", not only_text, ",".join(only_text))
        check("JSON exports no series the text lacks", not only_json, ",".join(only_json))
    else:
        mismatched, absent = [], []
        for jk, pk in pairs.items():
            if jk not in j:
                absent.append(jk)
                continue
            if pk not in values:
                absent.append(pk)
                continue
            want = 1.0 if j[jk] is True else 0.0 if j[jk] is False else float(j[jk])
            if abs(values[pk] - want) > 0:
                mismatched.append("%s=%g vs %s=%g" % (pk, values[pk], jk, want))
        check("text and JSON agree on every value", not mismatched and not absent,
              "; ".join(mismatched + absent))
        check("cpu_fallback maps to 1/0", values.get("pf_cpu_fallback") in (0.0, 1.0),
              str(values.get("pf_cpu_fallback")))

    if args.verbose:
        print()
        print(text)
    return finish()


def finish():
    failed = [n for n, ok, _ in RESULTS if not ok]
    print("")
    if failed:
        print("%d/%d checks failed" % (len(failed), len(RESULTS)))
        return 1
    print("%d/%d checks passed" % (len(RESULTS), len(RESULTS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())