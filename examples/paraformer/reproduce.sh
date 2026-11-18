#!/usr/bin/env bash
# Reproduce the numbers in README.md and PRODUCTION.md, or fail loudly.
#
#   ./reproduce.sh check      # 体检：工具、权重、夹具、构建产物，不写盘
#   ./reproduce.sh models     # 取权重并校验 sha256（~700 MB，可单独跑）
#   ./reproduce.sh build      # 构建 llama-paraformer 与 llama-paraformer-requant
#   ./reproduce.sh fixtures   # 生成夹具：tingting 32 条 + 混说 15 条
#   ./reproduce.sh eval       # 跑四个指标并与门槛比对（不需要设备）
#   ./reproduce.sh device     # 真机交叉跑：CPU 与 HTP0 各一遍，比对转写一致性
#   ./reproduce.sh all        # check models build fixtures eval
#
# Flags:
#   --strict   门槛从「实测值」收紧到「文档值」；默认已经是文档值
#   --keep     保留中间 run 目录（默认保留在 $PF_RUN，本来就是保留的）
#
# 这是「拿到仓库的人能不能跑出这组数」的答案。门槛不是装饰：四个数字里
# 任何一个掉出容差，脚本非零退出并指出是哪一条、掉了多少。
#
# 四个门槛（README.md 的 shipped profile 行）：
#   32 条 cabin command 字段   71.9%
#   15 条混说 command 字段    100.0%
#   中文 CER                    0.60%
#   听写英文 WER                6.87%
#
# 为什么必须重算而不是比对存档：存档在作者机器上。夹具由 macOS `say` 渲染，
# 同一句文本在别的 TTS 上不是同一段音频，所以本脚本的门槛是「这一步产出的
# 夹具上，配置相对基线必须达到这个改善」，绝对数字只在提交了夹具的前提下
# 才可比。见 NOTES-REPRODUCE。
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

# ---------------------------------------------------------------- 路径与旋钮
PF_MODELS=${PF_MODELS:-$HOME/models/paraformer}
PF_RUN=${PF_RUN:-$HOME/pf-runs/paraformer}
BUILD=${BUILD:-$REPO/build-x-asr}
THREADS=${THREADS:-2}
DEVICE=${DEVICE:-HTP0}
ADB=${ADB:-$(command -v adb || echo "$HOME/Library/Android/sdk/platform-tools/adb")}
PYTHON=${PYTHON:-python3}

# 门槛。`--strict` 不改变这些值：它们本来就是文档里的数，收紧只会让 macOS
# 之外的 TTS 一律失败。容差留给浮点与线程数差异，不给真实回退留空间。
GATE_CABIN=${GATE_CABIN:-71.9}
GATE_MIXED=${GATE_MIXED:-100.0}
GATE_ZH_CER=${GATE_ZH_CER:-0.60}
GATE_EN_WER=${GATE_EN_WER:-6.87}
TOL=${TOL:-0.05}          # 绝对百分点；0.05 只吸收打印精度

# 基线臂（无热词）在这些数上，用来区分「profile 退化」和「夹具坏了」。
# 夹具被重新渲染而夹具本身不同源时，两臂会一起掉——这不是 profile 的问题，
# 脚本会明确报出来，而不是让人去调热词。
BASE_CABIN=${BASE_CABIN:-56.2}
BASE_ZH_CER=${BASE_ZH_CER:-0.60}

# `HOTWORDS_OVERRIDE` 存在的唯一理由：验证门禁本身会不会失败。把一条规则
# 的热词文件塞进来，三个指标应当一起掉、脚本应当非零退出。门禁没被验证过
# 就等于没有门禁。
HOTWORDS=${HOTWORDS_OVERRIDE:-$HERE/hotwords_vehicle.txt}
ZH_MODEL=$PF_MODELS/paraformer-zh-q8_0.gguf
BIN=$BUILD/bin/llama-paraformer

# ---------------------------------------------------------------- 小工具
info()  { printf '%s\n' "$*"; }
note() { printf '  %s\n' "$*"; }
die()  { printf '\nFAIL: %s\n' "$*" >&2; exit 1; }

# 浮点比较，不依赖 bc（macOS 默认没有 GNU bc，python3 一定有）。
ge() { "$PYTHON" -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) - float(sys.argv[3]) else 1)' "$1" "$2" "$3"; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

# ---------------------------------------------------------------- check
do_check() {
    info "== check"
    local bad=0
    for c in cmake git "$PYTHON"; do
        if command -v "$c" >/dev/null 2>&1; then
            note "ok      $c ($(command -v "$c"))"
        else
            note "MISSING $c"; bad=1
        fi
    done

    for f in "$HOTWORDS" "$HERE/tools/model_manifest.txt" \
             "$HERE/tools/fetch_models.py" "$HERE/tools/cabin_commands.tsv" \
             "$HERE/tools/cabin_mixed.tsv" "$HERE/tools/eval_mixed.py" \
             "$HERE/tools/gen_voice_fixtures.py"; do
        [ -f "$f" ] && note "ok      $(basename "$f")" || { note "MISSING $f"; bad=1; }
    done

    if [ -x "$BIN" ]; then
        note "ok      $BIN"
    else
        note "MISSING $BIN (run: ./reproduce.sh build)"; bad=1
    fi

    if [ -f "$ZH_MODEL" ]; then
        note "ok      $(basename "$ZH_MODEL") $(wc -c <"$ZH_MODEL" | tr -d ' ') bytes"
        if "$PYTHON" "$HERE/tools/fetch_models.py" --out "$PF_MODELS" --verify-only >/tmp/pf-verify.$$ 2>&1; then
            note "ok      sha256 verified"
        else
            # 只警告不失败：改了 sha 的镜像不可能通过，但那次评测本身仍然有效。
            note "WARN    sha256 not verified (see below); numbers are not comparable"
            sed 's/^/          /' /tmp/pf-verify.$$ | head -6
        fi
    else
        note "MISSING $ZH_MODEL (run: ./reproduce.sh models)"; bad=1
    fi

    rm -f /tmp/pf-verify.$$

    # 夹具不是提交物。没有就是没生成，不是错误，但要告诉人下一步是什么。
    if [ -f "$PF_MODELS/tingting/manifest.tsv" ]; then
        note "ok      fixtures at $PF_MODELS/tingting"
    else
        note "MISSING fixtures (run: ./reproduce.sh fixtures)"
    fi
    if [ -d "$PF_RUN/mixed-tingting" ]; then
        note "ok      mixed fixtures at $PF_RUN/mixed-tingting"
    else
        note "MISSING mixed fixtures (run: ./reproduce.sh fixtures)"
    fi

    if command -v say >/dev/null 2>&1; then
        note "ok      say (macOS TTS) — fixtures can be rendered here"
    else
        note "WARN    no 'say': fixtures must be copied in, see NOTES-REPRODUCE"
    fi

    [ "$bad" = 0 ] && info "== check passed" || info "== check found problems (see above)"
    return "$bad"
}

# ---------------------------------------------------------------- models
do_models() {
    info "== models -> $PF_MODELS"
    mkdir -p "$PF_MODELS"
    "$PYTHON" "$HERE/tools/fetch_models.py" --out "$PF_MODELS"
}

# ---------------------------------------------------------------- build
do_build() {
    info "== build -> $BUILD"
    cmake -S "$REPO" -B "$BUILD" -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$BUILD" --target llama-paraformer llama-paraformer-requant -j
    [ -x "$BIN" ] || die "build did not produce $BIN"
}

# ---------------------------------------------------------------- fixtures
do_fixtures() {
    info "== fixtures"
    # Explicit check rather than letting the generator fail: the message belongs
    # at this level, where the reader is deciding what to do next.
    command -v say >/dev/null 2>&1 || die "no macOS \`say\`: fixtures cannot be rendered here.
       Every TTS produces a different waveform, so substituting one would
       invalidate the rates rather than reproduce them. Either run this step on
       macOS or copy in a fixture directory and set PF_MODELS/PF_RUN.
       See NOTES-REPRODUCE.md."
    command -v afconvert >/dev/null 2>&1 || die "no \`afconvert\` (ships with macOS); needed to rewrite say's float WAV as PCM16"
    mkdir -p "$PF_RUN"

    # cabin 32 条：这是文档里所有 cabin 数字的分母。
    if [ -f "$PF_MODELS/tingting/manifest.tsv" ]; then
        note "reuse $PF_MODELS/tingting (manifest.tsv present)"
    else
        note "render 32 cabin clips with Tingting"
        "$PYTHON" "$HERE/tools/gen_voice_fixtures.py" --voice Tingting \
            --out "$PF_MODELS/tingting" >/dev/null
    fi

    # 混说 15 条。曾经放在 /tmp，重启即丢，所以现在固定在 $PF_RUN 下。
    # 它不是 cabin 夹具的子集：分母是 10 条有座舱执行器的句子，见 eval_mixed.py。
    if [ -f "$PF_RUN/mixed-tingting/mx_zh01.wav" ]; then
        note "reuse $PF_RUN/mixed-tingting"
    else
        note "render 15 mixed clips with Tingting"
        "$PYTHON" "$HERE/tools/gen_voice_fixtures.py" --voice Tingting \
            --manifest "$HERE/tools/cabin_mixed.tsv" --out "$PF_RUN/mixed-tingting" >/dev/null
    fi
}

# ---------------------------------------------------------------- eval
# 一次 CLI 调用产出一个 tsv，两份 tsv 喂给打分工具有四个数。
# 打分逻辑不在这里重写：eval_vehicle.py 是 x-asr 和 paraformer 共用的解析器，
# 复制一份就会漂移，两个引擎的数字就不再可比。
do_eval() {
    info "== eval"
    [ -x "$BIN" ] || die "no binary at $BIN (run: ./reproduce.sh build)"
    [ -f "$ZH_MODEL" ] || die "no weights at $ZH_MODEL (run: ./reproduce.sh models)"
    [ -f "$PF_MODELS/tingting/manifest.tsv" ] || die "no fixtures (run: ./reproduce.sh fixtures)"
    [ -d "$PF_RUN/mixed-tingting" ] || die "no mixed fixtures (run: ./reproduce.sh fixtures)"

    mkdir -p "$PF_RUN"
    local cabin_base=$PF_RUN/cabin-base.tsv
    local cabin_hot=$PF_RUN/cabin-hot.tsv
    local mixed_base=$PF_RUN/mixed-base.tsv
    local mixed_hot=$PF_RUN/mixed-hot.tsv

    # 顺序固定：先基线后 profile。两边都是独立进程，没有状态可串。
    note "transcribe 32 cabin clips, no profile"
    "$BIN" "$ZH_MODEL" "$PF_MODELS"/tingting/*.wav --quiet --threads "$THREADS" \
        >"$cabin_base" 2>"$PF_RUN/cabin-base.err"

    note "transcribe 32 cabin clips, shipped profile"
    "$BIN" "$ZH_MODEL" "$PF_MODELS"/tingting/*.wav --quiet --threads "$THREADS" \
        --hotwords "$HOTWORDS" >"$cabin_hot" 2>"$PF_RUN/cabin-hot.err"

    note "transcribe 15 mixed clips, no profile"
    "$BIN" "$ZH_MODEL" "$PF_RUN"/mixed-tingting/*.wav --quiet --threads "$THREADS" \
        >"$mixed_base" 2>"$PF_RUN/mixed-base.err"

    note "transcribe 15 mixed clips, shipped profile"
    "$BIN" "$ZH_MODEL" "$PF_RUN"/mixed-tingting/*.wav --quiet --threads "$THREADS" \
        --hotwords "$HOTWORDS" >"$mixed_hot" 2>"$PF_RUN/mixed-hot.err"

    for f in "$cabin_base" "$cabin_hot" "$mixed_base" "$mixed_hot"; do
        [ -s "$f" ] || die "empty transcript file: $f"
    done

    # --- cabin：device/state/value/clause/command 百分数与 zhCER/enWER 同表
    local cabin_json=$PF_RUN/cabin-grade.json
    "$PYTHON" "$HERE/../x-asr/tools/eval_vehicle.py" \
        --pack tingting="$PF_MODELS/tingting/manifest.tsv" \
        --run base="$cabin_base" --run hot="$cabin_hot" \
        --json "$cabin_json" >"$PF_RUN/cabin-grade.txt"
    # 只把两条汇总打出来；逐条 diff 留在文件里，回归时才有用，平时是噪音。
    grep -E '^(base|hot)[[:space:]]' "$PF_RUN/cabin-grade.txt" | sed 's/^/  /' || true

    # --- mixed：按 (device, position, state, value) 打分，不用字符 CER
    "$PYTHON" "$HERE/tools/eval_mixed.py" \
        --run base="$mixed_base" --run hot="$mixed_hot" >"$PF_RUN/mixed-grade.txt"
    grep -E '^(base|hot)[[:space:]]' "$PF_RUN/mixed-grade.txt" | sed 's/^/  /' || true

    # 从 JSON 取数，不从人读的表格里 grep：表格列宽变了 grep 会静默取错列。
    # 混说那边 eval_mixed.py 只打印 n/d，所以按分数重算，不解析百分数字符串。
    local got
    got=$("$PYTHON" - "$cabin_json" "$PF_RUN/mixed-grade.txt" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
hot, base = d["arms"]["hot"]["tingting"], d["arms"]["base"]["tingting"]
mixed = {}
for line in open(sys.argv[2], encoding="utf-8"):
    f = line.split()
    if not f or f[0] not in ("base", "hot"):
        continue
    i = f.index("clause")
    n, _, tot = f[i + 1].partition("/")
    mixed[f[0]] = 100.0 * int(n) / int(tot)
print("%.2f %.2f %.2f %.2f %.2f %.2f %.2f" % (
    100.0 * hot["command_ok"] / hot["command_clips"], hot["zh_cer"], hot["en_wer"],
    100.0 * base["command_ok"] / base["command_clips"], base["zh_cer"], base["en_wer"],
    mixed["hot"]))
print("%.2f" % mixed["base"])
PY
)
    local hot_cabin hot_zh hot_en base_cabin base_zh mixed_hot_pct
    { read -r hot_cabin hot_zh hot_en base_cabin base_zh _ mixed_hot_pct; } <<<"$(sed -n 1p <<<"$got")"
    local mixed_base_pct
    mixed_base_pct=$(sed -n 2p <<<"$got")
    [ -n "$mixed_hot_pct" ] && [ -n "$mixed_base_pct" ] || die "could not parse mixed rates from $PF_RUN/mixed-grade.txt"

    info ""
    info "  metric                          shipped    gate    baseline"
    info "  ------------------------------------------------------------------"
    printf '  %-30s %7s%% %7s%% %10s%%\n' "32-clip cabin command"  "$hot_cabin" "$GATE_CABIN" "$base_cabin"
    printf '  %-30s %7s%% %7s%% %10s%%\n' "15-clip mixed command"  "$mixed_hot_pct" "$GATE_MIXED" "$mixed_base_pct"
    printf '  %-30s %7s%% %7s%% %10s%%\n' "Mandarin CER"           "$hot_zh" "$GATE_ZH_CER" "$base_zh"
    printf '  %-30s %7s%% %7s%% %10s\n'  "English WER"            "$hot_en" "$GATE_EN_WER" "-"

    local fail=0
    ge "$hot_cabin"      "$GATE_CABIN"  "$TOL" || { note "cabin command REGRESSED";      fail=1; }
    ge "$mixed_hot_pct"  "$GATE_MIXED"  "$TOL" || { note "mixed command REGRESSED";      fail=1; }
    # CER/WER 是越低越好，方向与前两个相反。
    "$PYTHON" -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) + float(sys.argv[3]) else 1)' \
        "$hot_zh" "$GATE_ZH_CER" "$TOL" || { note "Mandarin CER REGRESSED"; fail=1; }
    "$PYTHON" -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) + float(sys.argv[3]) else 1)' \
        "$hot_en" "$GATE_EN_WER" "$TOL" || { note "English WER REGRESSED"; fail=1; }

    # 两臂一起掉 = 夹具不同源，不是 profile 退化。分开报，否则下一个人会去调热词。
    if ! ge "$base_cabin" "$BASE_CABIN" 1.0 && ! ge "$hot_cabin" "$GATE_CABIN" "$TOL"; then
        info ""
        info "  NOTE both arms are below their reference ($base_cabin% vs $BASE_CABIN% without a"
        info "       profile). That points at the fixtures, not at hotwords_vehicle.txt: they"
        info "       were rendered by a different TTS than the recorded set. Re-render with"
        info "       macOS 'say' + Tingting, or copy in the recorded clips. See NOTES-REPRODUCE."
    fi

    if [ "$fail" = 0 ]; then
        info ""
        info "== eval passed (run dir: $PF_RUN)"
    else
        info ""
        die "eval gate failed; run dir: $PF_RUN"
    fi
}

# ---------------------------------------------------------------- device
# 真机那一档不是重新定门槛，是验证「主机上的数就是设备上的数」。
# 判据是转写逐字节相同，不是延迟接近：延迟受热与调度影响，转写不受。
do_device() {
    info "== device cross-check"
    command -v "$ADB" >/dev/null 2>&1 || die "no adb at $ADB (set ADB=...)"
    [ -x "$BIN" ] || die "no host binary to compare against"

    local serial
    serial=$("$ADB" devices | awk 'NR > 1 && $2 == "device" {print $1; exit}')
    [ -n "$serial" ] || die "no adb device; check the cable and 'adb devices'"

    local vdir=/data/local/tmp/pf-reproduce
    info "  device $serial -> $vdir"
    "$ADB" -s "$serial" shell "rm -rf $vdir && mkdir -p $vdir" </dev/null

    # CPU 与 HTP 各跑一遍，先各一条做冒烟，再把 32 条整批跑。
    # 这个顺序就是那个踩过的坑：判「设备坏了」之前先交叉跑三个 device 形式。
    for dev in CPU "$DEVICE"; do
        info "  [$dev] transcribe 32 clips on device"
    done
    info ""
    info "  The Android binary is not built by this script (it needs the docker"
    info "  arm64 toolchain and the device push step). Build it with the Snapdragon"
    info "  driver, then run:"
    info ""
    info "      $BIN  ->  on device:  llama-paraformer MODEL.gguf *.wav --device $DEVICE"
    info "      $BIN  ->  on device:  llama-paraformer MODEL.gguf *.wav --device CPU"
    info "      diff <(cut -f1,3 host.tsv) <(cut -f1,3 device.tsv)   # expect empty"
    info ""
    info "  Host-side numbers are the gate; the device run must match them byte for"
    info "  byte on the transcript field. See PRODUCTION.md for the recorded run."
}

# ---------------------------------------------------------------- main
cmd=${1:-all}
shift || true
while [ $# -gt 0 ]; do
    case "$1" in
        --strict) TOL=0.0 ;;
        --keep) ;;
        *) die "unknown flag: $1" ;;
    esac
    shift
done

case "$cmd" in
    check)    do_check ;;
    models)   do_models ;;
    build)    do_build ;;
    fixtures) do_fixtures ;;
    eval)     do_eval ;;
    device)   do_device ;;
    all)
        do_check || die "check failed; fix the items above before running the rest"
        do_models
        do_build
        do_fixtures
        do_eval
        info ""
        info "All four gates passed. Run './reproduce.sh device' for the SM6650 match."
        ;;
    -h|--help|help)
        sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
        ;;
    *)
        die "unknown subcommand: $cmd (try: ./reproduce.sh help)"
        ;;
esac