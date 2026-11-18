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
# 这是「拿到仓库的人能不能跑出这组数」的答案。门槛不是装饰：任何一个数字
# 掉出容差，脚本非零退出并指出是哪一条、掉了多少。
#
# 五个门槛（README.md / PRODUCTION.md 的 shipped profile 行）：
#   32 条 cabin command 字段   71.9%
#   15 条混说 command 字段    100.0%
#   中文 CER                    0.60%
#   听写英文 WER                6.87%
#   84 条扩展域 command 字段   91.7%   （夹具存在时才跑，见下面 have_exp）
#
# 前四个的分母是「窗、温度、座椅加热、氛围灯」这四类，第五个是雾灯/除霜/
# 空调模式/座椅/天窗/导航/点单/多媒体。分开设门槛的理由写在 GATE_CABIN_EXP 的
# 注释里：混在一起会让 71.9% 这个数失去含义。
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
# 七个新增域（雾灯/除霜/空调模式/座椅/天窗/导航/点单/多媒体）84 条夹具的
# command 字段门槛。这组夹具覆盖的是「产品要的车控」，而 32 条那组只有窗、温度、
# 座椅加热和氛围灯。两者分开设门槛，因为分母不同、难度不同：84 条里有专有名词
# （瑞幸、王府井、天安门）这些热词接不住的样本，混进 32 条那组的百分比里只会
# 让 71.9% 这个数变得没有含义。
GATE_CABIN_EXP=${GATE_CABIN_EXP:-91.7}
BASE_CABIN_EXP=${BASE_CABIN_EXP:-72.6}
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
             "$HERE/tools/gen_voice_fixtures.py" \
             "$HERE/tools/cabin_expanded.tsv" "$HERE/tools/vehicle_semantics_expanded.tsv" \
             "$HERE/tools/check_vocab_coverage.py" "$HERE/tools/check_hotword_rules.py"; do
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

    # 每条 span rewrite 的 token 必须真的在词表里，且末片必须收词。
    # 这两条错法的共同点是**静默**：解析器在加载时打一行
    # "replacement token not in vocab" 就丢掉规则，之后 deficit、命中数、
    # 分数全部正常，只有转写不变——读起来像模型到了极限，实际是规则写错。
    # 所以必须机械地查，不能靠读日志。
    if [ -f "$ZH_MODEL" ] && [ -x "$BIN" ]; then
        local vocab_dump=$PF_RUN/vocab-zh.txt
        mkdir -p "$PF_RUN"
        if [ ! -s "$vocab_dump" ]; then
            # A vocabulary dump needs one clip to trigger it (the CLI takes
            # MODEL WAV...), so borrow whichever fixture exists. The audio is
            # irrelevant: the dump is the token table.
            local probe_wav=
            for cand in "$PF_MODELS/tingting"/*.wav "$PF_RUN/cabin-expanded"/*.wav; do
                [ -f "$cand" ] && { probe_wav=$cand; break; }
            done
            if [ -n "$probe_wav" ]; then
                PF_DUMP_VOCAB= "$BIN" "$ZH_MODEL" "$probe_wav" --quiet \
                    >/dev/null 2>"$vocab_dump.tmp" || true
                grep '^pf-vocab: ' "$vocab_dump.tmp" | sed 's/^pf-vocab: //' >"$vocab_dump" || true
                rm -f "$vocab_dump.tmp"
            fi
        fi
        if [ -s "$vocab_dump" ]; then
            if "$PYTHON" "$HERE/tools/check_hotword_rules.py" \
                    --vocab "$vocab_dump" --hotwords "$HOTWORDS" >"$PF_RUN/hotword-rules.txt" 2>&1; then
                note "ok      hotword rules match the vocabulary ($(tail -1 "$PF_RUN/hotword-rules.txt"))"
            else
                note "FAIL    hotword rules do not match the vocabulary:"
                sed 's/^/          /' "$PF_RUN/hotword-rules.txt" | head -20
                bad=1
            fi
        else
            note "WARN    no vocabulary dump; hotword rule check skipped"
        fi
    fi

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
    # 扩展域夹具由 cabin_expanded.tsv 渲染，渲染需要 macOS say，所以这里只报
    # 有没有，不因此判失败：缺了它第五个门槛跳过，前四个照样跑完。
    if [ -f "$PF_RUN/cabin-expanded/manifest.tsv" ]; then
        note "ok      expanded-domain fixtures at $PF_RUN/cabin-expanded"
    else
        note "MISSING expanded-domain fixtures (5th gate will be skipped)"
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

    # 84 条扩展域。夹具名必须带 .wav：generator 用 id 列做文件名，缺后缀
    # 就渲染出一堆没有扩展名的文件，而 CLI 的 glob 只收 *.wav，于是 84 条
    # 全部"存在但没跑"，门禁读到空文件才报错。这个坑踩过一次。
    if [ -f "$PF_RUN/cabin-expanded/lt2_zh01.wav" ]; then
        note "reuse $PF_RUN/cabin-expanded"
    else
        note "render 84 expanded-domain clips with Tingting"
        "$PYTHON" "$HERE/tools/gen_voice_fixtures.py" --voice Tingting \
            --manifest "$HERE/tools/cabin_expanded.tsv" --out "$PF_RUN/cabin-expanded" >/dev/null
    fi
}

# ---------------------------------------------------------------- eval
# 一次 CLI 调用产出一个 tsv，两份 tsv 喂给打分工具有五个数。
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

    # 扩展域 84 条。第 5 个门禁，独立分母，只在夹具存在时跑：它不是所有
    # 移植都能复现的（需要同一套 TTS 渲染 + 同一棵中文权重），但一旦存在就
    # 必须过，否则新增的七个域退回「没人测过」的状态。
    local exp_base=$PF_RUN/cabin-expanded-base.tsv
    local exp_hot=$PF_RUN/cabin-expanded-hot.tsv
    local exp_dir=$PF_RUN/cabin-expanded
    local have_exp=0
    if [ -f "$exp_dir/manifest.tsv" ]; then
        have_exp=1
        note "transcribe 84 expanded-domain clips, no profile"
        "$BIN" "$ZH_MODEL" "$exp_dir"/*.wav --quiet --threads "$THREADS" \
            >"$exp_base" 2>"$PF_RUN/cabin-expanded-base.err"
        note "transcribe 84 expanded-domain clips, shipped profile"
        "$BIN" "$ZH_MODEL" "$exp_dir"/*.wav --quiet --threads "$THREADS" \
            --hotwords "$HOTWORDS" >"$exp_hot" 2>"$PF_RUN/cabin-expanded-hot.err"
        for f in "$exp_base" "$exp_hot"; do
            [ -s "$f" ] || die "empty transcript file: $f"
        done
    else
        note "skip 84 expanded-domain clips (no $exp_dir/manifest.tsv)"
    fi

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

    # --- 扩展域：同一个解析器，另一份标注，另一份夹具
    local exp_json=$PF_RUN/cabin-expanded-grade.json
    if [ "$have_exp" = 1 ]; then
        "$PYTHON" "$HERE/../x-asr/tools/eval_vehicle.py" \
            --semantics "$HERE/tools/vehicle_semantics_expanded.tsv" \
            --pack cabin-expanded="$exp_dir/manifest.tsv" \
            --run base="$exp_base" --run hot="$exp_hot" \
            --json "$exp_json" >"$PF_RUN/cabin-expanded-grade.txt"
        grep -E '^(base|hot)[[:space:]]' "$PF_RUN/cabin-expanded-grade.txt" | sed 's/^/  /' || true
    fi

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

    local exp_hot_cabin=- exp_base_cabin=-
    if [ "$have_exp" = 1 ]; then
        read -r exp_hot_cabin exp_base_cabin < <("$PYTHON" - "$exp_json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
hot, base = d["arms"]["hot"]["cabin-expanded"], d["arms"]["base"]["cabin-expanded"]
print("%.2f %.2f" % (100.0 * hot["command_ok"] / hot["command_clips"],
                     100.0 * base["command_ok"] / base["command_clips"]))
PY
)
    fi

    info ""
    info "  metric                          shipped    gate    baseline"
    info "  ------------------------------------------------------------------"
    printf '  %-30s %7s%% %7s%% %10s%%\n' "32-clip cabin command"  "$hot_cabin" "$GATE_CABIN" "$base_cabin"
    printf '  %-30s %7s%% %7s%% %10s%%\n' "15-clip mixed command"  "$mixed_hot_pct" "$GATE_MIXED" "$mixed_base_pct"
    printf '  %-30s %7s%% %7s%% %10s%%\n' "Mandarin CER"           "$hot_zh" "$GATE_ZH_CER" "$base_zh"
    printf '  %-30s %7s%% %7s%% %10s\n'  "English WER"            "$hot_en" "$GATE_EN_WER" "-"
    if [ "$have_exp" = 1 ]; then
        printf '  %-30s %7s%% %7s%% %10s%%\n' "84-clip expanded command" "$exp_hot_cabin" "$GATE_CABIN_EXP" "$exp_base_cabin"
    fi

    local fail=0
    ge "$hot_cabin"      "$GATE_CABIN"  "$TOL" || { note "cabin command REGRESSED";      fail=1; }
    ge "$mixed_hot_pct"  "$GATE_MIXED"  "$TOL" || { note "mixed command REGRESSED";      fail=1; }
    # CER/WER 是越低越好，方向与前两个相反。
    "$PYTHON" -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) + float(sys.argv[3]) else 1)' \
        "$hot_zh" "$GATE_ZH_CER" "$TOL" || { note "Mandarin CER REGRESSED"; fail=1; }
    "$PYTHON" -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) + float(sys.argv[3]) else 1)' \
        "$hot_en" "$GATE_EN_WER" "$TOL" || { note "English WER REGRESSED"; fail=1; }
    if [ "$have_exp" = 1 ]; then
        ge "$exp_hot_cabin" "$GATE_CABIN_EXP" "$TOL" \
            || { note "expanded-domain command REGRESSED"; fail=1; }
    fi

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
# 三臂都要跑（host / device-CPU / device-HTP）：只有 host 与 HTP 一对一错
# 时分不清是 HTP 传输层坏了还是参数解析吞了请求，那条坑踩过。
do_device() {
    info "== device cross-check"
    command -v "$ADB" >/dev/null 2>&1 || die "no adb at $ADB (set ADB=...)"
    [ -x "$BIN" ] || die "no host binary to compare against (run: ./reproduce.sh build)"
    [ -f "$ZH_MODEL" ] || die "no weights at $ZH_MODEL (run: ./reproduce.sh models)"
    [ -f "$PF_MODELS/tingting/manifest.tsv" ] || die "no fixtures (run: ./reproduce.sh fixtures)"

    local serial
    serial=$("$ADB" devices | awk 'NR > 1 && $2 == "device" {print $1; exit}')
    [ -n "$serial" ] || die "no adb device; check the cable and 'adb devices'"

    # 设备侧目录：二进制与权重通常已经在设备上（部署步骤不属于本脚本），
    # 但夹具是当前的，必须现推——推旧夹具比对的是两段不同的音频，会得出
    # 「HTP 与主机不一致」的假结论。
    local vbin=${DEVICE_BIN:-/data/local/tmp/llama.cpp}
    local vmodel=${DEVICE_MODEL:-/data/local/tmp/llama-paraformer/paraformer-zh-q8_0.gguf}
    local vdir=/data/local/tmp/pf-xcheck

    "$ADB" -s "$serial" shell "test -x $vbin/bin/llama-paraformer" </dev/null 2>/dev/null \
        || die "no llama-paraformer at $vbin/bin on the device.
       Build the Android target and push it, or set DEVICE_BIN=<dir>."
    "$ADB" -s "$serial" shell "test -f $vmodel" </dev/null 2>/dev/null \
        || die "no model at $vmodel on the device. Push it, or set DEVICE_MODEL=<path>."

    info "  device $serial"
    "$ADB" -s "$serial" shell "rm -rf $vdir && mkdir -p $vdir/wav" </dev/null
    # 推送本身不输出：每行一个文件的进度会把真正的结果埋掉。
    info "  push current fixtures (32 clips + profile)"
    "$ADB" -s "$serial" push "$PF_MODELS"/tingting/*.wav "$vdir/wav/" </dev/null >/dev/null 2>&1
    "$ADB" -s "$serial" push "$HOTWORDS" "$vdir/hotwords.txt" </dev/null >/dev/null 2>&1
    "$ADB" -s "$serial" shell "ls $vdir/wav/*.wav | wc -l" </dev/null | tr -d '\r' \
        | grep -qx 32 || die "expected 32 clips on the device, got a different count"

    mkdir -p "$PF_RUN"
    local host_tsv=$PF_RUN/xc-host.tsv
    local cpu_tsv=$PF_RUN/xc-device-cpu.tsv
    local htp_tsv=$PF_RUN/xc-device-htp.tsv

    info "  [host]      transcribe 32 clips"
    "$BIN" "$ZH_MODEL" "$PF_MODELS"/tingting/*.wav --quiet --threads "$THREADS" \
        --hotwords "$HOTWORDS" >"$host_tsv" 2>/dev/null

    # CPU 在前：它是已知能跑的那一臂。若 CPU 也不一致，问题在构建或权重，
    # 不在 HTP，这时不该去看 HTP 日志。
    local dev
    for dev in CPU "$DEVICE"; do
        info "  [$dev] transcribe 32 clips on device"
        # shell 里的 glob 由设备侧展开，不用引号——本地展开会传 32 个绝对路径。
        "$ADB" -s "$serial" shell \
            "cd $vbin && LD_LIBRARY_PATH=bin ./bin/llama-paraformer $vmodel $vdir/wav/*.wav \
             --device $dev --threads $THREADS --quiet --hotwords $vdir/hotwords.txt" \
            </dev/null >"$( [ "$dev" = CPU ] && echo "$cpu_tsv" || echo "$htp_tsv" )" 2>/dev/null
    done

    local tsv
    for tsv in "$host_tsv" "$cpu_tsv" "$htp_tsv"; do
        [ -s "$tsv" ] || die "empty transcript file: $tsv"
    done

    # 只比转写字段。路径不同（本地 vs /data/local/tmp）、wall_ms 必然不同，
    # 拿整行 diff 只会得到一堆噪音，然后没人再看这个检查。
    "$PYTHON" - "$host_tsv" "$cpu_tsv" "$htp_tsv" "$PF_RUN" <<'PY'
import os, sys

def load(path):
    rows = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        f = line.rstrip("\n").split("\t")
        if len(f) < 2:
            continue
        rows[os.path.basename(f[0])] = f[-1]
    return rows

host, cpu, htp = (load(p) for p in sys.argv[1:4])
run = sys.argv[4]

print("  clips: host=%d device-cpu=%d device-htp=%d" % (len(host), len(cpu), len(htp)))
bad = 0
for label, arm in (("device-CPU", cpu), ("device-HTP0", htp)):
    diff = [k for k in sorted(host) if host[k] != arm.get(k)]
    if not diff:
        print("  %-12s %d/%d transcripts identical to host" % (label, len(host), len(host)))
        continue
    bad = 1
    print("  %-12s %d/%d DIFFER" % (label, len(diff), len(host)))
    for k in diff[:8]:
        print("      %-14s host=%s" % (k, host[k]))
        print("      %-14s %-5s=%s" % ("", label[-4:], arm.get(k)))

# CPU 不一致说明构建或权重不同源，先修这个再看 HTP —— 顺序就是那条踩过的坑。
if any(host[k] != cpu.get(k) for k in host):
    print("\n  Device CPU already disagrees with the host. That is a build or weight")
    print("  mismatch, not an HTP problem: check that both binaries come from the")
    print("  same commit and that the pushed GGUF matches tools/model_manifest.txt.")

with open(os.path.join(run, "xcheck.tsv"), "w", encoding="utf-8") as fh:
    fh.write("clip\thost\thost_cpu\thost_htp\n")
    for k in sorted(host):
        fh.write("%s\t%s\t%s\t%s\n" % (k, host[k], cpu.get(k, ""), htp.get(k, "")))

sys.exit(1 if bad else 0)
PY

    local rc=$?
    info ""
    if [ "$rc" != 0 ]; then
        die "device transcripts differ from the host on the 32-clip set; see $PF_RUN/xcheck.tsv"
    fi
    info "== device matches the host on all 32 transcripts"

    # --- 扩展集：这里不能用「逐字节相同」做判据 -----------------------------
    # 84 条上实测不是逐字节一致：hvac2_en04 的 wrong side 在主机是
    # `de@@|fin@@|i@@|tely`（4 片）、在设备是 `de@@|fin@@|ite`（3 片），
    # seat2_en03 在 HTP 上是 `dri@@|ry|e@@|se@@|se@@|at`。同一段 wav、
    # 同一份权重（sha256 两侧相同）、同一份热词，分歧发生在 span rewrite
    # 之前，是量化卷积在不同 CPU/HTP 实现上的数值差，不是脚本或规则问题。
    #
    # 逐字节判据会把这种正常漂移判成失败，然后所有人学会忽略这个检查。
    # 真正要守的是「漂移不许吃掉分数」：所以这里比的是命令准确率，只有
    # 设备分数低于主机分数才算失败。形状不稳定的规则必须两条都写进
    # profile（见 hotwords_vehicle.txt 的 defog / driver 两处注释）。
    local exp_dir=$PF_RUN/cabin-expanded
    if [ ! -f "$exp_dir/manifest.tsv" ]; then
        info "== skip 84-clip device cross-check (no $exp_dir/manifest.tsv)"
        info ""
        info "== device matches the host on all 32 transcripts (run dir: $PF_RUN)"
        return 0
    fi

    info "  push expanded fixtures (84 clips)"
    "$ADB" -s "$serial" shell "mkdir -p $vdir/wav-exp" </dev/null
    "$ADB" -s "$serial" push "$exp_dir"/*.wav "$vdir/wav-exp/" </dev/null >/dev/null 2>&1
    "$ADB" -s "$serial" shell "ls $vdir/wav-exp/*.wav | wc -l" </dev/null | tr -d '\r' \
        | grep -qx 84 || die "expected 84 expanded clips on the device, got a different count"

    local xhost=$PF_RUN/xe-host.tsv
    local xcpu=$PF_RUN/xe-device-cpu.tsv
    local xhtp=$PF_RUN/xe-device-htp.tsv
    info "  [host]      transcribe 84 expanded clips"
    "$BIN" "$ZH_MODEL" "$exp_dir"/*.wav --quiet --threads "$THREADS" \
        --hotwords "$HOTWORDS" >"$xhost" 2>/dev/null
    for dev in CPU "$DEVICE"; do
        info "  [$dev] transcribe 84 expanded clips on device"
        "$ADB" -s "$serial" shell \
            "cd $vbin && LD_LIBRARY_PATH=bin ./bin/llama-paraformer $vmodel $vdir/wav-exp/*.wav \
             --device $dev --threads $THREADS --quiet --hotwords $vdir/hotwords.txt" \
            </dev/null >"$( [ "$dev" = CPU ] && echo "$xcpu" || echo "$xhtp" )" 2>/dev/null
    done
    for tsv in "$xhost" "$xcpu" "$xhtp"; do
        [ -s "$tsv" ] || die "empty expanded transcript file: $tsv"
    done

    # eval_vehicle 的键是 `<basename(dirname)>/<clip>`，设备上的目录叫
    # `wav-exp`、主机上叫 `cabin-expanded`，键对不上就会「84 条全部没有
    # 转写」然后静默给出只有 host 一行的报告。把设备侧的路径改写成主机
    # 的目录名，只是改键，不动转写字段。
    "$PYTHON" - "$xcpu" "$xhtp" "$exp_dir" <<'PY'
import sys
for path in sys.argv[1:3]:
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        f = line.rstrip("\n").split("\t")
        if len(f) < 2:
            continue
        f[0] = "%s/%s" % (sys.argv[3], f[0].rsplit("/", 1)[-1])
        rows.append("\t".join(f))
    open(path, "w", encoding="utf-8").write("\n".join(rows) + "\n")
PY

    "$PYTHON" "$HERE/../x-asr/tools/eval_vehicle.py" \
        --semantics "$HERE/tools/vehicle_semantics_expanded.tsv" \
        --pack cabin-expanded="$exp_dir/manifest.tsv" \
        --run host="$xhost" --run device-cpu="$xcpu" --run device-htp="$xhtp" \
        --json "$PF_RUN/xe-grade.json" >"$PF_RUN/xe-grade.txt" 2>/dev/null \
        || die "expanded-domain grading failed; see $PF_RUN/xe-grade.txt"
    grep -E '^(host|device)' "$PF_RUN/xe-grade.txt" | sed 's/^/  /' || true
    # 只有 host 一行说明设备臂一条都没匹配上（键错了），这不是「设备没问题」，
    # 是检查自己坏了。宁可红。
    [ "$(grep -cE '^## ' "$PF_RUN/xe-grade.txt")" -ge 3 ] \
        || die "grading saw fewer than 3 arms; check the pack/clip keys in $PF_RUN/xe-grade.txt"

    # 分数判据：设备任一臂低于主机就失败。容差给 1 个 clip 的分母（84 条
    # 上 1.2 个点），因为单条临界样本的漂移可能两边各错一条、方向不同，
    # 这种互不相欠的漂移不该判红。超过 1 条才开始怀疑规则在设备上失效。
    "$PYTHON" - "$PF_RUN/xe-grade.txt" <<'PY'
import re, sys

scores = {}
cur = None
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    line = line.strip()
    m = re.match(r"^## (\S+)$", line)
    if m:
        cur = m.group(1)
        continue
    if cur and "command=" in line:
        m = re.search(r"command=([\d.]+)%", line)
        if m:
            scores[cur] = float(m.group(1))

host = scores.get("host")
if host is None:
    print("  could not read the host score from", sys.argv[1])
    sys.exit(1)

# 不容差：设备分数必须与主机相等。曾经给过 1.3 个点（「两边各错一条、
# 互不相欠」），实测用 §33 之前的 profile 跑，两臂同时各掉一条、delta
# 恰好 -1.2，被容差吞掉、检查保持绿——容差把要抓的缺陷正好放过去了。
# 漂移本身不需要容差：转写不一致但分数不掉时 delta 就是 0。
bad = 0
for arm in ("device-cpu", "device-htp"):
    val = scores.get(arm)
    if val is None:
        print("  %-11s MISSING from the grading report" % arm)
        bad = 1
        continue
    delta = val - host
    ok = delta >= 0
    if not ok:
        bad = 1
    print("  %-11s command %.1f%% (host %.1f%%, delta %+.1f)  %s"
          % (arm, val, host, delta, "ok" if ok else "FAIL"))
sys.exit(1 if bad else 0)
PY
    local erc=$?
    info ""
    [ "$erc" = 0 ] || die "an expanded-domain device arm scored below the host; see $PF_RUN/xe-grade.txt"
    info "== device matches the host on all 32 transcripts and loses nothing on the 84 expanded clips (run dir: $PF_RUN)"
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