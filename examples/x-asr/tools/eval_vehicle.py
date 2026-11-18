#!/usr/bin/env python3
"""Grade cabin-command transcripts by field instead of by character error.

CER treats `seat peter` as six wrong characters out of ninety-five. The
actuator sees a command with no seat heater in it. This script parses each
transcript into (device, position, state, value) clauses, compares them with
the annotation in `vehicle_semantics.tsv`, and reports:

  device   the commanded device and position were recognised
  state    the commanded on/off direction was recognised
  value    the commanded level, percentage, temperature or colour was recognised
  clause   all three fields of one clause
  command  every clause of the utterance

Two failure modes get their own counters because they are not interchangeable
with the rest:

  wrong direction   the device was recognised with the opposite on/off state.
                    Saying "off" when the driver said "on" is a different class
                    of bug from dropping the clause.
  false trigger     a clip annotated `none` produced a device with a stated
                    state, so it would have been executed as a command.

The parser is a measuring instrument, so it is checked against the annotation
before it grades anything: `--check` re-parses every reference sentence and
fails if the fields it extracts differ from `vehicle_semantics.tsv`. Without
that gate a parser bug and an ASR error look the same in the report.

    python3 eval_vehicle.py --semantics tools/vehicle_semantics.tsv \
        --pack tingting=$PACK/tingting/manifest.tsv \
        --pack controls=$PACK/controls/manifest.tsv \
        --run off=off.tsv --run on=on.tsv

The `--pack` label must be the pack directory name under `hotword-eval`: that
is the prefix `load_transcripts` reads from each transcript path, and it is
what keeps the tuning and holdout packs, which share clip ids, apart.
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_cer import edit_distance, norm_en, norm_zh, strip_special  # noqa: E402

ON, OFF, UNKNOWN = "on", "off", "?"

# Longest and most specific first: 座椅通风 must win before anything generic,
# 阅读灯 before 灯, and 空调 needs the temperature/air-conditioning split below.
DEVICES = [
    ("seat_heat", (r"座椅加热", r"seat heat(?:er)?", r"seat heating")),
    ("seat_vent", (r"座椅通风", r"seat ventilation", r"seat vent\b")),
    ("massage", (r"按摩", r"massage")),
    ("ambient", (r"氛围灯", r"ambient light")),
    ("reading", (r"阅读灯", r"reading light")),
    ("dome", (r"顶灯", r"dome light")),
    ("footwell", (r"脚窝灯", r"footwell light")),
    ("rear_vents", (r"后排出风", r"rear vent")),
    ("recirc", (r"内循环", r"recirculation")),
    ("fan", (r"风量", r"\bfan\b")),
    ("window", (r"车窗", r"window")),
    # Lights beyond the ambient/reading/dome/footwell set, added with the
    # seven-domain expansion. 雾灯 before 灯 so the specific lamp wins.
    ("fog_lights", (r"雾灯", r"fog light")),
    ("headlights", (r"大灯", r"headlight")),
    ("hazard", (r"双闪", r"hazard light")),
    ("trunk_light", (r"后备箱灯", r"trunk light")),
    ("defrost", (r"除霜|除雾|风挡", r"defrost|defog")),
    ("air_purifier", (r"空气净化", r"air purifier")),
    ("sunroof", (r"天窗", r"sunroof")),
    ("sunshade", (r"遮阳帘", r"sunshade")),
    ("airflow", (r"吹脚|吹面|吹脸", r"airflow")),
    # Seat functions that are neither heat nor ventilation.
    ("seat_pos", (r"座椅位置|座椅往后|往后调|座椅靠背|放倒",
                  r"seat position|seat back|backrest|\brecline\b|\bmove the .*seat")),
    # Media and ordering are their own domains: a playback command never names
    # an actuator, so without these the parser returns no clause at all and
    # every clip in the domain scores as a miss regardless of the transcript.
    ("media", (r"播放|暂停|下一首|上一首|音量|收音机|歌单",
               r"\bplay\b|\bpause\b|\bnext track|\bvolume\b|\bradio\b|\bplaylist")),
    ("navigation", (r"导航", r"navigate")),
    ("ordering", (r"点一杯|来一杯|点单", r"\border\b|\bget me a\b")),
]

POSITIONS = [
    ("fr", (r"副驾", r"右前", r"前排右", r"front right", r"passenger")),
    ("fl", (r"主驾", r"驾驶位", r"左前", r"前排左", r"front left", r"driver")),
    ("rl", (r"左后", r"后排左", r"rear left")),
    ("rr", (r"右后", r"后排右", r"rear right")),
]

# Order matters. A negated verb first, because 别开 must not reach the plain
# 开 rule, and 别关 means the device ends up on.
STATE_RULES = [
    (r"(?:别|不要|不用|勿|切勿)\s*(?:打开|开启|开)", OFF),
    (r"(?:别|不要|不用|勿|切勿)\s*(?:关闭|关掉|关上|关)", ON),
    (r"保持(?:开启|打开|开着)", ON),
    (r"保持(?:关闭|关掉|关着)", OFF),
    (r"打开|开启", ON),
    # 只开右后车窗 and 开一半 state the direction with a bare 开.
    (r"开(?![关启])", ON),
    (r"关闭|关掉|关上|关(?![于闭键])", OFF),
    (r"调到|调成|调至|调整到|降到|降至|设为|设置|开到|调低|降低", ON),
    # 调高 was missing while 调低 was present, so `把主驾温度调高两度` parsed with
    # state "?" -- the direction was understood by the sentence and not by the
    # grader. Paired here so the relative pair cannot drift apart again.
    (r"调高|升高|提高|调大|增大", ON),
    # Seat and window adjustments that state a direction without naming an
    # on/off verb: `往后调`, `放倒`, `降下来`. All three move a device, which is
    # what the downstream layer acts on.
    (r"往后|往前|靠后|放倒|降下来|升起来|开一条缝", ON),
    (r"\bdo(?:es)? not (?:open|turn on|switch on)", OFF),
    (r"\bdo(?:es)? not (?:close|turn off|switch off|shut)", ON),
    (r"\bleave\b.{0,25}?\boff\b", OFF),
    (r"\bleave\b.{0,25}?\bon\b", ON),
    (r"\bkeep\b.{0,25}?\boff\b", OFF),
    (r"\bkeep\b.{0,25}?\bon\b", ON),
    # Cabin verbs the first profile never saw. `switch to ... mode` is a state
    # command even though it names no on/off word, and `move the seat back` is
    # a seat command; without them both domains parsed to nothing.
    (r"切换到|改成|调整到模式", ON),
    (r"\bmove\b|\brecline\b|\broll\b|\bswitch to\b", ON),
    # Playback names no on/off verb at all: `播放周杰伦的歌` and `Play some Jay
    # Chou` are start actions.
    (r"播放|\bplay\b", ON),
    # `turn the ventilation to the highest level` sets a level with "turn".
    (r"\bturn\b[^,.;]{0,30}\bto\b", ON),
    (r"\bturn on\b|\bswitch on\b|\bopen\b", ON),
    (r"\bturn off\b|\bswitch off\b|\bclose\b|\bshut\b", OFF),
    (r"\bset\b|\blower\b|\braise\b|\badjust\b", ON),
]

ZH_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
EN_NUMBERS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
              "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
              "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
              "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
              "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
              "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
              "ninety": 90}


def cn2int(text):
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text == "十":
        return 10
    if "十" in text:
        left, _, right = text.partition("十")
        tens = ZH_DIGITS.get(left, 1) if left else 1
        ones = ZH_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    if len(text) == 1:
        return ZH_DIGITS.get(text)
    return None


def en2int(text):
    total = 0
    seen = False
    for word in text.lower().split():
        word = word.strip("-")
        if word in EN_NUMBERS:
            total += EN_NUMBERS[word]
            seen = True
    return total if seen else None


def find_first(text, patterns):
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m
    return None


def split_clauses(text, lang):
    if lang == "zh":
        # 「和」 joins two commands the same way `and` does. Splitting on it is
        # what lets a clause inherit the verb instead of swallowing the second
        # device into the first clause, where it was simply not seen.
        parts = re.split(r"[，。；、,;!?！？]|和", text)
    else:
        parts = re.split(r"[,.;!?]|\band\b", text)
    return [p.strip() for p in parts if p.strip()]


def detect_device(text, lang):
    """Return (device, position, state) for one clause; device may be None.

    A device word without a stated direction is not a command, so callers must
    check `state` before treating the clause as actionable.
    """
    compact = re.sub(r"\s+", "", text)
    source = compact if lang == "zh" else text.lower()

    device = None
    for name, patterns in DEVICES:
        if any(re.search(p, source) for p in patterns):
            device = name
            break
    # 空调 is either the compressor or a temperature target. The temperature
    # reading is the one that names a number of degrees.
    # `rear window defog` names a window in English but is a defrost command.
    # The defrost pattern runs first so the window branch cannot claim it.
    if device == "window" and re.search(r"defrost|defog", source):
        device = "defrost"
    elif device == "window" and re.search(r"除霜|除雾", source):
        device = "defrost"
    if device is None and ("空调" in source or re.search(r"air conditioning", source)):
        has_degree = bool(re.search(r"[一二三四五六七八九十0-9]+\s*度", compact)) \
            or re.search(r"\b(?:\d+|" + "|".join(EN_NUMBERS) + r")\s*degrees?\b", source) \
            or "温度" in source or "temperature" in source
        device = "temp" if has_degree else "ac"
    elif device is None and re.search(r"temperature", source):
        device = "temp"
    elif device is None and re.search(r"\bclimate\b", source):
        device = "ac"
    # `把主驾温度调高两度` names the quantity, not the compressor, and it does so
    # without the word 空调. Before this branch existed the clause parsed to
    # nothing at all, so a Mandarin relative temperature command was invisible
    # to the grader -- the one field the requirements ask for most explicitly.
    if device is None and ("温度" in source
                           or (lang == "zh" and re.search(r"[一二三四五六七八九十两0-9]度", compact))):
        device = "temp"
        # 温度调低一度 states a size, not a target, so it is not a value for the
        # `value` field. detect_mode() reports it instead.

    position = "-"
    for name, patterns in POSITIONS:
        if any(re.search(p, source) for p in patterns):
            position = name
            break

    state = UNKNOWN
    for pattern, value in STATE_RULES:
        if re.search(pattern, source):
            state = value
            break
    return device, position, state


# Relative against absolute is a field of its own because the two take
# different downstream actions and are indistinguishable once both have been
# mapped to `on`. `把主驾温度调高两度` and `把主驾空调调到二十四度` are the pair
# the product requirements name: the first adds two to whatever is set, the
# second replaces it. A grader that only records the state scores both `on` and
# calls a confusion between them a success.
#
# Pure Python, not a dotted vector: `调高` contains `高` and `调低` contains
# `低`, which is the distinction, but `调到` must not match `调` alone.
RELATIVE_ZH = re.compile(r"调高|调低|调大|调小|升高|降低|加大|减小|提高|减少|往[前后]调|"
                         r"[大小高矮]一?点")
ABSOLUTE_ZH = re.compile(r"调到|调成|调至|调整到|设为|设置成?|锁定在")
RELATIVE_EN = re.compile(r"\b(?:raise|lower|increase|decrease|turn (?:it )?(?:up|down)|"
                         r"a little|a bit|slightly)\b")
# `Lower the window by one third` is an absolute position written with a
# relative verb -- the driver means "to 33 percent", and the annotation says
# 33. Treating it as a delta drops the value and the clip scores as a miss on
# a correct transcript, so a stated proportion wins over the verb.
PROPORTION_EN = re.compile(r"one third|a third|two thirds|a quarter|a half|halfway|"
                           r"\d+\s*(?:%|percent)")
PROPORTION_ZH = re.compile(r"三分之一|三分之二|四分之一|一半|半数|百分之[0-9一二三四五六七八九十百]+")
ABSOLUTE_EN = re.compile(r"\bset\b|\bto\s+(?:level\s+)?\d|\bto\s+(?:twenty|thirty|"
                         r"one|two|three|four|five|six|seven|eight|nine|ten)\b")


def detect_mode(text, lang):
    """`rel` (add/subtract), `abs` (replace), or None when the sentence does not
    say. Relative wins because 调高 is never also 调到 -- except when a
    proportion is stated, which fixes the target numerically."""
    if lang == "zh":
        compact = re.sub(r"\s+", "", text)
        if PROPORTION_ZH.search(compact):
            return "abs"
        if RELATIVE_ZH.search(compact):
            return "rel"
        if ABSOLUTE_ZH.search(compact):
            return "abs"
        return None
    lower = text.lower()
    if PROPORTION_EN.search(lower):
        return "abs"
    if RELATIVE_EN.search(lower):
        return "rel"
    if ABSOLUTE_EN.search(lower):
        return "abs"
    return None


def extract_amount(text, lang):
    """The delta for a relative command: `两度` -> 2, `by two degrees` -> 2.

    Only meaningful when the mode is `rel`; `extract_value` keeps reporting the
    absolute reading, so a relative command must not be graded on it.
    """
    if lang == "zh":
        m = re.search(r"([一二三四五六七八九十两0-9]+)\s*(?:度|档)", re.sub(r"\s+", "", text))
        if m:
            return cn2int(m.group(1))
        return None
    m = re.search(r"by\s+((?:twenty|thirty)[ -]?[a-z]+|" + "|".join(EN_NUMBERS) +
                  r"|\d+)\s*(?:degrees?|levels?)?", text.lower())
    if m:
        return en2int(m.group(1))
    return None


def extract_value(text, device, lang):
    if device == "headlights":
        if re.search(r"自动|\bauto\b", text.lower()):
            return "auto"
        return None
    if device == "ambient":
        lower = text.lower()
        if re.search(r"暖白|暖白色", text) or re.search(r"warm white", lower):
            return "warm_white"
        if re.search(r"冰蓝", text) or re.search(r"ice blue", lower):
            return "ice_blue"
        return None
    if device == "temp":
        m = re.search(r"([一二三四五六七八九十0-9]+)\s*度", re.sub(r"\s+", "", text))
        if m:
            return cn2int(m.group(1))
        m = re.search(r"((?:twenty|thirty)[ -]?[a-z]+|one|two|three|four|five|six|seven|"
                      r"eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
                      r"seventeen|eighteen|nineteen|twenty|thirty|\d+)\s*degrees?", text.lower())
        if m:
            return en2int(m.group(1))
        return None
    if device in ("ac", "temp"):
        if re.search(r"自动|auto", text.lower()):
            return "auto"
    if device in ("fan", "seat_heat", "seat_vent"):
        m = re.search(r"([一二三四五六七八九十两0-9]+)\s*档", re.sub(r"\s+", "", text))
        if m:
            return cn2int(m.group(1))
        m = re.search(r"level\s+((?:twenty|thirty)[ -]?[a-z]+|[a-z]+|\d+)", text.lower())
        if m:
            return en2int(m.group(1))
        return None
    if device in ("window", "footwell"):
        compact = re.sub(r"\s+", "", text)
        if "三分之一" in compact:
            return 33
        if "三分之二" in compact:
            return 67
        if "四分之一" in compact:
            return 25
        if "一半" in compact or "半数" in compact:
            return 50
        if "全部" in compact or "完全" in compact:
            return 100
        m = re.search(r"百分之([0-9]+|[一二三四五六七八九十百]+)", compact)
        if m:
            return cn2int(m.group(1))
        lower = text.lower()
        if "one third" in lower or "a third" in lower:
            return 33
        if "two thirds" in lower:
            return 67
        if "quarter" in lower:
            return 25
        if "half" in lower:
            return 50
        m = re.search(r"(\d+)\s*(?:%|percent)", lower)
        if m:
            return int(m.group(1))
        m = re.search(r"((?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
                      r"[ -]?[a-z]+|" + "|".join(EN_NUMBERS) + r")\s*percent", lower)
        if m:
            return en2int(m.group(1))
        return None
    return None


def parse(text, lang):
    """One clause per device mention, with the verb carried across `and`.

    `Turn off the dome light and the footwell lights` is two commands sharing
    one verb. Grading the second clause on `?` would let a transcript that
    drops the verb from `关闭顶灯和脚窝灯` score as a hit, so the state of the
    previous clause is inherited when a clause names a device and no direction.
    Only intra-utterance: nothing carries between clips.
    """
    clauses = []
    carried = None
    for part in split_clauses(text, lang):
        device, position, state = detect_device(part, lang)
        if device is None:
            continue
        # `only the rear right window` names a device for contrast, not a
        # direction: 别开...只开右后车窗 states its own 开, while the English
        # version leaves the verb out and the annotation grades it as `?`.
        # Inheriting "off" there would invent a command nobody gave.
        bare_contrast = re.search(r"\bonly\b|\bjust\b", part, re.I) and lang == "en"
        if state == UNKNOWN and carried in (ON, OFF) and not bare_contrast:
            state = carried
        if state in (ON, OFF):
            carried = state
        mode = detect_mode(part, lang)
        # A relative command states how far to move, not where to land, so the
        # number belongs in `amount` and `value` has to stay empty. Leaving it
        # in `value` scored `调高两度` and `调到两度` as the same command, which is
        # the confusion the requirements single out.
        if mode == "rel":
            value, amount = None, extract_amount(part, lang)
        else:
            value, amount = extract_value(part, device, lang), None
        clauses.append({"device": device, "position": position, "state": state,
                        "value": value, "mode": mode, "amount": amount, "text": part})
    return clauses


def load_semantics(path):
    table = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            clip_id, spec = parts[0], parts[1]
            if spec.strip() == "none":
                table[clip_id] = []
                continue
            clauses = []
            for chunk in spec.split(";"):
                fields = [f.strip() for f in chunk.split("|")]
                # Five fields is the extended clause: device|position|state|
                # value|mode. `mode` is `rel`, `abs` or `-`, and a relative
                # clause carries its delta in `value` because a relative command
                # has no absolute target to conflict with it. Four-field rows
                # written before the extension still load unchanged, so the
                # original 32-clip annotation did not have to be rewritten.
                if len(fields) == 5:
                    mode = None if fields[4] in ("-", "") else fields[4]
                elif len(fields) == 4:
                    mode = None
                else:
                    raise SystemExit("bad clause %r for %s" % (chunk, clip_id))
                value = fields[3]
                clauses.append({
                    "device": fields[0],
                    "position": fields[1],
                    "state": fields[2] if fields[2] in (ON, OFF, UNKNOWN) else UNKNOWN,
                    "value": None if value == "-" else coerce_value(fields[0], value),
                    "mode": mode,
                })
            table[clip_id] = clauses
    return table


def coerce_value(device, raw):
    if device == "ambient":
        return raw
    try:
        return int(raw)
    except ValueError:
        return raw


def load_manifest(path, pack=None):
    """`pack` names the key prefix when the manifest is a copy outside its dir."""
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = os.path.basename(parts[0])
            clip_id = name[:-4] if name.endswith(".wav") else name
            key = "%s/%s" % (pack, clip_id) if pack else clip_key(path, clip_id)
            rows[key] = {"id": clip_id, "lang": parts[1],
                         "text": "\t".join(parts[2:]), "manifest": path}
    return rows


def clip_key(path, clip_id):
    """`<pack>/<clip id>`: win_zh01 exists in both the tuning and holdout packs."""
    pack = os.path.basename(os.path.dirname(path))
    return "%s/%s" % (pack, clip_id) if pack else clip_id


def load_transcripts(path):
    """CLI output: `path<TAB>audio_ms=..<TAB>wall_ms=..<TAB>...<TAB>text`."""
    rows = {}
    text = open(path, encoding="utf-8", errors="replace").read()
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        name = os.path.basename(fields[0])
        clip_id = name[:-4] if name.endswith(".wav") else name
        meta = {}
        for field in fields[1:-1]:
            if "=" in field:
                key, _, value = field.partition("=")
                try:
                    meta[key] = float(value)
                except ValueError:
                    pass
        rows[clip_key(fields[0], clip_id)] = {"path": fields[0], "text": fields[-1],
                                              "meta": meta}
    return rows


def grade(ref_clauses, hyp_clauses):
    """Field-level comparison of one clip. `ref_clauses` empty means `none`."""
    if not ref_clauses:
        triggered = [h for h in hyp_clauses if h["state"] in (ON, OFF)]
        return {"none": True, "triggered": bool(triggered),
                "triggered_text": triggered[0]["text"] if triggered else None,
                "device_only": sorted({h["device"] for h in hyp_clauses}),
                "device": None, "state": None, "value": None, "clause": None,
                "command": None, "wrong_direction": False, "unstated": []}

    device_ok = state_ok = value_ok = clause_ok = 0
    wrong = []
    unstated = []
    for ref in ref_clauses:
        match = None
        for hyp in hyp_clauses:
            if hyp["device"] != ref["device"]:
                continue
            if ref["position"] != "-" and hyp["position"] != ref["position"]:
                continue
            match = hyp
            break
        if match is None:
            continue
        device_ok += 1
        if ref["state"] == UNKNOWN or match["state"] == ref["state"]:
            state_ok += 1
        elif match["state"] == UNKNOWN:
            # The device and value came through but no verb did: "said the
            # driver temperature to twenty one degrees". Nothing gets executed,
            # which is a failure but not the opposite of what was asked.
            unstated.append({"device": ref["device"], "position": ref["position"],
                             "want": ref["state"]})
        else:
            wrong.append({"device": ref["device"], "position": ref["position"],
                          "want": ref["state"], "got": match["state"]})
        # Only clauses that actually set a value enter the value denominator;
        # counting "no value expected" as a hit would push the ratio past 100%.
        if ref["value"] is not None and match["value"] == ref["value"]:
            value_ok += 1
        if match["state"] == ref["state"] or ref["state"] == UNKNOWN:
            if ref["value"] is None or match["value"] == ref["value"]:
                clause_ok += 1
    total = len(ref_clauses)
    return {"none": False, "device": device_ok, "state": state_ok, "value": value_ok,
            "clause": clause_ok, "command": clause_ok == total, "total": total,
            "wrong_direction": wrong, "unstated": unstated, "triggered": False}


def cer_wer(rows):
    zh_err = zh_len = en_err = en_len = 0
    for row in rows:
        lang = row["lang"]
        hyp, _ = strip_special(row.get("hyp", ""))
        if lang == "zh":
            ref_n, hyp_n = norm_zh(row["text"]), norm_zh(hyp)
            zh_err += edit_distance(ref_n, hyp_n)
            zh_len += len(ref_n)
        else:
            ref_n, hyp_n = norm_en(row["text"]).split(), norm_en(hyp).split()
            en_err += edit_distance(ref_n, hyp_n)
            en_len += len(ref_n)
    return {"zh_cer": 100.0 * zh_err / max(zh_len, 1), "zh_err": zh_err, "zh_len": zh_len,
            "en_wer": 100.0 * en_err / max(en_len, 1), "en_err": en_err, "en_len": en_len}


def percentiles(values, points=(50, 95, 99)):
    if not values:
        return {}
    ordered = sorted(values)
    out = {}
    for p in points:
        index = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
        out[p] = ordered[index]
    return out


def check_parser(packs, semantics):
    """Every reference sentence must parse back to its own annotation."""
    bad = 0
    for label, manifest in packs:
        for key, row in load_manifest(manifest, pack=label).items():
            clip_id = row["id"]
            ref = semantics.get(clip_id)
            if ref is None:
                print("MISSING ANNOTATION %s (%s)" % (clip_id, label))
                bad += 1
                continue
            got = parse(row["text"], row["lang"])
            if not ref:
                actionable = [c for c in got if c["state"] in (ON, OFF)]
                if actionable:
                    print("FALSE TRIGGER ON REF %s %r -> %s" % (
                        clip_id, row["text"], actionable))
                    bad += 1
                continue
            if len(got) != len(ref):
                print("CLAUSE COUNT %s %r: want %d got %d" % (
                    clip_id, row["text"], len(ref), len(got)))
                bad += 1
                continue
            for want, have in zip(ref, got):
                keys = ["device", "position", "state", "value"]
                # `mode` is only checked where the annotation states it. A
                # four-field row leaves it None, and comparing None against the
                # parser's `abs` for `调到` would fail every legacy clip.
                if want.get("mode"):
                    keys.append("mode")
                diffs = [k for k in keys if want.get(k) != have.get(k)]
                if diffs:
                    print("FIELD %s %r: %s want=%s got=%s" % (
                        clip_id, row["text"], diffs,
                        {k: want[k] for k in diffs}, {k: have[k] for k in diffs}))
                    bad += 1
    if bad:
        raise SystemExit("%d annotation/parser disagreements" % bad)
    print("parser check: %d references agree with the annotation" % sum(
        len(load_manifest(m)) for _, m in packs))


def summarise(rows):
    """Aggregate graded rows into the counters the report quotes."""
    out = {"clips": len(rows)}
    for key in ("device", "state", "value", "clause", "total"):
        out[key] = sum(r["grade"].get(key) or 0 for r in rows if not r["grade"]["none"])
    commands = [r for r in rows if not r["grade"]["none"]]
    out["command_clips"] = len(commands)
    out["command_ok"] = sum(1 for r in commands if r["grade"]["command"])
    out["wrong_direction"] = sum(len(r["grade"]["wrong_direction"]) for r in commands)
    out["unstated"] = sum(len(r["grade"]["unstated"]) for r in commands)
    negatives = [r for r in rows if r["grade"]["none"]]
    out["negative_clips"] = len(negatives)
    out["false_trigger"] = sum(1 for r in negatives if r["grade"]["triggered"])
    out["near_miss"] = sum(1 for r in negatives if r["grade"]["device_only"])
    out["latency"] = percentiles([r["wall_ms"] for r in rows if r.get("wall_ms")])
    out.update(cer_wer(rows))
    return out


def pct(num, den):
    return 100.0 * num / den if den else 0.0


def fmt(num, den):
    """`-` when a pack has no such field, so an empty set cannot read as 0%."""
    return "-" if not den else "%.1f%% (%d/%d)" % (pct(num, den), num, den)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantics", default=os.path.join(HERE, "vehicle_semantics.tsv"))
    parser.add_argument("--pack", action="append", default=[], metavar="LABEL=MANIFEST")
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=TSV")
    parser.add_argument("--check", action="store_true", help="parse references and stop")
    parser.add_argument("--json", help="write per-clip grades and summaries here")
    args = parser.parse_args()

    semantics = load_semantics(args.semantics)
    packs = []
    for spec in args.pack:
        label, _, path = spec.partition("=")
        packs.append((label, path))
    if args.check:
        check_parser(packs, semantics)
        return

    # Keyed by pack directory as well as clip id: the tuning and holdout packs
    # share clip ids and differ only in voice, so an id-only table silently
    # drops one of them.
    # The label is the pack directory name under `hotword-eval`, so it matches
    # the prefix that `load_transcripts` derives from each transcript path.
    clips = {}
    for label, manifest in packs:
        for key, row in load_manifest(manifest, pack=label).items():
            row["pack"] = label
            clips[key] = row

    report = {"arms": {}, "per_clip": {}}
    for spec in args.run:
        arm, _, path = spec.partition("=")
        transcripts = load_transcripts(path)
        rows = []
        missing = []
        for key, clip in clips.items():
            if key not in transcripts:
                missing.append(key)
                continue
            hyp = transcripts[key]["text"]
            ref = semantics.get(clip["id"])
            if ref is None:
                raise SystemExit("no annotation for %s" % key)
            graded = grade(ref, parse(hyp, clip["lang"]))
            rows.append({"id": clip["id"], "key": key, "pack": clip["pack"],
                         "lang": clip["lang"], "text": clip["text"], "hyp": hyp,
                         "grade": graded,
                         "wall_ms": transcripts[key]["meta"].get("wall_ms"),
                         "audio_ms": transcripts[key]["meta"].get("audio_ms"),
                         "value_total": sum(1 for c in ref if c["value"] is not None)})
        if missing:
            # A silently empty pack is how a key collision hides.
            raise SystemExit("%s: %d clips had no transcript, first is %s" % (
                arm, len(missing), missing[0]))
        by_pack = {}
        for pack in dict.fromkeys(r["pack"] for r in rows):
            pack_rows = [r for r in rows if r["pack"] == pack]
            summary = summarise(pack_rows)
            summary["value_total"] = sum(r["value_total"] for r in pack_rows)
            by_pack[pack] = summary
        report["arms"][arm] = by_pack
        report["per_clip"][arm] = rows

        print("## %s" % arm)
        for pack, s in by_pack.items():
            print("%-10s clips=%-3d device=%s state=%s value=%s clause=%s command=%s "
                  "wrong=%d unstated=%d false_trig=%d/%d near_miss=%d/%d "
                  "zhCER=%.2f%% enWER=%.2f%% wall p50/p95/p99=%.0f/%.0f/%.0f ms" % (
                      pack, s["clips"],
                      fmt(s["device"], s["total"]), fmt(s["state"], s["total"]),
                      fmt(s["value"], s["value_total"]), fmt(s["clause"], s["total"]),
                      fmt(s["command_ok"], s["command_clips"]),
                      s["wrong_direction"], s["unstated"], s["false_trigger"], s["negative_clips"],
                      s["near_miss"], s["negative_clips"],
                      s["zh_cer"], s["en_wer"],
                      s["latency"].get(50, 0), s["latency"].get(95, 0), s["latency"].get(99, 0)))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1)
        print("wrote %s" % args.json)

    if len(args.run) > 1:
        labels = [spec.partition("=")[0] for spec in args.run]
        print("\n## changed commands")
        for key, clip in clips.items():
            verdicts = {}
            for arm in labels:
                row = next((r for r in report["per_clip"][arm] if r["key"] == key), None)
                if row:
                    verdicts[arm] = row["grade"]["command"] if not row["grade"]["none"] \
                        else (not row["grade"]["triggered"])
            if len(set(verdicts.values())) > 1:
                first = next(r for r in report["per_clip"][labels[-1]] if r["key"] == key)
                print("%-22s %s  ref=%s  hyp=%s" % (
                    key, verdicts, first["text"], first["hyp"]))


if __name__ == "__main__":
    main()
