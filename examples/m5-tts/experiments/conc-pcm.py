import json, socket, time, threading, statistics as st, urllib.request

HOST, PORT = "127.0.0.1", 18080
TEXT = "前方三百米右转，随后靠左行驶。别着急，我们还有时间。"

def one(voice, tag, results):
    body = json.dumps({"input": TEXT, "voice": voice, "response_format": "pcm"}).encode()
    t0 = time.time()
    req = urllib.request.Request(f"http://{HOST}:{PORT}/v1/audio/speech", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        first = None; total = 0
        while True:
            chunk = r.read(65536)
            if not chunk: break
            if first is None:
                first = time.time() - t0
            total += len(chunk)
        results.append((tag, voice, first, time.time() - t0, total))

def batch(n, voices, label):
    results = []; threads = []
    t0 = time.time()
    for i in range(n):
        v = voices[i % len(voices)]
        th = threading.Thread(target=one, args=(v, f"{label}#{i}", results)); th.start(); threads.append(th)
    for th in threads: th.join()
    wall = time.time() - t0
    results.sort(key=lambda x: x[2])
    print(f"{label} conc={n} voices={'/'.join(sorted(set(voices)))} wall={wall:.2f}s")
    for tag, v, f, tot, sz in results:
        print(f"   {tag} voice={v} first_audio={f:.3f}s total={tot:.3f}s bytes={sz}")
    return wall

print("== A. same voice, real first-audio (pcm) ==")
for n in (1, 2, 4, 8):
    batch(n, ["serena"], "same")
print("\n== B. two voices interleaved ==")
for n in (2, 4):
    batch(n, ["serena", "官方样例"], "mix")
