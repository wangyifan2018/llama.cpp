import json, time, threading, urllib.request

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
            chunk = r.read1(4096)
            if not chunk: break
            if first is None: first = time.time() - t0
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
    print(f"{label} conc={n} wall={wall:.2f}s  first_audio: " +
          " ".join(f"{r[2]:.2f}" for r in results) +
          f"  bytes={sorted(set(r[4] for r in results))}")
    return wall

print("== same voice (pcm), first audible byte ==")
for n in (1, 2, 4, 8): batch(n, ["serena"], "same")
print("\n== two voices interleaved ==")
for n in (2, 4): batch(n, ["serena", "官方样例"], "mix")
