from concurrent.futures import ThreadPoolExecutor
import re
import struct
import subprocess
import sys
import time

from asr_client import Client, read_wav

port = int(sys.argv[1])
long, short = [read_wav(path) for path in sys.argv[2:4]]
mode, adb, log = sys.argv[4:]

def connect():
    client = Client("127.0.0.1", port)
    status, body = client.call(0)
    if status == 0:
        return client, struct.unpack("<I", body)[0]
    client.close()
    assert status == 2, (status, body)
    return None, None

def audio(client, sid, samples):
    client.send(1, sid, struct.pack(f"<{len(samples)}f", *samples))

def final(client, sid):
    status, text = client.call(2, sid)
    assert status == 1 and text, (status, text)
    assert client.call(4, sid)[0] == 0
    client.close()
    return text

def routes():
    original = subprocess.run([adb, "shell", "cat", log], text=True, check=True, capture_output=True).stdout
    return re.findall(r"xasr-route device=(\S+) idle_cpus=(-?\d+) htp_commit_age_ms=([\d.]+) htp_room=(\d+)", original)

time.sleep(.65)
if mode == "backlog":
    heavy, sid = connect()
    assert heavy
    audio(heavy, sid, long)
    with ThreadPoolExecutor() as pool:
        result = pool.submit(final, heavy, sid)
        time.sleep(.55)
        client, sid = connect()
        assert client
        audio(client, sid, short)
        assert final(client, sid).decode() == "请导航到上海虹桥机场"
        assert result.result(timeout=25)
    route = routes()
    assert [item[0] for item in route] == ["HTP0:0", "CPU"], route
    assert float(route[-1][2]) >= 80 and route[-1][3] == "1", route
elif mode == "no-headroom":
    active = [connect() for _ in range(4)]
    assert all(client for client, _ in active), active
    assert connect() == (None, None)
    route = routes()
    assert route[-1][0] == "busy" and int(route[-1][1]) < 2 and route[-1][3] == "0", route
    for client, sid in active:
        assert client.call(4, sid)[0] == 0
        client.close()
else:
    raise SystemExit(f"unknown mode: {mode}")

for item in route:
    print(*item)
