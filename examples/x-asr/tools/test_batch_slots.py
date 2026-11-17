#!/usr/bin/env python3
"""Check that cancelling and reusing a slot leaves other streams intact."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import struct
import threading

from asr_client import Client, read_wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("wavs", nargs=4)
    args = parser.parse_args()
    clips = [read_wav(path) for path in args.wavs]

    def open_stream():
        client = Client("127.0.0.1", args.port)
        status, body = client.call(0)
        assert status == 0, body
        sid, = struct.unpack("<I", body)
        return client, sid

    def send(client, sid, samples):
        client.send(1, sid, struct.pack(f"<{len(samples)}f", *samples))

    def transcribe(samples):
        client, sid = open_stream()
        try:
            send(client, sid, samples)
            status, body = client.call(2, sid)
            assert status == 1 and body, (status, body)
            assert client.call(4, sid)[0] == 0
            return body
        finally:
            client.close()

    expected = [transcribe(samples) for samples in clips]
    all_open = threading.Barrier(4, timeout=20)

    def concurrent(lane):
        client, sid = open_stream()
        try:
            all_open.wait()
            send(client, sid, clips[lane])
            if lane == 1:
                assert client.call(3, sid)[0] == 0
                assert client.call(2, sid)[0] == 2
                return
            status, body = client.call(2, sid)
            assert status == 1 and body == expected[lane], (lane, status, body, expected[lane])
            assert client.call(4, sid)[0] == 0
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(concurrent, range(4)))
    assert transcribe(clips[1]) == expected[1]
    print("slot cancellation, surviving-stream parity, and reuse: OK")


if __name__ == "__main__":
    main()
