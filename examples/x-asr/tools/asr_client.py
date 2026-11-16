#!/usr/bin/env python3
"""Replay a WAV into the X-ASR server over the length-prefixed frame protocol.

One thread per session, all sessions started on the same clock, so with
--stagger-ms the short sessions reach their endpoint first and commit while the
long one keeps sending audio.
"""

import argparse
import socket
import struct
import sys
import threading
import time
import wave

OP_OPEN, OP_APPEND, OP_COMMIT, OP_CANCEL, OP_CLOSE = 0, 1, 2, 3, 4
ST_OK, ST_FINAL, ST_ERROR = 0, 1, 2


def read_wav(path, limit_samples=None):
    with wave.open(path, "rb") as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2 and f.getframerate() == 16000, path
        data = f.readframes(f.getnframes())
    samples = list(struct.unpack("<%dh" % (len(data) // 2), data))
    if limit_samples is not None:
        samples = samples[:limit_samples]
    return [v / 32768.0 for v in samples]


class Client:
    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=30.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.lock = threading.Lock()

    def send(self, op, session=0, payload=b""):
        with self.lock:
            self.sock.sendall(struct.pack("<III", op, session, len(payload)) + payload)

    def call(self, op, session=0, payload=b""):
        with self.lock:
            self.sock.sendall(struct.pack("<III", op, session, len(payload)) + payload)
            head = self._recv_exact(8)
            status, size = struct.unpack("<II", head)
            body = self._recv_exact(size) if size else b""
        return status, body

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise RuntimeError("server closed the connection")
            buf += chunk
        return buf

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def run_session(args, samples, start, results, index, n_sessions):
    client = Client(args.host, args.port)
    status, body = client.call(OP_OPEN)
    if status != ST_OK or len(body) != 4:
        results[index] = {"error": "open failed: %s" % body.decode("utf-8", "replace")}
        client.close()
        return
    session = struct.unpack("<I", body)[0]
    endpoint_ms = 0.0
    sent = 0
    step = 320
    try:
        while sent < len(samples):
            n = min(step, len(samples) - sent)
            if args.paced:
                delay = (start + sent / 16000.0) - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            chunk = samples[sent:sent + n]
            client.send(OP_APPEND, session, struct.pack("<%df" % n, *chunk))
            sent += n
        # Endpoint is the scheduled arrival of the last packet in paced mode, so a
        # backlog shows up in the number instead of hiding in the sender.
        endpoint_schedule = start + len(samples) / 16000.0 if args.paced else time.monotonic()
        status, body = client.call(OP_COMMIT, session)
        done = time.monotonic()
        if status != ST_FINAL:
            results[index] = {"error": "commit failed: %s" % body.decode("utf-8", "replace")}
            return
        results[index] = {
            "session": index,
            "server_session": session,
            "audio_ms": len(samples) / 16.0,
            "endpoint_to_final_ms": (done - endpoint_schedule) * 1000.0,
            "text": body.decode("utf-8", "replace"),
        }
        client.call(OP_CLOSE, session)
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--stagger-ms", type=int, default=0,
                        help="the last session keeps this much more audio")
    parser.add_argument("--paced", action="store_true")
    parser.add_argument("--reps", type=int, default=1)
    args = parser.parse_args()

    full = read_wav(args.wav)
    for rep in range(args.reps):
        results = [None] * args.sessions
        threads = []
        start = time.monotonic() + 0.05
        for i in range(args.sessions):
            if args.stagger_ms and i < args.sessions - 1:
                samples = full[: max(0, len(full) - args.stagger_ms * 16)]
            else:
                samples = full
            t = threading.Thread(target=run_session, args=(args, samples, start, results, i, args.sessions))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        for row in results:
            if row is None or "error" in row:
                print("rep=%d error=%s" % (rep, row))
                continue
            print("rep=%d session=%d audio_ms=%.1f endpoint_to_final_ms=%.1f text=%s"
                  % (rep, row["session"], row["audio_ms"], row["endpoint_to_final_ms"], row["text"]))


if __name__ == "__main__":
    main()
