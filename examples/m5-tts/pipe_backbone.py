"""Backbone adapter for the official M5Onnx session dictionary."""

import struct
import subprocess

import numpy as np


class PipeBackbone:
    def __init__(self, command):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        # Popen types both pipes as IO[bytes] | None; keep the narrowed handles so
        # the call sites do not have to re-check what PIPE already guarantees.
        stdin, stdout = self.process.stdin, self.process.stdout
        if stdin is None or stdout is None:
            raise RuntimeError("runner must be started with stdin and stdout pipes")
        self.stdin = stdin
        self.stdout = stdout
        self.position = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.stdin.close()
            if exc_type is None and self.process.wait(timeout=15) != 0:
                raise RuntimeError("backbone runner exited unsuccessfully")
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.stdout.close()

    def run(self, outputs, feed):
        hidden = np.asarray(feed["hidden"], dtype="<f4", order="C")
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != 512:
            raise ValueError("expected [1, rows, 512] embeddings")
        rows = hidden.shape[1]
        end = self.position + rows
        if not rows or end > 2048:
            raise ValueError("embedding batch exceeds the context window")
        if not np.array_equal(feed["positions"], np.arange(self.position, end)[None]):
            raise ValueError("positions must be contiguous within a request")
        self.stdin.write(struct.pack("<I", rows))
        self.stdin.write(hidden.tobytes())
        self.stdin.flush()
        response = self.stdout.read(512 * 4)
        if len(response) != 512 * 4:
            raise RuntimeError("backbone runner returned an incomplete hidden state")
        self.position = end
        # llama_context owns the cache. M5Onnx passes this value back unchanged.
        return np.frombuffer(response, dtype="<f4").reshape(1, 512), None
