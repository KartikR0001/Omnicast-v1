"""Client for the isolated Chatterbox worker process."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ai.base import ComponentResult

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".tools/envs/chatterbox/bin/python"
WORKER = Path(__file__).with_name("chatterbox_worker.py")
MODEL_ID = "ResembleAI/chatterbox:multilingual (chatterbox-tts 0.1.7)"


class ChatterboxTTS:
    def __init__(self, device: str = "mps"):
        self.proc = subprocess.Popen([str(PY), "-I", str(WORKER), device], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        hello = self._read()
        if not hello.get("ready"):
            raise RuntimeError(f"worker failed to start: {hello}")
        self.device = hello["device"]
        self.load_seconds = hello["load_seconds"]

    def _read(self) -> dict:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("chatterbox worker exited")
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)

    def synthesize(self, text: str, language: str, prompt_wav: str, out_wav: str, seed: int = 0, **kw) -> ComponentResult:
        self.proc.stdin.write(json.dumps({"text": text, "language": language, "prompt_wav": prompt_wav,
                                          "out_wav": out_wav, "seed": seed, **kw}) + "\n")
        self.proc.stdin.flush()
        r = self._read()
        if not r.get("ok"):
            raise RuntimeError(r.get("error"))
        return ComponentResult(out_wav, MODEL_ID, r["seconds"], r)

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=60)
