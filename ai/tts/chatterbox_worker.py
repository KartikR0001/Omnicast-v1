"""Chatterbox Multilingual TTS worker. Runs INSIDE .tools/envs/chatterbox (its own pinned torch).

Protocol: one JSON request per stdin line -> one JSON response per stdout line.
  request : {"text", "language", "prompt_wav", "out_wav", "seed", optional sampling params}
  response: {"ok", "out_wav", "seconds", "audio_seconds", "sample_rate", "error"?}
Weights: ResembleAI/chatterbox (MIT). Outputs carry Resemble's Perth watermark.
"""
import json
import sys
import time

import soundfile as sf
import torch


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else ("mps" if torch.backends.mps.is_available() else "cpu")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    t0 = time.perf_counter()
    # Pass a *string*: chatterbox only maps CUDA-saved checkpoints to CPU when device in ["cpu", "mps"].
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    orig_t3_inf = model.t3.inference
    last_steps = [None]

    def _tracked_t3_inf(*a, **kw):
        toks = orig_t3_inf(*a, **kw)
        try:
            last_steps[0] = int(toks.shape[-1])
        except Exception:
            last_steps[0] = None
        return toks

    model.t3.inference = _tracked_t3_inf
    print(json.dumps({"ready": True, "device": device, "load_seconds": time.perf_counter() - t0}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        try:
            last_steps[0] = None
            torch.manual_seed(int(req.get("seed", 0)))
            t1 = time.perf_counter()
            wav = model.generate(req["text"], language_id=req["language"], audio_prompt_path=req.get("prompt_wav"),
                                 **{k: req[k] for k in ("exaggeration", "cfg_weight", "temperature") if k in req})
            if device == "mps":
                torch.mps.synchronize()
            dt = time.perf_counter() - t1
            x = wav.squeeze(0).detach().cpu().numpy()
            sf.write(req["out_wav"], x, model.sr)
            print(json.dumps({"ok": True, "out_wav": req["out_wav"], "seconds": dt,
                              "audio_seconds": len(x) / model.sr, "sample_rate": model.sr,
                              "tts_steps": last_steps[0]}), flush=True)
        except Exception as e:  # report, never fabricate output
            print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}), flush=True)


if __name__ == "__main__":
    main()
