#!/usr/bin/env python3
"""Check the multi-mode routing in app.py without cv2/gradio/model weights.

Run: python3 test_modes.py
Fails loudly if a mode routes wrong, tensor shapes mismatch, or the
upscale guard trips.
"""
import sys
import types
import numpy as np

# --- stub cv2 (app imports it at module level; load_rgb is monkeypatched) ---
cv2_stub = types.ModuleType("cv2")
cv2_stub.INTER_AREA = 3
cv2_stub.INTER_CUBIC = 2
cv2_stub.imread = lambda *a, **k: None
cv2_stub.cvtColor = lambda *a, **k: None
cv2_stub.COLOR_BGRA2RGBA = 0
sys.modules.setdefault("cv2", cv2_stub)

import app


class FakeSession:
    """Emulates both ONNX graphs: IS-Net (fixed 1200x1800) and Real-ESRGAN."""

    def __init__(self, kind):
        self.kind = kind

    def get_inputs(self):
        if self.kind == "isnet":
            return [types.SimpleNamespace(name="image")]
        return [types.SimpleNamespace(name="input")]

    def get_outputs(self):
        if self.kind == "isnet":
            return [types.SimpleNamespace(name="mask")]
        return [types.SimpleNamespace(name="output")]

    def run(self, out_names, feed):
        x = list(feed.values())[0]
        assert x.ndim == 4 and x.shape[1] == 3, f"bad NCHW {x.shape}"
        if self.kind == "isnet":
            assert x.shape[2:] == (1200, 1800), f"isnet saw {x.shape}"
            return [np.ones((x.shape[0], 1, x.shape[2], x.shape[3]),
                            dtype=np.float32)]
        assert (x.min(), x.max()) >= (0.0, 0.0) and x.max() <= 1.0
        h, w = x.shape[2], x.shape[3]
        return [x.repeat(4, 2).repeat(4, 3)]  # deterministic 4x stand-in


def main():
    rgb = (np.arange(64 * 96 * 3, dtype=np.uint8) % 251).reshape(64, 96, 3)
    app.load_rgb = lambda _path: rgb.copy()   # PIL/cv2 path needs no real file

    engine = app.Engine.__new__(app.Engine)
    engine.session = FakeSession("isnet")
    engine.input_name, engine.output_name = "image", "mask"
    engine.net_hw, engine.fixed_input, engine.provider = (1200, 1800), True, "CPU"
    # postprocess uses repo utils; fake a same-size alpha instead
    engine.mask_for = lambda im, ms: (
        np.full(im.shape[:2], 200, np.uint8),
        dict(prep=0, infer=0, post=0, orig=im.shape[:2], work=im.shape[:2],
             net=(1200, 1800), provider="CPU", fixed_input=True))

    ue = app.UpscaleEngine.__new__(app.UpscaleEngine)
    ue.session = FakeSession("esrgan")
    ue.input_name, ue.output_name = "input", "output"
    ue.provider = "CPU"

    # --- tensor handling: upscale() NCHW round trip ---
    out = ue.upscale(rgb)
    assert out.shape == (256, 384, 3) and out.dtype == np.uint8, out.shape
    try:
        ue.upscale(np.zeros((3000, 10, 3), np.uint8))
        raise AssertionError("oversize image passed the guard")
    except app.InputError:
        pass

    # --- mode 1: BG removal only, upscaler must never run ---
    calls = []
    real_upscale = app.UpscaleEngine.upscale
    def spy(self, im):
        calls.append(1)
        return real_upscale(self, im)
    app.UpscaleEngine.upscale = spy

    s = app.process_image(engine, ue, "x.jpg", "original", app.MODE_BG_ONLY)
    assert not calls and s["mode"] == app.MODE_BG_ONLY
    assert s["preview"].shape == (64, 96, 3) and s["out_mb"] > 0
    assert s["out_png"].endswith(".png")
    from PIL import Image
    assert Image.open(s["out_png"]).mode == "RGBA"

    # --- mode 2: upscale first, IS-Net sees the 4x image ---
    s = app.process_image(engine, ue, "x.jpg", "original",
                          app.MODE_UPSCALE_AND_BG)
    assert len(calls) == 1
    assert s["preview"].shape == (256, 384, 3), s["preview"].shape
    assert Image.open(s["out_png"]).mode == "RGBA"
    assert s["upscale"] > 0 and s["orig"] == (256, 384)

    # --- mode 3: upscale only, full RGB, no alpha, no IS-Net ---
    s = app.process_image(engine, ue, "x.jpg", "original",
                          app.MODE_UPSCALE_ONLY)
    assert s["preview"].shape == (256, 384, 3), s["preview"].shape
    assert Image.open(s["out_png"]).mode == "RGB"
    assert "upscaled" in s["out_png"]

    # mode 3 must fail loudly when the model is unavailable
    try:
        app.process_image(engine, None, "x.jpg", "original",
                          app.MODE_UPSCALE_ONLY)
        raise AssertionError("no upscale engine did not raise")
    except RuntimeError:
        pass

    # --- stats text knows the modes ---
    assert "RGB PNG (no transparency)" in app.stats_markdown(s)
    print("test_modes: ALL OK")


if __name__ == "__main__":
    main()
