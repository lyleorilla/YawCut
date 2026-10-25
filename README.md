# Background Remover (local, offline)

Small Gradio app that removes photo backgrounds using the
[Trendyol/background-removal](https://huggingface.co/Trendyol/background-removal)
IS-Net model in ONNX format. Everything runs **locally on your machine** —
no server, no upload, no CUDA. Works on macOS, Linux, and Windows. The ~176 MB weights are downloaded once into `model_cache/`
(sha256-verified) and every later run is fully offline.

Output is a transparent RGBA PNG at the **original resolution**, shown in the UI
over a checkerboard, saved to `outputs/`.

## Setup

```bash
# 1. Create a virtual environment (Python 3.8–3.11 recommended for the pins below)
python3 -m venv venv

# 2. Install dependencies
# Note: These specific versions ensure broad compatibility, including older systems.
# On modern OSes/architectures, you can drop the strict version pins if preferred.
./venv/bin/pip install \
    onnxruntime==1.16.3 \
    numpy==1.26.4 \
    opencv-python-headless==4.10.0.84 \
    huggingface_hub 'pillow>=11.3' requests gradio

# 3. First run downloads the model into model_cache/ (needs internet, once)
./venv/bin/python app.py --no-browser   # or just: ./venv/bin/python app.py
```

## Run

```bash
                 # UI on http://127.0.0.1:7860
./venv/bin/python app.py --no-browser   # don't open a browser tab
./venv/bin/python app.py --port 7861    # other port
./venv/bin/python app.py --bench IMG    # headless: process IMG twice, print timings
./venv/bin/python app.py --try-coreml   # benchmark CoreML EP vs CPU (macOS only)
```

After the first download you can cut the network entirely — try
`HF_HUB_OFFLINE=1 ./venv/bin/python app.py` to confirm.

## Input formats

Uploads and clipboard pastes accept **JPG, PNG, BMP, TIFF, WebP and AVIF**,
including AVIF with alpha, greyscale or 10/12-bit colour. Output is always a
transparent RGBA PNG at the original resolution.

AVIF is decoded by **Pillow's bundled libavif**, not by OS-level tools:

- AVIF read support needs the libavif built into Pillow's wheels (Pillow ≥ 11.3).
  `app.py` probes it at startup and logs either
  `AVIF decoder: libavif 1.4.2 (Pillow 12.3.0)` or a warning telling you to
  upgrade Pillow; without it, `.avif` uploads fail with that same instruction
  instead of a confusing decode error.
- Whatever the source profile, Pillow hands the pipeline 8-bit RGB/RGBA, so
  preprocessing, inference and compositing are unchanged.
- Some AVIF files are rejected by the decoder itself (conformance-suite files,
  some HDR/gain-map encodes). Those surface the decoder's own reason — re-export
  as PNG/JPEG/WebP — rather than looking like a corrupt upload.
- There is deliberately **no shell fallback** to OS-specific tools like `sips`
  or `ffmpeg`. The app relies entirely on Pillow's bundled libavif to ensure
  consistent behavior across macOS, Linux, and Windows.

A file whose extension is missing or wrong is judged by its contents, so a
renamed `photo.jpg` that is really a PNG (or a `.avif`) still works.

## Dependency Notes

The setup commands above use pinned versions chosen for broad cross-platform
compatibility, including older systems. If you are on a modern OS and architecture
(Apple Silicon, recent Linux, Windows), you can safely drop the strict version
pins and install the latest releases instead.

| package | pin | reason |
|---|---|---|
| `onnxruntime` | `1.16.3` | Pinned for broad compatibility, including older Intel macOS systems. 1.17+ may require newer OS versions. Newer versions work fine on modern platforms. |
| `numpy` | `1.26.4` | onnxruntime 1.16.3 is built against the NumPy 1.x ABI; NumPy 2.x breaks it at import. |
| `opencv-python-headless` | `4.10.0.84` | Known to build cleanly across platforms; headless = no GUI deps. |
| `pillow` | `>=11.3` | AVIF input depends on the libavif bundled in Pillow's own wheels, which shipped from 11.3; older Pillow decodes no AVIF at all. `app.py` probes it at startup. |
| Python | `3.8–3.11` | onnxruntime 1.16.3 publishes wheels for cp38–cp311. If you unpin onnxruntime, newer Python versions (3.12+) work as well. |

## Python Version Requirements

With the pinned dependencies above, **Python 3.8–3.11** is required because
onnxruntime 1.16.3 only publishes wheels for those versions. If you're using
a newer Python (3.12+), simply drop the strict version pins during install
and use the latest compatible releases.

If your system doesn't have a compatible Python version readily available,
[python-build-standalone](https://github.com/astral-sh/python-build-standalone)
provides self-contained CPython binaries that need no system installer.
A portable interpreter can optionally be placed in `./python311/` (gitignored)
and used to create the virtual environment.

## Max side length

This particular ONNX export has a **fixed input of 1200×1800** (only the batch
dim is symbolic), so the network always infers at 1200×1800. The "Max side"
dropdown changes the size the image is *processed* at (decode → resize →
composite), which affects pre/post CPU time and memory — not inference time.
The final PNG is always written at the original resolution. The UI prints this
in its status line, and the engine adapts automatically if you ever swap in a
dynamically-shaped export.

## CoreML (macOS only)

By default, the app uses onnxruntime's `CPUExecutionProvider`, which works on
all platforms (macOS, Linux, Windows). CoreML acceleration is an optional,
Apple-specific feature.

If your onnxruntime build does not include the CoreML execution provider,
`--try-coreml` will report its unavailability and stay on CPU. If you install
a build with the CoreML EP, the flag benchmarks both providers on a synthetic
image and keeps the faster one that produces a usable (non-blank) mask — CoreML
exports sometimes return empty masks, which the benchmark detects and rejects.

## About `utils.py` and torch

The model repo's `utils.py` does the pre/post-processing but imports `torch` at
module level. To stay lightweight and cross-platform, this app **does not
install torch**: `app.py` tries to import the repo's `utils.py` and uses it
verbatim when available, otherwise it falls back to a built-in NumPy/OpenCV
port of the exact same maths and constants (bilinear resize, `(x-0.5)/1.0`
normalisation, min-max mask normalisation, threshold 25,
`area_limit = 50000*h*w/(1800*1200)`, 9×9 ellipse dilate × 2). This NumPy
fallback ensures the app works everywhere without requiring PyTorch.
Install torch if you want the verbatim path.

## Files

- `app.py` — the whole app (UI, engine, CLI)
- `model_cache/` — downloaded `model.onnx` + `utils.py` (gitignored)
- `outputs/` — result PNGs (gitignored)
- `python311/` — optional portable interpreter (gitignored, see Python notes above)
- `venv/` — virtual environment (gitignored, recreatable)

## License / attribution

- The model weights and `utils.py` are © Trendyol, licensed
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). They are
  downloaded at runtime from Hugging Face and are **not** redistributed by this
  repository (`model_cache/` is gitignored). Share-alike applies to derivative
  works of the model — check the terms before commercial use.
- This repository's code (`app.py`, `README.md`) is yours to license as you
  see fit.
