#!/usr/bin/env python3
"""Local background remover - IS-Net ONNX (Trendyol/background-removal) + Gradio.

Everything runs on this Mac: no server, no upload, no CUDA. The weights are
fetched once from Hugging Face into ./model_cache and every later run is offline.

Usage:
    python app.py                    # UI on http://127.0.0.1:7860
    python app.py --try-coreml       # benchmark CoreML against CPU, keep the winner
    python app.py --bench IMG        # headless timing run (first vs second pass)
    python app.py --no-browser       # do not open a browser window

Model weights (c) Trendyol, CC BY-SA 4.0 - see README.md before commercial use.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import cv2
import PIL
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_cache"
OUT_DIR = BASE_DIR / "outputs"

REPO_ID = "Trendyol/background-removal"
MODEL_FILE = "model.onnx"
UTILS_FILE = "utils.py"

# Verified against the HF API: model.onnx is exactly 176,215,273 bytes.
MODEL_BYTES = 176_215_273
MODEL_BYTES_TOL = 2_000_000
MODEL_SHA256 = "b9ef43bab5c1a5538916d31066d5ae843a2200d1fd83ed5bac320643076332d0"
UTILS_BYTES = 4837

# utils.py sets INPUT_SIZE = [1200, 1800]; the shipped ONNX graph is also
# hard-wired to 1x3x1200x1800 (only the batch dim is symbolic), so on this model
# the net always sees 1200x1800 no matter what the caller asks for.
REPO_INPUT_HW = (1200, 1800)
INTRA_OP_THREADS = 4

MAX_SIDE_CHOICES = ["original", "1800", "1200", "1024", "768"]
DEFAULT_MAX_SIDE = "1024"

MAX_INPUT_BYTES = 60 * 1024 * 1024   # reject absurd uploads before decoding
MAX_INPUT_PIXELS = 60_000_000        # 60 MP; RGBA + mask + resize would thrash 16 GB
PREVIEW_MAX_SIDE = 1200              # display only - the PNG download stays full res

# --- Real-ESRGAN upscale model (SRVGGNetCompact, 4x) ---
# The official xinntao / ai-forever releases ship .pth only. This community
# repo exports the official realesr-general-x4v3 weights to ONNX; size and
# sha256 below were verified against the live file, and the graph takes
# "input" [1,3,H,W] float32 RGB 0..1 and returns "output" [1,3,4H,4W].
UPSCALE_REPO_ID = "vlad-grig/realesr-general-x4v3-onnx"
UPSCALE_REPO_FILE = "onnx/model.onnx"             # path inside the repo
UPSCALE_MODEL_FILE = "realesr-general-x4v3.onnx"  # local cache name
UPSCALE_MODEL_BYTES = 4_866_420
UPSCALE_MODEL_BYTES_TOL = 200_000
UPSCALE_MODEL_SHA256 = "7b10e80021ec3e0ff35017ae5022791879d93a799b2e201e1010a37510202266"
MAX_UPSCALE_INPUT_SIDE = 2048   # 4x output caps at ~8K; keeps VRAM/RAM sane

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
                 ".avif"}
# A suffix is only a hint - renamed files exist and clipboard pastes arrive as
# .webp - so an unknown one is confirmed with Pillow's own format detection
# rather than rejected sight unseen.
SNIFFABLE_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "BMP": ".bmp",
                     "TIFF": ".tif", "WEBP": ".webp", "AVIF": ".avif"}


def log(msg: str) -> None:
    print(f"[app] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Model cache: download once, then stay offline
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download_direct(filename: str, dest: Path, repo_id: str = REPO_ID) -> None:
    """Fallback when huggingface_hub fails: plain streaming GET to a temp file."""
    import requests

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"direct download {url}")
    with requests.get(url, stream=True, timeout=1800) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    tmp.replace(dest)


def _fetch(filename: str, dest: Path, repo_id: str = REPO_ID) -> None:
    """Fetch one repo file into MODEL_DIR. huggingface_hub is imported lazily so
    a warm cache never touches the network or the import."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id=repo_id, filename=filename, revision="main",
                        local_dir=str(MODEL_DIR))
    except Exception as exc:  # proxy quirks, hub outage, xet problems
        log(f"hf_hub_download failed for {filename} "
            f"({exc.__class__.__name__}: {str(exc)[:120]})")
        _download_direct(filename, dest, repo_id)


def ensure_model_files() -> tuple[Path, Path]:
    """Return (model_path, utils_path), downloading only what is missing or broken."""
    model_path = MODEL_DIR / MODEL_FILE
    utils_path = MODEL_DIR / UTILS_FILE

    # utils.py first: its constants define the whole pipeline.
    if not (utils_path.exists() and utils_path.stat().st_size == UTILS_BYTES):
        log(f"downloading {UTILS_FILE} ...")
        _fetch(UTILS_FILE, utils_path)
        log(f"{UTILS_FILE}: {utils_path.stat().st_size:,} bytes")
    else:
        log(f"utils.py cached ({UTILS_BYTES:,} bytes) - no network needed")

    size = model_path.stat().st_size if model_path.exists() else -1
    if abs(size - MODEL_BYTES) <= MODEL_BYTES_TOL:
        log(f"model cached: {size/1e6:.1f} MB - running fully offline")
    else:
        if model_path.exists():
            log(f"cached model.onnx looks truncated ({size:,} of ~{MODEL_BYTES:,} "
                "bytes) - re-downloading")
            model_path.unlink()
        else:
            log("model.onnx not cached - downloading ~176 MB (one time) ...")
        t0 = time.perf_counter()
        _fetch(MODEL_FILE, model_path)
        size = model_path.stat().st_size if model_path.exists() else -1
        if abs(size - MODEL_BYTES) > MODEL_BYTES_TOL:
            model_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"model.onnx is still truncated after download ({size:,} of "
                f"{MODEL_BYTES:,} bytes). Check your connection and re-run - the "
                "bad file was deleted, so it will be fetched again.")
        digest = _sha256(model_path)
        if digest != MODEL_SHA256:
            model_path.unlink(missing_ok=True)
            raise RuntimeError(
                "model.onnx checksum mismatch - corrupt download, or Trendyol "
                "published a new version. Re-run to refetch; if the repo really "
                "changed, update MODEL_SHA256 in app.py.")
        log(f"model.onnx ok: {size/1e6:.1f} MB in {time.perf_counter()-t0:.1f}s, "
            "sha256 verified")

    return model_path, utils_path


def ensure_upscale_model() -> Path:
    """Return path to realesr-general-x4v3.onnx, downloading if missing."""
    up_path = MODEL_DIR / UPSCALE_MODEL_FILE
    size = up_path.stat().st_size if up_path.exists() else -1
    if abs(size - UPSCALE_MODEL_BYTES) <= UPSCALE_MODEL_BYTES_TOL:
        log(f"upscale model cached: {size/1e6:.1f} MB")
        return up_path
    if up_path.exists():
        log(f"cached {UPSCALE_MODEL_FILE} looks truncated - re-downloading")
        up_path.unlink()
    else:
        log(f"{UPSCALE_MODEL_FILE} not cached - downloading ~{UPSCALE_MODEL_BYTES/1e6:.0f} MB ...")
    t0 = time.perf_counter()
    # ponytail: one streaming GET beats the hf_hub local_dir dance for a 4.9 MB
    # file in a repo subfolder; the hub path would land it as onnx/model.onnx.
    _download_direct(UPSCALE_REPO_FILE, up_path, UPSCALE_REPO_ID)
    size = up_path.stat().st_size if up_path.exists() else -1
    if abs(size - UPSCALE_MODEL_BYTES) > UPSCALE_MODEL_BYTES_TOL:
        up_path.unlink(missing_ok=True)
        raise RuntimeError(f"{UPSCALE_MODEL_FILE} truncated after download ({size:,} bytes)")
    digest = _sha256(up_path)
    if digest != UPSCALE_MODEL_SHA256:
        up_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{UPSCALE_MODEL_FILE} checksum mismatch - corrupt download or new version. "
            "Re-run to refetch; if the repo changed, update UPSCALE_MODEL_SHA256.")
    log(f"{UPSCALE_MODEL_FILE} ok: {size/1e6:.1f} MB in {time.perf_counter()-t0:.1f}s")
    return up_path


# ---------------------------------------------------------------------------
# The repo's utils.py: reused when importable, ported when not
# ---------------------------------------------------------------------------
# utils.py is downloaded above and imported here when possible. It does
# `import torch` at module level and PyTorch ships no Intel-macOS wheels past
# 2.2.2 (the repo pins 2.7.1), so on this Mac the import fails and the NumPy /
# OpenCV port further down runs instead. Both paths implement the same maths and
# constants from utils.py: bilinear resize, (x-0.5)/1.0 normalisation, min-max
# mask normalisation, threshold 25, area_limit = 50000*h*w/(1800*1200), 9x9
# ellipse dilate with 2 iterations.
REPO_UTILS: Optional[Any] = None


def load_repo_utils(utils_path: Path) -> None:
    global REPO_UTILS
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("trendyol_utils", str(utils_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)                     # needs torch + torchvision
        REPO_UTILS = mod
        log(f"repo utils.py active (INPUT_SIZE={mod.INPUT_SIZE}, torch path)")
    except Exception as exc:
        REPO_UTILS = None
        first = str(exc).splitlines()[0][:90] if str(exc) else exc.__class__.__name__
        log(f"repo utils.py not importable ({exc.__class__.__name__}: {first}) "
            "-> using the built-in NumPy port of the same pre/post-processing")


# ---------------------------------------------------------------------------
# Pre / post-processing
# ---------------------------------------------------------------------------
# Every constant and operation below is taken from the model repo's utils.py
# (INPUT_SIZE=[1200,1800]; F.upsample bilinear; divide by 255; normalize with
# mean=[0.5,0.5,0.5] std=[1,1,1]; min-max the upsampled mask; threshold 25;
# area_limit = 50000*h*w/(1800*1200); 9x9 ellipse dilate, 2 iterations). When
# utils.py imports cleanly its functions are called directly; otherwise these
# NumPy/OpenCV equivalents run. torch's F.upsample and cv2.resize differ by a
# hair on pixel-grid alignment, which min-max normalisation absorbs.

def scaled_hw(h: int, w: int, max_side: Optional[int]) -> tuple[int, int]:
    """Longest side capped at max_side, aspect kept. None/0 = untouched."""
    if not max_side or max(h, w) <= max_side:
        return h, w
    k = max_side / float(max(h, w))
    return max(32, int(round(h * k))), max(32, int(round(w * k)))


def preprocess_input(im: np.ndarray, net_hw: tuple[int, int]) -> np.ndarray:
    """RGB uint8 (H,W,3) -> float32 NCHW normalised for the net."""
    if REPO_UTILS is not None:
        return np.ascontiguousarray(
            REPO_UTILS.preprocess_input(np.ascontiguousarray(im)).cpu().numpy())
    if im.ndim == 2:
        im = im[:, :, None]
    if im.shape[2] == 4:
        im = im[:, :, :3]
    net_h, net_w = net_hw
    resized = cv2.resize(im, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x -= 0.5                                    # normalize([0.5]*3, [1.0]*3)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None, ...])


def keep_large_components(a: np.ndarray) -> np.ndarray:
    """utils.keep_large_components: drop blobs smaller than area_limit."""
    if REPO_UTILS is not None:
        return REPO_UTILS.keep_large_components(a)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    a_mask = (a > 25).astype(np.uint8) * 255
    total_labels, label_ids, values, _ = cv2.connectedComponentsWithStats(
        a_mask, 4, cv2.CV_32S)
    h, w = a.shape[:2]
    area_limit = 50000 * (h * w) / (REPO_INPUT_HW[1] * REPO_INPUT_HW[0])
    keep = [i for i in range(1, total_labels)
            if values[i, cv2.CC_STAT_AREA] > area_limit]
    if keep:
        final_mask = np.zeros_like(a, dtype=np.uint8)
        for i in keep:
            final_mask = cv2.bitwise_or(
                final_mask, (label_ids == i).astype(np.uint8) * 255)
        final_mask = cv2.dilate(final_mask, dilate_kernel, iterations=2)
        a = cv2.bitwise_and(a, final_mask)
        a = a.reshape((a.shape[0], a.shape[1], 1))
    return a


def postprocess_output(result: np.ndarray, orig_hw: tuple[int, int]) -> np.ndarray:
    """Net output -> uint8 alpha (H,W) at the original image size."""
    out = np.asarray(result)
    if REPO_UTILS is not None:
        # utils.postprocess_output expects (1,H,W) for this export.
        if out.ndim == 3:
            out = out[0][None, ...]
        alpha = REPO_UTILS.postprocess_output(np.ascontiguousarray(out), tuple(orig_hw))
    else:
        while out.ndim > 2:
            out = out[0]
        h, w = orig_hw
        up = cv2.resize(out.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        lo, hi = float(up.min()), float(up.max())
        # hi == lo would divide by zero (utils.py yields NaN there -> 0); blank is
        # the honest answer when the model is flat.
        up = (up - lo) / (hi - lo) if hi > lo else np.zeros_like(up)
        alpha = keep_large_components((up * 255.0).astype(np.uint8)[:, :, None])
    alpha = np.asarray(alpha)
    while alpha.ndim > 2:
        alpha = alpha[..., 0]
    return alpha


# ---------------------------------------------------------------------------
# ONNX session (loaded once, kept in memory)
# ---------------------------------------------------------------------------
CPU = "CPUExecutionProvider"
COREML = "CoreMLExecutionProvider"


def build_session(model_path: Path, provider: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = INTRA_OP_THREADS
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=[provider])


def synthetic_test_image(side: int = 1024) -> np.ndarray:
    """Deterministic stand-in photo for the CoreML benchmark (no download): a
    bright ellipse over a noisy gradient, so a working mask is never blank."""
    img = np.zeros((side, side, 3), np.uint8)
    yy, xx = np.mgrid[0:side, 0:side]
    img[..., 0] = (200 * xx / side).astype(np.uint8)
    img[..., 1] = (200 * yy / side).astype(np.uint8)
    img[..., 2] = 110
    cv2.ellipse(img, (side // 2, int(side * 0.55)), (side // 5, side // 3), 0, 0, 360,
                (240, 230, 220), -1)
    rng = np.random.default_rng(0)
    return cv2.add(img, rng.integers(0, 20, img.shape, dtype=np.uint8))


def mask_is_usable(raw_out: np.ndarray, alpha: np.ndarray) -> bool:
    """False if the provider returned a flat or empty mask (CoreML can do this)."""
    raw = np.asarray(raw_out)
    if float(raw.max()) - float(raw.min()) < 1e-6:
        return False
    return float((np.asarray(alpha) > 25).mean()) > 0.001


class Engine:
    """One InferenceSession held for the whole process."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.session = None
        self.provider = CPU
        self.input_name = None
        self.output_names = None
        self.net_hw = REPO_INPUT_HW
        self.fixed_input = True

    def load(self, provider: str = CPU) -> None:
        t0 = time.perf_counter()
        self.session = build_session(self.model_path, provider)
        self.provider = provider
        self._bind()
        log(f"session ready on {provider} in {time.perf_counter() - t0:.1f}s "
            f"(intra_op_num_threads={INTRA_OP_THREADS})")

    def _bind(self) -> None:
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = list(inp.shape)

        def as_int(v):
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        h, w = (as_int(shape[2]), as_int(shape[3])) if len(shape) >= 4 else (None, None)
        self.fixed_input = bool(h and w)
        self.net_hw = (h, w) if self.fixed_input else REPO_INPUT_HW
        # This graph also returns 11 intermediate feature maps (~140 MB of dead
        # weight per call on 16 GB). Requesting just the mask was verified
        # np.array_equal to run(None)[0], so ask for one output only.
        self.output_names = [self.session.get_outputs()[0].name]
        log(f"input '{self.input_name}' shape={shape} -> "
            f"{'FIXED' if self.fixed_input else 'dynamic'} net size {self.net_hw}, "
            f"output '{self.output_names[0]}' of "
            f"{len(self.session.get_outputs())} available")


    def setup_provider(self, try_coreml: bool) -> str:
        """Pick CPU or (when asked and usable) CoreML by measuring both."""
        import gc

        import onnxruntime as ort

        available = ort.get_available_providers()
        if not try_coreml:
            self.load(CPU)
            return self.provider
        if COREML not in available:
            log(f"--try-coreml: this onnxruntime build has no CoreML EP "
                f"(available: {', '.join(available)}). Staying on CPU.")
            self.load(CPU)
            return self.provider

        log("--try-coreml: benchmarking CoreML vs CPU on a 1024x1024 test image "
            "(2 runs each, reporting the 2nd)")
        img = synthetic_test_image(1024)
        scores = {}
        for provider in (CPU, COREML):
            sess = None
            try:
                sess = build_session(self.model_path, provider)
                name = sess.get_inputs()[0].name
                outs = [sess.get_outputs()[0].name]
                sh = list(sess.get_inputs()[0].shape)
                try:
                    net_hw = (int(sh[2]), int(sh[3]))
                except (TypeError, ValueError, IndexError):
                    net_hw = REPO_INPUT_HW
                x = preprocess_input(img, net_hw)
                elapsed = float("inf")
                for _ in range(2):
                    t0 = time.perf_counter()
                    out = sess.run(outs, {name: x})[0]
                    elapsed = time.perf_counter() - t0
                usable = mask_is_usable(out, postprocess_output(out, img.shape[:2]))
                scores[provider] = (elapsed, usable, None)
            except Exception as exc:
                scores[provider] = (float("inf"), False, exc)
            finally:
                sess = None
                gc.collect()

        for provider, (t, usable, exc) in scores.items():
            note = "OK" if usable else (
                f"{exc.__class__.__name__}: {str(exc)[:60]}" if exc else "BLANK MASK")
            log(f"    {provider:24s} {t:6.2f}s  {note}")

        usable_scores = {p: v for p, v in scores.items() if v[1]}
        winner = min(usable_scores, key=lambda p: usable_scores[p][0]) if usable_scores else CPU
        log(f"    -> keeping {winner}")
        self.load(winner)
        return self.provider

    def mask_for(self, rgb: np.ndarray, max_side: Optional[int]) -> tuple[np.ndarray, dict]:
        """RGB uint8 -> alpha uint8 at the ORIGINAL size, plus stage timings."""
        h0, w0 = rgb.shape[:2]
        work_hw = scaled_hw(h0, w0, max_side)
        if self.fixed_input:
            net_hw = self.net_hw            # this export: always (1200, 1800)
        else:
            net_hw = (max(32, work_hw[0] // 32 * 32), max(32, work_hw[1] // 32 * 32))
        if work_hw != (h0, w0):
            work = cv2.resize(rgb, (work_hw[1], work_hw[0]), interpolation=cv2.INTER_AREA)
        else:
            work = rgb

        t0 = time.perf_counter()
        x = preprocess_input(work, net_hw)
        t_prep = time.perf_counter() - t0

        t0 = time.perf_counter()
        out = self.session.run(self.output_names, {self.input_name: x})[0]
        t_infer = time.perf_counter() - t0

        t0 = time.perf_counter()
        alpha = postprocess_output(out, (h0, w0))
        t_post = time.perf_counter() - t0

        stats = {"prep": t_prep, "infer": t_infer, "post": t_post, "total": None,
                 "orig": (h0, w0), "work": work_hw, "net": net_hw,
                 "provider": self.provider, "fixed_input": self.fixed_input}
        return alpha, stats


# ---------------------------------------------------------------------------
# Processing modes
# ---------------------------------------------------------------------------
MODE_BG_ONLY = "Background Removal Only"
MODE_UPSCALE_AND_BG = "Upscale & Remove Background"
MODE_UPSCALE_ONLY = "Upscale Only (No Background Removal)"
PROCESSING_MODES = [MODE_BG_ONLY, MODE_UPSCALE_AND_BG, MODE_UPSCALE_ONLY]


# ---------------------------------------------------------------------------
# Real-ESRGAN upscaler engine
# ---------------------------------------------------------------------------
class UpscaleEngine:
    """Lazy-loaded Real-ESRGAN ONNX session for 4x upscaling."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.session = None
        self.provider = CPU

    def load(self, provider: str = CPU) -> None:
        t0 = time.perf_counter()
        self.session = build_session(self.model_path, provider)
        self.provider = provider
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        log(f"upscale session ready on {provider} in {time.perf_counter() - t0:.1f}s")

    def upscale(self, rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 (H,W,3) -> RGB uint8 (H*4,W*4,3) via Real-ESRGAN."""
        h, w = rgb.shape[:2]
        if max(h, w) > MAX_UPSCALE_INPUT_SIDE:
            raise InputError(
                f"Image is {w}x{h}. Upscale mode requires both sides <= "
                f"{MAX_UPSCALE_INPUT_SIDE}px to keep memory sane. "
                "Resize first or use Background Removal Only.")
        # ponytail: Real-ESRGAN ONNX expects RGB float32 [0,1] NCHW
        x = rgb.astype(np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None, ...])
        out = self.session.run([self.output_name], {self.input_name: x})[0]
        out = np.clip(out[0].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        return out


# ---------------------------------------------------------------------------
# Image IO
# ---------------------------------------------------------------------------
class InputError(Exception):
    """Anything the user can fix: wrong file, broken file, absurdly big file."""


@lru_cache(maxsize=1)
def avif_decoder_version() -> Optional[str]:
    """Return the bundled libavif version if Pillow can decode AVIF, else None.

    Pillow wheels bundle libavif starting with Pillow 11.0 / 11.3; a system without
    that plugin raises ValueError on check_module('avif') or returns False.
    """
    try:
        from PIL import features
        if features.check_module("avif"):
            ver = features.version("avif")
            return f"libavif {ver}" if ver else "present"
    except Exception:
        pass
    return None


def _looks_like_avif(path: Path) -> bool:
    """True when the file starts with an ISOBMFF 'ftyp' box naming an AVIF brand.

    Distinguishes "libavif could not decode this AVIF" from "this is not an AVIF
    file at all" (wrong extension, truncated download, saved error page).
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(64)
    except OSError:
        return False
    return len(head) >= 12 and head[4:8] == b"ftyp" and (
        b"avif" in head or b"avis" in head)


def describe_decode_error(path: Path, exc: Exception) -> str:
    """Turn an open/load exception into advice the user can act on."""
    cname = exc.__class__.__name__
    reason = str(exc).strip().replace("\n", " ")[:120]

    if path.suffix.lower() == ".avif":
        dec = avif_decoder_version()
        if not _looks_like_avif(path):
            return (f"{path.name} has an .avif extension but is not an AVIF file "
                    f"(no 'ftyp' header). It may be a truncated download or a "
                    f"different format under the wrong name ({cname}: {reason}).")
        if not dec:
            return (f"Cannot read {path.name}: this environment has no AVIF decoder. "
                    "Run `./venv/bin/pip install -U 'pillow>=11.3'` to enable it "
                    f"({cname}: {reason}).")
        return (f"Could not decode {path.name}: {dec} could not parse this AVIF. "
                "Conformance-suite, HDR and gain-map encodes are often unsupported; "
                f"re-exporting it as PNG, JPEG or WebP will work ({cname}: {reason}).")

    known = path.suffix.lower() in SUPPORTED_EXT
    hint = "" if known else f" Supported: {', '.join(sorted(SUPPORTED_EXT))}."
    return (f"Could not read {path.name} as an image ({cname}: {reason}). "
            f"It may be corrupt, truncated or not a picture at all.{hint}")


def load_rgb(path_str: str) -> np.ndarray:
    """Validate + decode a user file into RGB uint8 (H,W,3)."""
    if not path_str:
        raise InputError("No image selected. Drop a photo in the box first.")
    path = Path(str(path_str))
    if not path.exists() or not path.is_file():
        raise InputError(f"File not found: {path}")

    # Fast-path suffix check. Unknown extensions fall through to header sniffing
    # so renamed files and clipboard pastes still work.
    has_known_ext = path.suffix.lower() in SUPPORTED_EXT

    size = path.stat().st_size
    if size == 0:
        raise InputError(f"{path.name} is empty (0 bytes).")
    if size > MAX_INPUT_BYTES:
        raise InputError(f"{path.name} is {size/1e6:.0f} MB, over the "
                         f"{MAX_INPUT_BYTES//(1<<20)} MB limit. Export a smaller copy.")

    # Guard: if it's explicitly .avif but this Pillow has no decoder, tell them right away.
    if path.suffix.lower() == ".avif" and not avif_decoder_version():
        raise InputError(
            f"Cannot open {path.name}: AVIF decoding is not available in this "
            "Pillow install. Run `./venv/bin/pip install -U 'pillow>=11.3'` to enable it.")

    try:
        with Image.open(path) as im:
            if not has_known_ext and im.format not in SNIFFABLE_FORMATS:
                raise InputError(
                    f"Unsupported file type '{path.suffix or '(none)'}' "
                    f"(detected as {im.format or 'unknown'}). Supported: "
                    f"{', '.join(sorted(SUPPORTED_EXT))}")

            # Reject absurd dimensions from the header BEFORE decoding pixels.
            # Large AVIFs can otherwise thrash memory before the check is reached.
            w, h = im.size
            if w * h > MAX_INPUT_PIXELS:
                raise InputError(
                    f"{path.name} is {w}x{h} ({w*h/1e6:.0f} MP), over the "
                    f"{MAX_INPUT_PIXELS//1_000_000} MP limit this 16 GB Mac can "
                    "handle in one pass. Downscale it and try again.")

            im.load()
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                flat = Image.new("RGB", im.size, (255, 255, 255))
                rgba = im.convert("RGBA")
                flat.paste(rgba, mask=rgba.getchannel("A"))
                im = flat
            else:
                im = im.convert("RGB")
            return np.asarray(im, dtype=np.uint8)
    except InputError:
        raise
    except Exception as exc:
        raise InputError(describe_decode_error(path, exc)) from None


def checkerboard(h: int, w: int, square: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    parity = ((yy // square + xx // square) % 2).astype(np.uint8)
    return np.where(parity[..., None].astype(bool), 204, 255).astype(np.uint8)


def composite_preview(rgba: np.ndarray, max_side: int = PREVIEW_MAX_SIDE) -> np.ndarray:
    """Flatten RGBA over a checkerboard for display only (the PNG keeps alpha)."""
    h, w = rgba.shape[:2]
    ph, pw = scaled_hw(h, w, max_side)
    if (ph, pw) != (h, w):
        small = cv2.resize(rgba, (pw, ph), interpolation=cv2.INTER_AREA)
    else:
        small = rgba
    bg = checkerboard(ph, pw)
    a = small[..., 3:4].astype(np.float32) / 255.0
    return (small[..., :3] * a + bg * (1.0 - a)).astype(np.uint8)


def _save_output(arr: np.ndarray, mode_str: str, stem: str) -> Path:
    """Save an RGBA or RGB array to outputs/; return the path."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = "upscaled" if mode_str == MODE_UPSCALE_ONLY else "nobg"
    out_png = OUT_DIR / f"{stem}_{suffix}_{stamp}.png"
    Image.fromarray(arr, "RGBA" if arr.shape[2] == 4 else "RGB").save(
        out_png, format="PNG", compress_level=1)
    return out_png


def process_image(engine: Engine, upscale_engine: Optional[UpscaleEngine],
                  path_str: str, max_side_choice: str,
                  mode: str = MODE_BG_ONLY) -> dict:
    """Full pipeline for one image, routed by processing mode. Raises InputError
    on bad input; RuntimeError on an inference failure. Returns timings, paths
    and the preview array."""
    t_all = time.perf_counter()
    rgb = load_rgb(path_str)
    max_side = None if str(max_side_choice).lower() == "original" \
        else int(max_side_choice)
    stem = Path(str(path_str)).stem or "image"

    if mode == MODE_UPSCALE_ONLY:
        if upscale_engine is None:
            raise RuntimeError("Upscale model is not available")
        t0 = time.perf_counter()
        upscaled = upscale_engine.upscale(rgb)
        t_up = time.perf_counter() - t0
        out_png = _save_output(upscaled, mode, stem)
        h, w = upscaled.shape[:2]
        return {
            "total": time.perf_counter() - t_all, "prep": 0.0,
            "infer": t_up, "post": 0.0, "upscale": t_up,
            "orig": rgb.shape[:2], "work": rgb.shape[:2], "net": (h, w),
            "provider": upscale_engine.provider, "fixed_input": False,
            "out_png": str(out_png), "out_mb": out_png.stat().st_size / 1e6,
            "max_side": max_side_choice, "fg_pct": 100.0, "mode": mode,
            "preview": upscaled,}

    # Both remaining modes end in an IS-Net cutout; only the input differs.
    work_rgb, t_up = rgb, 0.0
    if mode == MODE_UPSCALE_AND_BG:
        if upscale_engine is None:
            raise RuntimeError("Upscale model is not available")
        t0 = time.perf_counter()
        work_rgb = upscale_engine.upscale(rgb)
        t_up = time.perf_counter() - t0

    alpha, stats = engine.mask_for(work_rgb, max_side)
    rgba = np.dstack([work_rgb, alpha])
    if not (rgba.shape[0], rgba.shape[1]) == stats["orig"]:
        raise RuntimeError("internal error: output size does not match the input")

    out_png = _save_output(rgba, mode, stem)
    stats["total"] = time.perf_counter() - t_all
    stats["out_png"] = str(out_png)
    stats["out_mb"] = out_png.stat().st_size / 1e6
    stats["max_side"] = max_side_choice
    stats["fg_pct"] = float((alpha > 127).mean() * 100.0)
    stats["mode"] = mode
    stats["upscale"] = t_up
    stats["preview"] = composite_preview(rgba)
    return stats


def stats_markdown(s: dict) -> str:
    h, w = s["orig"]
    mode = s.get("mode", MODE_BG_ONLY)
    fmt = "RGB PNG (no transparency)" if mode == MODE_UPSCALE_ONLY else "RGBA PNG"
    fixed = ("The ONNX graph is fixed at {0}x{1}, so Max side only changes the "
             "CPU pre/post work, not inference time."
             ).format(s["net"][0], s["net"][1]) if s["fixed_input"] else \
            "Max side feeds the network directly."
    up = f"upscale **{s['upscale']:.2f}s**, " if s.get("upscale", 0) > 0 else ""
    fg = (f", foreground {s['fg_pct']:.1f}% of pixels" if mode != MODE_UPSCALE_ONLY
          else "")
    return (
        f"**Done in {s['total']:.2f}s** ({mode})  |  {up}"
        f"model **{s['infer']:.2f}s**, "
        f"pre {s['prep']:.2f}s, post {s['post']:.2f}s  |  "
        f"provider **{s['provider']}**<br>"
        f"Output **{w}x{h}** {fmt}, {s['out_mb']:.1f} MB -> `{s['out_png']}`<br>"
        f"Working size {s['work'][1]}x{s['work'][0]}, network "
        f"{s['net'][1]}x{s['net'][0]}{fg}. "
        f"{fixed}"
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui(engine: Engine, upscale_model_path: Optional[Path],
             default_max_side: str, startup_note: str):
    import gradio as gr

    # ponytail: lazy-load the upscale session on first upscale request (~65 MB
    # of RAM is not worth paying when only background removal is used)
    _uc: list[Optional[UpscaleEngine]] = [None]

    def get_upscale() -> Optional[UpscaleEngine]:
        if upscale_model_path is None:
            return None
        if _uc[0] is None:
            _uc[0] = UpscaleEngine(upscale_model_path)
            _uc[0].load(engine.provider)
        return _uc[0]

    def run(image_path, mode, max_side_choice):
        try:
            ue = None if mode == MODE_BG_ONLY else get_upscale()
            s = process_image(engine, ue, image_path, max_side_choice, mode)
        except InputError as exc:
            return None, None, f"**{exc}**"
        except Exception as exc:
            return None, None, (
                f"**Processing failed** ({exc.__class__.__name__}): {str(exc)[:300]}\n\n"
                "The image was not saved. See the console for details.")
        # DownloadButton takes the file path straight from the second output
        return s["preview"], s["out_png"], stats_markdown(s)

    with gr.Blocks(title="Background Remover") as demo:
        gr.Markdown("## Background Remover & Upscaler (local, offline)\n"
                    "IS-Net ONNX by Trendyol + Real-ESRGAN running on this Mac. "
                    "Nothing is uploaded anywhere. Results land in `outputs/`.\n\n"
                    "Accepts JPG, PNG, BMP, TIFF, WebP and AVIF (including AVIF "
                    "with alpha and 10/12-bit colour); pasted images arrive "
                    "re-encoded as WebP.")
        gr.Markdown(startup_note)
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(type="filepath", label="Photo",
                                  sources=["upload", "clipboard"])
                mode_radio = gr.Radio(choices=PROCESSING_MODES,
                                      value=MODE_BG_ONLY, label="Processing Mode")
                side_dd = gr.Dropdown(choices=MAX_SIDE_CHOICES, value=default_max_side,
                                      label="Max side length (processed size)")
                btn = gr.Button("Process", variant="primary")
            with gr.Column(scale=1):
                preview = gr.Image(type="numpy", label="Result (checkerboard = "
                                                       "transparent)", interactive=False)
                dl = gr.DownloadButton("Download result", size="lg")
        status = gr.Markdown("Pick a photo, choose a processing mode, and press "
                             "**Process**.")
        btn.click(run, inputs=[img_in, mode_radio, side_dd],
                  outputs=[preview, dl, status], show_progress="full")
    return demo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def bench(engine: Engine, upscale_engine: Optional[UpscaleEngine],
          image_path: str, max_side_choice: str, mode: str = MODE_BG_ONLY) -> int:
    log(f"bench: {image_path} (mode={mode}, max side: {max_side_choice})")
    for run_no in (1, 2):
        try:
            s = process_image(engine, upscale_engine, image_path,
                              max_side_choice, mode)
        except InputError as exc:
            log(f"bench run {run_no}: {exc}")
            return 1
        log(f"bench run {run_no}: total {s['total']:.2f}s "
            f"(pre {s['prep']:.2f} / model {s['infer']:.2f} / post {s['post']:.2f}) "
            f"on {s['provider']}, out {s['orig'][1]}x{s['orig'][0]} "
            f"{s['out_mb']:.1f} MB -> {s['out_png']}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Local background remover (IS-Net ONNX + Gradio, CPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--try-coreml", action="store_true",
                    help="benchmark the CoreML provider against CPU at startup and "
                         "keep the faster one (falls back to CPU on error/blank mask)")
    ap.add_argument("--bench", metavar="IMAGE",
                    help="process IMAGE twice and print timings instead of starting "
                         "the UI")
    ap.add_argument("--mode", choices=PROCESSING_MODES, default=MODE_BG_ONLY,
                    help=f"processing mode for --bench (default: {MODE_BG_ONLY})")
    ap.add_argument("--max-side", choices=MAX_SIDE_CHOICES, default=DEFAULT_MAX_SIDE,
                    help=f"default processed size (default: {DEFAULT_MAX_SIDE})")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true",
                    help="do not open a browser window")
    args = ap.parse_args(argv)

    log(f"python {sys.version.split()[0]} ({sys.platform}), "
        f"intra_op_num_threads={INTRA_OP_THREADS}")
    if sys.version_info < (3, 11) or sys.version_info >= (3, 12):
        log("WARNING: onnxruntime 1.16.3 (the last build with Intel-macOS "
            "Monterey support) only has cp311 wheels. Recreate ./venv with "
            "./python311 - see README.md")

    av = avif_decoder_version()
    if av:
        log(f"AVIF decoder: {av} (Pillow {PIL.__version__})")
    else:
        log(f"WARNING: Pillow {PIL.__version__} has no AVIF decoder - .avif "
            "uploads will fail. Enable it with: "
            "./venv/bin/pip install -U 'pillow>=11.3'")

    try:
        model_path, utils_path = ensure_model_files()
    except Exception as exc:
        log(f"cannot get the model files: {exc}")
        log("put model.onnx + utils.py into "
            f"{MODEL_DIR} by hand, or run this while online once.")
        return 1
    load_repo_utils(utils_path)

    # Upscale model: download/check now, load into memory lazily in the UI
    upscale_model_path: Optional[Path] = None
    try:
        upscale_model_path = ensure_upscale_model()
    except Exception as exc:
        log(f"WARNING: upscale model unavailable ({exc}) - "
            "upscale modes will be disabled")

    engine = Engine(model_path)
    try:
        engine.setup_provider(args.try_coreml)
    except Exception as exc:
        log(f"failed to load the ONNX session: {exc.__class__.__name__}: {exc}")
        return 1

    if args.bench:
        ue = None
        if args.mode != MODE_BG_ONLY and upscale_model_path:
            ue = UpscaleEngine(upscale_model_path)
            ue.load(engine.provider)
        return bench(engine, ue, args.bench, args.max_side, args.mode)

    up_note = (f", upscale model {UPSCALE_MODEL_BYTES/1e6:.0f} MB"
               if upscale_model_path else ", upscale model **unavailable**")
    note = (f"*Models cached in `model_cache/` (IS-Net {MODEL_BYTES/1e6:.0f} MB"
            f"{up_note}), inference on **{engine.provider}**, network input "
            f"{engine.net_hw[1]}x{engine.net_hw[0]}.*")
    demo = build_ui(engine, upscale_model_path, args.max_side, note)
    log(f"UI on http://{args.host}:{args.port} (Ctrl-C to stop)")
    demo.launch(server_name=args.host, server_port=args.port,
                inbrowser=not args.no_browser, share=False)
    return 0


if __name__ == "__main__":
    sys_exit = 0
    try:
        sys_exit = main()
    except KeyboardInterrupt:
        print()
    raise SystemExit(sys_exit)






