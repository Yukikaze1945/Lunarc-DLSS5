"""CPU decoders for HDR stills into linear Rec.709 scRGB."""

from __future__ import annotations

import os
import subprocess

import numpy as np

import pipeline
from jxr_image_decode import EXTENSIONS as JXR_EXTENSIONS


_PQ_TRANSFERS = {"smpte2084", "smpte-st-2084"}


def probe_hdr(path) -> dict | None:
    """Return pipeline metadata for supported HDR stills, otherwise ``None``."""
    extension = os.path.splitext(os.fspath(path))[1].lower()
    if extension in JXR_EXTENSIONS:
        import jxr_image_decode
        return jxr_image_decode.probe_hdr(path)
    if extension != ".avif":
        return None
    info = pipeline.probe_pixel_info(os.fspath(path))
    if not isinstance(info, dict):
        return None
    if str(info.get("color_transfer", "")).lower() not in _PQ_TRANSFERS:
        return None
    if str(info.get("color_primaries", "")).lower() != "bt2020":
        return None
    if str(info.get("color_space", "")).lower() not in {"bt2020nc", "bt2020_ncl"}:
        return None
    if str(info.get("color_range", "")).lower() not in {"pc", "full", "tv", "mpeg"}:
        return None
    if int(info.get("bit_depth") or 0) < 10:
        return None
    if int(info.get("width") or 0) <= 0 or int(info.get("height") or 0) <= 0:
        return None
    return info


def decode_hdr(path, info=None) -> np.ndarray:
    """Decode one supported HDR still to float16 RGBA linear Rec.709 scRGB.

    scRGB is normalized at 80 cd/m² per unit. Components may be negative or
    greater than one; only alpha is constrained to the coverage range [0, 1].
    """
    path = os.fspath(path)
    if os.path.splitext(path)[1].lower() in JXR_EXTENSIONS:
        import jxr_image_decode
        return jxr_image_decode.decode_hdr(path, info)
    if info is None:
        info = probe_hdr(path)
    elif not isinstance(info, dict):
        raise ValueError("HDR AVIF metadata must be a dictionary")
    if info is None:
        raise ValueError("input is not a supported PQ-tagged AVIF")

    width, height = int(info.get("width") or 0), int(info.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("HDR AVIF metadata has invalid dimensions")
    transfer = str(info.get("color_transfer", "")).lower()
    if transfer not in _PQ_TRANSFERS:
        raise ValueError("HDR AVIF must use SMPTE ST 2084 PQ")
    if str(info.get("color_primaries", "")).lower() != "bt2020":
        raise ValueError("HDR AVIF must be tagged with BT.2020 primaries")
    if str(info.get("color_space", "")).lower() not in {"bt2020nc", "bt2020_ncl"}:
        raise ValueError("HDR AVIF must use the BT.2020 non-constant matrix")
    source_range = str(info.get("color_range", "")).lower()
    if source_range in {"pc", "full"}:
        range_in = "full"
    elif source_range in {"tv", "mpeg"}:
        range_in = "limited"
    else:
        raise ValueError("HDR AVIF must declare full or limited range")

    ffmpeg = pipeline._bundle_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("bundled FFmpeg is unavailable")
    filt = (
        f"zscale=rangein={range_in}:matrixin=bt2020nc:primariesin=bt2020:"
        "transferin=smpte2084:range=full:matrix=gbr:primaries=bt709:"
        "transfer=linear:npl=80,format=gbrapf32le"
    )
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-i", path,
        "-map", "0:v:0", "-frames:v", "1", "-vf", filt,
        "-f", "rawvideo", "-pix_fmt", "gbrapf32le", "pipe:1",
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=180, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"FFmpeg could not decode HDR AVIF: {exc}") from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"FFmpeg HDR AVIF decode failed: {detail}")

    expected = width * height * 4 * np.dtype("<f4").itemsize
    if len(result.stdout) != expected:
        raise RuntimeError(
            f"FFmpeg returned {len(result.stdout)} bytes; expected {expected}"
        )
    # gbrapf32le stores four full-resolution planes in G, B, R, A order.
    planes = np.frombuffer(result.stdout, dtype="<f4").reshape(4, height, width)
    if not np.isfinite(planes).all():
        raise ValueError("HDR AVIF decode produced non-finite components")
    alpha = planes[3]
    if np.any(alpha < 0.0) or np.any(alpha > 1.0):
        raise ValueError("HDR AVIF decode produced alpha outside [0, 1]")
    rgba = np.stack((planes[2], planes[0], planes[1], alpha), axis=-1)
    if np.any(np.abs(rgba) > np.finfo(np.float16).max):
        raise ValueError("HDR AVIF components exceed float16 range")
    decoded = rgba.astype(np.float16)
    if not np.isfinite(decoded).all():
        raise ValueError("HDR AVIF components overflowed float16")
    return decoded
