"""RIFE frame interpolation helper.

Wraps `rife-ncnn-vulkan.exe` (https://github.com/nihui/rife-ncnn-vulkan) to
double or triple the frame rate of an MP4 produced by Wan 2.2 (16fps native).

Why ncnn/Vulkan and not CUDA RIFE?
- Doesn't share VRAM pool with ComfyUI - works after Comfy generation finishes,
  no conflict with the model still warm in CUDA memory.
- Single .exe, no Python dependencies, no pytorch version pin.
- ~3-5 sec per 80-frame chunk on a 3080 Ti via Vulkan.

Setup (one-time):
    1. Download `rife-ncnn-vulkan-<date>-windows.zip` from
       https://github.com/nihui/rife-ncnn-vulkan/releases
    2. Extract somewhere stable, e.g. `C:\\Tools\\rife-ncnn-vulkan\\`.
       The folder must contain `rife-ncnn-vulkan.exe` and a `rife-v4.6` (or
       newer like `rife-v4.26`) subfolder with the model weights.
    3. Set `RIFE_EXE` below or pass `rife_exe=` to the function.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# Configure once. Override with rife_exe= argument if needed.
RIFE_EXE = os.environ.get(
    "RIFE_EXE",
    r"C:\Users\Loopy\Desktop\rife-ncnn-vulkan-20221029-windows\rife-ncnn-vulkan.exe",
)
RIFE_MODEL = os.environ.get("RIFE_MODEL", "rife-v4.6")  # or "rife-v4.26"


def _probe_fps(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "csv=p=0",
            str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    num, den = out.split("/")
    return float(num) / float(den)


def rife_interpolate(
    in_mp4: str | Path,
    out_mp4: str | Path,
    multiplier: int = 2,
    target_fps: float | None = None,
    rife_exe: str | None = None,
    model: str | None = None,
    crf: int = 18,
) -> Path:
    """Interpolate frames in `in_mp4`, write `out_mp4` with multiplied fps.

    Parameters
    ----------
    in_mp4 :
        Input video (e.g. Wan 2.2 chunk at 16 fps).
    out_mp4 :
        Output path.
    multiplier :
        Frame multiplier. 2 -> 16fps becomes 32fps. 3 -> 48fps. RIFE only
        supports integer multipliers natively.
    target_fps :
        Optional override for the output container fps. If None, uses
        input_fps * multiplier. If you want exactly 30 fps from a
        16-fps source, set multiplier=2 and target_fps=30 - ffmpeg
        will drop 2 of every 32 frames during re-encode.
    rife_exe, model :
        Override defaults if your install path or model name differ.
    crf :
        H.264 CRF for the re-encode. 18 is visually lossless, 23 is the
        ffmpeg default. Lower is bigger file, higher quality.

    Returns
    -------
    Path to the written `out_mp4`.
    """
    in_mp4 = Path(in_mp4).resolve()
    out_mp4 = Path(out_mp4).resolve()
    rife_exe = rife_exe or RIFE_EXE
    model = model or RIFE_MODEL

    if not in_mp4.exists():
        raise FileNotFoundError(in_mp4)
    if not Path(rife_exe).exists():
        raise FileNotFoundError(
            f"RIFE binary not found at {rife_exe}. Set RIFE_EXE env var or pass rife_exe=."
        )
    if multiplier < 2:
        raise ValueError("multiplier must be >= 2")

    in_fps = _probe_fps(in_mp4)
    out_fps = target_fps if target_fps is not None else in_fps * multiplier

    with tempfile.TemporaryDirectory(prefix="rife_") as tmpdir:
        tmp = Path(tmpdir)
        in_pngs = tmp / "in"
        out_pngs = tmp / "out"
        in_pngs.mkdir()
        out_pngs.mkdir()

        # 1. Extract input frames as PNGs
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(in_mp4),
                "-vsync", "0",
                str(in_pngs / "%08d.png"),
            ],
            check=True,
        )
        n_in = len(list(in_pngs.glob("*.png")))
        if n_in < 2:
            raise RuntimeError(f"ffmpeg extracted {n_in} frames from {in_mp4}")

        # 2. RIFE interpolate. Use -n (output frame count) for exact multiplier.
        n_out = (n_in - 1) * multiplier + 1
        subprocess.run(
            [
                rife_exe,
                "-i", str(in_pngs),
                "-o", str(out_pngs),
                "-m", model,
                "-n", str(n_out),
                "-f", "%08d.png",
                "-j", "1:2:2",
            ],
            check=True,
            cwd=str(Path(rife_exe).parent),
        )

        # 3. Encode back to MP4 at the requested output fps
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(out_fps),
                "-i", str(out_pngs / "%08d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", str(crf),
                "-preset", "medium",
                str(out_mp4),
            ],
            check=True,
        )

    return out_mp4


def rife_interpolate_inplace(
    mp4: str | Path,
    multiplier: int = 2,
    target_fps: float | None = None,
    **kwargs,
) -> Path:
    """Same as `rife_interpolate` but replaces the input file in place.

    Useful in pipelines that already have chunk paths committed - just
    upgrade each chunk before concat.
    """
    mp4 = Path(mp4).resolve()
    tmp_out = mp4.with_suffix(".rife.mp4")
    rife_interpolate(mp4, tmp_out, multiplier=multiplier, target_fps=target_fps, **kwargs)
    shutil.move(str(tmp_out), str(mp4))
    return mp4


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("-m", "--multiplier", type=int, default=2)
    p.add_argument("--fps", type=float, default=None,
                   help="Target output fps. Default = input_fps * multiplier.")
    p.add_argument("--model", default=None)
    p.add_argument("--exe", default=None)
    args = p.parse_args()

    rife_interpolate(
        args.input, args.output,
        multiplier=args.multiplier,
        target_fps=args.fps,
        rife_exe=args.exe,
        model=args.model,
    )
    print(f"OK -> {args.output}")
