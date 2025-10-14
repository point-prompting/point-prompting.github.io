#!/usr/bin/env python3
"""
Create a 3x5 grid video from 15 source videos with per-video cropping, frame-wrapping,
resizing to square, speed matching to a common length, and black borders.

And then 
ffmpeg -f concat -safe 0 -i <(printf "file '%s'\nfile '%s'\n" "$(realpath input.mp4)" "$(realpath input.mp4)") -c copy output_concat.mp4

USAGE
-----
python grid_video_builder.py \
  --config config.json \
  --output out.mp4 \
  --square-size 512 \
  --common-length 8 \
  --fps 30 \
  --inner-border 4 \
  --outer-border 12

CONFIG FILE FORMAT (JSON)
-------------------------
{
  "videos": [
    {
      "path": "video1.mp4",
      "center": [640, 360],                 # (x, y) in source pixel coords
      "crop_size": [600, 600],              # [width, height]
      "start_percent": 0.33                 # 0..1, 0=start, 1=last frame
    },
    { "path": "video2.mp4", "center": [500, 500], "crop_size": [700, 700], "start_percent": 0.0 },
    ... (13 more entries) ...
  ]
}

NOTES
-----
- All times are derived from each clip's native FPS; start_percent is mapped to time as start_t = min(start_percent * duration, duration - 1/fps).
- If crop_size extends beyond the frame, MoviePy will clamp the crop.
- If a per-clip parameter is omitted, sensible defaults are used (center = frame center; crop_size = centered square using min dimension; start_percent = 0).
- Audio is disabled by default; enable if needed by removing audio=False in write_videofile.
"""

import argparse
import json
import os
import platform
import subprocess
from typing import List, Tuple

# --- Pillow ≥10 compatibility shim (MoviePy expects Image.ANTIALIAS) ---
from PIL import Image as _PIL_Image
try:
    _Resampling = _PIL_Image.Resampling  # Pillow ≥10
    if not hasattr(_PIL_Image, "ANTIALIAS"):
        _PIL_Image.ANTIALIAS = _Resampling.LANCZOS
    if not hasattr(_PIL_Image, "BICUBIC"):
        _PIL_Image.BICUBIC = _Resampling.BICUBIC
    if not hasattr(_PIL_Image, "BILINEAR"):
        _PIL_Image.BILINEAR = _Resampling.BILINEAR
except Exception:
    # Older Pillow: attributes already exist
    pass

from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclips,
    vfx,
    clips_array,
)


def wrap_clip_by_start_percent(clip: VideoFileClip, start_percent: float) -> VideoFileClip:
    """Rotate the clip so that it starts at a percentage into the clip (0..1).
    0 means first frame; 1 maps to the last frame time (duration - 1/fps).
    """
    try:
        p = float(start_percent)
    except Exception:
        p = 0.0
    if p <= 0.0:
        return clip
    p = max(0.0, min(p, 1.0))
    fps = clip.fps or 30
    duration = float(clip.duration or 0.0)
    if duration <= 0.0:
        return clip
    last_frame_t = max(0.0, duration - 1.0 / float(fps))
    start_t = min(p * duration, last_frame_t)
    if start_t <= 0.0:
        return clip
    head = clip.subclip(start_t)
    tail = clip.subclip(0, start_t)
    return concatenate_videoclips([head, tail])


def safe_crop_to_square(
    clip: VideoFileClip,
    center: Tuple[float, float] | None,
    crop_size: Tuple[int, int] | None,
    square_size: int,
) -> VideoFileClip:
    """Crop around a provided center and size; then resize to a square of `square_size`.
    If center/crop_size are None, choose a centered square crop.
    """
    w, h = clip.size

    if center is None:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = center

    if crop_size is None:
        side = min(w, h)
        cw, ch = side, side
    else:
        cw, ch = crop_size

    # Enforce positive crop sizes and clamp to at least 2px
    cw = max(2, int(cw))
    ch = max(2, int(ch))

    cropped = clip.fx(vfx.crop, x_center=cx, y_center=cy, width=cw, height=ch)
    return cropped.resize(newsize=(square_size, square_size))


def match_duration_with_speed(clip: VideoFileClip, target_len: float) -> VideoFileClip:
    """Speed up or slow down the clip so its duration equals `target_len`.
    Uses vfx.speedx where factor >1 speeds up and <1 slows down.
    """
    if target_len <= 0:
        return clip
    cur = max(1e-6, clip.duration)
    factor = cur / float(target_len)
    sped = clip.fx(vfx.speedx, factor=factor)
    # Guard for tiny numeric drift
    if abs(sped.duration - target_len) > 1e-3:
        if sped.duration > target_len:
            sped = sped.subclip(0, target_len)
        else:
            # Loop to fill the gap (seamless for looped playback)
            sped = sped.fx(vfx.loop, duration=target_len)
    return sped.set_duration(target_len)


def add_margin(clip: VideoFileClip, mar: int, color=(0, 0, 0)) -> VideoFileClip:
    """Add a uniform margin around the clip."""
    if mar <= 0:
        return clip
    return clip.margin(mar=mar, color=color)


def build_grid(clips: List[VideoFileClip], rows: int, cols: int) -> VideoFileClip:
    assert len(clips) == rows * cols, f"Expected {rows*cols} clips, got {len(clips)}"
    grid = []
    idx = 0
    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append(clips[idx])
            idx += 1
        grid.append(row)
    return clips_array(grid)


def process_videos(
    entries: List[dict],
    square_size: int,
    common_length: float,
    fps: int,
    inner_border: int,
    outer_border: int,
) -> VideoFileClip:
    assert len(entries) == 15, f"Config must contain exactly 15 videos, got {len(entries)}"

    processed: List[VideoFileClip] = []
    for i, e in enumerate(entries):
        path = e.get("path")
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Video {i+1}: '{path}' not found")
        clip = VideoFileClip(path, audio=False)

        # 1) wrap by start percent (0..1)
        start_percent = float(e.get("start_percent", 0.0) or 0.0)
        clip = wrap_clip_by_start_percent(clip, start_percent)

        # 2) crop around center/size and resize to square
        center = e.get("center")
        crop_size = e.get("crop_size")
        if center is not None:
            cx, cy = float(center[0]), float(center[1])
            center = (cx, cy)
        if crop_size is not None:
            cw, ch = int(crop_size[0]), int(crop_size[1])
            crop_size = (cw, ch)
        clip = safe_crop_to_square(clip, center=center, crop_size=crop_size, square_size=square_size)

        # 3) speed to match duration
        clip = match_duration_with_speed(clip, common_length)

        # 4) per-tile border
        clip = add_margin(clip, inner_border, color=(0, 0, 0))

        # Standardize FPS to avoid A/V writer warnings
        clip = clip.set_fps(fps)
        processed.append(clip)

    # 5) build 3x5 grid
    grid = build_grid(processed, rows=3, cols=5)

    # 6) outer border around the whole grid
    grid = add_margin(grid, outer_border, color=(0, 0, 0))

    # Ensure exact duration/FPS
    grid = grid.set_duration(common_length).set_fps(fps)
    return grid
def write_video_with_ffmpeg_pipe(
    clip: VideoFileClip,
    filename: str,
    fps: int,
    codec: str,
    threads: int,
    preset: str | None,
):
    """Write video using a direct ffmpeg pipe to avoid MoviePy's preset behavior.

    This streams raw RGB frames to ffmpeg via stdin and lets us control flags precisely.
    """
    w, h = clip.size
    ffmpeg_bin = os.environ.get("FFMPEG_BINARY", "ffmpeg")

    cmd: List[str] = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel", "warning",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(int(fps)),
        "-i", "-",
        "-an",
        "-vcodec", codec,
    ]

    if threads and threads > 0:
        cmd += ["-threads", str(int(threads))]

    # Encoder-specific options
    if codec == "libx264":
        # Constant rate factor for quality and yuv420p output
        cmd += ["-pix_fmt", "yuv420p", "-crf", "18"]
        if preset and preset.lower() != "none":
            cmd += ["-preset", preset]
        cmd += ["-movflags", "+faststart"]
    elif codec == "h264_videotoolbox":
        # Hardware encoder prefers a bitrate; ensure yuv420p output
        cmd += [
            "-b:v", "8M",
            "-maxrate", "8M",
            "-bufsize", "16M",
            "-pix_fmt", "yuv420p",
        ]
    elif codec == "mpeg4":
        cmd += ["-q:v", "3", "-pix_fmt", "yuv420p"]
    else:
        # Default to yuv420p if unknown
        cmd += ["-pix_fmt", "yuv420p"]

    cmd += [filename]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        for frame in clip.iter_frames(fps=fps, dtype="uint8"):
            if proc.poll() is not None:
                break
            assert proc.stdin is not None
            try:
                proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                break
    finally:
        if proc.stdin:
            try:
                proc.stdin.close()
            except Exception:
                pass

    stderr = b""
    try:
        if proc.stderr:
            stderr = proc.stderr.read()
    except Exception:
        pass
    retcode = proc.wait()
    if retcode != 0:
        cmd_str = " ".join(cmd)
        raise RuntimeError(
            "ffmpeg failed with code "
            + str(retcode)
            + "\nCommand: "
            + cmd_str
            + "\nStderr:\n"
            + stderr.decode("utf-8", errors="ignore")
        )



def parse_args():
    p = argparse.ArgumentParser(description="Create a 2x5 grid video from 10 inputs with crop/wrap/speed/borders.")
    p.add_argument("--config", required=True, help="Path to JSON config file (see header for schema)")
    p.add_argument("--output", required=True, help="Output video path, e.g. out.mp4")
    p.add_argument("--square-size", type=int, default=512, help="Per-tile square resolution in pixels (default 512)")
    p.add_argument("--common-length", type=float, default=8.0, help="Final video duration in seconds (default 8.0)")
    p.add_argument("--fps", type=int, default=30, help="Output FPS (default 30)")
    p.add_argument("--inner-border", type=int, default=4, help="Black border (px) around each tile (default 4)")
    p.add_argument("--outer-border", type=int, default=12, help="Black border (px) around the whole grid (default 12)")
    p.add_argument("--codec", default="libx264", help="FFmpeg video codec (e.g., libx264, h264_videotoolbox, mpeg4)")
    p.add_argument(
        "--preset",
        default="",
        help="x264 preset; set to 'none' or leave empty to omit (e.g., ultrafast, veryfast, medium)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    entries = cfg.get("videos", [])

    final = process_videos(
        entries=entries,
        square_size=args.square_size,
        common_length=args.common_length,
        fps=args.fps,
        inner_border=args.inner_border,
        outer_border=args.outer_border,
    )

    # Choose codec/params; change as needed for your platform.
    def ffmpeg_has_encoder(name: str) -> bool:
        try:
            out = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            ).stdout
            return (name in out)
        except Exception:
            return False

    # Determine requested codec and preset intent
    requested_codec = (args.codec or "libx264")
    requested_preset = args.preset or ""

    # Select a safe codec based on availability and platform.
    # On macOS, prefer hardware encoder to avoid x264 preset handling issues.
    chosen_codec = requested_codec
    is_macos = platform.system().lower() == "darwin"
    if requested_codec == "libx264":
        if is_macos and ffmpeg_has_encoder("h264_videotoolbox"):
            chosen_codec = "h264_videotoolbox"
        elif not ffmpeg_has_encoder("libx264"):
            chosen_codec = "mpeg4"

    # Only pass x264 preset when actually using libx264; otherwise force None to override MoviePy default
    should_pass_preset = (chosen_codec == "libx264") and bool(requested_preset) and (requested_preset.lower() != "none")

    # Build kwargs to avoid accidentally re-introducing default preset
    write_kwargs = dict(
        codec=chosen_codec,
        audio=False,
        fps=args.fps,
        threads=(os.cpu_count() or 4),
        remove_temp=True,
        verbose=True,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    if should_pass_preset:
        write_kwargs["preset"] = requested_preset

    print(f"Using codec={chosen_codec} preset={'<omitted>' if not should_pass_preset else requested_preset}")

    # Use direct ffmpeg pipe to fully control flags and avoid unwanted -preset
    write_preset = requested_preset if should_pass_preset else None
    write_video_with_ffmpeg_pipe(
        clip=final,
        filename=args.output,
        fps=args.fps,
        codec=chosen_codec,
        threads=(os.cpu_count() or 4),
        preset=write_preset,
    )


if __name__ == "__main__":
    main()
