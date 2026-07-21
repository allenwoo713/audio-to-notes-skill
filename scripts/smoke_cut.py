#!/usr/bin/env python3
"""Cut a long recording down to a short smoke-test subset.

Why: speaker diarization / SER on a 90-minute audio can take ~1h. For fast
iteration we slice the first N seconds of the ORIGINAL audio AND filter the
corresponding segments.json so the two stay aligned (a segment whose `end`
exceeds the cut is clamped; segments starting after the cut are dropped).

This avoids the classic trap of cutting only the audio while leaving
segments.json covering the full length, which makes diarize.py misalign
segments against a shorter waveform.

Usage:
  python smoke_cut.py --src <original.mp3> --segments <full/segments.json> \
      --out-dir <smoke/> [--duration 900]

Outputs (under --out-dir):
  recording_cut.wav     : first N seconds, 16kHz mono (matches pipeline input)
  segments_ref.json     : filtered segments (reference for diffing the re-transcribed subset)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys


def find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError("ffmpeg not found. Install imageio-ffmpeg or system ffmpeg.")


def cut_audio(src, dst, duration, ffmpeg):
    # -ss before -i = fast seek; -t = duration of the kept slice.
    cmd = [ffmpeg, "-y", "-ss", "0", "-t", str(duration), "-i", src,
           "-ar", "16000", "-ac", "1", "-f", "wav", dst]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def filter_segments(segments, duration):
    out = []
    for s in segments:
        start, end = s.get("start", 0.0), s.get("end", 0.0)
        if start >= duration:
            continue  # entirely after the cut
        if end > duration:
            end = duration  # clamp a segment that straddles the boundary
        if end - start < 0.05:
            continue  # degenerate after clamping
        out.append({"start": round(start, 2), "end": round(end, 2), "text": s.get("text", "")})
    out.sort(key=lambda x: x["start"])
    return out


def main():
    ap = argparse.ArgumentParser(description="Slice a recording to a short smoke-test subset")
    ap.add_argument("--src", required=True, help="original audio/video file")
    ap.add_argument("--segments", required=True, help="full-length segments.json (from transcribe.py)")
    ap.add_argument("--out-dir", required=True, help="where to write recording_cut.wav + segments_ref.json")
    ap.add_argument("--duration", type=float, default=900.0, help="seconds to keep from the start (default 900 = 15 min)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ffmpeg = find_ffmpeg()

    print(f"[smoke_cut] 切割音频前 {args.duration:g}s：{os.path.basename(args.src)}", flush=True)
    wav_dst = os.path.join(args.out_dir, "recording_cut.wav")
    cut_audio(args.src, wav_dst, args.duration, ffmpeg)
    print(f"[smoke_cut] 音频 -> {wav_dst}", flush=True)

    segs = json.load(open(args.segments, encoding="utf-8"))
    kept = filter_segments(segs, args.duration)
    ref_path = os.path.join(args.out_dir, "segments_ref.json")
    json.dump(kept, open(ref_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[smoke_cut] 段过滤：{len(segs)} -> {len(kept)} 段 -> {ref_path}", flush=True)
    print(f"[smoke_cut] 完成。后续全链路在该子集上跑，分轨可从 ~1h 降到 ~1-2 分钟。", flush=True)


if __name__ == "__main__":
    main()
