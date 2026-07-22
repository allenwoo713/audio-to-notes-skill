#!/usr/bin/env python3
"""OPTIONAL speech emotion recognition (SER) as an independent, pluggable module.

Why a separate module?  The sherpa-onnx SenseVoice ONNX export produces text-only
output: its emotion/event tags are STRIPPED by sherpa-onnx's post-processing and
there is no API flag to keep them (verified on sherpa-onnx 1.13.4 -- the model
*token* table still contains <|HAPPY|> <|ANGRY|> ... but the recognizer returns
plain text). So emotion is a distinct acoustic task, layered on top of any ASR
backend. This keeps the heavy dependency (FunASR + torch) OUT of the default
transcription path -- it is only imported when the user asks for emotion.

Engine:
  emotion2vec+  (Alibaba, via FunASR AutoModel).  9-class SER, robust across zh/en.
  Install (opt-in, ~torch heavy):  pip install -U funasr modelscope
  Model auto-downloads on first use: iic/emotion2vec_plus_large (default, ~1.94GB) or _base (~1.12GB) / _seed.
  Run time is similar across sizes; large gives a more reasonable emotion distribution
  (base is more conservative and shows angry noise on very short segments -- see compare_emotion.py).

This file also exposes load_emotion_model() / classify_emotions() so that
transcribe.py can run emotion inline (its --with-emotion flag) and reuse the
already-decoded waveform -- no second ffmpeg decode, one process end to end.

Input : segments.json (from transcribe.py) + the original audio (or a decoded
        float32 array, when called from transcribe.py).
Method: cut each segment's audio (in-memory, 16k mono) -> classify -> attach label.
Output (under the segments.json's dir, unless --out given):
  segments_emotion.json       : segments with added {"emotion": "...", "emo_score": float}
  transcript_timed_emotion.txt : "[emotion] [mm:ss - mm:ss] text"

If funasr is unavailable, exits gracefully (code 2) so the skill can either skip
the emotion dimension or fall back to LLM-based inference from text (see SKILL.md).

Usage (standalone):
  python emotion.py --segments <transcripts>/segments.json --audio input.m4a \
      [--model iic/emotion2vec_plus_large] [--out <dir or file>] [--topk 1]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave


def fmt(sec):
    m, s = int(sec // 60), int(sec % 60)
    return f"{m:02d}:{s:02d}"


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


def to_wav16k(src, dst):
    cmd = [find_ffmpeg(), "-y", "-i", src, "-ar", "16000", "-ac", "1", "-f", "wav", dst]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_wav_f32(path):
    import numpy as np
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype("float32") / 32768.0, sr


def _clean_label(lbl):
    """emotion2vec+ labels look like '生气/angry' or 'happy'. Keep a compact form."""
    if not lbl:
        return "unknown"
    return lbl.strip()


def load_emotion_model(model_id="iic/emotion2vec_plus_large"):
    """Lazy-import funasr + load emotion2vec+. Raises if funasr is missing.

    Model cache is redirected to this skill's own models/emotion2vec dir
    (overridable via A2N_EMOTION_CACHE) instead of inheriting a global
    MODELSCOPE_CACHE set by other tools (e.g. mineru). This keeps the
    emotion model self-contained next to the SenseVoice/silero weights.
    """
    import os
    cache = os.environ.get("A2N_EMOTION_CACHE")
    if not cache:
        here = os.path.dirname(os.path.abspath(__file__))
        skill_dir = os.path.dirname(here)
        cache = os.path.join(skill_dir, "models", "emotion2vec")
    os.environ["MODELSCOPE_CACHE"] = cache
    os.environ["MODELSCOPE_HOME"] = cache
    from funasr import AutoModel
    import numpy as np  # noqa: F401  (kept for callers that slice arrays)
    os.makedirs(cache, exist_ok=True)
    print(f"[emotion] 加载 emotion2vec+：{model_id}（首次会自动下载到 {cache}）...", flush=True)
    return AutoModel(model=model_id)


def classify_emotions(segments, audio, sr, model, min_dur=0.30, topk=1):
    """Attach {'emotion':..., 'emo_score':...} to each segment in place.

    audio : float32 numpy array at sample rate sr (16k expected).
    Returns the number of segments actually classified.
    """
    import numpy as np
    n = len(segments)
    done = 0
    print(f"[emotion] 逐段情绪识别（{n} 段）...", flush=True)
    for i, s in enumerate(segments):
        dur = s["end"] - s["start"]
        if dur < min_dur:
            s["emotion"], s["emo_score"] = "unknown", 0.0
            continue
        a = int(s["start"] * sr)
        b = int(s["end"] * sr)
        if b <= a:
            s["emotion"], s["emo_score"] = "unknown", 0.0
            continue
        clip = audio[a:b]
        try:
            res = model.generate(clip, granularity="utterance",
                                 extract_embedding=False, disable_pbar=True)
            r0 = res[0] if isinstance(res, list) else res
            labels = r0.get("labels", [])
            scores = r0.get("scores", [])
            if labels and scores:
                order = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
                top = order[:max(1, topk)]
                s["emotion"] = "/".join(_clean_label(labels[k]) for k in top)
                s["emo_score"] = round(float(scores[order[0]]), 4)
                done += 1
            else:
                s["emotion"], s["emo_score"] = "unknown", 0.0
        except Exception as e:
            s["emotion"], s["emo_score"] = "unknown", 0.0
            sys.stderr.write(f"[emotion] 段 {i} 识别失败：{e}\n")
        if (i + 1) % 50 == 0:
            print(f"[emotion] {i + 1}/{n} ...", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser(description="Optional SER via emotion2vec+ (FunASR)")
    ap.add_argument("--segments", required=True, help="transcribe.py 产出的 segments.json")
    ap.add_argument("--audio", required=True, help="原始音频文件")
    ap.add_argument("--model", default="iic/emotion2vec_plus_large",
                    help="emotion2vec+ 模型 id：_large(默认,~1.94GB)/_base(~1.12GB)/_seed")
    ap.add_argument("--out", default=None, help="输出目录或文件（默认与 segments.json 同目录）")
    ap.add_argument("--topk", type=int, default=1, help="每段保留前 k 个情绪标签")
    ap.add_argument("--min-dur", type=float, default=0.30,
                    help="小于该秒数的段跳过情绪识别（太短不可靠）")
    args = ap.parse_args()

    try:
        from funasr import AutoModel  # noqa
        import numpy as np  # noqa
    except Exception as e:
        sys.stderr.write(
            "\n[emotion] funasr 未安装：%s\n"
            "[emotion] 启用可选情绪维度：pip install -U funasr modelscope\n"
            "[emotion] （较重，含 torch；不需要情绪时无需安装。也可改用 LLM 从文本推断情绪。）\n" % e
        )
        sys.exit(2)

    segs = json.load(open(args.segments, encoding="utf-8"))
    out_dir = os.path.dirname(os.path.abspath(args.segments))
    if args.out and os.path.isdir(args.out):
        out_dir = args.out
    emo_json = (args.out if (args.out and not os.path.isdir(args.out))
                else os.path.join(out_dir, "segments_emotion.json"))
    emo_txt = os.path.join(out_dir, "transcript_timed_emotion.txt")

    model = load_emotion_model(args.model)

    # 任务隔离的临时目录：进程唯一，退出（含异常）自动清理，避免并发互相覆盖。
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "emo.wav")
        to_wav16k(args.audio, wav)
        audio, sr = read_wav_f32(wav)

        classify_emotions(segs, audio, sr, model, min_dur=args.min_dur, topk=args.topk)

        json.dump(segs, open(emo_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        with open(emo_txt, "w", encoding="utf-8") as f:
            for s in segs:
                f.write(f"[{s.get('emotion','unknown')}] [{fmt(s['start'])} - {fmt(s['end'])}] {s['text']}\n")

        print(f"[emotion] 完成 -> {emo_json}\n[emotion] 人读版 -> {emo_txt}", flush=True)


if __name__ == "__main__":
    main()
