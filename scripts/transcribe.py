#!/usr/bin/env python3
"""Local speech-to-text. No API key required. Two pluggable backends:

  --backend sensevoice  (default)  sherpa-onnx SenseVoice (ONNX, CPU, no torch).
                                   Stronger on Chinese / zh-en code-switch;
                                   ~13x faster than whisper-small on the same 15-min
                                   clip in our local benchmark (RTF 0.04 vs 0.52).
  --backend whisper                faster-whisper (CPU int8). Transformer ASR,
                                   self-segmenting, strong on English.
                                   segments via silero-VAD. NOTE: sherpa-onnx strips
                                   the model's emotion/event tags in post-processing
                                   (the token table still contains <|HAPPY|> etc., but
                                   there is no API flag to keep them -- verified on
                                   sherpa-onnx 1.13.4). For emotion, use --with-emotion
                                   (lazy funasr, reuses the decoded waveform) or run
                                   scripts/emotion.py separately.

Pipeline:
  input audio/video -> ffmpeg -> 16kHz mono WAV -> <backend> -> transcripts

Outputs (identical schema for both backends, under <workdir>/transcripts/):
  segments.json        : [{"start":float,"end":float,"text":str}, ...]
  transcript_full.txt  : plain full text
  transcript_timed.txt : "[mm:ss - mm:ss] text" per segment

Model locations are resolved from CLI flags -> env vars -> skill 'models/' ->
project cache. NO machine-specific absolute paths are hardcoded (portability rule).
See references/transcription.md for how to fetch each model with curl.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave

AUDIO_EXTS = {"mp3", "m4a", "wav", "ogg", "aac", "flac", "webm", "mp4", "mkv", "mov", "avi"}


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError("ffmpeg not found. Install `imageio-ffmpeg` (pip) or system ffmpeg.")


def to_wav(src, dst, ffmpeg):
    cmd = [ffmpeg, "-y", "-i", src, "-ar", "16000", "-ac", "1", "-f", "wav", dst]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_wav_f32(path):
    """Read a 16k mono PCM16 wav into a float32 numpy array in [-1, 1]."""
    import numpy as np
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype("float32") / 32768.0
    return data, sr


def fmt_ts(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


def _dir_candidates(arg_dir, env_name, skill_dir, subname):
    """Ordered directories to probe for a model folder named <subname>."""
    cands = []
    if arg_dir:
        cands.append(arg_dir)
    env = os.environ.get(env_name)
    if env:
        cands.append(env)
    # 模型放在 skill 根目录的 models/ 下（scripts/ 的上一级）
    cands.append(os.path.join(os.path.dirname(skill_dir), "models", subname))
    # a sibling project cache (e.g. <project>/_asr_models/<subname>)
    proj = os.path.abspath(os.path.join(skill_dir, "..", "..", "..", "..", "_asr_models", subname))
    cands.append(proj)
    cands.append(os.path.expanduser(f"~/.cache/audio-to-notes/{subname}"))
    return cands


# --------------------------------------------------------------------------- #
# backend: faster-whisper
# --------------------------------------------------------------------------- #
def _whisper_model_dir(size, arg_dir, skill_dir):
    for c in _dir_candidates(arg_dir, "A2N_MODEL_DIR", skill_dir, f"faster-whisper-{size}"):
        # arg/env may point directly at the model dir, or at a parent containing it
        for p in (c, os.path.join(c, f"faster-whisper-{size}")):
            if os.path.isfile(os.path.join(p, "model.bin")):
                return p
    # also reuse legacy project cache _whisper_models/
    legacy = os.path.abspath(os.path.join(skill_dir, "..", "..", "..", "..", "_whisper_models", f"faster-whisper-{size}"))
    if os.path.isfile(os.path.join(legacy, "model.bin")):
        return legacy
    try:
        from faster_whisper import download_model
        target = os.path.join(os.path.expanduser("~/.cache/audio-to-notes"), f"faster-whisper-{size}")
        download_model(size, target)
        return target
    except Exception as e:
        sys.stderr.write(
            f"\n[transcribe] 无法定位/下载 faster-whisper-{size}：{e}\n"
            f"[transcribe] 请按 references/transcription.md 用 curl 下载模型，或设置 A2N_MODEL_DIR。\n"
        )
        sys.exit(1)


def run_whisper(args, wav_path, skill_dir):
    from faster_whisper import WhisperModel
    model_dir = _whisper_model_dir(args.model, args.model_dir, skill_dir)
    print(f"[transcribe] backend=whisper 模型：{model_dir}", flush=True)
    model = WhisperModel(model_dir, device=args.device, compute_type=args.compute_type,
                         cpu_threads=max(1, os.cpu_count() or 4))
    print(f"[transcribe] 开始听译（language={args.language}）...", flush=True)
    segments = []
    try:
        seg_gen, _ = model.transcribe(wav_path, language=args.language, beam_size=5,
                                      vad_filter=True, condition_on_previous_text=True)
    except Exception:
        seg_gen, _ = model.transcribe(wav_path, language=args.language, beam_size=5,
                                      vad_filter=False, condition_on_previous_text=True)
    for i, seg in enumerate(seg_gen):
        segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()})
        if (i + 1) % 50 == 0:
            print(f"[transcribe] {i + 1} 段 ...", flush=True)
    return segments


# --------------------------------------------------------------------------- #
# backend: sherpa-onnx SenseVoice (+ silero VAD)
# --------------------------------------------------------------------------- #
def _find_file(arg_val, env_name, skill_dir, subname, filename):
    """Resolve a single model file path across candidate dirs."""
    if arg_val and os.path.isfile(arg_val):
        return arg_val
    for c in _dir_candidates(arg_val, env_name, skill_dir, subname):
        p = c if os.path.isfile(c) else os.path.join(c, filename)
        if os.path.isfile(p):
            return p
    return None


def run_sensevoice(args, wav_path, skill_dir):
    try:
        import sherpa_onnx  # noqa
        import numpy as np
    except Exception as e:
        sys.stderr.write(
            "\n[transcribe] sherpa-onnx 未安装：%s\n"
            "[transcribe] 启用 SenseVoice 后端：pip install sherpa-onnx\n"
            "[transcribe] 并按 references/transcription.md 下载 SenseVoice 与 silero-vad 模型。\n" % e
        )
        sys.exit(2)

    sv_model = _find_file(args.sensevoice_model, "A2N_SENSEVOICE_DIR", skill_dir,
                          "sherpa-onnx-sense-voice", "model.int8.onnx")
    sv_tokens = _find_file(args.sensevoice_tokens, "A2N_SENSEVOICE_DIR", skill_dir,
                           "sherpa-onnx-sense-voice", "tokens.txt")
    vad_model = _find_file(args.vad_model, "A2N_VAD_MODEL", skill_dir,
                           "silero-vad", "silero_vad.onnx")
    missing = [n for n, v in [("SenseVoice model.int8.onnx", sv_model),
                              ("SenseVoice tokens.txt", sv_tokens),
                              ("silero_vad.onnx", vad_model)] if not v]
    if missing:
        sys.stderr.write(
            "\n[transcribe] 缺少 SenseVoice/VAD 模型文件：%s\n"
            "[transcribe] 下载方法见 references/transcription.md『SenseVoice 后端』；"
            "或用 --sensevoice-model/--sensevoice-tokens/--vad-model 显式指定，"
            "或设 A2N_SENSEVOICE_DIR / A2N_VAD_MODEL。\n" % ", ".join(missing)
        )
        sys.exit(2)

    print(f"[transcribe] backend=sensevoice\n  asr={sv_model}\n  vad={vad_model}", flush=True)
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=sv_model, tokens=sv_tokens, use_itn=True,
        num_threads=max(1, os.cpu_count() or 4),
    )

    audio, sr = read_wav_f32(wav_path)
    if sr != 16000:
        sys.stderr.write(f"[transcribe] 期望 16k，实际 {sr}Hz；请确认 ffmpeg 转码。\n")

    vad_cfg = sherpa_onnx.VadModelConfig()
    vad_cfg.silero_vad.model = vad_model
    vad_cfg.silero_vad.min_silence_duration = 0.25
    vad_cfg.silero_vad.min_speech_duration = 0.25
    vad_cfg.sample_rate = 16000
    vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=100)

    segments = []
    window = 512  # silero frame
    i = 0
    n = len(audio)
    count = 0

    def drain():
        nonlocal count
        while not vad.empty():
            seg = vad.front
            start = seg.start / 16000.0
            samples = seg.samples
            end = start + len(samples) / 16000.0
            st = recognizer.create_stream()
            st.accept_waveform(16000, samples)
            recognizer.decode_stream(st)
            text = st.result.text.strip()
            if text:
                segments.append({"start": round(start, 2), "end": round(end, 2), "text": text})
                count += 1
                if count % 50 == 0:
                    print(f"[transcribe] {count} 段 ...", flush=True)
            vad.pop()

    while i < n:
        chunk = audio[i:i + window]
        vad.accept_waveform(chunk)
        i += window
        drain()
    vad.flush()
    drain()
    segments.sort(key=lambda s: s["start"])
    return segments


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Local speech-to-text (pluggable backend)")
    ap.add_argument("--input", required=True, help="audio/video file path")
    ap.add_argument("--workdir", required=True, help="transcripts go to <workdir>/transcripts/")
    ap.add_argument("--backend", default="sensevoice", choices=["whisper", "sensevoice"],
                    help="sensevoice (default, ~13x faster on zh) | whisper (faster-whisper small)")
    ap.add_argument("--model", default="small", help="faster-whisper size (whisper backend only)")
    ap.add_argument("--language", default="zh", help="ISO code, e.g. zh / en (whisper backend)")
    ap.add_argument("--model-dir", default=None, help="explicit faster-whisper model dir")
    ap.add_argument("--sensevoice-model", default=None, help="SenseVoice model.int8.onnx path (or its dir)")
    ap.add_argument("--sensevoice-tokens", default=None, help="SenseVoice tokens.txt path (or its dir)")
    ap.add_argument("--vad-model", default=None, help="silero_vad.onnx path (sensevoice backend)")
    ap.add_argument("--device", default="cpu", help="whisper: cpu (default) / cuda")
    ap.add_argument("--compute-type", default="int8", help="whisper: int8 (default) / float16 / float32")
    ap.add_argument("--force", action="store_true", help="re-transcribe even if segments.json exists")
    ap.add_argument("--with-emotion", action="store_true",
                    help="(optional) 转写同时复用已解码波形跑 emotion2vec+ 打情绪标签。"
                         "惰性 import funasr，默认路径仍零 torch。")
    ap.add_argument("--emotion-model", default="iic/emotion2vec_plus_large",
                    help="--with-emotion 时使用的 emotion2vec+ 模型 id（默认 _large ~1.94GB；可选 _base ~1.12GB / _seed）。"
                         "运行时间相近，large 情绪分布更合理（见 compare_emotion.py）。")
    args = ap.parse_args()

    skill_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(args.workdir, "transcripts")
    os.makedirs(out_dir, exist_ok=True)
    seg_path = os.path.join(out_dir, "segments.json")

    wav_path = None
    if os.path.isfile(seg_path) and not args.force:
        print(f"[transcribe] 复用已有 {seg_path}（--force 可强制重跑）", flush=True)
        segments = json.load(open(seg_path, encoding="utf-8"))
    else:
        ffmpeg = find_ffmpeg()
        wav_path = os.path.join(tempfile.gettempdir(), "a2n_input.wav")
        print("[transcribe] 转码为 16kHz 单声道 WAV ...", flush=True)
        to_wav(args.input, wav_path, ffmpeg)

        if args.backend == "sensevoice":
            segments = run_sensevoice(args, wav_path, skill_dir)
        else:
            segments = run_whisper(args, wav_path, skill_dir)

        json.dump(segments, open(seg_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[transcribe] 完成 {len(segments)} 段 -> {seg_path}", flush=True)

    full = "\n".join(s["text"] for s in segments)
    with open(os.path.join(out_dir, "transcript_full.txt"), "w", encoding="utf-8") as f:
        f.write(full)
    with open(os.path.join(out_dir, "transcript_timed.txt"), "w", encoding="utf-8") as f:
        for s in segments:
            f.write(f"[{fmt_ts(s['start'])} - {fmt_ts(s['end'])}] {s['text']}\n")

    # ---- optional: inline emotion recognition (approach 1) ----
    # 复用转写时已解码的 16k 波形，逐段跑 emotion2vec+，零二次 ffmpeg 解码。
    # funasr 惰性 import；未安装则优雅跳过、保留转写结果（不致命）。
    if args.with_emotion:
        try:
            sys.path.insert(0, skill_dir)
            from emotion import load_emotion_model, classify_emotions
            wav_for_emo = wav_path
            if not wav_for_emo or not os.path.isfile(wav_for_emo):
                wav_for_emo = os.path.join(tempfile.gettempdir(), "a2n_input.wav")
                to_wav(args.input, wav_for_emo, find_ffmpeg())
            audio_emo, sr_emo = read_wav_f32(wav_for_emo)
            model_emo = load_emotion_model(args.emotion_model)
            n_cls = classify_emotions(segments, audio_emo, sr_emo, model_emo)
            json.dump(segments, open(os.path.join(out_dir, "segments_emotion.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            with open(os.path.join(out_dir, "transcript_timed_emotion.txt"), "w", encoding="utf-8") as f:
                for s in segments:
                    f.write(f"[{s.get('emotion', 'unknown')}] [{fmt_ts(s['start'])} - {fmt_ts(s['end'])}] {s['text']}\n")
            print(f"[transcribe] 情绪标注完成（{n_cls}/{len(segments)} 段有效）"
                  f" -> {out_dir}/segments_emotion.json", flush=True)
        except ImportError as e:
            sys.stderr.write(
                f"\n[transcribe] funasr 未安装，跳过情绪标注：{e}\n"
                f"[transcribe] 启用：pip install -U funasr modelscope\n")
        except Exception as e:
            sys.stderr.write(f"\n[transcribe] 情绪标注失败（已保留转写结果）：{e}\n")

    total = segments[-1]["end"] if segments else 0
    print(f"[transcribe] 全文约 {len(full)} 字，时长约 {fmt_ts(total)}。"
          f" 产物：{out_dir}/{{segments.json, transcript_full.txt, transcript_timed.txt}}", flush=True)


if __name__ == "__main__":
    main()
