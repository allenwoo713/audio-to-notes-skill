#!/usr/bin/env python3
"""Speaker diarization -> merge speaker turns into transcribe.py segments.

Two engines:
  --engine sherpa   (default)  sherpa-onnx OfflineSpeakerDiarization.
                               pyannote segmentation (ONNX) + 3D-Speaker embedding.
                               NO torch, NO HuggingFace token. CPU friendly.
  --engine pyannote            original pyannote.audio path (needs torch + HF_TOKEN).
                               Kept as a fallback for parity / comparison.

Both read segments.json and emit a speaker-tagged transcript:
  [SPEAKER_00] [mm:ss - mm:ss] text

If the chosen engine's package/models are unavailable, exits gracefully (code 2/3)
so the skill can fall back to LLM-based attribution (see SKILL.md step 4).

Usage (sherpa, default):
  python diarize.py --segments segments.json --audio input.m4a \
      --out transcript_timed_speaker.txt [--num-speakers N]
      [--seg-model .../sherpa-onnx-pyannote-segmentation-3-0/model.onnx]
      [--emb-model .../3dspeaker_..._16k.onnx]

Usage (pyannote fallback):
  python diarize.py --engine pyannote --segments ... --audio ... --out ... [--token $HF_TOKEN]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave

# 自动用满 CPU 核数跑分轨。
# sherpa-onnx 的分轨重计算（segmentation + 3D-Speaker 嵌入）走 OpenMP 并行，
# 若不显式设置，默认往往只起少数线程，长音频（>30min）分轨会慢数倍乃至十倍。
# 仅在用户未显式设置 OMP_NUM_THREADS 时生效，便于手动覆盖。
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)
# ACTIVE 策略让 OpenMP 线程保持忙等，减少长批量任务的调度延迟（纯计算场景收益明显）。
if not os.environ.get("OMP_WAIT_POLICY"):
    os.environ["OMP_WAIT_POLICY"] = "ACTIVE"


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


def _resolve(arg_val, env_name, filenames, extra_dirs=None):
    """Resolve a model file: explicit arg -> env dir -> extra_dirs -> None.

    filenames   = candidate basenames (e.g. ["model.onnx", "model.int8.onnx"]).
    extra_dirs  = fallback dirs to probe (e.g. skill's models/diar dir).
    """
    if arg_val and os.path.isfile(arg_val):
        return arg_val
    search = list(extra_dirs or [])
    base = os.environ.get(env_name)
    if base:
        search.insert(0, base)
    if arg_val and os.path.isdir(arg_val):
        search.append(arg_val)
    for d in search:
        if os.path.isfile(d):
            return d
        for fn in filenames:
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                return p
    return None


# --------------------------------------------------------------------------- #
# engine: sherpa-onnx
# --------------------------------------------------------------------------- #
def diarize_sherpa(args):
    try:
        import sherpa_onnx  # noqa
    except Exception as e:
        sys.stderr.write(
            "\n[diarize] sherpa-onnx 未安装：%s\n"
            "[diarize] 启用：pip install sherpa-onnx；模型下载见 references/transcription.md『说话人分轨(sherpa)』。\n" % e
        )
        sys.exit(2)

    skill_dir = os.path.dirname(os.path.abspath(__file__))
    # 模型放在 skill 根目录的 models/ 下（scripts/ 的上一级）
    diar_model_dir = os.path.join(os.path.dirname(skill_dir), "models", "diar")
    seg_model = _resolve(args.seg_model, "A2N_DIAR_SEG_MODEL",
                         ["model.onnx", "model.int8.onnx"], extra_dirs=[diar_model_dir])
    emb_model = _resolve(args.emb_model, "A2N_DIAR_EMB_MODEL",
                         ["3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx", "model.onnx"],
                         extra_dirs=[diar_model_dir])
    missing = [n for n, v in [("segmentation model", seg_model), ("embedding model", emb_model)] if not v]
    if missing:
        sys.stderr.write(
            "\n[diarize] 缺少分轨模型：%s\n"
            "[diarize] 用 --seg-model/--emb-model 指定，或设 A2N_DIAR_SEG_MODEL / A2N_DIAR_EMB_MODEL。\n"
            "[diarize] 下载见 references/transcription.md。\n" % ", ".join(missing)
        )
        sys.exit(2)

    # num_threads 是分轨性能的关键：sherpa 据此对 ONNX session 设 intra_op_num_threads。
    # 若不显式传入，sherpa 默认按 1（单线程）跑，CPU 大核几乎闲置 → 长音频分轨极慢。
    # 旧版仅设 OMP_NUM_THREADS 环境变量无效（sherpa 不读该变量），此处必须落到 config。
    nt = args.num_threads or int(os.environ.get("A2N_DIAR_THREADS", os.cpu_count() or 4))
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg_model),
            num_threads=nt,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb_model, num_threads=nt),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=args.num_speakers, threshold=args.cluster_threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    sys.stderr.write("[diarize] sherpa 引擎线程数 num_threads=%d\n" % nt)
    if not cfg.validate():
        sys.stderr.write("[diarize] 配置校验失败：请检查模型文件是否存在且匹配。\n")
        sys.exit(2)
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)

    # 任务隔离的临时目录：进程唯一，退出（含异常）自动清理，避免并发互相覆盖。
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "dia.wav")
        to_wav16k(args.audio, wav)
        audio, sr = read_wav_f32(wav)
        if sr != sd.sample_rate:
            sys.stderr.write(f"[diarize] 采样率不匹配：期望 {sd.sample_rate}，实际 {sr}。\n")

        print("[diarize] sherpa-onnx 分轨中（与音频长度成正比）...", flush=True)

        def cb(done, total):
            if total:
                print(f"[diarize] 进度 {done / total * 100:.1f}%", flush=True)
            return 0

        result = sd.process(audio, callback=cb).sort_by_start_time()
        turns = [(r.start, r.end, f"SPEAKER_{r.speaker:02d}") for r in result]
        return turns
    # TemporaryDirectory 退出时自动清理 wav


# --------------------------------------------------------------------------- #
# engine: pyannote (fallback)
# --------------------------------------------------------------------------- #
def diarize_pyannote(args):
    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        sys.stderr.write(
            "\n[diarize] pyannote.audio 未安装：%s\n"
            "[diarize] pip install torch pyannote.audio；接受 pyannote 门控协议并设 HF_TOKEN。\n" % e
        )
        sys.exit(2)
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        sys.stderr.write("[diarize] 缺少 HF token（--token 或 HF_TOKEN）。\n")
        sys.exit(3)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    print("[diarize] pyannote 分轨中 ...", flush=True)
    # 已知人数时显式传给 pyannote（num_speakers=-1 时传 None 让其自动聚类）
    diar = pipeline(args.audio, num_speakers=args.num_speakers if args.num_speakers and args.num_speakers > 0 else None)
    return [(t.start, t.end, sp) for t, _, sp in diar.itertracks(yield_label=True)]


# --------------------------------------------------------------------------- #
# 时间重叠归因（纯函数，便于单测）：给定 segment 与全部 turn，返回归属说话人。
def speaker_at(seg, turns):
    # 按时间重叠比例归因：收集与 segment 时间区间重叠的所有 turn，
    # 取重叠时长最大者；覆盖率（重叠总长 / 段长）过低则标 SPEAKER_??
    # （跨多个说话人的段不应整体归给一人）。
    s0, e0 = seg["start"], seg["end"]
    best_sp, best_ov = None, 0.0
    covered = 0.0
    for s, e, sp in turns:
        ov = min(e0, e) - max(s0, s)
        if ov > 0:
            covered += ov
            if ov > best_ov:
                best_ov, best_sp = ov, sp
    if best_sp is None:
        return "SPEAKER_??"
    if covered < (e0 - s0) * 0.5:
        return "SPEAKER_??"
    return best_sp


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Speaker diarization over transcript segments")
    ap.add_argument("--segments", required=True, help="transcribe.py 产出的 segments.json")
    ap.add_argument("--audio", required=True, help="原始音频文件（用于声纹分析）")
    ap.add_argument("--out", required=True, help="输出带说话人标签的转录稿")
    ap.add_argument("--engine", default="sherpa", choices=["sherpa", "pyannote"],
                    help="sherpa (default, no torch/token) | pyannote (needs torch + HF_TOKEN)")
    ap.add_argument("--num-speakers", type=int, default=-1, help="已知说话人数则指定，否则 -1 自动聚类")
    ap.add_argument("--cluster-threshold", type=float, default=0.5,
                    help="num-speakers=-1 时的聚类阈值：越小人越多")
    ap.add_argument("--seg-model", default=None, help="sherpa 分割模型 model.onnx（或其目录）")
    ap.add_argument("--emb-model", default=None, help="sherpa 说话人嵌入 onnx（或其目录）")
    ap.add_argument("--token", default=None, help="pyannote 引擎的 HuggingFace token")
    ap.add_argument("--num-threads", type=int, default=None,
                    help="ONNX 推理线程数（sherpa 引擎关键性能项）。默认取 CPU 核数；"
                         "不设则 sherpa 用单线程，长音频分轨极慢。也可用 A2N_DIAR_THREADS 环境变量覆盖。")
    args = ap.parse_args()

    segs = json.load(open(args.segments, encoding="utf-8"))
    turns = diarize_sherpa(args) if args.engine == "sherpa" else diarize_pyannote(args)

    # 存盘原始 turn 级分轨，便于只调合并逻辑/阈值而不必重跑耗时分轨。
    turns_path = os.path.join(os.path.dirname(args.out), "diar_turns.json")
    json.dump([{"start": s, "end": e, "speaker": sp} for s, e, sp in turns],
              open(turns_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    with open(args.out, "w", encoding="utf-8") as f:
        for s in segs:
            sp = speaker_at(s, turns)
            f.write(f"[{sp}] [{fmt(s['start'])} - {fmt(s['end'])}] {s['text']}\n")

    n_spk = len({t[2] for t in turns})
    print(f"[diarize] 完成（engine={args.engine}，识别 {n_spk} 位说话人）-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
