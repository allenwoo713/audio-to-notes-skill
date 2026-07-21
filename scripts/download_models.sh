#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# audio-to-notes — 模型一键下载脚本（可移植版）
#
# 适用：本机可访问 GitHub releases 时，拉取所有 sherpa-onnx / whisper 模型。
# 可移植铁律：不写死任何本机绝对路径；根目录默认是 skill 自身的 models/，
#             也可由 $MODEL_ROOT 覆盖。
#
# 用法（在 skill 目录下，用 bash / Git Bash 运行）：
#   bash scripts/download_models.sh                      # 默认下载到 skill 的 models/
#   MODEL_ROOT=/your/path bash scripts/download_models.sh  # 自定义根目录
#
# 默认落点（skill 内，结构简单、随 skill 走、好找）：
#   <skill>/models/sherpa-onnx-sense-voice/
#   <skill>/models/silero-vad/
#   <skill>/models/diar/
# transcribe.py / diarize.py 会自动探测以上路径，无需设环境变量即可运行。
# 脚本末尾也会打印一段 `export`，若你自定义了根目录可复制进 shell。
#
# 注（Windows 用户）：请用 Git Bash 或 WSL 运行；`tar xjf` 在 PowerShell cmd 下不可用。
# emotion2vec+ 不在本脚本内——它经 funasr 首次运行时自动下载，见文末说明。
# ---------------------------------------------------------------------------

set -u

# ---- 可配置项（仅此处一处可改） -------------------------------------------
# 默认下载到 skill 自身的 models/ 目录（结构简单、随 skill 走、好找）。
# 若要放到别处，用 MODEL_ROOT=/your/path 覆盖即可。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_ROOT="${MODEL_ROOT:-$SKILL_DIR/models}"
REPO="https://github.com/k2-fsa/sherpa-onnx/releases/download"

# 资源名（已逐一核对 GitHub releases / 官方文档 2026-07）
# 用 int8 精简包：整包约 250MB（含 model.int8.onnx 228MB + tokens.txt）。
# 切勿用不带 -int8 的「全家桶」(sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2025-09-09.tar.bz2，
# 999MB，其中 894MB 是 CPU 用不上的 fp32 权重)。transcribe.py 始终显式加载 model.int8.onnx。
# 注意：k2-fsa 后续又发布了 2025-09-09 版（比 2024-07-17 更新），这里用最新版。
SV_TARBALL="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2"
SV_DIR="$MODEL_ROOT/sherpa-onnx-sense-voice"
VAD_URL="$REPO/asr-models/silero_vad.onnx"
VAD_DIR="$MODEL_ROOT/silero-vad"

SEG_TARBALL="sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
SEG_DIR="$MODEL_ROOT/diar"
EMB_FILE="3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
EMB_URL="$REPO/speaker-recongition-models/$EMB_FILE"
# ---------------------------------------------------------------------------

# 模型权重体积大（数百 MB），不应纳入 skill 版本控制。
# 确保 models/ 存在，并写入 .gitignore（仅忽略权重、保留 .gitignore 自身）。
mkdir -p "$MODEL_ROOT"
if [ ! -f "$MODEL_ROOT/.gitignore" ]; then
  printf '# 模型权重体积大，不纳入 skill 版本控制（运行 download_models.sh 生成）\n*\n!.gitignore\n' \
    > "$MODEL_ROOT/.gitignore"
  log "已写入 $MODEL_ROOT/.gitignore（权重不会被提交）"
fi

GREEN=''; YEL=''; RED=''; NC=''
# 非 TTY 时关闭彩色，避免日志里出现控制字符
if [ -t 1 ]; then GREEN='\033[0;32m'; YEL='\033[0;33m'; RED='\033[0;31m'; NC='\033[0m'; fi

log()  { printf "%b[download]%b %s\n" "$GREEN" "$NC" "$1"; }
warn() { printf "%b[skip]%b    %s\n" "$YEL" "$NC" "$1"; }
err()  { printf "%b[ERROR]%b   %s\n" "$RED" "$NC" "$1"; }

# download_file <url> <out_path>  —— 带重试、断点续传、非空校验
download_file() {
  local url="$1" out="$2"
  mkdir -p "$(dirname "$out")"
  if [ -s "$out" ]; then
    warn "已存在且非空，跳过：$out"
    return 0
  fi
  log "下载：$url"
  local i rc=1
  for i in 1 2 3; do
    if curl -L -C - -f -o "$out" "$url"; then
      if [ -s "$out" ]; then rc=0; break; fi
    fi
    err "第 $i 次尝试失败，重试…"
    rm -f "$out"
  done
  if [ "$rc" -ne 0 ]; then
    err "下载失败：$url"
    return 1
  fi
  log "完成：$out"
  return 0
}

# extract_tar <tarball> <dest_dir>  —— 解包并扁平一层
extract_tar() {
  local tb="$1" dest="$2"
  if [ ! -f "$tb" ]; then err "压缩包不存在：$tb"; return 1; fi
  mkdir -p "$dest"
  if tar xjf "$tb" -C "$dest" --strip-components=1 2>/dev/null; then
    log "解包：$dest"
  else
    # 某些环境 bzip2 路径不同，回退到不剥离层级
    tar xjf "$tb" -C "$dest" 2>/dev/null && log "解包（含顶层目录）：$dest" \
      || { err "解包失败：$tb"; return 1; }
  fi
  return 0
}

overall=0

# ---- 1. SenseVoice（zh/en/ja/ko/yue）--------------------------------------
SV_TB="$MODEL_ROOT/$SV_TARBALL"
download_file "$REPO/asr-models/$SV_TARBALL" "$SV_TB" || overall=1
extract_tar "$SV_TB" "$SV_DIR" || overall=1
# silero VAD
download_file "$VAD_URL" "$VAD_DIR/silero_vad.onnx" || overall=1

# ---- 2. 说话人分轨：分割 + 嵌入 -------------------------------------------
SEG_TB="$SEG_DIR/$SEG_TARBALL"
download_file "$REPO/speaker-segmentation-models/$SEG_TARBALL" "$SEG_TB" || overall=1
extract_tar "$SEG_TB" "$SEG_DIR" || overall=1
download_file "$EMB_URL" "$SEG_DIR/$EMB_FILE" || overall=1

# ---- 3. whisper（可选；faster-whisper 通常自带下载，这里给 curl 兜底）------
# 默认不下载（体积大）。如需，取消下一行注释并改 size：
# WHISPER_SIZE="small"; WH_TB="$MODEL_ROOT/_asr_models/faster-whisper-$WHISPER_SIZE"
# download_file "https://huggingface.co/Systran/faster-whisper-$WHISPER_SIZE/resolve/main/model.bin" "$WH_TB/model.bin" || overall=1

echo ""
if [ "$overall" -ne 0 ]; then
  err "部分资源下载失败，请检查网络 / 代理后重试。"
  exit 1
fi

log "全部模型就绪，根目录：$MODEL_ROOT"
echo ""
echo "# 将以下 export 复制到你的 shell（或写入 ~/.bashrc）："
echo "export A2N_SENSEVOICE_DIR=\"$SV_DIR\""
echo "export A2N_VAD_MODEL=\"$VAD_DIR/silero_vad.onnx\""
echo "export A2N_DIAR_SEG_MODEL=\"$SEG_DIR/model.onnx\""
echo "export A2N_DIAR_EMB_MODEL=\"$SEG_DIR/$EMB_FILE\""
echo "# export A2N_MODEL_DIR=\"$MODEL_ROOT/_asr_models\"   # 仅 whisper 手动下载时需要"
echo ""
echo "# 随后可直接跑："
echo "python scripts/transcribe.py --input <音频> --workdir <目录> --backend sensevoice"
echo "python scripts/diarize.py   --segments <目录>/transcripts/segments.json --audio <音频> --out <目录>/transcripts/transcript_timed_speaker.txt"
echo ""
echo "# 情绪识别（emotion2vec+）：不含在本脚本。首次运行自动下载："
echo "#   pip install -U funasr modelscope"
echo "#   python scripts/emotion.py --segments <目录>/transcripts/segments.json --audio <音频>"
exit 0
