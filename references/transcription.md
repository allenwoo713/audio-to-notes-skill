# 听译 / 分轨 / 情绪识别：环境搭建与排错

覆盖三段可选能力：ASR 后端（whisper / sensevoice）、说话人分轨（sherpa / pyannote）、情绪识别（emotion2vec+）。

> **一键下载**：本机可访问 GitHub 时，直接跑 `bash scripts/download_models.sh` 即可拉齐 SenseVoice / silero-vad / 分割 / 嵌入四个模型，并打印所需 `export` 环境变量。下文各节保留手动 curl 步骤，供只想下部分模型或排错时参考。

> **可移植性铁律**：本文件与所有脚本中**禁止硬编码本机绝对路径**（如 `C:\Users\...`、`D:\...`）。模型位置一律通过环境变量或相对目录解析，便于分享。

> **模型根目录约定**：下文的 `MR` 指「模型根目录」。默认即本 skill 的 `models/` 子目录——`bash scripts/download_models.sh` 会自动把模型解包到这里，且 `transcribe.py` / `diarize.py` 会**自动探测**该路径，**无需设任何环境变量**即可运行。自定义位置：`export MR=/your/path` 后运行下载脚本。

## 0. 依赖总览（按需安装，非全部必需）

| 能力 | pip 依赖 | 轻重 | 备注 |
|------|---------|------|------|
| ASR: sensevoice（默认） | `sherpa-onnx imageio-ffmpeg` | 轻 | ONNX，无 torch；zh/en 更强；本地基准比 whisper-small 快约 13 倍 |
| ASR: whisper（可选） | `faster-whisper imageio-ffmpeg` | 中 | CPU int8，自带 ffmpeg；英文/通用更强 |
| 分轨: sherpa（默认） | `sherpa-onnx` | 轻 | 无 torch、无 HF token |
| 分轨: pyannote（回退） | `torch pyannote.audio` | 重 | 需 HF token + 接受协议 |
| 情绪: emotion2vec+ | `funasr modelscope` | 重 | 含 torch；仅需情绪时装 |

安装示例（任意 Python，建议独立 venv；**不写死解释器路径**）：
```bash
python -m pip install faster-whisper imageio-ffmpeg sherpa-onnx
# 默认后端即 sensevoice（中文/中英混说首选，~13x 快于 whisper-small，无 torch）
python scripts/transcribe.py --input <音频> --workdir <目录>
# 纯英文 / 需最高鲁棒性才切 whisper
python scripts/transcribe.py --input <音频> --workdir <目录> --backend whisper
```

## 1. ffmpeg

`transcribe.py` / `diarize.py` / `emotion.py` 均优先用 `imageio_ffmpeg.get_ffmpeg_exe()`（自带，无需系统安装），找不到时回退系统 `ffmpeg`。二者皆无才报错退出。

## 2. ASR 后端 A：faster-whisper（whisper 后端，可选）

自动探测本地缓存，命中则不下载。探测顺序见 `transcribe.py` 的 `_whisper_model_dir`：`--model-dir` → `A2N_MODEL_DIR` → `<skill>/models/` → 同级项目缓存 `_asr_models/` 与旧缓存 `_whisper_models/` → `MR`（及 `~/.cache/audio-to-notes/`）回退。

若本地无模型，脚本先试 `faster_whisper.download_model()`；沙箱常拦 HF Python 直连，此时用 curl 手动下载（已验证）：
```bash
MODEL_DIR="$MR/faster-whisper-small"
mkdir -p "$MODEL_DIR"
BASE="https://huggingface.co/Systran/faster-whisper-small/resolve/main"
for f in config.json tokenizer.json vocabulary.txt README.md .gitattributes; do
  curl -L -o "$MODEL_DIR/$f" "$BASE/$f"
done
curl -L -C - -o "$MODEL_DIR/model.bin" "$BASE/model.bin"   # ~480MB，断点续传
```
然后 `--model-dir "$MODEL_DIR"` 或设 `A2N_MODEL_DIR` 指向其父目录。其它 size 把 URL/目录里的 `small` 替换即可。

## 3. ASR 后端 B：sherpa-onnx SenseVoice（zh/en 推荐，**默认后端**）

**特点**：ONNX Runtime、纯 CPU、**无 torch / tensorflow / modelscope**；中文与中英混说准确率优于同级 whisper-small。
**关于「轻量」的准确说法**：轻的是**运行时依赖**（没有 torch 那套），模型权重本身并不小——SenseVoice 是 ~400M 参数级模型，int8 量化后 `model.int8.onnx` 仍有 **228MB**。不要误以为它「权重轻」。
**限制**：sherpa-onnx 的 SenseVoice 解码器会在后处理阶段**剥离情绪/事件标签**（模型词表本身含 `<|HAPPY|> <|ANGRY|> <|NEUTRAL|>` 等，但 `from_sense_voice` 无保留开关，实测 1.13.4 只返回纯文本）——情绪维度请用第 5 节的 `emotion.py` 或转写时加 `--with-emotion`（见 SKILL.md 步骤 2）。

需要两个模型：SenseVoice 识别模型（**务必用 int8 精简包**）+ silero-VAD（长音频分段）。

> **为什么用 int8 包而非「全家桶」**：不带 `-int8` 的 `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2025-09-09.tar.bz2` 整包 **999MB**，其中 894MB 是 CPU 根本用不上的 fp32 权重 `model.onnx`；int8 包 `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2` 仅 **~250MB**（只含 `model.int8.onnx` 228MB + `tokens.txt`）。`transcribe.py` 始终显式加载 `model.int8.onnx`，所以两者功能等价，直接下 int8 包最省。k2-fsa 的 SenseVoice 已迭代到 **2025-09-09** 版（比早期 2024-07-17 更新），脚本与下方命令均用最新版。
```bash
# SenseVoice（zh/en/ja/ko/yue，int8 精简包 ~250MB，最新 2025-09-09）
D="$MR/sherpa-onnx-sense-voice"
mkdir -p "$D" && cd "$D"
curl -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2
tar xjf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2 --strip-components=1
# silero VAD
curl -L -o "$MR/silero-vad/silero_vad.onnx" \
  --create-dirs https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
```
运行（默认已落在 `$MR` 下，脚本自动探测，export 可省略；下方仅作显式示范）：
```bash
export A2N_SENSEVOICE_DIR="$MR/sherpa-onnx-sense-voice"
export A2N_VAD_MODEL="$MR/silero-vad/silero_vad.onnx"
python scripts/transcribe.py --input <音频> --workdir <目录> --backend sensevoice
```
或用 `--sensevoice-model/--sensevoice-tokens/--vad-model` 显式指定文件路径。

## 4. 说话人分轨

### 4A. sherpa-onnx（默认，无 torch/token）

需要「分割模型（pyannote 转 ONNX）+ 说话人嵌入模型（3D-Speaker）」：
```bash
S="$MR/diar"
mkdir -p "$S" && cd "$S"
# 分割
curl -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar xjf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
# 说话人嵌入（3D-Speaker，中文优化）
curl -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
```
运行：
```bash
export A2N_DIAR_SEG_MODEL="$S/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
export A2N_DIAR_EMB_MODEL="$S/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
python scripts/diarize.py --segments <transcripts>/segments.json --audio <音频> \
  --out <transcripts>/transcript_timed_speaker.txt --num-speakers <已知人数或省略>
```
- 已知说话人数：`--num-speakers N`（更稳）。未知：省略，用 `--cluster-threshold`（越小人越多）。
- **重要**：人数已知时务必**精确**设 N，不要「+1 留余量」。多设 1 类反而更容易把某个真实说话人（声纹内部有差异，如爱笑、语速多变）拆成两个类，凭空造出「假的第 N 人」。留余量只在「人数未知、怕漏掉偶尔插话者」时才考虑。
- 自动模式（`--num-speakers` 省略，即 `-1`）在长音频 + 默认阈值下极易过分割（实测 92 分钟 3 人播客被碎成 134 类）。只要能从转写稿确认人数，就显式指定。
- 已把原始 turns 存盘为 `<transcripts>/diar_turns.json`；改 K 或阈值后想重聚类，复用该文件即可，不必重跑整段分轨。

### 4B. pyannote（可选回退，需 torch + token）
```bash
python -m pip install torch pyannote.audio
# 在 https://huggingface.co/pyannote/speaker-diarization-3.1 与 segmentation-3.1 各点 Accept
python scripts/diarize.py --engine pyannote --segments ... --audio ... --out ... --token $HF_TOKEN
```

未安装 / 无模型 / 失败时，`diarize.py` 优雅退出，SKILL.md「步骤 4」自动回退到 LLM 归属 + 占位符方案。

## 5. 情绪识别（可选，emotion2vec+）

**独立模块**，与 ASR 后端解耦，只有需要情绪时才引入重依赖：
```bash
python -m pip install -U funasr modelscope   # 含 torch，较重
python scripts/emotion.py --segments <transcripts>/segments.json --audio <音频>
# 首次自动下载 iic/emotion2vec_plus_large（默认，~1.94GB；可选 _base ~1.12GB / _seed）。
# 运行时间相近，large 情绪分布更合理；base 更保守、超短段偶有 angry 噪声。
```
产出 `segments_emotion.json`（每段加 `emotion`/`emo_score`）与 `transcript_timed_emotion.txt`。
未装 funasr 时脚本优雅退出——此时可让 LLM 仅凭文本推断情绪倾向（弱于声学 SER，仅作辅助）。

> **模型存放**：emotion2vec+ 由 funasr/modelscope 自动下载，默认收归本 skill 的 `models/emotion2vec/`（与 SenseVoice/silero 同目录，自包含），**不再继承全局 `MODELSCOPE_CACHE`**（避免被塞进其他工具的缓存，如 mineru）。可用环境变量 `A2N_EMOTION_CACHE` 覆盖落盘位置。

## 6. 排错速查

| 现象 | 原因 / 对策 |
|------|------------|
| `CUDA` 静默崩溃 / segfault | 沙箱 CUDA 运行时不匹配。whisper 默认 `cpu/int8`，勿改 CUDA。 |
| Python 下载模型报 `LocalEntryNotFoundError` / SSL 失败 | 沙箱拦 HF Python 客户端。改用 curl 手动下载 + 环境变量/`--*-model`。 |
| `HeaderNotFoundError`（mutagen） | 扩展名与容器不符（如 `.mp3` 实为 M4A）。脚本经 ffmpeg 统一转 WAV，不受影响。 |
| sensevoice 报缺 `model.int8.onnx/tokens.txt/silero_vad.onnx` | 未下载或路径未设。见第 3 节，设 `A2N_SENSEVOICE_DIR`/`A2N_VAD_MODEL`。 |
| diarize 报缺分割/嵌入模型 | 见第 4A 节，设 `A2N_DIAR_SEG_MODEL`/`A2N_DIAR_EMB_MODEL`。 |
| Bash/Python 偶发返回空、退出码 `-1073741819` | 沙箱间歇 access-violation 抖动。重试 2–3 次；用脚本文件代替 heredoc；勿据此误判文件缺失。 |
| 长音频听译中断 | 复用已存在的 `segments.json`；`--force` 才覆盖。small 约 3–4× 实时。 |
| `pip install funasr` 报 `SAFE_DELETE_FAIL_CLOSED` / 卸载 numpy 失败 | 沙箱删除策略 fail-closed：pip 升级/卸载现有包（如把 numpy 降级）需删除文件，被沙箱拒绝并整体中止。本沙箱内**无法**装 funasr/torch。正常机器上 `pip install -U funasr modelscope` 即可；若想避开对 numpy 的变动，用 `pip install --no-cache-dir --no-deps funasr` 再单独装其余依赖。 |
| diarize 挂死（进度停在某 % 不动）/ 退出码 139 段错误 | sherpa-onnx 原生 ONNX 推理在此构建或特定音频上偶发不稳定（非 Python 层可控）。同环境下 92 分钟全量自动聚类曾成功产出，15 分钟子集偶发卡在 ~25%。遇此不要反复重试，直接回退到 SKILL.md「步骤 4」的 LLM 内容归属（Q1 默认路径），或换 `--engine pyannote`（需 torch + HF token）。 |

## 7. 长音频快速验证（开发调试建议）

全量分轨/情绪在 90 分钟级音频上可能要 ~1 小时。若要快速验证全链路，可把原音频切出前 N 秒子集，但**必须同步过滤 segments.json**，否则分轨时间戳会错位。

用 ffmpeg 切音频（16k 单声道，匹配管线输入）：
```bash
ffmpeg -y -ss 0 -t 900 -i "<原始录音.mp3>" -ar 16000 -ac 1 -f wav "<工作目录>/smoke/recording_cut.wav"
```
再按 `start/end` 过滤 `segments.json`：仅保留 `start < 900` 的段，跨越边界的段把 `end` 截断到 900，得到 `<工作目录>/smoke/segments_ref.json`。后续步骤改用该子集，分轨可从 ~1h 降到 ~1–2 分钟。

> 切分 + 过滤脚本 `smoke_cut.py` 仅用于开发调试，不随 skill 发布；发布版请用上面的 ffmpeg 命令 + 手动过滤。
