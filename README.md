# audio-to-notes

把任意「声音 / 文字源」变成一份**可复用、可溯源**的结构化笔记 —— 完全本地运行，**无需 API key**。

这是一个 [WorkBuddy](https://www.codebuddy.cn/) skill：AI 读取 `SKILL.md` 后按其编排流程执行「听译 → （可选）分轨/情绪 → 套用模板 → 产出笔记」。核心是**模板化**：同一段听译文字，套用不同的「提示词模板」就能产出不同形态的产物（播客深度笔记、会议纪要、待办清单……）。新增一种产出形态 = 新增一个 `templates/*.md`，无需改动主逻辑。

## 特性

- **本地优先、零 API key**：听译、分轨、情绪全部在本机 CPU 完成，隐私可控。
- **可插拔 ASR 后端**：
  - `sensevoice`（**默认**）—— sherpa-onnx SenseVoice（ONNX，无 torch），中文/中英混说更强；本地基准比 whisper-small **快约 13 倍**（同一 15 分钟片段 RTF 0.04 vs 0.52）。
  - `whisper` —— faster-whisper（CPU int8），英文/通用更鲁棒。
- **说话人区分（可选、手动触发）**：默认由 LLM 按内容归属角色；仅当显式要求「本地分轨」或给出人数时才跑声学分轨 `diarize.py`（sherpa-onnx，无需 torch / HF token）。
- **情绪识别（可选）**：`emotion2vec+`（默认 `_large`，9 类情绪），转写时加 `--with-emotion` 可复用已解码波形一并打标。
- **多输入**：本地音频/视频文件、已转录文档/纯文本、音频直链或网页 URL。
- **可移植**：脚本与文档中**不硬编码任何本机绝对路径**，模型位置经环境变量或相对目录解析。

## 安装

### 1. 获取 skill

```bash
git clone https://github.com/allenwoo713/audio-to-notes-skill.git
# WorkBuddy 用户可放到 ~/.workbuddy/skills/ 下即被识别：
#   git clone https://github.com/allenwoo713/audio-to-notes-skill.git ~/.workbuddy/skills/audio-to-notes
```

### 2. 安装 Python 依赖（建议独立 venv）

依赖是**分层**的，起步只需核心 + 默认后端：

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 中的可选块（whisper 后端、情绪、pyannote 分轨）默认注释，按需解除或单独安装。例如启用情绪识别：

```bash
python -m pip install funasr modelscope torch==2.11.0 torchaudio==2.11.0
```

> **注意**：`torch` 与 `torchaudio` 版本必须一致（实测 `2.11.0` 组合可用），否则 funasr 加载会报错。

### 3. 下载模型

模型权重体积较大（不入库，已由 `.gitignore` 排除），首次使用前下载：

```bash
# 本机可访问 GitHub releases 时，一键拉取 SenseVoice / silero-VAD / 分轨模型
bash scripts/download_models.sh
# 默认落到 <skill>/models/，脚本与各脚本会自动探测，无需设环境变量。
# 自定义位置：MODEL_ROOT=/your/path bash scripts/download_models.sh
```

- 情绪模型 `emotion2vec+` 不在该脚本内 —— 由 funasr 首次运行 `--with-emotion` 时自动下载到 `<skill>/models/emotion2vec/`。
- Windows 用户请用 **Git Bash / WSL** 运行 `download_models.sh`（`tar xjf` 在 cmd/PowerShell 下不可用）。
- 手动逐个下载与国内镜像见 `references/transcription.md`。

## 快速开始（直接调脚本）

也可脱离 WorkBuddy 单独把脚本当命令行工具用。产物统一位于 `<workdir>/transcripts/`。

```bash
# 1) 听译（默认 sensevoice 后端）
python scripts/transcribe.py --input "meeting.m4a" --workdir "./out"

# 纯英文 / 需最高鲁棒性：切 whisper
python scripts/transcribe.py --input "talk.mp3" --workdir "./out" --backend whisper --language en

# 2) 听译 + 情绪一并标注（复用波形，惰性 import funasr）
python scripts/transcribe.py --input "podcast.mp3" --workdir "./out" --with-emotion

# 3) 说话人分轨（可选，声学；已知人数就显式给 --num-speakers）
python scripts/diarize.py --segments "./out/transcripts/segments.json" \
    --audio "podcast.mp3" --out "./out/transcripts/transcript_timed_speaker.txt" --num-speakers 3

# 4) URL 输入（音频直链则下载后听译；网页则抓正文为文本）
python scripts/fetch_input.py --url "https://example.com/ep01.mp3" --workdir "./out"

```

听译产物（两后端 schema 一致）：

- `segments.json` —— 段落级 `[{start, end, text}]`
- `transcript_full.txt` —— 纯全文
- `transcript_timed.txt` —— `[mm:ss - mm:ss] text` 逐段
- （`--with-emotion` 时）`segments_emotion.json` / `transcript_timed_emotion.txt`

## 目录结构

```
audio-to-notes/
├── SKILL.md                    # WorkBuddy 编排入口（AI 读取并执行）
├── requirements.txt            # 分层依赖
├── scripts/
│   ├── transcribe.py           # 本地听译（sensevoice 默认 / whisper 可选，+ --with-emotion）
│   ├── diarize.py              # 说话人分轨（sherpa 默认 / pyannote 回退）
│   ├── emotion.py              # 情绪识别（emotion2vec+，独立/可内联）
│   ├── fetch_input.py          # URL → 音频下载或网页正文抽取
│   └── download_models.sh      # 一键下载 sherpa 模型（SenseVoice / silero-vad / 分轨）
├── templates/                  # 提示词模板（产出形态，可扩展）
│   ├── blog_deepnote.md        # 文章/播客深度笔记
│   ├── meeting_minutes.md      # 会议纪要
│   └── README.md
├── references/
│   ├── transcription.md        # 依赖、模型下载、后端细节、排错
│   └── adding_templates.md     # 如何新增模板
└── models/                     # 模型权重（.gitignore 排除，需自行下载）
```

## 模型与体积速览

| 用途 | 模型 | 体积 | 依赖 |
|------|------|------|------|
| ASR 默认 | SenseVoice int8 (2025-09-09) | ~237MB | sherpa-onnx |
| VAD 分段 | silero-vad | ~2MB | sherpa-onnx |
| ASR 可选 | faster-whisper-small | ~480MB | faster-whisper |
| 情绪 默认 | emotion2vec+ large | ~1.94GB | funasr + torch |
| 情绪 可选 | emotion2vec+ base | ~1.12GB | funasr + torch |
| 分轨 默认 | pyannote-seg-3 + 3D-Speaker | ~100MB | sherpa-onnx |

## 设计原则

1. **本地优先、零 API key**；模型缓存或按需下载。
2. **模板即提示词**：每个 `templates/<id>.md` 自带 SOP + Output Protocol，AI 严格照其执行。
3. **说话人识别默认走 LLM 内容归属**，本地声学分轨为可选、手动触发。
4. **情绪维度按需启用**，不进默认听译路径。
5. **反幻觉、强溯源**：每个事实/数字/引文可溯源到转录稿，无法确认处显式标注。
6. **可移植性铁律**：禁止硬编码本机绝对路径。

## 许可

- **本仓库代码**：MIT License（见 [LICENSE](./LICENSE)）。
- **第三方模型**：各自遵循其上游许可，**使用前请查阅对应上游仓库**。汇总见 [THIRD_PARTY_MODELS.md](./THIRD_PARTY_MODELS.md)：
  - SenseVoice / silero-vad / sherpa-onnx 系列（k2-fsa）：Apache-2.0
  - emotion2vec+（iic / Alibaba）：ModelScope 社区许可（免费用于研究/商业需符合其条款）
  - pyannote / faster-whisper：MIT（pyannote 模型权重另受 HuggingFace 门控协议约束）
  - 3D-Speaker 嵌入：相关论文/仓库许可
