---
name: audio-to-notes
description: 将音频文件、已转录文档或 URL 转化为结构化笔记/纪要的本地工作流。内置多套可扩展的「提示词模板」（播客/文章深度笔记、会议纪要等）：自动完成听译（本地转录，无需 API key，可选 faster-whisper 或 sherpa-onnx SenseVoice 后端）、可选说话人分轨（sherpa-onnx，无需 torch/token）、可选情绪识别（emotion2vec+）、并套用选定模板的 SOP 生成最终产出。触发词：听译、转写、音频转笔记、语音转文字、会议纪要、播客笔记、把这段音频整理成笔记、用XX模板处理。
agent_created: true
---

# audio-to-notes

把任意「声音/文字源」变成一份可复用、可溯源的结构化笔记。核心是**模板化**：同一段听译文字，套用不同「提示词模型」（模板）就能产出不同形态的产物（深度笔记、会议纪要、待办清单……）。新增一种产出形态 = 新增一个 `templates/*.md`，无需改本文件逻辑。

## 何时使用

- 用户提供音频文件（mp3/m4a/wav/...) 要求整理成笔记/纪要。
- 用户提供已转录的文档（txt/md/docx）或一段文本，要求按某种结构重构。
- 用户提供音频/文章 URL，要求听译或抓取后整理。
- 用户提到「会议纪要 / 播客笔记 / 深度笔记 / 整理这段录音」等意图。

## 设计原则

1. **本地优先、零 API key**：听译走 `scripts/transcribe.py`，两个可插拔后端——`sensevoice`（**默认**，sherpa-onnx，无 torch，中文/中英混说更强，本地基准比 whisper-small 快约 13 倍）与 `whisper`（faster-whisper CPU int8，英文/通用更强）。模型缓存或按需下载（见 `references/transcription.md`）。
2. **模板即提示词**：每个 `templates/<id>.md` 自带 frontmatter 元数据 + 完整 SOP + Output Protocol。WorkBuddy 读取后**严格照其 SOP 执行**。
3. **说话人识别：本地声学分轨为「可选、手动触发」，默认走 LLM 内容归属**。`scripts/diarize.py`（`sherpa` 引擎，pyannote 分割转 ONNX + 3D-Speaker 嵌入，**无需 torch/HF token**）**仅当用户显式要求「本地分轨 / 用声纹区分」或给出了具体人数（如「3 个人」→ `--num-speakers 3`）时才跑**——它慢（与音频时长成正比），识别的是「声纹身份」而非「角色」。其余情况（尤其播客/访谈/师生等角色从内容可辨）默认由 LLM 按内容把段落归属到「主持人/嘉宾」「提问方/回答方」等角色，或结合用户提供的参会者名单；无法确认者标 `（发言人待确认）` 或 `发言人A/B/C` 占位符。**绝不在无依据时臆测发言人身份。**
4. **情绪维度按需启用**：情绪是独立的声学任务（SenseVoice ONNX 已剥离情绪标签），由**可选**模块 `scripts/emotion.py`（emotion2vec+，9 类情绪）承担，不进默认听译路径；未装依赖时可让 LLM 仅凭文本推断情绪（弱于声学 SER）。
5. **反幻觉、强溯源**：所有模板必须显式要求「每个事实/数字/引文可溯源到转录稿」「无法确认处标（不确定）」「未知字段填未知」。这是底线，模板 SOP 已内置，执行时不得打折。
6. **可移植性铁律**：SKILL.md / references / scripts 内**禁止硬编码本机绝对路径**。模型位置只经环境变量或相对目录解析。

## 编排流程（按顺序执行）

### 1. 识别输入类型
- 音频文件（扩展名在 `mp3 m4a wav ogg aac flac webm` 等常见音频/视频格式）→ 走「听译」。
- 文档/文本（`.txt/.md/.docx` 或直接粘贴的文字）→ 直接作为转录稿，**跳过听译**。
- URL → 先跑 `scripts/fetch_input.py`：音频直链则下载后听译；网页则抓取正文后作为文本。

### 2. 文字化（仅音频/音频 URL 需要）
运行（任意装好依赖的 Python 即可；推荐独立 venv，先 `cd` 到 skill 目录以便相对引用）：
```bash
# 默认后端已是 sensevoice（中文/中英混说首选，~13x 快于 whisper-small，无 torch）
python scripts/transcribe.py --input "<音频路径>" --workdir "<工作目录>"
# 纯英文 / 需最高鲁棒性：切 whisper 后端
python scripts/transcribe.py --input "<音频路径>" --workdir "<工作目录>" --backend whisper [--model small] [--language en]
# 转写同时一并做情绪识别（方案1：复用已解码波形，惰性 import funasr，默认路径仍零 torch）
python scripts/transcribe.py --input "<音频路径>" --workdir "<工作目录>" --with-emotion
```
**后端选择建议**：以中文或中英混说为主 → 默认 `sensevoice`；纯英文或需最高鲁棒性 → `--backend whisper`。
依赖与模型下载详见 `references/transcription.md`（whisper 用 `faster-whisper`；sensevoice 用 `sherpa-onnx` + SenseVoice/silero-vad 模型）。
产出位于 `<工作目录>/transcripts/`（两后端 schema 一致）：
- `segments.json`：段落级 `{start,end,text}`（供分轨与精确定位）
- `transcript_full.txt`：纯全文
- `transcript_timed.txt`：`[mm:ss - mm:ss] 文本`（供分节映射与人读）

若用户已提供转录稿，跳过本步，直接读取。

### 3. 选定模板
- 若用户点名模板（如「用会议纪要模板」「转成播客深度笔记」），读取对应 `templates/<id>.md`。
- 否则，列出 `templates/` 下所有模板（读取各自 frontmatter 的 `name` 与 `description`），用选择题让用户挑。
- 读取选定模板**全文**到上下文，后续严格按其 SOP 与 Output Protocol 产出。

### 4. 说话人分轨（可选、手动触发；默认用 LLM 内容归属）

**默认不做本地声学分轨。** 以下任一条件成立才跑 `scripts/diarize.py`（默认 `sherpa` 引擎，无需 torch/token）：
- 用户显式要求「本地分轨 / 用声纹区分说话人」；
- 用户给出了已知说话人数（如「3 个人」）→ 直接 `--num-speakers 3`，**不要 +1 留余量**（多设 1 类反而易把真实说话人拆成两类，造出假的第 N 人）。

否则（模板 `requires_speakers: true` 但无本地分轨需求）：由 LLM 按内容把段落归属到角色（主持人/嘉宾、提问方/回答方等），或结合用户提供的参会者名单；无法确认者标 `（发言人待确认）` 或 `发言人A/B/C` 占位符。**绝不在无依据时臆测发言人身份。**

需要本地分轨时运行（需先按 `references/transcription.md` §4A 下载分割/嵌入模型）：
```bash
python scripts/diarize.py --segments "<工作目录>/transcripts/segments.json" \
  --audio "<音频路径>" --out "<工作目录>/transcripts/transcript_timed_speaker.txt" \
  [--num-speakers <已知人数>]
```
- 可选回退引擎：`--engine pyannote --token $HF_TOKEN`（需 torch + 接受协议）。
- 若模型/依赖不可用或运行失败：回退到上面的 LLM 归属 + 占位符方案，并向用户说明。

### 4.5 情绪识别（可选；模板含情绪维度 / 用户点名要情绪时）
- **合并到转写（推荐，方案1）**：转写时加 `--with-emotion`，复用已解码波形直接跑 emotion2vec+，单进程、零二次 ffmpeg 解码；未装 `funasr` 时优雅跳过、保留转写结果。见步骤 2 的命令。
- 独立补情绪（已转写、只想加情绪维度时）：
  ```bash
  python scripts/emotion.py --segments "<工作目录>/transcripts/segments.json" --audio "<音频路径>"
  ```
  产出 `segments_emotion.json`（每段 `emotion`/`emo_score`）与 `transcript_timed_emotion.txt`，可作为纪要「情绪/语气」列的依据。
- 若 `funasr` 未装：不强行安装，改由 LLM 仅凭文本推断情绪倾向，并在产出中标注「（情绪为文本推断，非声学识别）」。

### 5. 套用模板 SOP 产出
按模板要求：通读转录稿 → 抽取论点骨架 → 按逻辑分 3–7 节（带时间戳）→ 逐节展开（概览→细节含数字/引文→方法步骤→存疑标注）→ 必要时抽取框架/心智模型 → 自洽性校验 → 按 Output Protocol 格式化。

### 6. 写出产物
- 默认写入 `<项目根>/deliverables/<标题>_<模板id>.md`。
- 告知用户文件路径与关键产物，并附「如需调整分节粒度 / 补引文原文 / 换模板，可继续」。

## 目录速查
- `scripts/transcribe.py` — 本地听译（必用；`--backend whisper|sensevoice`）
- `scripts/diarize.py` — 说话人分轨（可选、手动触发；`--engine sherpa|pyannote`，默认 sherpa 无需 token）
- `scripts/emotion.py` — 情绪识别（可选；emotion2vec+，需 funasr；也可经 transcribe.py `--with-emotion` 合并执行）
- `scripts/smoke_cut.py` — 把长录音切出前 N 秒子集 + 同步过滤 segments.json，供快速全链路冒烟测试
- `scripts/fetch_input.py` — URL 输入处理
- `scripts/download_models.sh` — 一键下载所有 sherpa/whisper 模型（本机有 GitHub 访问权时）；默认落在本 skill 的 `models/` 目录，脚本会自动探测，无需设环境变量
- `templates/` — 提示词模板库（新增形态在此加文件）
- `references/transcription.md` — 模型下载 / ffmpeg / 分轨启用 / 排错
- `references/adding_templates.md` — 如何新增一套模板

## 扩展新模板（用户未来自助）
复制 `templates/README.md` 里的骨架，新建 `templates/<新id>.md`，填好 frontmatter（`requires_speakers` 等）与 SOP，即可被本流程自动发现并选用。无需改 SKILL.md。
