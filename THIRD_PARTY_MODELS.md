# 第三方模型与依赖许可清单（THIRD_PARTY_MODELS.md）

本仓库的**脚本代码**以 MIT 许可证分发（见 [LICENSE](./LICENSE)）。但脚本在运行时会下载并调用若干**第三方模型权重**与 **Python 依赖**，它们各自遵循其上游许可。使用前请查阅对应上游仓库，确保你的使用场景符合其条款。

> 模型权重体积较大，**不纳入本仓库版本控制**（由 `.gitignore` 排除，运行时通过 `scripts/download_models.sh` 或各依赖自动下载）。

## 1. 声学模型（由 `download_models.sh` 下载）

| 模型 | 上游 / 仓库 | 许可 | 备注 |
|------|------------|------|------|
| SenseVoice (int8) | k2-fsa/sherpa-onnx | Apache-2.0 | ASR 默认后端 |
| silero-vad | snakers4/silero-vad（经 sherpa-onnx 打包） | MIT | 语音活动检测 |
| sherpa-onnx-pyannote-segmentation-3-0 | k2-fsa/sherpa-onnx | Apache-2.0 | 分轨分割 |
| 3D-Speaker 嵌入 | k2-fsa/sherpa-onnx（源自 3D-Speaker） | Apache-2.0 | 说话人嵌入 |

## 2. 情绪识别模型（由 funasr 首次运行时自动下载）

| 模型 | 上游 / 仓库 | 许可 | 备注 |
|------|------------|------|------|
| emotion2vec+ (large/base/seed) | iic/emotion2vec_plus（Alibaba, ModelScope） | ModelScope 社区许可（免费研究；商业需符合其条款） | `--with-emotion` 或 `scripts/emotion.py` |

## 3. 可选后端 / 回退引擎的模型

| 模型 | 上游 / 仓库 | 许可 | 备注 |
|------|------------|------|------|
| faster-whisper (small 等) | Systran/faster-whisper | MIT | `--backend whisper` |
| pyannote/speaker-diarization-3.1 | pyannote/pyannote-audio | MIT（**模型权重另受 HuggingFace 门控协议约束**，需登录并接受协议） | `--engine pyannote`，需 `HF_TOKEN` |

## 4. 关键 Python 依赖及其许可（摘要）

| 依赖 | 许可 | 用途 |
|------|------|------|
| sherpa-onnx | Apache-2.0 | SenseVoice / VAD / 分轨推理 |
| faster-whisper | MIT | 可选 ASR 后端 |
| funasr / modelscope | Apache-2.0 / 自定义 | 情绪识别 |
| torch / torchaudio | BSD-3-Clause | 情绪 / pyannote 后端 |
| numpy | BSD-3-Clause | 波形处理 |
| imageio-ffmpeg | MIT | 自带 ffmpeg 二进制 |

> 依赖的确切许可以各自 PyPI / 上游仓库发布时声明的 LICENSE 文件为准。本清单仅供快速索引，不替代上游许可原文。

## 5. 合规建议

- **研究 / 个人使用**：上述模型大多允许免费使用，注意 ModelScope / HuggingFace 门控类需接受其协议。
- **商业 / 产品集成**：SenseVoice、emotion2vec+ 等有各自商用条款，请在上线前完成合规确认（必要时获取商业授权）。
- **分发本 skill**：你只需以 MIT 分发本仓库的脚本与文档；**不要**将模型权重打包进你的分发物，应引导用户自行下载并遵守上游许可。
