# 模板库（提示词模型）

本目录下的每个 `.md` 文件就是一套「提示词模板」（prompt model）。audio-to-notes 的 SKILL.md 会自动发现并列出它们，供用户选择。同一段听译文字，套用不同模板即可产出不同形态。

## 新增一套模板

1. 复制本文件下方的骨架，新建 `templates/<你的id>.md`。
2. 填好顶部 frontmatter（见下）。
3. 在正文写完整 SOP + Output Protocol（直接作为给 AI 的指令/提示词）。
4. 完成。无需改 SKILL.md——下次运行会自动出现为新选项。

## Frontmatter 字段

```yaml
---
id: my_template          # 唯一 id，文件名建议与之相同；用户点名时用
name: 我的模板           # 展示名
description: 一句话说明用途与触发场景  # 用于自动列出时让用户理解
requires_speakers: false # true=会议纪要类，需要说话人识别（触发分轨分支）
input_types: [audio, doc, url]  # 该模板接受哪些输入类型
---
```

## 模板正文建议结构

- **# Role & Context**：界定 AI 在该模板下的角色与读者。
- **# Task**：明确要做什么。
- **# System Constraints**：硬性底线（反幻觉、溯源、（不确定）标注、未知填未知等）。**强烈建议保留这些反幻觉约束**，它们是 audio-to-notes 的质量底座。
- **# Standard Operating Procedure (SOP)**：分步骤的可执行流程。
- **# Output Protocol**：最终产物的精确格式（用 `<output_format>...</output_format>` 包裹，便于 AI 精确遵循）。
- **# Initialization**：结尾的触发句，提示 AI 接收转录稿开始工作。

## 现有模板

| id | name | requires_speakers | 说明 |
|----|------|------------------|------|
| blog_deepnote | 播客/文章深度笔记 | false | 深度内容重构为可复用笔记 |
| meeting_minutes | 会议纪要 | true | 会议转录稿→结构化纪要，含发言人归因 |

## 说话人识别（requires_speakers: true 时）

SKILL.md 会优先尝试 `scripts/diarize.py` 的 **sherpa 引擎**（pyannote 分割转 ONNX + 3D-Speaker 嵌入，**无需 torch / HF_TOKEN**）；仅当用户显式要求 pyannote 或 sherpa 不可用时，才回退到 `--engine pyannote`（需 `HF_TOKEN`）。声学分轨都不可用时，最终回退到「用户提供参会者名单 + LLM 按内容归属 + 发言人A/B 占位符 + （发言人待确认）」。模板 SOP 内应明确：无法确认归属时标 `（发言人待确认）`，不得猜测。
