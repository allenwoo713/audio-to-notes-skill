# 扩展模板（新增一套提示词模型）

完整步骤与 frontmatter 字段说明见 **`templates/README.md`**（唯一权威来源，避免重复）。

要点速记：
1. 在 `templates/` 新建 `<id>.md`，填好 frontmatter（`id / name / description / requires_speakers / input_types`）。
2. 正文写 Role & Context → Task → System Constraints（强烈保留反幻觉/溯源/（不确定）约束）→ SOP → Output Protocol（用 `<output_format>` 包裹）→ Initialization。
3. 无需改 SKILL.md，下次运行自动发现。

会议纪要类模板设 `requires_speakers: true` 即可激活说话人分轨分支。
