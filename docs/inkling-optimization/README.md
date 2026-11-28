# Inkling-2-Mini Apple Silicon 推理优化 — 工作档案

实验知识中枢、迁移指南、引擎补丁参考、测试工具。主线证据链在 `SKILL.md`(按时间倒序追加)。

## 当前战况速览(2026-11-28)

1. no-spec 基线 34.3 tok/s(M6 24GB);tree 投机输出正确但零收益,block host verify 上游损坏
2. verify batch 前向正确(spec-check 仪器:窗口 logits vs B=1 max dev 0.0055),腐败在 rollback/重喂状态循环
3. 融合 unembed+topk mm8 kernel 就绪(单测 128/128 精确),verify 修复后兑现收益
4. fused kernel 在 nd≥5 verify 时激活但产生垃圾输出 —— 与 verify 腐败同源待查

## 目录

- `SKILL.md` — 全部实验证据与方法论(125KB,先读这个)
- `MIGRATION-M1MAX.md` — M1 Max 迁移指南与安全操作规则
- `patches/` — 引擎关键文件参考实现(基于 0.2.22 快照,上游已 0.2.23,合并需人工 diff)
- `tools/` — 测试 harness(spd/ab 探针、并发发散、跨模式验证)
- `kaggle-train/` — engram 云训练管线(Kaggle T4x2,自包含 kernel)
