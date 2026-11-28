# Inkling-2-Mini AI Infra 优化 — M1 Max 迁移指南

## 这台机器上有什么(~/dageno-modelbench,已全部 git 提交)

| 路径 | 内容 |
|---|---|
| `strata-fork/` | 推理引擎 fork(含全部优化:kernels、verify 路径、测试仪器)。`cd strata-fork/build && make -j6 strata-inkling` 数秒编译 |
| `strata-ink67-fp8/` | 生产二进制 + 历史备份(.bak-preverifycand/.orig-nov27/.pre-topk-backup) |
| `sq10.strata` (10.47GB) | SQ-10 量化权重,**必须从这台机器拷贝**(git 不含,重新下载极慢) |
| `Inkling-2-Mini-Vision.strata` | 视觉权重 |
| `ai.dageno.dageno-inkling.inkling/` | 生产服务配置(config.json + launchd plist 在 ~/Library/LaunchAgents/) |
| `kaggle-train/` | engram 云训练管线(build_kernel.py 一键重建自包含 kernel) |
| `dflash-assets/`、`distill-v2/`、`engram-hy4-q4/` | draft 蒸馏与 engram 资产 |
| `*.md` | 各轮实验记录(中文) |
| 知识中枢 | `/Users/sonders/.codex/skills/inkling-strata-optimization/SKILL.md`(在新机器上要手动拷过去) |

## 拷贝清单(rsync 命令参考)

```bash
# 在 M1 Max 上执行(假设这台机器 IP/hostname 为 M6HOST):
rsync -avP M6HOST:~/dageno-modelbench/ ~/dageno-modelbench/
# 大头是 sq10.strata 10.5GB + inkling-hf 19.5GB(官方 HF 权重,训练用,可选)
# 知识 skill:
rsync -avP M6HOST:~/.codex/skills/inkling-strata-optimization ~/.codex/skills/
# 生产服务 launchd plist(可选,M1 Max 上建议重新配置):
rsync -avP M6HOST:~/Library/LaunchAgents/ai.dageno.dageno-inkling.inkling.plist ~/Library/LaunchAgents/
```

## 当前战况(2026-11-28)

### 已确证的事实
1. **no-spec 基线 34.3 tok/s**(M6 24GB,swap 噪声 ±10%),tree 模式输出与 no-spec 逐字节一致
2. **block 模式 host verify 上游就坏**:draft 接受率 3.9%,输出垃圾。nd=1/2 时输出连贯但第 28/14 字符发散;nd≥4 即时垃圾
3. **verify batch 前向数学是对的**:spec-check 仪器证明 M=2/3/4/8 窗口 logits 与 B=1 偏差 max 0.0055(无 argmax flip 可能)
4. **spec-check 的崩溃是测试基建 bug**(TapCapture 在 Metal 上为空),已修 + 加 STRATA_SPEC_CHECK_BLOCK 强制走 host verify 链
5. **dsp_ctx 污染机理已证实**:被拒 draft token 的 tap 特征留在 draft 上下文里永不修正;repair-step 实验已做但单独不够——腐败仍有残余,说明 trunk KV/ShortConv 也有未处理的错位(下一个调查点:repair 后 cur_pos 簿记与 verify 环的交互)
6. **融合 unembed+topk mm8 kernel 已就绪**(合成数据单测 128/128 精确),verify 修复后立即兑现收益
7. M=2 并发 decode vs solo 有 1/6 概率 near-tie 发散(slot2 专属,loop 放大)——与 verify 腐败是不同的、更轻微的问题

### 破案工具(全部已提交)
- `strata-fork` 内:`--batch-check`(inter-slot batch 正确性)、`--spec-check`(B=1 vs verify 窗口,含 STRATA_SPEC_CHECK_BLOCK/QUICK/IDS/STEPS 开关)、INK_TIMELINE(GPU 段计时)、vdebug stderr 仪器
- `/tmp/spd.sh` `/tmp/ab_test.sh`(/tmp 会被重启清空,建议挪到 ~/dageno-modelbench/tools/)
- **验证铁律**:固定 spec 模式对比、跨 tree/block/no-spec 三模态逐字节验证、GPU TIMELINE 替代端到端测速

### 下一步路线(按优先级)
1. **修 block host verify**:residual corruption 定位(从 repair-step 之后的状态簿记查起;备选方案——干脆弃用 rollback,改成 verify 前不做任何写状态、verify 后重放 accepted 前缀,代价是重放计算但逻辑 trivially 正确)
2. 修好后 fused topk/amax 收益兑现,目标 40-55 tok/s(对照 no-spec 34.3)
3. 全部提交 strata 上游 PR(upstream 在 /tmp/strata-upstream-check 的 sparse clone,会被重启清掉,重新 clone 只要几秒)
4. megakernel:verify unembed 多行共享、48 层小 kernel 融合(M=1 理论极限 ~12ms,当前 22ms,空泡大头)
5. nvfp4 Metal 原生支持 PR(megakernel 之后)
6. 远期:DSA/YOCO/engram/latentmoe 架构改造 + Muon 训练(云管线在 kaggle-train/)

## M1 Max 的优势
- 32GB 统一内存 vs M6 的 24GB:权重 10.5GB + KV + 系统余量大得多,swap 压力小
- **内存带宽 M1 Max 400GB/s vs M6 估计 ~270GB/s**:这个模型的 decode 是纯内存带宽 bound,mm8 unembed、MoE 权重流都快 ~1.5x
- 代价:GPU 代数老(无 Apple10 的某些指令),`is_apple10` 门控的 mm8 内核会自动退化到 mv——首次 bench 时确认走哪条路
- M6 机器从此只跑业务,不再有内存双重负载

## 安全操作规则(血泪教训,机器重启 5 次换来)
1. 绝不同时跑两个加载权重的进程(bootout 生产后才能跑独立测试实例)
2. /tmp 会被重启清空——一切重要文件放 ~/dageno-modelbench
3. zsh 禁止 `rm -f`(用 python shutil);后台进程会被会话管理器杀,长任务前台跑
4. 重活前 `memory_pressure` + `sysctl vm.swapusage`,wired 高就不动手
5. 生产服务重启只用 `launchctl kickstart -k`(4 秒重载),bootout 后要手动 bootstrap 回来
