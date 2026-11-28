# LoopCD 移植与验证计划(2026-11-28)

## 现状
- strata fork 已有 LoopCD-Hidden 固定权重实现(ink_metal_runner.h:2096-2110),env INK_LCD_W 控制,默认 0=关闭
- 数学:h' = h_R + w*(h_R - h_1),h_R=最终 pass 隐藏态,h_1=pass-0 快照,axpby 融合 kernel,代价可忽略
- oMLX 0.8.4 有 logits-adaptive 完整版(margin 自适应 + thinking-aware + control-mass preservation)
- nanoattention 有 hidden/logits/logits-adaptive 三种 + fused dual-unembed 优化

## 分层验证计划(每步有明确的通过/拒绝标准)

### 第一层:固定权重 hidden(今天可做)
1. 训练窗口结束后恢复服务,INK_LCD_W=0.3(论文/nsight 常用区间 0.2-0.5)
2. P3 打标基准对比:w=0 vs w=0.3 vs w=0.5,看标签正确数变化
3. 事实性测试:闭卷问答 20 题(AA-Omniscience 风格),统计幻觉率
4. 速度验证:tok/s 变化应 <1%(hidden 模式只加一个 axpby)
- 通过标准:P3 不退化 + 事实性提升或持平 → 默认开启
- 拒绝标准:P3 标签错误增加 → 回退 w=0,尝试更小 w

### 第二层:logits-adaptive(质量上限更高,成本 9-13%)
- 需要 fused dual-unembed(unembed 权重只读一遍)才能压到 5% 以内
- nanoattention 的 mk_bf16_linear_4x2_softcap_dual 是参考实现
- 只有第一层证明 LoopCD 方向有效后才值得做

### 第三层:weak-pass 消融(研究价值)
- pass1 vs pass2 作为 weak 的对比,论文暗示 pass1 更优(对比度最大)
- 低成本:改 INK_LCD 快照点即可

## 注意
- 我们的模型是 3-pass loop;hidden 的 round=bf16 会和生产 sq10 的数值有互动,验证时必须用生产 sq10 服务测
- control token(199998+)的 logit 质量必须保持:logits 变体需要 control-mass preservation(oMLX 已实现);hidden 变体天然免疫
