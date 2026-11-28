# DFlash2 精确架构规格（从 ink_metal_dflash2.h/.metal 逆向，训练用）

## 常量
H=2304, L=5, QH=32, KVH=8, HD=128, Q=4096, KV=1024, KV2=2048, FF=6912,
DYN=576, RANK=256, BLOCK=8, DRAFT=7, WIN=2048, RING=2112, TOPK=16,
MASK_ID=200056, ROPE_THETA=10000.0 (非 target 的 1e6!)

## 权重清单 (safetensors, bf16)
- fc.weight [2304, 9216]            — ctx 压缩: concat(tap0..3) -> c
- hidden_norm / norm (rmsnorm 权重)
- layers.{l}.attention_conv.kernel_projection.weight [576, 2304]
- layers.{l}.attention_conv.base_kernel [2,2,2304]?? (base_a)
- layers.{l}.mlp_conv.kernel_projection.weight [576, 2304], base_kernel (base_f)
- layers.{l}.self_attn.{q_proj.weight[4096,2304], o_proj.weight[2304,4096],
  q_norm, k_norm, k/v 打包 [2048,2304] (前1024=K, 后1024=V)}
- layers.{l}.mlp.{gate,up}.weight [6912,2304], down.weight [2304,6912]
- layers.{l}.{attn_norm, post_norm}
- candidate_selector.hidden_projection.weight [256, 2304]
- candidate_selector.predecessor_codebook [V, 256]
- candidate_selector.successor_codebook [V, 256]

## Ctx ingest (每个 target prefill token)
1. c = fc @ concat(taps4)  (9216->2304)
2. cn = rms_norm(c, hidden_norm)
3. 每层: kv = Wkv[l] @ cn  (2304->2048, 前1024 K 后1024 V)
4. k_norm + rope(theta=1e4, pos=token_pos) -> 写入 ring[l][pos % 2112]
   ring 滑窗 keep = 2112-16 = 2096

## Block-8 draft forward (一次并行出 7 草稿)
x[0] = rms_norm(embed[anchor])*embed_norm; x[1..7] = mask_row (mask token 占位)
每层 l:
  xn = rms_norm(x, attn_norm)
  dyn = kernel_proj_a @ xn            # [8,576]
  c = conv_pre(xn, dyn, base_a)       # 动态深度卷积, 沿 block 序列维, kernel size 2? (base [2,2,2304])
  q = Wq @ c  (q8)                    # [8,4096]
  kv = Wkv @ c (bf16)                 # [8,2048] block 内 KV
  q,k 各自 per-head rmsnorm + rope(anchor_pos + row offset?, theta 1e4)
  attn = 非因果 SWA-2048 32:8 GQA, attend 对象 = block 内 kv + ctx ring
         (split-K 8 份, fp32 部分和, 然后 reduce)
  z = Wo @ attn (q8)
  x = conv_post_add(x, z, dyn, base_a)   # 残差 + 动态卷积 post
  xn = rms_norm(x, post_norm)
  dyn = kernel_proj_f @ xn
  c = conv_pre(xn, dyn, base_f)
  g = Wg@c, u = Wu@c (q8); act = silu(g)*u
  z = Wd @ act (q8)
  x = conv_post_add(x, z, dyn, base_f)
head:
  hd = rms_norm(x, norm)
  每行 d=0..6: logits = hd[d] @ unemb_q8^T -> top16 (id,val) 候选
  gsel = sel_h @ hd  (2304->256)      # 每行 rank-256 投影
  select_chain: 在 16^? 候选中用 bilinear 打分选链:
    score 用 anchor_tok 与候选 pred/succ codebook 的 rank-256 双线性:
    predecessor: P[tok]·gsel, successor: S[tok]·gsel 链接约束
    输出 7 个 id + conf

## 训练要点 (自有化)
- 监督目标: row d 预测 target 在 anchor+d+1 位置的 token (并行块, 非自回归)
- ctx taps 来源复用 INK_TAPDUMP (distill_dataset.npz 已有 4 taps!)
- loss = CE(row d vs target token anchor+d+1) 均值, 可加 chain-consistency
- 量化训练: q8 矩阵在 forward 做 fake-quant ( STE ) 或先 bf16 训后 PTQ 校准
- conv 语义需读 .metal 的 df2_conv_pre/conv_post_add 确认 (kernel size, causal?)

## 已确认 kernel 细节 (ink_metal_dflash2.metal)
- conv: 2-tap 深度因果动态卷积, 每通道 base 权重 [4,2304], 动态分量按 16 通道分组 (DYN=576=2304/16×4 taps)
  conv_pre:  y[t] = w0*x[t] + w1*x[t-1]  (w0=base[0]+dyn[g], w1=base[1]+dyn[144+g])
  conv_post: x[t] += w2*z[t] + w3*z[t-1]  (dyn 偏移 288/432)
- rope: NeoX half-split (pair d,d+64), pos = pos0 + row_index, theta=1e4
- q/k norm: per-head rmsnorm (128 维), eps=DF2_EPS
- attention: 非因果 block 内 + ctx, cell 范围 [max(lo, pos0+t-2047), pos0+7], SWA 窗 2048
- select_chain: 逐行贪心 d=0..6:
    pg = pred[prev] * gsel[d+1]  (elementwise, rank-256)
    dt(c) = <pg, succ[c]>  (bilinear 链接得分)
    score(c) = raw_logit(c) + dt(c)
    取 top1 -> ids[d]; conf = softcap 归一化概率 (tail 用剩余质量近似)
    prev = ids[d]  (链式依赖, 非独立并行!)
- conf_pow 可调, verify 宽度由 ceiling.py MoE E_union 模型自适应 (1-7)

## 自有化改造自由度 (用户决策: 量化+训练+架构改造)
1. 量化: q_proj/o_proj/gate/up/down 已是 q8row; 可探索 sf4/mxfp4 + 校准;
   fc/wkv 是 bf16 (占大头), 量化到 q8 可省一半 ctx ingest 成本
2. 架构: 层数 5→更少/更多; conv kernel 2→3; RANK 256→128 (codebook 是大头: 2×V×256×2B ≈ 103MB×2)
3. 训练: 用 INK_TAPDUMP taps + distill_dataset 序列, CE on 7 行 + chain 一致性;
   关键: 必须对齐 SQ-10 量化 target 的真实 tap 分布 (与 bf16 target 有分布偏移 — 这是 native checkpoint 接受率崩塌的根因假设)
