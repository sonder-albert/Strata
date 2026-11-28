"""Kaggle: DSpark block-8 draft distillation on SQ-10-collected taps.

Goal: the native dspark checkpoint (mtp.safetensors) was trained against bf16
target taps; on the SQ-10 pack its acceptance collapses (acc 0/124 measured).
Fix = fine-tune the SAME architecture on taps collected ON the SQ-10 engine
(INK_TAPDUMP + --tap-collect, see inkling_tap_collect.py).

Architecture is a faithful port of engine/src/inkling2/metal/ink_metal_dspark.metal:
  init:   row0 = rmsnorm(embed[anchor])*embed_norm ; rows1..7 = mask_embed
  layer x3: rmsnorm -> q/k/v (+ctx k/v from taps) -> qk per-head rmsnorm ->
            NeoX half-split rope(theta=1e6, pos=apos+row) ->
            non-causal 8x8 + ctx attention (ctx K already normed+roped at store
            time; SWA layers 0,2 keep last 1025 ctx) -> o -> +x -> rmsnorm ->
            swiglu(mlp) -> +x
  head:   rmsnorm(cur); base = unembed(cur[1..8] + hidden_out.bias)
  markov: for s=1..7: joint=[state|markov_w1[tok_{s-1}]|cur[s]|f_rnn(apos-1)]
          proj=rnn_proj(joint)+b; state'=cand+gate*(state-cand);
          latent=tanh(clamp(proj[512:],-15,15))
          logits_s = base[s-1] + MEff @ latent   (MEff = unembed @ hidden_out.w)
  loss:   CE(logits_s, gold=stream[apos+s]) with linear position decay.

Frozen: embed, unembed (shared target weights); trained: everything in mtp.
Muon for ndim>=2 params, AdamW for vectors (same dual-opt as engram trainer).
"""
from __future__ import annotations
import math, os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D, NH, NKV, HD, RANK, BLK, NDRAFT = 2304, 24, 8, 128, 256, 8, 7
JOINT = 512 + 2 * D  # 5120
SWA_LAYERS = {0, 2}
SWA_WIN = 1024 + 1  # engine: sliding_window_size + 1 -> keep c0 = apos-1024
ROPE_TH = 1e6
EPS = 1e-6


def rmsnorm(x, w, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_half(x, pos, theta=ROPE_TH):
    """x: [..., 128]; pos: [B,T] int -> NeoX half-split rope (d, d+64)."""
    d = x.shape[-1]
    half = d // 2
    freq = 1.0 / (theta ** (torch.arange(0, half, device=x.device, dtype=torch.float32) * 2.0 / d))
    ang = pos.unsqueeze(-1).to(torch.float32) * freq          # [B,T,half]
    cos, sin = ang.cos(), ang.sin()                            # broadcast to heads
    x0, x1 = x[..., :half], x[..., half:]
    return torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)


class DSpark(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse_weight = nn.Parameter(torch.zeros(4, 4))
        self.mask_embed = nn.Parameter(torch.zeros(D))
        self.norm = nn.Parameter(torch.ones(D))
        L = nn.Module
        self.layers = nn.ModuleList()
        for _ in range(3):
            l = L()
            l.dflare_norm = nn.Parameter(torch.ones(D))
            l.input_layernorm = nn.Parameter(torch.ones(D))
            l.post_attention_layernorm = nn.Parameter(torch.ones(D))
            l.q_proj = nn.Linear(D, 24 * HD, bias=False)
            l.k_proj = nn.Linear(D, NKV * HD, bias=False)
            l.v_proj = nn.Linear(D, NKV * HD, bias=False)
            l.k_proj_ctx = nn.Linear(D, NKV * HD, bias=False)
            l.v_proj_ctx = nn.Linear(D, NKV * HD, bias=False)
            l.o_proj = nn.Linear(24 * HD, D, bias=False)
            l.q_norm = nn.Parameter(torch.ones(HD))
            l.k_norm = nn.Parameter(torch.ones(HD))
            l.k_norm_ctx = nn.Parameter(torch.ones(HD))
            l.gate_proj = nn.Linear(D, 9216, bias=False)
            l.up_proj = nn.Linear(D, 9216, bias=False)
            l.down_proj = nn.Linear(9216, D, bias=False)
            self.layers.append(l)
        mh = L()
        mh.dflare_norm = nn.Parameter(torch.ones(D))
        mh.hidden_out = nn.Linear(RANK, D, bias=True)
        mh.markov_w1 = nn.Parameter(torch.zeros(201024, RANK))
        mh.rnn_proj = nn.Linear(JOINT, 768, bias=True)
        self.markov_head = mh

    def mix(self, taps4, row):
        """taps4 [B,T,4,D] -> rmsnorm(softmax(fuse_weight[row]) . taps)*dflare-ish
        (norm_w lives per-layer as dflare_norm; caller passes it)."""
        w = torch.softmax(self.fuse_weight[row], dim=0)          # [4]
        return (taps4 * w.view(1, 1, 4, 1)).sum(2)

    def ctx_kv(self, taps4, pos, li):
        """Build per-layer ctx K/V for positions pos [B,T]. K is rmsnormed per
        head with k_norm_ctx THEN roped at store time (cache semantics)."""
        l = self.layers[li]
        x = rmsnorm(self.mix(taps4, li), l.dflare_norm)
        k = l.k_proj_ctx(x).view(*x.shape[:-1], NKV, HD)
        v = l.v_proj_ctx(x).view(*x.shape[:-1], NKV, HD)
        k = rmsnorm(k, l.k_norm_ctx)
        k = rope_half(k, pos)
        return k, v                                          # [B,T,NKV,HD]

    def f_rnn_ctx(self, taps4):
        """context_vec for the markov joint = dflare(row3 mix) of the LAST
        committed position; the engine keeps it per slot. The stored value is
        rmsnormed with markov_head.dflare_norm (dspark_mix_taps with row=3 and
        the markov head's norm weight)."""
        return rmsnorm(self.mix(taps4, 3), self.markov_head.dflare_norm)

    def block_forward(self, anchor_tok, anchor_pos, taps4_ctx, pos_ctx,
                      embed, embed_norm):
        """anchor_tok [B]; taps4_ctx [B,C,4,D] already per-layer-mixed? No:
        raw taps; we build ctx k/v per layer inside. pos_ctx [B,C].
        Returns cur [B,8,D] (post final norm) and cur_pre [B,8,D]."""
        B = anchor_tok.shape[0]
        dev = anchor_tok.device
        cur = torch.empty(B, BLK, D, device=dev, dtype=torch.float32)
        cur[:, 0] = rmsnorm(embed[anchor_tok].float(), embed_norm)
        cur[:, 1:] = self.mask_embed
        for li, l in enumerate(self.layers):
            x = rmsnorm(cur, l.input_layernorm)
            q = l.q_proj(x).view(B, BLK, NH, HD)
            kd = l.k_proj(x).view(B, BLK, NKV, HD)
            vd = l.v_proj(x).view(B, BLK, NKV, HD)
            q = rmsnorm(q, l.q_norm)
            kd = rmsnorm(kd, l.k_norm)
            pos_blk = anchor_pos.view(B, 1, 1) + torch.arange(BLK, device=dev).view(1, BLK, 1)
            q = rope_half(q, pos_blk)
            kd = rope_half(kd, pos_blk)
            # ctx kv (built once per call; could hoist but cheap)
            kc, vc = self.ctx_kv(taps4_ctx, pos_ctx, li)
            if li in SWA_LAYERS:
                c0 = (anchor_pos - SWA_WIN + 1).clamp_min(0)   # engine: max(0, apos-1025+1)
                keep = pos_ctx >= c0.view(B, 1)               # [B,C]
            else:
                keep = torch.ones(B, taps4_ctx.shape[1], dtype=torch.bool, device=dev)
            # attention: non-causal block + ctx, GQA 3:1
            # keys [B, C+8, NKV, HD] with keep mask; values same
            kh_idx = torch.arange(NH, device=dev) // 3
            out = torch.zeros(B, BLK, NH, HD, device=dev)
            scale = 1.0 / math.sqrt(HD)
            # build per-b padded ctx tensors
            C = taps4_ctx.shape[1]
            kall = torch.cat([kc, kd], dim=1)                 # [B,C+8,NKV,HD]
            vall = torch.cat([vc, vd], dim=1)
            kmask = torch.cat([keep, torch.ones(B, BLK, dtype=torch.bool, device=dev)], 1)
            qf = (q * scale).permute(0, 2, 1, 3)              # [B,NH,8,HD]
            kk = kall.permute(0, 2, 1, 3)                     # [B,NKV,C+8,HD]
            vv = vall.permute(0, 2, 1, 3)
            kk = kk[:, kh_idx]                                # [B,NH,C+8,HD]
            vv = vv[:, kh_idx]
            att = torch.einsum('bhid,bhjd->bhij', qf, kk)     # [B,NH,8,C+8]
            att = att.masked_fill(~kmask.view(B, 1, 1, -1), float('-inf'))
            p = att.softmax(-1)
            out = torch.einsum('bhij,bhjd->bhid', p, vv).permute(0, 2, 1, 3)
            cur = cur + l.o_proj(out.reshape(B, BLK, NH * HD))
            x = rmsnorm(cur, l.post_attention_layernorm)
            h = F.silu(l.gate_proj(x)) * l.up_proj(x)
            cur = cur + l.down_proj(h)
        return rmsnorm(cur, self.norm)

    def markov_logits(self, cur_norm, ctx_vec, gold_prev, MEff,
                      unembed, hidden_out_bias):
        """7 teacher-forced steps. gold_prev [B,7] = stream[apos..apos+6]
        (anchor at index 0). Returns logits [B,7,V] for CE and latents."""
        B = cur_norm.shape[0]
        state = torch.zeros(B, RANK, device=cur_norm.device)
        outs = []
        for s in range(1, BLK):
            prev = gold_prev[:, s - 1]
            joint = torch.cat([state,
                               self.markov_head.markov_w1[prev],
                               cur_norm[:, s],
                               ctx_vec], dim=1)               # [B,5120]
            proj = self.markov_head.rnn_proj(joint)            # [B,768]
            p0, p1, p2 = proj[:, :256], proj[:, 256:512], proj[:, 512:]
            gate = torch.sigmoid(p0)
            cand = torch.tanh(p1.clamp(-15, 15))
            state = cand + gate * (state - cand)
            latent = torch.tanh(p2.clamp(-15, 15))
            base = (cur_norm[:, s:s + 1] + hidden_out_bias).squeeze(1) @ unembed.T.float()
            outs.append(base + latent @ MEff.T)                # [B,V]
        return torch.stack(outs, 1)                            # [B,7,V]


# ---- Muon (copied from train_engram_cloud.py: momentum orthogonalized by NS) ----
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-3, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))

    @staticmethod
    def zeropower_ns(G, steps):
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.to(torch.bfloat16)
        if G.size(-2) > G.size(-1):
            X = X.mT
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(-2) > G.size(-1):
            X = X.mT
        return X.to(G.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if 'momentum_buffer' not in st:
                    st['momentum_buffer'] = torch.zeros_like(g)
                buf = st['momentum_buffer']
                buf.mul_(group['momentum']).add_(g)
                g = g.add(buf, alpha=group['momentum']) if group['nesterov'] else buf
                if p.ndim >= 2:
                    upd = self.zeropower_ns(g, group['ns_steps'])
                    upd = upd * (max(1.0, p.size(-2) / p.size(-1)) ** 0.5)
                else:
                    upd = g
                if group['weight_decay']:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(upd, alpha=-group['lr'])


def load_mtp(path):
    """Load native dspark weights (mtp.safetensors, bf16) into DSpark."""
    from safetensors.torch import load_file
    sd = load_file(path)
    m = DSpark()
    m.fuse_weight.data.copy_(sd['fuse_weight'].float())
    m.mask_embed.data.copy_(sd['mask_embed'].float())
    m.norm.data.copy_(sd['norm.weight'].float())
    for i, l in enumerate(m.layers):
        p = f'layers.{i}.'
        for k in ('dflare_norm', 'input_layernorm', 'post_attention_layernorm'):
            getattr(l, k).data.copy_(sd[p + k + '.weight'].float())
        l.q_proj.weight.data.copy_(sd[p + 'self_attn.q_proj.weight'].float())
        l.k_proj.weight.data.copy_(sd[p + 'self_attn.k_proj.weight'].float())
        l.v_proj.weight.data.copy_(sd[p + 'self_attn.v_proj.weight'].float())
        l.k_proj_ctx.weight.data.copy_(sd[p + 'self_attn.k_proj_ctx.weight'].float())
        l.v_proj_ctx.weight.data.copy_(sd[p + 'self_attn.v_proj_ctx.weight'].float())
        l.o_proj.weight.data.copy_(sd[p + 'self_attn.o_proj.weight'].float())
        l.q_norm.data.copy_(sd[p + 'self_attn.q_norm.weight'].float())
        l.k_norm.data.copy_(sd[p + 'self_attn.k_norm.weight'].float())
        l.k_norm_ctx.data.copy_(sd[p + 'self_attn.k_norm_ctx.weight'].float())
        l.gate_proj.weight.data.copy_(sd[p + 'mlp.gate_proj.weight'].float())
        l.up_proj.weight.data.copy_(sd[p + 'mlp.up_proj.weight'].float())
        l.down_proj.weight.data.copy_(sd[p + 'mlp.down_proj.weight'].float())
    mh = m.markov_head
    mh.dflare_norm.data.copy_(sd['markov_head.dflare_norm.weight'].float())
    mh.hidden_out.weight.data.copy_(sd['markov_head.hidden_out.weight'].float())
    mh.hidden_out.bias.data.copy_(sd['markov_head.hidden_out.bias'].float())
    mh.markov_w1.data.copy_(sd['markov_head.markov_w1.weight'].float())
    mh.rnn_proj.weight.data.copy_(sd['markov_head.rnn_proj.weight'].float())
    mh.rnn_proj.bias.data.copy_(sd['markov_head.rnn_proj.bias'].float())
    return m


def export_mtp(m, path):
    """Write back bf16 safetensors with the original key names."""
    from safetensors.torch import save_file
    sd = {}
    sd['fuse_weight'] = m.fuse_weight.data.bfloat16()
    sd['mask_embed'] = m.mask_embed.data.bfloat16()
    sd['norm.weight'] = m.norm.data.bfloat16()
    for i, l in enumerate(m.layers):
        p = f'layers.{i}.'
        for k in ('dflare_norm', 'input_layernorm', 'post_attention_layernorm'):
            sd[p + k + '.weight'] = getattr(l, k).data.bfloat16()
        sd[p + 'self_attn.q_proj.weight'] = l.q_proj.weight.data.bfloat16()
        sd[p + 'self_attn.k_proj.weight'] = l.k_proj.weight.data.bfloat16()
        sd[p + 'self_attn.v_proj.weight'] = l.v_proj.weight.data.bfloat16()
        sd[p + 'self_attn.k_proj_ctx.weight'] = l.k_proj_ctx.weight.data.bfloat16()
        sd[p + 'self_attn.v_proj_ctx.weight'] = l.v_proj_ctx.weight.data.bfloat16()
        sd[p + 'self_attn.o_proj.weight'] = l.o_proj.weight.data.bfloat16()
        sd[p + 'self_attn.q_norm.weight'] = l.q_norm.data.bfloat16()
        sd[p + 'self_attn.k_norm.weight'] = l.k_norm.data.bfloat16()
        sd[p + 'self_attn.k_norm_ctx.weight'] = l.k_norm_ctx.data.bfloat16()
        sd[p + 'mlp.gate_proj.weight'] = l.gate_proj.weight.data.bfloat16()
        sd[p + 'mlp.up_proj.weight'] = l.up_proj.weight.data.bfloat16()
        sd[p + 'mlp.down_proj.weight'] = l.down_proj.weight.data.bfloat16()
    sd['markov_head.dflare_norm.weight'] = m.markov_head.dflare_norm.data.bfloat16()
    sd['markov_head.hidden_out.weight'] = m.markov_head.hidden_out.weight.data.bfloat16()
    sd['markov_head.hidden_out.bias'] = m.markov_head.hidden_out.bias.data.bfloat16()
    sd['markov_head.markov_w1.weight'] = m.markov_head.markov_w1.data.bfloat16()
    sd['markov_head.rnn_proj.weight'] = m.markov_head.rnn_proj.weight.data.bfloat16()
    sd['markov_head.rnn_proj.bias'] = m.markov_head.rnn_proj.bias.data.bfloat16()
    save_file(sd, path)


def main():
    WORK = os.environ.get('DSPARK_WORK', '/kaggle/working')
    DATA = os.environ.get('DSPARK_DATA', '/kaggle/input/inkling-sq10-distill-taps')
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- data ---
    npz = np.load(os.path.join(DATA, 'distill_windows.npz'))
    taps = npz['taps']            # [N,77,4,2304] fp32
    toks = npz['tokens']          # [N,77]
    gen = npz['is_generated']
    N, W = taps.shape[0], taps.shape[1]
    abspos = npz['pos'] if 'pos' in npz else np.tile(
        np.arange(W, dtype=np.int32), (N, 1))  # absolute stream positions
    print(f'[data] {N} windows x {W} pos', flush=True)

    # --- weights ---
    mtp = os.path.join(WORK, 'mtp.safetensors')
    if not os.path.exists(mtp):
        from huggingface_hub import hf_hub_download
        mtp = hf_hub_download('thinkingmachines/Inkling-2-Mini', 'mtp.safetensors')
    model = load_mtp(mtp).to(dev)

    # target embed + unembed rows needed: embed rows for anchors (~200k x 2304 is
    # 925MB bf16 - load once, it's the shared matrix anyway)
    st = os.path.join(WORK, 'model-00001-of-00001.safetensors')
    if not os.path.exists(st):
        from huggingface_hub import hf_hub_download
        st = hf_hub_download('thinkingmachines/Inkling-2-Mini',
                             'model-00001-of-00001.safetensors')
    from safetensors import safe_open
    with safe_open(st, framework='pt') as f:
        embed = f.get_tensor('model.llm.embed.weight').float().to(dev)
        unembed = f.get_tensor('model.llm.unembed.weight').float().to(dev)
        embed_norm = f.get_tensor('model.llm.embed_norm.weight').float().to(dev)
    # hidden_out.weight only ever enters logits through MEff, which the engine
    # precomputes ONCE at init. Keep it frozen: recompute would be a [V,256]
    # matmul per step, and the latent->correction map is already pretrained.
    model.markov_head.hidden_out.weight.requires_grad_(False)
    MEff = (unembed @ model.markov_head.hidden_out.weight.detach().float())

    # --- opts: Muon for matrices, AdamW for vectors ---
    mats, vecs = [], []
    for n_, p in model.named_parameters():
        (mats if p.ndim >= 2 else vecs).append(p)
    muon = Muon(mats, lr=1e-3, weight_decay=0.0)
    adamw = torch.optim.AdamW(vecs, lr=3e-4, weight_decay=0.0)

    # --- training ---
    EPOCHS = int(os.environ.get('EPOCHS', '2'))
    MAXSTEPS = int(os.environ.get('MAXSTEPS', '3000'))
    BATCH = int(os.environ.get('BATCH', '2'))
    rng = np.random.RandomState(1234)

    # valid anchors need a+8 <= W-1 (gold chain inside window). Any region -
    # taps are on-policy either way; generation-region anchors also carry the
    # response distribution the draft sees in production.
    anchors = [(w_, a_) for w_ in range(N) for a_ in range(0, W - 8)]
    print(f'[data] {len(anchors)} anchors')

    step = 0
    t0 = time.time()
    for ep in range(EPOCHS):
        rng.shuffle(anchors)
        for i in range(0, len(anchors) - BATCH + 1, BATCH):
            idx = anchors[i:i + BATCH]
            # gather
            Cs = [a_ + 1 for (_, a_) in idx]
            C = max(Cs)
            t4 = np.zeros((BATCH, C, 4, D), np.float32)
            pos_ctx = np.zeros((BATCH, C), np.int64)
            at = np.zeros(BATCH, np.int64)
            gp = np.zeros((BATCH, NDRAFT), np.int64)
            gl = np.zeros((BATCH, NDRAFT), np.int64)
            apos = np.zeros(BATCH, np.int64)
            for bi, (w_, a_) in enumerate(idx):
                t4[bi, :a_ + 1] = taps[w_, :a_ + 1]
                pos_ctx[bi, :a_ + 1] = abspos[w_, :a_ + 1]
                at[bi] = toks[w_, a_]
                gp[bi] = toks[w_, a_:a_ + 7]
                gl[bi] = toks[w_, a_ + 1:a_ + 8]
                apos[bi] = abspos[w_, a_]
            tt4 = torch.tensor(t4, device=dev)
            tpos = torch.tensor(pos_ctx, device=dev)
            tat = torch.tensor(at, device=dev)
            tgp = torch.tensor(gp, device=dev)
            tgl = torch.tensor(gl, device=dev)
            tapos = torch.tensor(apos, device=dev)

            cur_norm = model.block_forward(tat, tapos, tt4, tpos,
                                           embed, embed_norm)
            # ctx_vec = f_rnn of the anchor position = last committed row (a_)
            # build a per-batch slice [B,4,D] at each anchor's last ctx row
            anchor_taps = torch.stack([
                tt4[bi, Cs[bi] - 1] for bi in range(BATCH)])  # [B,4,D]
            ctx_vec = rmsnorm(
                (anchor_taps * torch.softmax(model.fuse_weight[3], 0)
                 .view(1, 4, 1)).sum(1), model.markov_head.dflare_norm)
            logits = model.markov_logits(cur_norm, ctx_vec, tgp,
                                         MEff, unembed, model.markov_head.hidden_out.bias)
            # position-decay CE (DFlash recipe): w_s = 1/(1+0.4 s)
            w = torch.tensor([1.0 / (1 + 0.4 * s) for s in range(7)], device=dev)
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 tgl.reshape(-1), reduction='none')
            loss = (ce.view(BATCH, 7) * w).sum() / w.sum() / BATCH
            muon.zero_grad(set_to_none=True)
            adamw.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            muon.step()
            adamw.step()
            step += 1
            if step % 20 == 0:
                # acceptance proxy: greedy prefix agreement
                pred = logits.argmax(-1)
                acc = (pred == tgl).float()
                run = (acc.cumprod(1) * acc).sum(1).mean()
                print(f'[step {step}] loss {loss.item():.4f} acc {acc.mean().item():.3f} '
                      f'run {run.item():.2f}/7 ({time.time()-t0:.0f}s)', flush=True)
            if step % 500 == 0:
                export_mtp(model, os.path.join(WORK, 'mtp_ft.safetensors'))
            if step >= MAXSTEPS:
                break
        if step >= MAXSTEPS:
            break
    export_mtp(model, os.path.join(WORK, 'mtp_ft.safetensors'))
    print('[done] wrote mtp_ft.safetensors', flush=True)


if __name__ == '__main__':
    main()
