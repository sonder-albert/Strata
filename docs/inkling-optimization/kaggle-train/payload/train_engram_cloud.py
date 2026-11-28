"""Engram graft adaptation training on cloud GPU (H100/A100 80GB).

Mirrors engram_graft_train.py logic, on transformers' native inkling2_loop:
- backbone: bf16-converted checkpoint (run convert_nvfp4_to_bf16.py first), FROZEN
- insertion: after layer 3 forward in the pre-loop region
- trainable: key_proj/value_proj/norms/conv/out_scale (~7.1M), zero-init out side
- optimizer: Muon for matrices, AdamW for vectors/scalars
- loss: next-token CE through the frozen downstream (real autograd, no patches)
- acceptance: seed-binding contrast (layer 2 = correct donor seed, layer 5 = control)
  plus heldout with/without engram perplexity gap

Data: engram needs natural diverse text; default fineweb-style jsonl via --data.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D_HID = 2304
E_DIM = 1536
INSERT_AFTER_LAYER = 3


def log(*a):
    print(*a, flush=True)


class EngramAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_proj = nn.Parameter(torch.randn(E_DIM, D_HID) * 0.02)
        self.value_proj = nn.Parameter(torch.randn(E_DIM, D_HID) * 0.02)
        self.norm_k = nn.Parameter(torch.ones(D_HID))
        self.norm_q = nn.Parameter(torch.ones(D_HID))
        conv = torch.zeros(D_HID, 4)
        conv[:, -1] = 1.0
        self.conv_w = nn.Parameter(conv)
        self.out_scale = nn.Parameter(torch.zeros(1))

    @staticmethod
    def rmsnorm(x, w, eps=1e-6):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w

    def forward(self, h, embeds):
        """h: [B,T,2304], embeds: [B,T,1536] -> delta [B,T,2304]"""
        k = embeds @ self.key_proj
        nk = self.rmsnorm(k, self.norm_k)
        nq = self.rmsnorm(h, self.norm_q)
        gate = (nk * nq).sum(-1) / math.sqrt(D_HID)
        gate = torch.sqrt(gate.abs() + 1e-6) * gate.sign()
        gate = torch.sigmoid(gate)
        v = gate[..., None] * (embeds @ self.value_proj)
        T = v.shape[1]
        vp = F.pad(v, (0, 0, 3, 0))
        cv = sum(vp[:, 3 - i:3 - i + T] * self.conv_w[:, i][None, None, :] for i in range(4))
        return self.out_scale * (v + cv)


class Muon(torch.optim.Optimizer):
    """Muon optimizer (momentum orthogonalized by Newton-Schulz iteration)."""

    def __init__(self, params, lr=3e-3, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

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
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=group['momentum'])
                else:
                    g = buf
                if p.ndim >= 2:
                    upd = self.zeropower_ns(g, group['ns_steps'])
                    scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                    upd = upd * scale
                else:
                    upd = g
                if group['weight_decay']:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(upd, alpha=-group['lr'])


class EngramTablesFp16:
    """memmap-backed fp16 table reader for tables built by build_engram_tables_fp16.py.
    Same lookup semantics as engram_adapter.EngramTables (cloud-native, no q4 needed)."""
    def __init__(self, table_dir, donor_layer=1):
        import json as _json
        man = _json.load(open(f'{table_dir}/fp16_manifest.json'))
        self.tables = {}
        self.primes = {}
        for t in man['tables']:
            key = (t['ngram_order'], t['head'])
            self.primes[key] = t['rows']
            self.tables[key] = np.memmap(f"{table_dir}/{t['file']}", dtype=np.float16,
                                         mode='r', shape=(t['rows'], 96))
        assert len(self.tables) == 16, f'need 16 tables, got {len(self.tables)}'

    def lookup(self, cids_window):
        """cids_window: [T,3] canonical ids. returns [T,1536] fp32."""
        T = len(cids_window)
        out = np.zeros((T, 16, 96), dtype=np.float32)
        mults = self._mults
        for order in (2, 3):
            span = cids_window[:, 3-order:]
            valid = (span >= 0).all(axis=1)
            if not valid.any():
                continue
            idxs = np.nonzero(valid)[0]
            span_v = span[idxs]
            m = mults[order][:order]
            for head in range(8):
                key = (order, head)
                mod = self.primes[key]
                rows = np.empty(len(idxs), dtype=np.int64)
                for i, w in enumerate(span_v):
                    mix = np.int64(0)
                    for k, c in enumerate(w):
                        mix = np.bitwise_xor(mix, np.int64(c) * m[k])
                    rows[i] = int(mix % np.int64(mod))
                vecs = self.tables[key][rows].astype(np.float32)
                out[idxs, (order-2)*8 + head, :] = vecs
        return out.reshape(T, 1536)

    def bind_multipliers(self, layer_id):
        def layer_multipliers(layer_id, max_ngram=3, seed=0, n_canonical=88501):
            PRIME_1 = 10007
            max_long = np.iinfo(np.int64).max
            M_max = int(max_long // n_canonical)
            half_bound = max(1, M_max // 2)
            g = np.random.default_rng(int(seed + PRIME_1 * int(layer_id)))
            r = g.integers(low=0, high=half_bound, size=(max_ngram,), dtype=np.int64)
            return r * 2 + 1
        lm = layer_multipliers(layer_id)
        self._mults = {2: lm[:2], 3: lm[:3]}


def load_engram_tables(table_dir, donor_layer=1, device='cuda'):
    if os.path.exists(f'{table_dir}/fp16_manifest.json'):
        return EngramTablesFp16(table_dir, donor_layer)
    sys_path = os.path.dirname(os.path.abspath(__file__))
    if sys_path not in os.sys.path:
        os.sys.path.insert(0, sys_path)
    from engram_adapter import EngramTables
    return EngramTables(table_dir, donor_layer=donor_layer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, help='bf16-converted checkpoint dir')
    ap.add_argument('--tables', default='/data/engram-hy4-q4')
    ap.add_argument('--tokenizer', default='/data/hy4-tokenizer.json')
    ap.add_argument('--data', required=True, help='jsonl with "text" field')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--seq-len', type=int, default=512)
    ap.add_argument('--batchsize', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seed-layer', type=int, default=2)
    ap.add_argument('--out', default='engram_adapter_cloud.npz')
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--quant4', action='store_true',
                    help='load backbone NF4 quantized (fits T4 16GB; backbone is frozen so grads unaffected)')
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoConfig
    # tensor-core dtype: T4/P100 (cc<8) have NO bf16 units - bf16 there is
    # cuda-core emulation, ~10x slower. fp16 hits the 65 TFLOPS tensor cores.
    cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    compute_dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
    log(f'cuda cc{cc[0]}.{cc[1]} -> compute dtype {compute_dtype}')
    log('loading backbone...')
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(args.checkpoint)
    if args.quant4:
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype,
                                     bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.checkpoint, quantization_config=bnb, device_map='auto'
            )
            log('backbone loaded NF4-quantized across available GPUs')
        except Exception as e:
            log(f'NF4 load failed ({e}); falling back to {compute_dtype} single-GPU')
            model = AutoModelForCausalLM.from_pretrained(
                args.checkpoint, torch_dtype=compute_dtype, device_map='cuda'
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint, torch_dtype=compute_dtype, device_map='cuda'
        )
    for p in model.parameters():
        p.requires_grad_(False)
    log(f'backbone loaded in {time.time()-t0:.0f}s')

    # transformers maps model.llm.* -> model.language_model.* on load
    inner = model.model if hasattr(model, 'model') else model
    text_model = getattr(inner, 'language_model', None) or getattr(inner, 'llm', None) or inner
    layers = text_model.layers
    log(f'{len(layers)} layers; injecting after layer {INSERT_AFTER_LAYER}')

    # adapter must live where layer INSERT_AFTER_LAYER emits activations:
    # early layers sit on the first GPU under device_map='auto' (cuda:0).
    first_dev = next(model.parameters()).device
    adapter = EngramAdapter().to(first_dev).float()
    mat_params = [p for p in adapter.parameters() if p.ndim >= 2]
    vec_params = [p for p in adapter.parameters() if p.ndim < 2]
    muon = Muon(mat_params, lr=args.lr)
    adamw = torch.optim.AdamW(vec_params, lr=args.lr * 0.1)

    tables = load_engram_tables(args.tables)
    tables.bind_multipliers(args.seed_layer)

    from tokenizers import Tokenizer
    hy4t = Tokenizer.from_file(args.tokenizer)
    # id mapping table must be supplied on cloud too (engram_serving_sim artifact)
    id2cid = np.load('/data/hy4_id2cid.npy') if os.path.exists('/data/hy4_id2cid.npy') else None
    assert id2cid is not None, 'copy hy4_id2cid.npy to the instance (built by engram_serving_sim)'

    texts = []
    with open(args.data) as f:
        for line in f:
            try:
                texts.append(json.loads(line)['text'])
            except Exception:
                texts.append(line.strip())
    log(f'{len(texts)} training texts')

    log('tokenizing...')
    samples = []
    from transformers import AutoTokenizer
    inkt = AutoTokenizer.from_pretrained(args.checkpoint)
    for text in texts:
        enc = inkt(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc['input_ids'][:args.seq_len + 1]
        if len(ids) < 33:
            continue
        offs = enc['offset_mapping'][:args.seq_len + 1]
        eh = hy4t.encode(text)
        cids = id2cid[np.array(eh.ids, dtype=np.int64)]
        h_ends = {e: i for i, (s, e) in enumerate(eh.offsets)}
        pos_map = []
        last = -1
        for s, e in offs:
            j = h_ends.get(e)
            if j is not None:
                last = j
            pos_map.append(last)
        samples.append((ids, cids, pos_map))
        if len(samples) >= args.batchsize * 200:
            break
    log(f'{len(samples)} samples ready')

    def aligned_emb(cids_donor, pos_map):
        T = len(pos_map)
        out = np.zeros((T, 3), dtype=np.int64)
        for t in range(T):
            j = pos_map[t]
            if j < 0:
                out[t] = -1
                continue
            for kk in range(3):
                jj = j - (2 - kk)
                out[t, kk] = cids_donor[jj] if 0 <= jj < len(cids_donor) else 0
        emb = tables.lookup(out).astype(np.float32)
        emb[np.array(pos_map) < 0] = 0.0
        return emb

    def forward_with_injection(ids, embs):
        """Replicates text_model.forward pre-loop region with injection after layer 3."""
        raise NotImplementedError('subclass hook - see train loop')

    # The injection uses a forward hook on layer INSERT_AFTER_LAYER.
    injected = {}
    def hook(module, inputs, output):
        istuple = isinstance(output, tuple)
        h = output[0] if istuple else output
        emb = injected.get('emb')
        if emb is not None:
            if emb.device != h.device:
                emb = emb.to(h.device)
                injected['emb'] = emb
            h = h + adapter(h.float(), emb).to(h.dtype)
        return (h,) + output[1:] if istuple else h

    handle = layers[INSERT_AFTER_LAYER].register_forward_hook(hook)

    # industrial loss path: fused linear+CE (liger), softcap 30 faithful to the model.
    # Naive path materializes [B,T,V] fp32 logits *twice* (model + cast) - tens of GB
    # of memory traffic per step on a 160k vocab. FLCE chunks it with no materialization.
    flce = None
    base = None
    lm_w = None
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        try:
            flce = LigerFusedLinearCrossEntropyLoss(softcap=30.0)
        except TypeError:
            flce = LigerFusedLinearCrossEntropyLoss()
            log('WARNING: liger build lacks softcap - loss objective differs from model math!')
        base = text_model  # decoder stack without lm_head
        lm_w = model.get_output_embeddings().weight
        log('liger FLCE active (chunked, no logit materialization)')
    except Exception as e:
        log(f'liger unavailable ({e}); using naive fp32-logits CE - expect slower, more memory')

    tlosses = []
    log('training...')
    rng = np.random.default_rng(0)
    for step in range(args.steps):
        batch = [samples[rng.integers(len(samples))] for _ in range(args.batchsize)]
        maxlen = max(len(s[0]) for s in batch)
        ids = np.full((len(batch), maxlen), inkt.pad_token_id or 0, dtype=np.int64)
        emb_batch = np.zeros((len(batch), maxlen, E_DIM), dtype=np.float32)
        for i, (s_ids, cids, pos_map) in enumerate(batch):
            ids[i, :len(s_ids)] = s_ids
            e = aligned_emb(cids, pos_map)
            emb_batch[i, :len(s_ids)] = e[:len(s_ids)]
        x = torch.tensor(ids, device=first_dev)
        emb = torch.tensor(emb_batch, device=first_dev)
        injected['emb'] = emb
        tgt = x[:, 1:]
        pad = inkt.pad_token_id or 0
        if flce is not None:
            hidden = base(input_ids=x).last_hidden_state
            shift_h = hidden[:, :-1].contiguous()
            shift_t = tgt.clone()
            shift_t[shift_t == pad] = -100
            loss = flce(lm_w, shift_h, shift_t)
        else:
            out = model(input_ids=x)
            logits = out.logits.float()
            # logit softcap 30 as in the model's own tail
            capped = 30.0 * torch.tanh(logits / 30.0)
            logp = capped - torch.logsumexp(capped, dim=-1, keepdim=True)
            lp = logp[:, :-1]
            tok = -torch.gather(lp, -1, tgt[..., None]).squeeze(-1)
            valid = (tgt != pad).float()
            loss = (tok * valid).sum() / valid.sum().clamp(min=1)
        loss.backward()
        muon.step(); muon.zero_grad()
        adamw.step(); adamw.zero_grad()
        tlosses.append(float(loss))
        if step % 20 == 0:
            log(f'step {step}: loss {float(loss):.4f} (mean20 {np.mean(tlosses[-20:]):.4f})')
        if step % args.eval_every == 0 and step > 0:
            torch.save({k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                       f'engram_adapter_step{step}.pt')
            log(f'checkpoint step {step} saved')

    handle.remove()
    torch.save({k: v.detach().cpu() for k, v in adapter.state_dict().items()}, args.out)
    log(f'final loss curve: {tlosses[0]:.3f} -> {np.mean(tlosses[-20:]):.3f}; adapter saved to {args.out}')


if __name__ == '__main__':
    main()
