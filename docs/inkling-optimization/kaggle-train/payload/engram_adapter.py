#!/usr/bin/env python3
"""Engram grafting adapter for Inkling: table lookup + gating + shortconv.

Design verified against deepseek Engram demo (/tmp/engram_demo.py) and Hy4 manifest:
- 16 tables per donor layer: heads 0-7 bigram, heads 8-15 trigram, dim 96 each
- concat embeddings = 2*8*96 = 1536 dims
- hash: XOR of (canonical_id * odd_multiplier) per ngram position, mod per-head prime
- pad semantics: unmappable inkling tokens contribute zero rows

Adaptation choices for Inkling (hidden 2304, single stream, no hyper-connections):
- donor gates per hc stream; inkling single-stream -> single gate path
- trainable: key_proj, value_proj (1536->2304 each), two RMSNorms, depthwise
  shortconv k=4, out_scale zero-init so insertion is identity at step 0
- backbone frozen; loss = next-token CE replayed through frozen inkling forward
  with injection at insertion layer; perplexity recovery curve is acceptance metric
"""
import sys, json
sys.path.insert(0, '/Users/sonders/dageno-modelbench/train-venv/lib/python3.13/site-packages')
import numpy as np

sys.path.insert(0, '/Users/sonders/dageno-modelbench')
from engram_lookup import NgramQTable, layer_multipliers, ngram_hash

D_HID = 2304
E_DIM = 1536  # 16 heads x 96

class EngramTables:
    """Lazy SSD-streamed access to one donor layer's 16 quantized tables."""
    def __init__(self, qdir, donor_layer):
        m = json.load(open(f'{qdir}/ngram_manifest.json'))
        self.tables = {}
        self.primes = {}
        for t in m['tables']:
            if t['layer'] != donor_layer:
                continue
            key = (t['ngram_order'], t['head'] % 8)  # manifest numbers trigram heads 8-15
            self.primes[key] = t['rows']
            self.tables[key] = dict(shard=f"{qdir}/{t['shard']}", rows=t['rows'], dim=t['dim'])
        self.handles = {}
        assert len(self.tables) == 16, f"expected 16 tables for layer {donor_layer}, got {len(self.tables)}"

    def handle(self, key):
        if key not in self.handles:
            t = self.tables[key]
            self.handles[key] = NgramQTable(t['shard'], t['rows'], t['dim'], bits=4)
        return self.handles[key]

    def lookup(self, cids_window):
        """cids_window: [T, 3] canonical ids for positions (t-2, t-1, t) per row.
        Returns [T, 1536] fp32 concatenated rows (bigram heads then trigram heads).
        Windows containing -1 (unmappable) contribute zeros."""
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
            for head in range(8):
                key = (order, head)
                mod = self.primes[key]
                m = mults[order][:order]
                rows = np.empty(len(idxs), dtype=np.int64)
                for i, w in enumerate(span_v):
                    rows[i] = ngram_hash(list(w), m, mod)
                vecs = self.handle(key).read_rows(rows)
                out[idxs, (order-2)*8 + head, :] = vecs
        return out.reshape(T, E_DIM)

    def bind_multipliers(self, donor_layer_id_for_seed):
        """Multipliers depend on donor layer id (seed + 10007*layer_id).
        Manifest zero-based 'layer' vs donor one-based layer_ids is unresolved;
        bind explicitly, validate both in perplexity test."""
        lm = layer_multipliers(donor_layer_id_for_seed)
        self._mults = {2: lm[:2], 3: lm[:3]}


def cids_windows(id2cid, token_ids, pad_cid=0):
    """Build [T,3] canonical-id windows, left-padded with pad_cid.
    Unmappable inkling tokens (id2cid=-1) stay -1 -> zero rows."""
    tok = np.asarray(token_ids, dtype=np.int64)
    cids = id2cid[tok]
    T = len(cids)
    win = np.full((T, 3), pad_cid, dtype=np.int64)
    for t in range(T):
        lo = max(0, t - 2)
        seg = cids[lo:t+1]
        win[t, 3-len(seg):] = seg
    return win


class EngramAdapterNP:
    """Numpy forward reference for validation. Training uses MLX port."""
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.key_proj = rng.normal(0, 0.02, (E_DIM, D_HID)).astype(np.float32)
        self.value_proj = rng.normal(0, 0.02, (E_DIM, D_HID)).astype(np.float32)
        self.norm_k = np.ones(D_HID, dtype=np.float32)
        self.norm_q = np.ones(D_HID, dtype=np.float32)
        self.conv_w = np.zeros((D_HID, 4), dtype=np.float32)
        self.conv_w[:, -1] = 1.0  # identity init
        self.out_scale = np.float32(0.0)  # zero-init: insertion is identity

    @staticmethod
    def rmsnorm(x, w, eps=1e-6):
        ms = np.mean(x * x, axis=-1, keepdims=True)
        return x / np.sqrt(ms + eps) * w

    def forward(self, hidden, embeds):
        """hidden: [T, 2304] at insertion layer input; embeds: [T, 1536].
        Returns injection delta [T, 2304]."""
        k = embeds @ self.key_proj
        nk = self.rmsnorm(k, self.norm_k)
        nq = self.rmsnorm(hidden, self.norm_q)
        gate = (nk * nq).sum(-1) / np.sqrt(D_HID)
        gate = np.sqrt(np.abs(gate) + 1e-6) * np.sign(gate)  # signed-sqrt
        gate = 1.0 / (1.0 + np.exp(-gate))
        v = gate[:, None] * (embeds @ self.value_proj)
        T = v.shape[0]
        vp = np.pad(v, ((3, 0), (0, 0)))
        cv = np.zeros_like(v)
        for t in range(T):
            cv[t] = (vp[t:t+4] * self.conv_w[:, ::-1].T).sum(axis=0)
        return self.out_scale * (v + cv)


if __name__ == '__main__':
    qdir = '/Users/sonders/dageno-modelbench/engram-hy4-q4'
    tables = EngramTables(qdir, donor_layer=1)
    print('num table keys:', len(tables.primes))
    print('bigram rows sample:', tables.primes[(2, 0)])
    print('trigram rows sample:', tables.primes.get((3, 0), 'MISSING - still downloading'))
    tables.bind_multipliers(2)
    print('multipliers bigram:', tables._mults[2])
    ad = EngramAdapterNP()
    h = np.random.randn(8, D_HID).astype(np.float32) * 0.1
    e = np.random.randn(8, E_DIM).astype(np.float32) * 0.1
    delta = ad.forward(h, e)
    print('adapter forward ok, delta shape', delta.shape, 'zero-init max abs:', np.abs(delta).max())

# ---------------------------------------------------------------------------
# Training design note (recorded 2026-11-27):
# Exact gradient for adapter params requires downstream Jacobian dl logits/d hidden
# at insertion layer. Backbone is frozen but x_ins DEPENDS on adapter params, so
# backprop must flow through frozen layers 12..end. Two viable implementations:
#   A) full MLX re-implementation of inkling forward (48 visits, looped pass 2-3x,
#      MoE 48-visit routing, attn-res) - exact but large engineering surface
#   B) single-pass finite-difference Jacobian probe per batch: run frozen forward
#      twice with perturbed injection, fit linear J locally - approximate but
#      unbiased-ish for small deltas; cheap at seq 256
# Validation metric regardless of training path: perplexity on held-out tap spans
# must be UNCHANGED at init (out_scale=0 identity check) and RECOVER below
# baseline as training proceeds. Seed binding (layer_id 1 vs 2) decided by
# 200-step smoke runs under identical data; winner = lower CE.
# Memory gate: requires exclusive weight access - run only in service windows
# when collection is paused, or after collection completes.
