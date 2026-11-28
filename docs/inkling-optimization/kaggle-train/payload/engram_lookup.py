#!/usr/bin/env python3
"""Engram lookup layer for grafting Hy4 tables onto Inkling.

Replicates exactly:
1. Hy4 CompressedTokenizer (NFKC/NFD/strip-accents/lower/space-collapse canonicalization,
   canonical ids assigned in donor vocab scan order)
2. Per-layer hash multipliers (seed + 10007*layer_id, odd randoms)
3. Prime table sizes from Hy4 config / ngram manifest
4. ngram-q lloyd-row-f16 decode for SSD-streamed table rows

Inkling adaptation: map inkling token -> normalized text -> hy4 canonical id when the
normalized text exists in hy4's canonical space; unmapped tokens use a reserved
'unknown' behavior (skip: contribute zero vector, matching Engram pad semantics)."""
import sys, json, unicodedata
sys.path.insert(0, '/Users/sonders/dageno-modelbench/train-venv/lib/python3.13/site-packages')
import numpy as np

def normalize(text):
    text = unicodedata.normalize('NFKC', text)
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = text.lower()
    text = ' '.join(text.split())
    return text

def build_hy4_canonical(tokenizer_json_path):
    """Replicate donor CompressedTokenizer: scan vocab in id order, assign canonical ids
    in first-occurrence order. Returns (hy4_tok2cid dict tokenstr->cid, key2cid, n_canonical)."""
    hy4j = json.load(open(tokenizer_json_path))
    vocab = hy4j['model']['vocab']  # tokenstr -> id
    ordered = sorted(vocab.items(), key=lambda x: x[1])
    key2cid = {}
    tok2cid = {}
    for tok, _id in ordered:
        key = normalize(tok.replace('\u0120', ' ')) or tok
        cid = key2cid.setdefault(key, len(key2cid))
        tok2cid[tok] = cid
    return tok2cid, key2cid, len(key2cid)

def build_inkling_to_hy4(inkling_vocab_path, hy4_key2cid):
    """inkling token id -> hy4 canonical id (or -1 if unmappable)"""
    ink = json.load(open(inkling_vocab_path))
    id2cid = np.full(max(ink.values()) + 1, -1, dtype=np.int32)
    for tok, tid in ink.items():
        key = normalize(tok.replace('\u0120', ' ')) or tok
        cid = hy4_key2cid.get(key, -1)
        id2cid[tid] = cid
    return id2cid

def layer_multipliers(layer_id, max_ngram=3, seed=0, n_canonical=88501):
    PRIME_1 = 10007
    max_long = np.iinfo(np.int64).max
    M_max = int(max_long // n_canonical)
    half_bound = max(1, M_max // 2)
    base_seed = int(seed + PRIME_1 * int(layer_id))
    g = np.random.default_rng(base_seed)
    r = g.integers(low=0, high=half_bound, size=(max_ngram,), dtype=np.int64)
    return r * 2 + 1

def ngram_hash(cids, multipliers, mod):
    """cids: compressed ids for positions [t-n+1..t]; multipliers per position.
    hash = xor of (cid * mult) mod table prime size."""
    mix = np.int64(0)
    for k, c in enumerate(cids):
        mix = np.bitwise_xor(mix, np.int64(c) * multipliers[k])
    return int(mix % np.int64(mod))

class NgramQTable:
    """lloyd-row-f16 table reader: random-access rows from a quantized shard."""
    def __init__(self, shard_path, rows, dim, bits=4):
        self.rows, self.dim, self.bits = rows, dim, bits
        import json as _json, struct as _struct
        bf = open(shard_path, 'rb')
        hlen = _struct.unpack('<Q', bf.read(8))[0]
        header = _json.loads(bf.read(hlen))
        self.bf = bf
        self.data_base = 8 + hlen
        self.header = header
        # tensor names inside shard
        names = list(k for k in header if k != '__metadata__')
        self.codes_name = [x for x in names if x.endswith('.codes')][0]
        self.scale_name = [x for x in names if x.endswith('.scale')][0]
        self.codebook_name = [x for x in names if x.endswith('.codebook')][0]
        self.row_bytes = (dim * bits + 7) // 8
        # load codebook once (tiny)
        info = header[self.codebook_name]
        bf.seek(self.data_base + info['data_offsets'][0])
        raw = bf.read(info['data_offsets'][1] - info['data_offsets'][0])
        self.codebook = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        self.codes_off = self.data_base + header[self.codes_name]['data_offsets'][0]
        self.scale_off = self.data_base + header[self.scale_name]['data_offsets'][0]

    def read_rows(self, row_ids):
        """row_ids: array of row indices. returns [n, dim] fp32 decoded."""
        row_ids = np.asarray(row_ids)
        out = np.zeros((len(row_ids), self.dim), dtype=np.float32)
        scales = np.zeros(len(row_ids), dtype=np.float32)
        for i, r in enumerate(row_ids):
            self.bf.seek(self.scale_off + int(r) * 2)
            scales[i] = np.frombuffer(self.bf.read(2), dtype=np.float16)[0]
        for i, r in enumerate(row_ids):
            if scales[i] == 0.0:
                continue
            self.bf.seek(self.codes_off + int(r) * self.row_bytes)
            raw = self.bf.read(self.row_bytes)
            b = np.frombuffer(raw, dtype=np.uint8)
            if self.bits == 4:
                lo = b & 0x0F
                hi = b >> 4
                codes = np.empty(len(b) * 2, dtype=np.uint8)
                codes[0::2] = lo
                codes[1::2] = hi
                codes = codes[:self.dim]
            else:
                raise NotImplementedError('5-bit unpack not yet needed')
            out[i] = scales[i] * self.codebook[codes.astype(np.int32)]
        return out

if __name__ == '__main__':
    # self-test: canonical build + remap coverage on sample tokens
    tok2cid, key2cid, ncanon = build_hy4_canonical('/tmp/hy4-tokenizer.json')
    print('hy4 canonical size:', ncanon)
    id2cid = build_inkling_to_hy4(
        '/Users/sonders/dageno-modelbench/ai.dageno.dageno-inkling.inkling/tokenizer/vocab.json',
        key2cid)
    mapped = int((id2cid >= 0).sum())
    print(f'inkling tokens mapped: {mapped}/{len(id2cid)}')
    # sample hash
    mults = layer_multipliers(2)
    print('layer2 multipliers:', mults)
    h = ngram_hash([100, 200], mults[:2], 10312007)
    print('sample bigram hash:', h)
