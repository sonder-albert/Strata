#!/usr/bin/env python3
"""Serving-path engram integration simulator: validates the re-tokenize design
end to end on real generation text, measuring what production integration will cost.

Pipeline per generated token (as in production serving):
1. text accumulates; donor tokenizer re-tokenizes incrementally
   (production: rust tokenizers on the new suffix only - here simulated with
   full re-tokenize per measurement batch, correctness equivalent)
2. donor ids -> canonical ids (static table lookup)
3. bigram/trigram hashes per position -> row indices per head
4. rows fetched: bigram rows from simulated RAM LRU cache (34% dispersion means
   high hit rate), trigram rows from async prefetch queue (indices known 1-2
   tokens ahead, hidden latency)
Outputs: cache hit rates, prefetch coverage, rows/sec bandwidth requirements.
"""
import sys, json, time
sys.path.insert(0, '/Users/sonders/dageno-modelbench/train-venv/lib/python3.13/site-packages')
sys.path.insert(0, '/Users/sonders/dageno-modelbench/ms-venv/lib/python3.13/site-packages')
sys.path.insert(0, '/Users/sonders/dageno-modelbench')
import numpy as np
from collections import OrderedDict
from tokenizers import Tokenizer
from engram_lookup import build_hy4_canonical, layer_multipliers

DIMS = 96
BYTES_PER_ROW = 50  # 4bit packed 96 dim + scale

class LRU:
    def __init__(self, cap):
        self.cap = cap
        self.d = OrderedDict()
        self.hits = 0
        self.misses = 0
    def access(self, key):
        if key in self.d:
            self.hits += 1
            self.d.move_to_end(key)
            return True
        self.misses += 1
        self.d[key] = 1
        self.d.move_to_end(key)
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
        return False

def build_hy4_id2cid():
    import unicodedata
    def normalize(text):
        text = unicodedata.normalize('NFKC', text)
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        text = text.lower()
        return ' '.join(text.split())
    j = json.load(open('/tmp/hy4-tokenizer.json'))
    vocab = j['model']['vocab']
    added = {t['content']: t['id'] for t in j.get('added_tokens', [])}
    id2tok = {}
    for tok, tid in vocab.items(): id2tok[tid] = tok
    for tok, tid in added.items(): id2tok.setdefault(tid, tok)
    key2c = {}
    m = np.full(max(id2tok) + 1, -1, dtype=np.int32)
    for tid in sorted(id2tok):
        tok = id2tok[tid]
        key = normalize(tok.replace('Ġ', ' ')) or tok
        cid = key2c.setdefault(key, len(key2c))
        m[tid] = cid
    return m

def ngram_hash_batch(cids, mults, mod):
    mix = np.int64(0)
    for k, c in enumerate(cids):
        mix = np.bitwise_xor(mix, np.int64(int(c)) * mults[k])
    return int(mix % np.int64(mod))

def main():
    hy4t = Tokenizer.from_file('/tmp/hy4-tokenizer.json')
    id2cid = build_hy4_id2cid()
    mults = layer_multipliers(2)  # donor layer 2 = our layer-1 shards
    primes_bi = [10312007, 10312021, 10312039, 10312069, 10312091, 10312121, 10312123, 10312147]
    primes_tri = [10312177, 10312187, 10312207, 10312213, 10312229, 10312241, 10312249, 10312259]

    # reconstruct generation text from tap stream tokens (inkling ids -> text via hf tokenizer)
    PER_ROW = 8 + 4 * 2304 * 4
    raw = open('/tmp/tapd0.bin', 'rb').read(PER_ROW * 800)
    rows = len(raw) // PER_ROW
    arr = np.frombuffer(raw[:rows * PER_ROW], dtype=np.uint8).reshape(rows, PER_ROW)
    inkt = Tokenizer.from_file('/Users/sonders/dageno-modelbench/strata-tok/tokenizer.hf.json')
    toks = arr[:, 4:8].copy().view(np.int32).reshape(-1)
    # decode pieces individually to reconstruct raw text approximately
    text = inkt.decode([int(t) for t in toks[:1200] if 0 <= t < 200000])
    print(f'reconstructed text: {len(text)} chars from tap stream')
    print('sample:', text[:150].replace('\n', ' '))

    # donor re-tokenize
    h_ids = hy4t.encode(text).ids
    cids = id2cid[np.array(h_ids, dtype=np.int64)]
    print(f'donor tokens: {len(h_ids)}, canonical mapped: {(cids>=0).mean()*100:.1f}%')

    # simulate serving: token-by-token, bigram LRU + trigram prefetch
    for cache_cap in (2048, 8192, 32768):
        lru_bi = [LRU(cache_cap // 8) for _ in range(8)]
        lru_tri = [LRU(cache_cap // 8) for _ in range(8)]
        n = len(cids)
        for i in range(1, n):
            if cids[i] < 0 or cids[i-1] < 0: continue
            for h in range(8):
                r = ngram_hash_batch([cids[i-1], cids[i]], mults[:2], primes_bi[h])
                lru_bi[h].access(('b', h, r))
            if i >= 2 and cids[i-2] >= 0:
                for h in range(8):
                    r = ngram_hash_batch([cids[i-2], cids[i-1], cids[i]], mults[:3], primes_tri[h])
                    lru_tri[h].access(('t', h, r))
        hit_bi = sum(l.hits for l in lru_bi) / max(1, sum(l.hits + l.misses for l in lru_bi))
        hit_tri = sum(l.hits for l in lru_tri) / max(1, sum(l.hits + l.misses for l in lru_tri))
        uniq_rows = sum(len(l.d) for l in lru_bi + lru_tri)
        print(f'cache cap {cache_cap}: bigram hit {hit_bi*100:.0f}%, trigram hit {hit_tri*100:.0f}%, '
              f'unique rows resident {uniq_rows} (~{uniq_rows*BYTES_PER_ROW/1024:.0f}KB)')

if __name__ == '__main__':
    main()
