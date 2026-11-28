"""tools/inkling_tap_collect.py - on-policy INK_TAPDUMP collection driver for draft distill.

WHY: DSpark/DFlash draft checkpoints trained against bf16 taps collapse on the SQ-10
quantized target (acceptance ~0/124 measured). The fix is on-policy distillation:
collect the *target's own* taps on the *quantized* pack. `INK_TAPDUMP=<file>` makes the
metal runner append, per forward row: int32 pos + int32 tok + 4 x D fp32 taps
(visits {5,23,41} of b_cur + final b_h_norm), for both prefill and decode rows.

This script:
  1. builds token-id sequences (fineweb streaming corpus -> raw chunks, or chat-shaped
     prompts via the pack's chat template) using strata_tokenizer.py,
  2. invokes `strata-inkling-metal --tap-collect ids.txt --tap-meta meta.txt --gen N
     --draft off` with INK_TAPDUMP set (one engine process, all sequences),
  3. parses the binary dump + meta and emits `distill_windows.npz` with windows of
     W consecutive positions: taps[W,4,D] fp32, tokens[W] int32, is_generated[W] bool.

SAFETY: run a single engine process only - never concurrently with the serving engine
(:8392 holds ~10GB locked). Check `sysctl vm.swapusage` first.

Usage:
  python3 inkling_tap_collect.py --seqs 4 --gen 64 --prompt-tokens 128 \
      --out /tmp/taps --bin ~/dageno-modelbench/strata-ink67-fp8/strata-inkling-metal \
      --model ~/dageno-modelbench/sq10.strata --tok ~/dageno-modelbench/tok_inkling
"""
from __future__ import annotations
import argparse, io, json, os, struct, subprocess, sys, urllib.request
import numpy as np
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import strata_tokenizer as ST

D = 2304            # hidden size; taps = 4 x D fp32
ROW_BYTES = 4 + 4 + 4 * D * 4
BOS = 200028        # <|begin_of_text|>  (add_bos_token=false -> we add it)
EOS = 200006


def load_tokenizer(tok_dir: str) -> ST.Tokenizer:
    """Same load path as terminal/serve/server.py (vocab.json + merges.txt + token_type.json)."""
    p = Path(tok_dir)
    vocab = json.loads((p / "vocab.json").read_text(encoding="utf-8"))
    tokens: list[str | None] = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (p / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((p / "token_type.json").read_text())
    cfg = {}
    try:
        cfg = json.loads((p / "tokenizer.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return ST.Tokenizer(tokens, merges, types, cfg.get("pre") or "qwen35", cfg.get("special_ids") or None)


def _system_proxy() -> str | None:
    p = os.environ.get("https_proxy")
    if p or sys.platform != "darwin":
        return p
    try:
        sc = subprocess.run(["scutil", "--proxy"], capture_output=True, text=True, timeout=5).stdout
        host = port = None
        for line in sc.splitlines():
            s = line.split()
            if len(s) >= 3 and s[0] == "HTTPSProxy":
                host = s[2]
            if len(s) >= 3 and s[0] == "HTTPSPort":
                port = s[2]
        if host and port:
            return f"http://{host}:{port}"
    except Exception:
        pass
    return None


def _curl_range(url: str, lo: int, hi: int, proxy: str | None, tries: int = 4) -> bytes | None:
    cmd = ["curl", "-sL", "-m", "120", "-o", "-", "-r", f"{lo}-{hi}"]
    if proxy:
        cmd += ["--proxy", proxy]
    cmd.append(url)
    for _ in range(tries):
        r = subprocess.run(cmd, capture_output=True, timeout=180)
        if r.returncode == 0 and r.stdout:
            return r.stdout
    return None


class _RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range GETs (curl subprocess; the local
    proxy breaks urllib's TLS). Only used by the pyarrow reader below."""

    def __init__(self, url: str, size: int, proxy: str | None):
        self.url, self.size, self.proxy = url, size, proxy
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        b = _curl_range(self.url, self.pos, self.pos + n - 1, self.proxy)
        if b is None:
            raise IOError("range fetch failed")
        self.pos += len(b)
        return b

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def fineweb_texts(n: int, shard_url: str | None = None) -> list[str]:
    """Pull ~n texts from a fineweb sample-10BT parquet shard via HTTP Range reads
    (the shards are ~2GB; we only fetch the footer + the first row groups' text
    column, ~5-10MB total). Fallback: the repo's own docs corpus."""
    api = shard_url or (
        "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb/parquet/sample-10BT/train/0.parquet")
    texts: list[str] = []
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        proxy = _system_proxy()
        # resolve the 302 to the CDN url and get content-length
        head = subprocess.run(
            ["curl", "-sIL", "-m", "30"] + (["--proxy", proxy] if proxy else []) + [api],
            capture_output=True, text=True, timeout=60)
        loc = size = None
        for ln in head.stdout.splitlines():
            s = ln.lower()
            if s.startswith("location:"):
                loc = ln.split(":", 1)[1].strip()
            if s.startswith("content-length:"):
                size = int(s.split(":")[1].strip())
        url = loc or api
        # footer: last few MB hold the thrift metadata
        foot = _curl_range(url, max(0, size - 4194304), size - 1, proxy)
        mlen = struct.unpack("<I", foot[-8:-4])[0]
        meta = foot[-8 - mlen:-8]
        f = pq.ParquetFile(io.BytesIO(b"PAR1" + meta + foot[-8:]))
        cols = {c: f.metadata.row_group(0).column(c) for c in range(f.metadata.num_row_groups and 1 or 1)}
        tcol = f.metadata.row_group(0).column(0)
        need = tcol.file_offset + tcol.total_compressed_size + 4096
        head_bytes = _curl_range(url, 0, need, proxy)
        pf = pq.ParquetFile(io.BytesIO(head_bytes + meta + foot[-8:]))
        texts = []
        for rg in range(min(4, pf.metadata.num_row_groups)):
            try:
                texts += pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
            except Exception:
                break
        texts = [t for t in texts if t and len(t) > 400][:n]
    except Exception as e:
        print(f"[collect] fineweb fetch failed ({e}); falling back to repo docs corpus", file=sys.stderr)
        docs = Path(os.environ.get("TAP_CORPUS", TOOLS.parents[1] / "docs"))
        for f in sorted(docs.rglob("*.md")):
            try:
                t = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(t) > 400:
                texts.append(t)
            if len(texts) >= n:
                break
    return texts


def build_prompts(tok: ST.Tokenizer, texts: list[str], n_seq: int, prompt_tokens: int,
                  chat_frac: float = 0.4, seed: int = 0) -> list[list[int]]:
    """n_seq id-lists. chat_frac of them wrapped as user-message chat prompts (the
    production traffic shape); the rest are raw text chunks (continuation shape)."""
    rng = np.random.RandomState(seed)
    seqs: list[list[int]] = []
    ti = 0
    while len(seqs) < n_seq and ti < len(texts):
        text = texts[ti]; ti += 1
        ids = tok.encode(text)
        if len(ids) < prompt_tokens + 8:
            continue
        hi = max(1, len(ids) - prompt_tokens - 4)
        off = int(rng.randint(0, hi))
        chunk = ids[off:off + prompt_tokens]
        if rng.random_sample() < chat_frac:
            # production shape: [BOS] <|message_user|><|content_text|> ... <|end_message|>
            #                   <|message_model|>
            prefix = tok.encode("<|message_user|><|content_text|>", parse_special=True)
            suffix = tok.encode("<|end_message|><|message_model|>", parse_special=True)
            body = chunk[:max(8, prompt_tokens - len(prefix) - len(suffix))]
            seq = [BOS] + prefix + body + suffix
        else:
            seq = [BOS] + chunk
        seqs.append(seq)
    return seqs


def run_engine(binary: str, model: str, seqs: list[list[int]], gen: int, workdir: Path,
               extra: list[str] | None = None) -> tuple[Path, Path]:
    ids_file = workdir / "collect_ids.txt"
    meta_file = workdir / "collect_meta.txt"
    dump_file = workdir / "collect_taps.bin"
    for f in (meta_file, dump_file):
        if f.exists():
            f.unlink()
    ids_file.write_text("\n".join(" ".join(map(str, s)) for s in seqs) + "\n")
    env = dict(os.environ)
    env["INK_TAPDUMP"] = str(dump_file)
    cmd = [binary, "--model", model, "--draft", "off", "--gen", str(gen),
           "--tap-collect", str(ids_file), "--tap-meta", str(meta_file), "--no-warm"]
    cmd += extra or []
    print("[collect] " + " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=36000)
    sys.stderr.write(r.stderr[-4000:])
    if r.returncode != 0:
        raise RuntimeError(f"engine exited {r.returncode}")
    return dump_file, meta_file


def read_rows(dump: Path, meta: Path) -> list[np.ndarray]:
    """Return per-seq arrays [rows, 2+4*D] (pos, tok, taps) split by meta row counts."""
    data = np.fromfile(dump, dtype=np.uint8)
    assert data.size % ROW_BYTES == 0, f"dump size {data.size} not multiple of {ROW_BYTES}"
    rows = data.reshape(-1, ROW_BYTES)
    seqs = []
    off = 0
    for line in meta.read_text().splitlines():
        _, idx, n, plen, glen = line.split()
        n = int(n)
        blk = rows[off:off + n]
        assert blk.shape[0] == n, f"seq {idx}: meta says {n} rows, got {blk.shape[0]}"
        seqs.append(blk)
        off += n
    assert off == len(rows), f"meta covers {off} rows, dump has {len(rows)}"
    return seqs


def to_windows(seqs: list[np.ndarray], prompt_lens: list[int], window: int = 77,
               stride: int = 37) -> dict[str, np.ndarray]:
    """Slice each sequence into overlapping [window, ...] windows.
    taps [N,W,4,D] fp32; tokens [N,W] int32; is_generated [N,W] bool; src [N,3]."""
    taps_l, toks_l, gen_l, src_l = [], [], [], []
    for si, blk in enumerate(seqs):
        pos = blk[:, 0:4].view(np.int32)[:, 0]
        tokv = blk[:, 4:8].view(np.int32)[:, 0]
        taps = blk[:, 8:].view(np.float32).reshape(-1, 4, D)
        n = len(pos)
        plen = prompt_lens[si]
        s = 0
        while s + window <= n:
            taps_l.append(taps[s:s + window])
            toks_l.append(tokv[s:s + window])
            g = pos[s:s + window] >= plen - 1  # rows written by decode steps
            gen_l.append(g)
            src_l.append([si, s, plen])
            s += stride
    return {
        "taps": np.stack(taps_l) if taps_l else np.zeros((0, window, 4, D), np.float32),
        "tokens": np.stack(toks_l) if toks_l else np.zeros((0, window), np.int32),
        "is_generated": np.stack(gen_l) if gen_l else np.zeros((0, window), bool),
        "src": np.asarray(src_l, np.int32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=str(Path.home() / "dageno-modelbench/strata-ink67-fp8/strata-inkling-metal"))
    ap.add_argument("--model", default=str(Path.home() / "dageno-modelbench/sq10.strata"))
    ap.add_argument("--tok", default=str(Path.home() / "dageno-modelbench/tok_inkling"))
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--window", type=int, default=77)
    ap.add_argument("--stride", type=int, default=37)
    ap.add_argument("--chat-frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids-file", help="skip corpus: file of space-separated id sequences")
    ap.add_argument("--parse-only", action="store_true", help="only parse existing dump+meta in --out")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dump, meta = out / "collect_taps.bin", out / "collect_meta.txt"

    if not args.parse_only:
        tok = load_tokenizer(args.tok)
        if args.ids_file:
            seqs = [[int(x) for x in ln.split()] for ln in Path(args.ids_file).read_text().splitlines() if ln.strip()]
        else:
            texts = fineweb_texts(max(args.seqs * 3, 16))
            seqs = build_prompts(tok, texts, args.seqs, args.prompt_tokens, args.chat_frac, args.seed)
        print(f"[collect] {len(seqs)} sequences, lens {[len(s) for s in seqs[:8]]}...", file=sys.stderr)
        (out / "prompts.json").write_text(json.dumps(seqs))
        dump, meta = run_engine(args.bin, args.model, seqs, args.gen, out)

    seqs_rows = read_rows(dump, meta)
    plens = [int(l.split()[3]) for l in meta.read_text().splitlines()]
    w = to_windows(seqs_rows, plens, args.window, args.stride)
    npz = out / "distill_windows.npz"
    np.savez_compressed(npz, **w)
    print(f"[collect] wrote {npz}: {w['taps'].shape[0]} windows x {args.window} pos x taps[4,{D}]")


if __name__ == "__main__":
    main()
