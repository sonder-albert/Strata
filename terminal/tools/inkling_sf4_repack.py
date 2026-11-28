"""tools/inkling_sf4_repack.py - repack an Inkling-2-Mini .strata, converting Q8_ROW trunk
attention projections to SF4 (NVFP4: E2M1 codes + F8_E4M3 block-16 scales + F32 outer).

    python3 tools/inkling_sf4_repack.py --in sq10.strata --out sq10-sf4trunk.strata

What it does:
  * Copies Sections 3 (expert dir + vram order), 4 (expert bodies) and 5 (self-check) verbatim.
  * Rewrites Section 1: trunk tensors selected by --convert (default: every 2-D Q8_ROW
    attention/shared/mlp projection, never embed/unembed) are dequantized from Q8_ROW
    ([N x f32 scales][N*K int8]) and re-encoded as the canonical SF4 dense body
    [16 B hdr: amax, denom][N*K/2 E2M1 codes][N*K/16 E4M3 scales], matching
    sk_codecs.hpp dot_sf4_f32_row and the Metal ink_linear_sf4_mv* kernels.
  * `embed.weight` stays Q8_ROW (the embed gather kernel has no SF4 path) and
    `unembed.weight` stays S6_K (sensitive; fused argmax kernels are BF16/Q8/S6-only).

QUALITY WARNING: Q8_ROW -> SF4 is a REQUANTIZATION, not the native QAD checkpoint. This
artifact is for kernel/path validation and speed measurement; generation parity against
the source pack is NOT bit-exact. For production SF4 the native SF4 tensors from the
original checkpoint must be packed instead.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

import numpy as np
import zstandard as zstd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools" / "squant"))
from strata_format import (  # noqa: E402
    MAGIC, PAGE_SIZE, StrataWriter, PROFILE_TIERED,
)

Q8_ROW = "Q8_ROW"
E2M1_VALS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
# thresholds mirror host_f32_to_e2m1_nib: a < 0.25 -> 0 ... a >= 5.0 -> 7
E2M1_BOUNDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float64)


def e4m3_pos_encode(v: np.ndarray) -> np.ndarray:
    """Vectorized host_f32_to_e4m3_pos (positive-only E4M3FN, NaN bin saturated to 0x7E)."""
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros(v.shape, dtype=np.uint8)
    pos = v > 0
    hi = pos & (v >= 448.0)
    out[hi] = 0x7E
    sub = pos & (v < 1.0 / 64.0)
    out[sub] = np.clip(np.rint(v[sub] * 512.0), 0, 7).astype(np.uint8)
    mid = pos & ~hi & ~sub
    if mid.any():
        frac, exp = np.frexp(v[mid])
        e_bits = (exp - 1) + 7
        m = np.rint((frac * 2.0 - 1.0) * 8.0)
        over = m >= 8
        e_bits = e_bits + over.astype(np.int64)
        m = np.where(over, 0.0, m)
        sat = e_bits >= 15
        code = np.where(sat, 0x7E, (e_bits.astype(np.int64) << 3) | m.astype(np.int64))
        code = np.where(sat & (m <= 6), (15 << 3) | np.clip(m, 0, 6).astype(np.int64), code)
        out[mid] = code.astype(np.uint8)
    return out


def e4m3_pos_decode(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.int64) & 0x7F
    e = (b >> 3) & 0xF
    m = b & 7
    return np.where(e == 0, m * (1.0 / 512.0), (1.0 + m * 0.125) * np.exp2(e - 7))


def sf4_quant(w: np.ndarray):
    """w: [N, K] float64 -> (codes N*K/2, scales N*K/16, outer). Mirrors host_quant_row_sf4."""
    N, K = w.shape
    assert K % 16 == 0
    outer = float(np.abs(w).max() / 6.0)
    if outer <= 0.0:
        outer = 1.0
    inv_outer = 1.0 / outer
    blk = w.reshape(N, K // 16, 16)
    am = np.abs(blk).max(axis=2)                            # [N, K/16]
    sb = e4m3_pos_encode((am / 6.0) * inv_outer)            # block scales
    sc = e4m3_pos_decode(sb) * outer                        # [N, K/16]
    inv = np.where(sc > 0, 1.0 / sc, 0.0)
    q = blk * inv[:, :, None]
    mag = np.searchsorted(E2M1_BOUNDS, np.abs(q), side="right")
    nib = (mag & 7) | np.where(q < 0, 8, 0)
    packed = (nib[:, :, 0::2] | (nib[:, :, 1::2] << 4)).astype(np.uint8).reshape(N, K // 2)
    return packed.tobytes(), sb.astype(np.uint8).tobytes(), outer


def sf4_dequant_err(w: np.ndarray, codes: bytes, scales: bytes, outer: float) -> float:
    N, K = w.shape
    pk = np.frombuffer(codes, dtype=np.uint8).reshape(N, K // 2)
    sb = np.frombuffer(scales, dtype=np.uint8).reshape(N, K // 16)
    nib = np.empty((N, K), dtype=np.int64)
    nib[:, 0::2] = pk & 0xF
    nib[:, 1::2] = pk >> 4
    val = E2M1_VALS[nib & 7] * np.where(nib & 8, -1.0, 1.0)
    sc = e4m3_pos_decode(sb) * outer
    deq = val * np.repeat(sc, 16, axis=1)
    return float(np.linalg.norm(deq - w) / max(np.linalg.norm(w), 1e-30))


def q8row_dequant(raw: bytes, N: int, K: int) -> np.ndarray:
    buf = np.frombuffer(raw, dtype=np.uint8)
    assert buf.size == N * 4 + N * K, f"Q8_ROW payload {buf.size} != {N * 4 + N * K}"
    scales = np.frombuffer(buf[: N * 4].tobytes(), dtype=np.float32)
    codes = np.frombuffer(buf[N * 4:].tobytes(), dtype=np.int8).reshape(N, K)
    return codes.astype(np.float64) * scales[:, None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--convert", nargs="*", default=None,
                    help="substrings; default converts 2-D Q8_ROW .attn./.mlp./shared_expert trunk tensors")
    ap.add_argument("--build-tag", default="SF4TRUNK-RQ")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    with open(src, "rb") as f:
        frame = f.read(PAGE_SIZE)
    assert frame[:8] == MAGIC, "not a .strata file"
    secs = []
    for i in range(6):
        e = struct.unpack_from("<QQ32sIIQ", frame[0x10 + i * 64:0x10 + (i + 1) * 64])
        secs.append(dict(off=e[0], len=e[1], flags=e[4]))
    with open(src, "rb") as f:
        f.seek(secs[0]["off"])
        man = json.loads(zstd.ZstdDecompressor().decompress(f.read(secs[0]["len"]),
                                                          max_output_size=256 * 1024 * 1024))
    trunk = man["trunk_tensors"]

    def want(t) -> bool:
        if t["format"] != Q8_ROW or len(t["shape"]) != 2:
            return False
        n = t["name"]
        if args.convert is not None:
            return any(s in n for s in args.convert)
        if "embed" in n or "unembed" in n:
            return False
        return (".attn.wq_du." in n or ".attn.wk_dv." in n or ".attn.wv_dv." in n or
                ".attn.wr_du." in n or ".attn.wo_ud." in n or ".attn.gate_attn." in n or
                "shared_expert" in n or ".mlp." in n)

    with open(src, "rb") as f:
        writer = StrataWriter(str(args.dst), profile=PROFILE_TIERED,
                              engram_shard_sha256=frame[0x190:0x1B0])
        s1 = writer.begin_streaming_section(1)
        new_meta = []
        n_conv = 0
        saved = 0
        for t in trunk:
            rem = s1.length % 64
            if rem:
                s1.write(b"\x00" * (64 - rem))
            off = s1.length
            f.seek(secs[1]["off"] + t["offset"])
            raw = f.read(t["bytes"])
            fmt = t["format"]
            if want(t):
                N, K = int(t["shape"][0]), int(t["shape"][1])
                w = q8row_dequant(raw, N, K)
                codes, scales, outer = sf4_quant(w)
                hdr = struct.pack("<4f", outer, 1.0, 0.0, 0.0)
                raw = hdr + codes + scales
                fmt = "SF4"
                err = sf4_dequant_err(w, codes, scales, outer)
                saved += (t["bytes"] - len(raw))
                n_conv += 1
                print(f"  [SF4] {t['name']} [{N}x{K}] outer={outer:.6f} "
                      f"rms_err={err:.4%} {t['bytes'] / 1048576:.1f}->{len(raw) / 1048576:.1f} MB", flush=True)
            s1.write(raw)
            new_meta.append({"name": t["name"], "format": fmt, "shape": t["shape"],
                             "offset": off, "bytes": len(raw)})
        s1.pad_4k()
        s1e = s1.finish()
        for sid in (3, 4, 5):
            sec = secs[sid]
            if sec["len"] == 0:
                continue
            f.seek(sec["off"])
            data = f.read(sec["len"])
            writer.write_section_bytes(sid, data, flags=sec["flags"])
        man["trunk_tensors"] = new_meta
        man["build"] = f'{man.get("build", "?")}+{args.build_tag}'
        man["section1_bytes"] = s1e.byte_length
        man["repack_note"] = ("Q8_ROW->SF4 requantization of attention projections; "
                              "kernel/path validation artifact, NOT native QAD weights")
        writer.write_manifest(man)
        writer.finalize()
    print(f"[DONE] converted {n_conv} tensors, saved {saved / 1048576:.1f} MB of trunk DRAM traffic per token")
    return 0


if __name__ == "__main__":
    sys.exit(main())
