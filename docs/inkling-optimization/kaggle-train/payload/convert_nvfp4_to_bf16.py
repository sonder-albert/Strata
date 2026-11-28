"""Inkling-2-Mini NVFP4/MXFP8 mixed checkpoint -> uniform BF16 converter.

Verified against the official checkpoint layout (thinkingmachines/Inkling-2-Mini):
- layers 0-19 experts: NVFP4. weight U8 [E,R,K/2] (low nibble first),
  scale F8_E4M3 [E,R,K/16], amax F32 [E], denom F32 [1].
  dequant = e2m1(nib) * e4m3(scale)*16-group-repeat * amax/denom
- layers 20-23 experts: MXFP8. weight F8_E4M3 [E,R,K], scale U8-as-e8m0 [E,R,K/32]
  dequant = e4m3(w) * e8m0(scale)*32-group-repeat
- everything else streams through unchanged.

Output: a single bf16 safetensors file (~26GB for the LLM part; vision tower
and everything else is preserved as bf16 too) plus a patched config.json
without quantization_config. Written with pure numpy - no GPU needed.
"""
import json
import os
import struct
import sys
import numpy as np

E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def e2m1(nib):
    sign = np.where((nib & 8) != 0, -1.0, 1.0).astype(np.float32)
    return sign * E2M1_MAG[nib & 7]


def f8e4m3(u8):
    u8 = u8.astype(np.uint8)
    sign = np.where((u8 & 0x80) != 0, -1.0, 1.0).astype(np.float32)
    exp = ((u8 >> 3) & 0xF).astype(np.int32)
    man = (u8 & 0x7).astype(np.float32)
    val = np.where(exp == 0, man / 8.0 * 2.0 ** -6, (1.0 + man / 8.0) * 2.0 ** (exp - 7))
    return sign * val.astype(np.float32)


def e8m0(u8):
    u8 = u8.astype(np.uint32)
    return np.where(u8 == 0, 2.0 ** -127, np.ldexp(1.0, u8.astype(np.int32) - 127)).astype(np.float32)


def read_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def read_raw(path, meta, base):
    begin, end = meta['data_offsets']
    return np.memmap(path, dtype=np.uint8, mode='r', offset=base + begin, shape=(end - begin,))


def bf16_bytes(f32):
    bits = (np.ascontiguousarray(f32.astype(np.float32)).view(np.uint32) + 0x8000) >> 16
    return bits.astype(np.uint16).tobytes()


def dequant_nvfp4_layer(w_u8, scale_f8, amax, denom, chunk=8):
    """Chunked over experts: peak ~600MB at chunk=8 for production shapes."""
    E, R, halfK = w_u8.shape
    out = np.empty((E, R, halfK * 2), dtype=np.float32)
    gsc = (amax.reshape(-1) / float(denom.reshape(-1)[0])).astype(np.float32)
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        w = w_u8[e0:e1]
        lo = w & 0xF
        hi = w >> 4
        nib = np.empty((e1 - e0, R, halfK * 2), dtype=np.uint8)
        nib[:, :, 0::2] = lo
        nib[:, :, 1::2] = hi
        vals = e2m1(nib)
        scales = np.repeat(f8e4m3(scale_f8[e0:e1]), 16, axis=-1)
        out[e0:e1] = vals * scales * gsc[e0:e1].reshape(-1, 1, 1)
    return out


def dequant_mxfp8_layer(w_f8, scale_u8, chunk=16):
    E = w_f8.shape[0]
    out = np.empty(w_f8.shape, dtype=np.float32)
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        vals = f8e4m3(w_f8[e0:e1])
        scales = np.repeat(e8m0(scale_u8[e0:e1]), 32, axis=-1)
        out[e0:e1] = vals * scales
    return out


def main(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    hdr, base = read_header(src)
    real = {k: v for k, v in hdr.items() if k != '__metadata__'}

    expert_layers = sorted({
        (int(k.split('.')[3]), k.split('experts.')[1].split('_weight')[0])
        for k in real if '.mlp.experts.' in k and k.endswith('_weight')
    })
    nvfp4_layers = sorted({li for li, _ in expert_layers
                           if f'model.llm.layers.{li}.mlp.experts.w13_weight_amax' in real})
    done_expert = set()

    # Two passes with a fixed emit order: pass 1 computes out_meta offsets,
    # pass 2 streams each tensor to disk. Bytes are never all held at once
    # (the previous single-pass accumulator OOM'd the kernel at layer 11).
    def emit_order():
        for li in sorted({li for li, _ in expert_layers}):
            prefix = f'model.llm.layers.{li}.mlp'
            for proj in ('w13', 'w2'):
                yield ('dq', f'{prefix}.experts.{proj}.weight', li, proj)
        for k, meta in real.items():
            if ('.mlp.experts.' in k and k.endswith(('_weight', '_weight_scale',
                        '_weight_amax', '_weight_outer_scale_denominator'))):
                continue
            yield ('raw', k, meta['dtype'], meta['shape'])

    def dq_bytes(name):
        """BF16 bytes for one converted expert tensor (one at a time)."""
        li = int(name.split('.')[3])
        proj = 'w13' if '.w13.weight' in name else 'w2'
        prefix = f'model.llm.layers.{li}.mlp'
        key = f'{prefix}.experts.{proj}_weight'
        meta = real[key]
        arr = read_raw(src, meta, base)
        w = arr.reshape(meta['shape'])
        s = read_raw(src, real[key + '_scale'], base).reshape(real[key + '_scale']['shape'])
        if li in nvfp4_layers:
            amax_raw = np.frombuffer(bytes(read_raw(src, real[key + '_amax'], base)), dtype=np.float32)
            amax = amax_raw.reshape(-1)
            den = np.frombuffer(bytes(read_raw(src, real[key + '_outer_scale_denominator'], base)), dtype=np.float32)
            dq = dequant_nvfp4_layer(w, s, amax, den)
        else:
            dq = dequant_mxfp8_layer(w, s)
        return dq

    print('pass 1: offsets...', flush=True)
    out_meta = {}
    cursor = 0
    for kind, name, a, b in emit_order():
        if kind == 'dq':
            li, proj = a, b
            meta = real[f'model.llm.layers.{li}.mlp.experts.{proj}_weight']
            shape = list(meta['shape'])
            shape[-1] = shape[-1] * 2 if li in nvfp4_layers else shape[-1]
            n = int(np.prod(shape)) * 2
            out_meta[name] = {'dtype': 'BF16', 'shape': shape,
                              'data_offsets': [cursor, cursor + n]}
            cursor += n
        else:
            meta = real[name]
            begin, end = meta['data_offsets']
            out_meta[name] = {'dtype': meta['dtype'], 'shape': meta['shape'],
                              'data_offsets': [cursor, cursor + (end - begin)]}
            cursor += end - begin

    header = json.dumps(out_meta).encode()
    pad = (8 - (len(header) + 8) % 8) % 8
    header += b' ' * pad
    out_path = os.path.join(dst_dir, 'model.safetensors')
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<Q', len(header)))
        f.write(header)
        print('pass 2: streaming to disk...', flush=True)
        for kind, name, a, b in emit_order():
            if kind == 'dq':
                li, proj = a, b
                dq = dq_bytes(name)
                bits = (np.ascontiguousarray(dq).view(np.uint32) + 0x8000) >> 16
                u16 = bits.astype(np.uint16)
                CH = 64 * 1024 * 1024
                for o in range(0, u16.size * 2, CH):
                    f.write(u16.ravel()[o // 2:(o + CH) // 2].tobytes())
                del dq
                print(f'  layer {li} {proj}: done', flush=True)
            else:
                meta = real[name]
                raw = read_raw(src, meta, base)
                CH = 64 * 1024 * 1024
                for o in range(0, len(raw), CH):
                    f.write(raw[o:o + CH])

    # patched config without quantization_config, single-file index
    cfg_src = os.path.join(os.path.dirname(src), 'config.json')
    if os.path.exists(cfg_src):
        cfg = json.load(open(cfg_src))
        cfg.pop('quantization_config', None)
        json.dump(cfg, open(os.path.join(dst_dir, 'config.json'), 'w'), indent=2)
    print(f'written {out_path} ({cursor/1e9:.1f} GB)', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
