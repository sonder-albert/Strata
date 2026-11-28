"""Rebuild Hy4 engram tables on a cloud instance WITHOUT uploading them.

The engram embedding matrix `model.layers.{L}.engram_layer.multi_head_embedding.embedding.weight`
is a [164994380, 96] BF16 tensor stored CONTIGUOUSLY as the only tensor in shard
model-0000{4,3,2,1}-of-00148.safetensors of Tencent-Hunyuan/Hy4-flash (31.68GB per shard).

The 16 per-head tables of layer L are contiguous row slices in ngram_manifest order:
8 bigram heads then 8 trigram heads (verified: 16 x ~10.3M rows == 164994380 total).

This script streams ONLY the byte ranges belonging to the requested donor layer via
HTTP range requests against modelscope.cn (domestic-fast, no upload bottleneck) and
writes one fp16 .npy memmap-compatible file per table plus a manifest.

Memory: ~200MB working set (chunked). Disk: 16 x rows*96*2 fp16 bytes (~31.6GB for a layer).
Network: same amount downloaded once.

Usage:
  python build_engram_tables_fp16.py --layer 1 --out /data/engram-hy4-fp16
"""
import argparse, json, os, struct, sys, time
import numpy as np
import requests

MS_REPO = 'https://modelscope.cn/api/v1/models/Tencent-Hunyuan/Hy4-flash/repo'
# HF serves the same safetensors over CDN with range support - much faster outside China (e.g. kaggle).
HF_REPO = 'https://huggingface.co/Tencent-Hunyuan/Hy4-flash/resolve/main'
SHARD_FOR_LAYER = {1: 'model-00004-of-00148.safetensors',
                   11: 'model-00003-of-00148.safetensors',
                   21: 'model-00002-of-00148.safetensors',
                   31: 'model-00001-of-00148.safetensors'}
# verified row counts per layer-1 table (bigram heads 0-7, trigram heads 8-15),
# from ngram_manifest.json of the q4 serving tables. Other layers differ slightly;
# read them from the donor config at runtime (prime sizes are config-derived).
LAYER1_ROWS = [10312007, 10312021, 10312039, 10312069, 10312091, 10312121, 10312123, 10312147,
               10312177, 10312187, 10312207, 10312213, 10312229, 10312241, 10312249, 10312259]
DIM = 96
BYTES_PER_ROW_BF16 = DIM * 2
CHUNK_ROWS = 262144  # 50MB bf16 per chunk


def log(*a):
    print(*a, flush=True)


def shard_header(url):
    r = requests.get(url, headers={'Range': 'bytes=0-7'}, timeout=30)
    hlen = struct.unpack('<Q', r.content[:8])[0]
    r2 = requests.get(url, headers={'Range': f'bytes=8-{8+hlen-1}'}, timeout=60)
    return json.loads(r2.content), 8 + hlen


def stream_range(url, start, end, dst_write):
    """stream bytes [start,end) of url into dst_write(buf) callback"""
    pos = start
    sess = requests.Session()
    while pos < end:
        stop = min(pos + 64 * 1024 * 1024 - 1, end - 1)
        for attempt in range(5):
            try:
                with sess.get(url, headers={'Range': f'bytes={pos}-{stop}'}, timeout=120, stream=True) as r:
                    assert r.status_code == 206, f'range status {r.status_code}'
                    for chunk in r.iter_content(1024 * 1024):
                        dst_write(chunk)
                    pos = stop + 1
                break
            except Exception as e:
                log(f'  range retry {attempt}: {e}')
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError('range download failed after retries')


def bf16_to_f16_inplace(buf):
    """convert a bf16 bytes buffer to fp16 numpy array (rounding, numpy)"""
    u16 = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 2)
    # reconstruct fp32 from bf16 then cast to fp16
    f32bits = np.zeros((u16.shape[0], 4), dtype=np.uint8)
    f32bits[:, 2] = u16[:, 0]
    f32bits[:, 3] = u16[:, 1]
    f32 = f32bits.copy().view('<f4').reshape(-1)
    return f32.astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=int, default=1, choices=[1, 11, 21, 31])
    ap.add_argument('--out', required=True)
    ap.add_argument('--rows-manifest', default=None,
                    help='ngram_manifest.json to read row counts from (defaults to built-in layer1)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    shard = SHARD_FOR_LAYER[args.layer]
    src = os.environ.get('ENGRAM_SRC', 'hf')
    if src == 'hf':
        url = f'{HF_REPO}/{shard}'
    else:
        url = f'{MS_REPO}?FilePath={shard}'
    # probe: some CDNs redirect; follow with the session inside stream_range
    try:
        rp = requests.get(url, headers={'Range': 'bytes=0-7'}, timeout=30, allow_redirects=True)
        if rp.status_code not in (200, 206) or len(rp.content) < 8:
            raise RuntimeError(f'probe status {rp.status_code}')
    except Exception as e:
        log(f'{src} probe failed ({e}); falling back to modelscope')
        url = f'{MS_REPO}?FilePath={shard}'

    # row counts per table
    if args.rows_manifest:
        man = json.load(open(args.rows_manifest))
        row_counts = [t['rows'] for t in man['tables'] if t['layer'] == args.layer]
        assert len(row_counts) == 16, f'need 16 tables for layer {args.layer}'
    else:
        assert args.layer == 1, 'built-in row counts only valid for layer 1; pass --rows-manifest'
        row_counts = LAYER1_ROWS

    log(f'fetching shard header for {shard}...')
    hdr, base = shard_header(url)
    tensor_key = [k for k in hdr if k != '__metadata__'][0]
    info = hdr[tensor_key]
    log(f'{tensor_key}: dtype={info["dtype"]} shape={info["shape"]}')
    assert info['shape'][1] == DIM and info['dtype'] == 'BF16'
    file_row0 = base + info['data_offsets'][0]  # absolute byte offset of row 0

    total_rows = sum(row_counts)
    assert total_rows == info['shape'][0], f'row mismatch {total_rows} vs {info["shape"][0]}'

    manifest = {'format': 'ngram-fp16', 'layer': args.layer, 'dim': DIM, 'tables': []}
    row_start = 0
    t0 = time.time()
    downloaded = 0
    for ti, nrows in enumerate(row_counts):
        fname = f'layer{args.layer}_table{ti:02d}.f16.npy'
        fpath = os.path.join(args.out, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) > nrows * DIM * 2:
            log(f'table {ti} exists, skipping')
            row_start += nrows
            continue
        # memmap the npy file: write header manually then memmap data
        header = np.lib.format.header_data_from_array_1_0(
            np.empty((nrows, DIM), dtype=np.float16))
        arr = np.memmap(fpath, dtype=np.float16, mode='w+', shape=(nrows, DIM),
                        offset=len(np.lib.format.dumps(np.empty(0))) - len(np.lib.format.dumps(np.empty(0))))
        # simpler: use np.lib.format.open_memmap
        arr = np.lib.format.open_memmap(fpath, mode='w+', dtype=np.float16, shape=(nrows, DIM))
        start_byte = file_row0 + row_start * BYTES_PER_ROW_BF16
        end_byte = start_byte + nrows * BYTES_PER_ROW_BF16
        log(f'table {ti}: rows {row_start}..{row_start+nrows} ({nrows/1e6:.1f}M rows, '
            f'{(end_byte-start_byte)/1e9:.2f}GB)')

        state = {'row_in_table': 0, 'carry': b''}
        def consume(chunk, arr=arr, nrows=nrows, state=state):
            nonlocal downloaded
            buf = state['carry'] + chunk
            nwhole = (len(buf) // BYTES_PER_ROW_BF16) * BYTES_PER_ROW_BF16
            state['carry'] = buf[nwhole:]
            buf = buf[:nwhole]
            rows_in_chunk = len(buf) // BYTES_PER_ROW_BF16
            f16 = bf16_to_f16_inplace(buf)
            f16 = f16.reshape(rows_in_chunk, DIM)
            r0 = state['row_in_table']
            r1 = min(r0 + rows_in_chunk, nrows)
            arr[r0:r1] = f16[:r1 - r0]
            state['row_in_table'] = r1
            downloaded += len(buf)

        stream_range(url, start_byte, end_byte, consume)
        arr.flush()
        del arr
        manifest['tables'].append({'index': ti, 'rows': nrows, 'file': fname,
                                   'head': ti % 8, 'ngram_order': 2 if ti < 8 else 3})
        el = time.time() - t0
        rate = downloaded / max(el, 1) / 1e6
        log(f'  table {ti} done. cumulative {downloaded/1e9:.1f}GB at {rate:.0f}MB/s')
        row_start += nrows

    json.dump(manifest, open(os.path.join(args.out, 'fp16_manifest.json'), 'w'), indent=1)
    log(f'ALL DONE in {(time.time()-t0)/60:.0f} min')


if __name__ == '__main__':
    main()
