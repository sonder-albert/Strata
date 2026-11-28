"""Kaggle kernel v2: Inkling-2-Mini engram graft training on T4 x2 / L4.

Budget-aware phases (kaggle GPU sessions cap at ~9-12h wall):
  1. HF snapshot download of the official checkpoint (~19.5GB, kaggle net is fast)
  2. NVFP4/MXFP8 -> bf16 conversion, streaming chunked (~10-20 min CPU)
  3. fp16 engram tables: range-streamed from the HF mirror of Hy4-flash (31.6GB disk)
  4. Muon adapter training, backbone frozen in NF4 (fits T4 16GB, spreads over both)

Resumable: every phase skips work whose artifact already exists in /kaggle/working,
which persists across restarts as kernel output.
"""
import os, sys, time, json, shutil

WORK = '/kaggle/working'
INPUT = '/kaggle/input'
os.environ.setdefault('ENGRAM_SRC', 'hf')


def log(*a):
    print(*a, flush=True)


def phase_checkpoint():
    src_dir = f'{WORK}/hf-orig'
    out_dir = f'{WORK}/ckpt-bf16'
    if os.path.exists(f'{out_dir}/model.safetensors'):
        log('bf16 checkpoint exists, reusing')
        return out_dir
    if not os.path.exists(f'{src_dir}/model-00001-of-00001.safetensors'):
        log('downloading official checkpoint from HF...')
        os.makedirs(src_dir, exist_ok=True)
        try:
            os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
            import hf_transfer  # noqa
        except Exception:
            pass
        from huggingface_hub import snapshot_download
        snapshot_download('thinkingmachines/Inkling-2-Mini', local_dir=src_dir,
                          ignore_patterns=['*.md', '.gitattributes'])
    log('converting experts to bf16...')
    sys.path.insert(0, WORK)
    import convert_nvfp4_to_bf16 as conv
    conv.main(f'{src_dir}/model-00001-of-00001.safetensors', out_dir)
    for f in os.listdir(src_dir):
        if f.endswith('.json') or f.startswith('tokenizer') or f.endswith('.jinja') or f == 'mtp.safetensors':
            dst = os.path.join(out_dir, f)
            if not os.path.exists(dst):
                shutil.copy(os.path.join(src_dir, f), dst)
    shutil.rmtree(src_dir, ignore_errors=True)
    log('conversion done; raw nvfp4 checkpoint removed')
    return out_dir


def phase_tables():
    out = f'{WORK}/tables-fp16'
    if os.path.exists(f'{out}/fp16_manifest.json'):
        log('fp16 tables exist, reusing')
        return out
    import subprocess
    subprocess.run([sys.executable, f'{WORK}/build_engram_tables_fp16.py',
                    '--layer', '1', '--out', out], check=True)
    return out


def phase_data():
    import subprocess
    subprocess.run([sys.executable, f'{WORK}/build_train_data.py'], check=True, cwd=WORK)
    return f'{WORK}/train_texts.jsonl'


def main():
    t0 = time.time()
    sys.path.insert(0, WORK)

    log('== env probe ==')
    os.system('nvidia-smi | head -12')
    os.system('df -h /kaggle/working | tail -1; free -g | head -2')

    ckpt = phase_checkpoint()
    log(f'checkpoint phase: {(time.time()-t0)/60:.1f} min')
    tables = phase_tables()
    log(f'tables phase: {(time.time()-t0)/60:.1f} min total')
    data = phase_data()

    log('== training ==')
    sys.argv = [
        'train_engram_cloud.py',
        '--checkpoint', ckpt,
        '--tables', tables,
        '--tokenizer', f'{WORK}/hy4-tokenizer.json',
        '--data', data,
        '--steps', os.environ.get('STEPS', '2000'),
        '--seq-len', os.environ.get('SEQ_LEN', '512'),
        '--batchsize', os.environ.get('BATCH', '4'),
        '--seed-layer', os.environ.get('SEED_LAYER', '2'),
        '--out', f'{WORK}/engram_adapter_seed{os.environ.get("SEED_LAYER", "2")}.pt',
        '--eval-every', '50',
    ]
    if os.environ.get('INK_BF16') != '1':
        sys.argv.append('--quant4')
    os.makedirs('/data', exist_ok=True)
    src, dst = f'{WORK}/hy4_id2cid.npy', '/data/hy4_id2cid.npy'
    if os.path.exists(src) and not os.path.exists(dst):
        os.symlink(src, dst)
    import train_engram_cloud
    train_engram_cloud.main()
    log(f'total wall: {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
