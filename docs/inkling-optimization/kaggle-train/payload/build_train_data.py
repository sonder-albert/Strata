"""Build training text jsonl on the cloud instance: diverse natural text (fineweb sample).
Engram adaptation needs natural diverse token streams, not synthetic JSON."""
import json
out = 'train_texts.jsonl'
try:
    with open(out) as f:
        n = sum(1 for _ in f)
    if n >= 2000:
        print(f'{out} already has {n} lines')
        raise SystemExit
except FileNotFoundError:
    pass
try:
    from datasets import load_dataset
    try:
        ds = load_dataset('HuggingFaceFW/fineweb', 'sample-10BT', split='train', streaming=True)
    except Exception:
        # modelscope-hosted alternative if HF unreachable from the instance
        ds = load_dataset('swift/fineweb', split='train', streaming=True)
    n = 0
    with open(out, 'w') as f:
        for row in ds:
            text = row['text'][:2000]
            if len(text) > 200:
                f.write(json.dumps({'text': text}) + '\n')
                n += 1
                if n >= 4000:
                    break
    print(f'wrote {n} training texts')
except Exception as e:
    print('fineweb failed:', e)
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train', streaming=True)
    n = 0
    with open(out, 'w') as f:
        for row in ds:
            t = row['text'].strip()
            if len(t) > 200:
                f.write(json.dumps({'text': t[:2000]}) + '\n')
                n += 1
                if n >= 4000:
                    break
    print(f'wrote {n} wikitext training texts')
