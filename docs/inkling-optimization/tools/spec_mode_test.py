# cross-mode correctness harness: greedy output must be identical across no-spec/tree/block
# usage: python3 spec_mode_test.py <ref-file>  (captures current output and compares)
import json, subprocess, sys

def gen():
    r = subprocess.run(['curl', '-s', '--max-time', '120', 'http://127.0.0.1:8392/v1/chat/completions',
        '-H', 'Content-Type: application/json', '-d', json.dumps({
        'model': 'inkling-2-mini-sq10',
        'messages': [{'role': 'user', 'content': '用中文写一段关于秋天杭州西湖的描写,大约两百字。'}],
        'max_tokens': 150, 'temperature': 0})], capture_output=True, text=True)
    d = json.loads(r.stdout)
    return d['choices'][0]['message']['content']

out = gen()
if len(sys.argv) > 1:
    ref = open(sys.argv[1]).read()
    n = next((j for j in range(min(len(ref), len(out))) if ref[j] != out[j]), None)
    print('diverge at char', n, '| match' if ref == out else '| DIFFERENT')
else:
    print(out)
