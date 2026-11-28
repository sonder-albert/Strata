# concurrent-vs-solo divergence probe: two slots in parallel must match solo output bitwise
import json, subprocess
from concurrent.futures import ThreadPoolExecutor

def gen(prompt, n=120):
    r = subprocess.run(['curl', '-s', '--max-time', '120', 'http://127.0.0.1:8392/v1/chat/completions',
        '-H', 'Content-Type: application/json', '-d', json.dumps({
        'model': 'inkling-2-mini-sq10',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': n, 'temperature': 0})], capture_output=True, text=True)
    d = json.loads(r.stdout)
    return d['choices'][0]['message']['content']

p1 = '用中文写一段关于冬天哈尔滨的描写,大约一百五十字。'
p2 = '写一首关于夏天海边日落的短诗。'
a1 = gen(p1); a2 = gen(p2)
with ThreadPoolExecutor(2) as ex:
    f1 = ex.submit(gen, p1)
    f2 = ex.submit(gen, p2)
    b1, b2 = f1.result(), f2.result()
print('slot1 concurrent == solo:', a1 == b1)
print('slot2 concurrent == solo:', a2 == b2)
