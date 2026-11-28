#!/bin/bash
# greedy A/B output probe: prints content + speed line; compare across binaries/modes
curl -s --max-time 120 http://127.0.0.1:8392/v1/chat/completions -H 'Content-Type: application/json' -d '{
 "model":"inkling-2-mini-sq10",
 "messages":[{"role":"user","content":"用中文写一段关于秋天杭州西湖的描写,大约两百字。"}],
 "max_tokens":150,"temperature":0
}' | python3 -c "
import sys, json
d = json.load(sys.stdin)
t = d.get('timings',{})
print(d['choices'][0]['message']['content'])
print('---SPEED---', t.get('predicted_per_second'), 'accept', t.get('draft_n_accepted'),'/',t.get('draft_n'))
"
