#!/bin/bash
# quick decode speed probe: fixed prompt, count tokens and time
curl -s --max-time 120 http://127.0.0.1:8392/v1/chat/completions -H 'Content-Type: application/json' -d '{
 "model":"inkling-2-mini-sq10",
 "messages":[{"role":"user","content":"用中文写一段关于秋天杭州西湖的描写,大约两百字。"}],
 "max_tokens":200,
 "temperature":0
}' | python3 -c "
import sys, json
d = json.load(sys.stdin)
u = d.get('usage', {})
t = d.get('timings', {})
print('completion_tokens:', u.get('completion_tokens'))
print('timings:', json.dumps(t))
"
