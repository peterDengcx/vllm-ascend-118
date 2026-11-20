#!/usr/bin/env python3
"""Verify both backends, then nonstream and streaming requests through PD proxy."""
import json, urllib.request, os
p=os.getenv('P_IP','141.61.141.22'); d=os.getenv('D_IP','141.61.141.26')
proxy=f'http://{p}:{os.getenv("PROXY_PORT","18000")}'
def get(url):
 with urllib.request.urlopen(url,timeout=10) as r: assert r.status==200
for url in [f'http://{p}:18010/health',f'http://{d}:18020/health',proxy+'/healthcheck']:
 get(url); print('PASS',url,flush=True)
for stream in (False,True):
 body={'model':'hy4','messages':[{'role':'user','content':'请简短回答：1+1等于多少？'}],'max_tokens':32,'temperature':0,'stream':stream}
 req=urllib.request.Request(proxy+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=300) as r: data=r.read().decode()
 if stream:
  assert 'data: [DONE]' in data, data
  assert '"delta"' in data, data
 else:
  result=json.loads(data); assert result['choices'],result
 print('PASS stream='+str(stream),data,flush=True)
