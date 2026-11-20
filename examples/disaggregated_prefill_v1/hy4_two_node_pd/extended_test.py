import json,time,urllib.request,concurrent.futures
from pathlib import Path
base='http://141.61.141.22:18000';direct='http://141.61.141.26:18020'
def query(endpoint,prompt,max_tokens=16):
 body={'model':'hy4','prompt':prompt,'temperature':0,'max_tokens':max_tokens,'seed':42}
 t=time.time();req=urllib.request.Request(endpoint+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=300) as r:out=json.load(r)
 assert out['choices'] and out['usage']['completion_tokens']>0,out
 return {'text':out['choices'][0]['text'],'usage':out['usage'],'seconds':round(time.time()-t,2)}
prompt=('The following is a reference note about fruits. Apples are red. Bananas are yellow. Oranges are orange.\n'*12)+'Question: What color are bananas? Answer:'
a=query(direct,prompt);b=query(base,prompt)
print('DIRECT',json.dumps(a),flush=True);print('PD',json.dumps(b),flush=True)
assert a['usage']['prompt_tokens']>128,a
print('DIRECT_PD_EXACT_MATCH', a['text']==b['text'],flush=True)
print('PASS multi-block request completed',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 results=list(pool.map(lambda p:query(base,p,8),['The capital of France is','The capital of Japan is']))
print('PASS concurrent=2',json.dumps(results),flush=True)
Path('/mnt/share/g00887675/Hy4/two_node_pd/validation_results.json').write_text(json.dumps({'direct':a,'pd':b,'concurrent':results,'exact_match':a['text']==b['text']},indent=2))
