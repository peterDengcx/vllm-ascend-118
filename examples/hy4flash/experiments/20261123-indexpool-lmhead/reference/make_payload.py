import hashlib, itertools, json, pathlib, string
from transformers import AutoTokenizer
r=pathlib.Path('/home/d00943518/hy4/runs/hy4-prefix90-128k1k-latest-20261123-02')
t=AutoTokenizer.from_pretrained('/home/d00943518/weights/Hy4-flash-FP8',trust_remote_code=True)
v=[]
for n in (1,2,3):
    for chars in itertools.product(string.ascii_lowercase,repeat=n):
        s=' '+''.join(chars)
        ids=t.encode(s,add_special_tokens=False)
        if len(ids)==1 and ids[0] not in {x[1] for x in v}:
            v.append((s,ids[0]))
        if len(v)==128: break
    if len(v)==128: break
assert len(v)==128,len(v)
prefix=' a'*115200
prefix_ids=t.encode(prefix,add_special_tokens=False)
assert len(prefix_ids)==115200
out=r/'prefix90_128k_frozen.jsonl'
with out.open('w',encoding='utf8') as f:
    for s,tid in v:
        q=prefix+s+' b'*12799
        ids=t.encode(q,add_special_tokens=False)
        assert len(ids)==128000,(s,len(ids))
        assert ids[:115200]==prefix_ids,s
        assert ids[115200]==tid,(s,tid,ids[115200])
        f.write(json.dumps({'question':q,'answer':'','max_out_len':1024},ensure_ascii=False,separators=(',',':'))+'\n')
print(json.dumps({'count':128,'input_tokens':128000,'shared_tokens':115200,'suffix_tokens':12800,'first_suffix_ids_unique':len(set(x[1] for x in v)),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
