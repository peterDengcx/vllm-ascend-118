"""Validate the completed AISBench run and reconstruct full-precision decode metrics."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import statistics
import struct
import sys

R=Path(sys.argv[1])
completion=json.loads((R/'completion.json').read_text())
assert completion['completed']
common=completion['metrics']
result=Path(completion['result'])
with result.with_suffix('.csv').open() as f:latency={x['Performance Parameters']:x for x in csv.DictReader(f)}
details=[json.loads(x) for x in result.with_name(result.stem+'_details.jsonl').read_text().splitlines()]
assert len(details)==len({x['id'] for x in details})==128
assert all(x['success'] and not x['error_info'] and x['input_tokens']==128000 and x['output_tokens']==1024 for x in details)
assert common['Total Generated Tokens']['total']==131072
assert common['Total Input Tokens']['total']==16384000
assert common['Success Requests']['total']==128 and common['Failed Requests']['total']==0
assert common['Max Concurrency']['total']==32
assert json.loads((R/'aisbench-exit.json').read_text())['exit_code']==0

def stats(values):
    values=sorted(values)
    def percentile(p):
        index=(len(values)-1)*p
        lower=int(index);upper=min(lower+1,len(values)-1)
        return values[lower]+(values[upper]-values[lower])*(index-lower)
    return {'n':len(values),'mean':statistics.mean(values),'p50':percentile(.5),
            'p90':percentile(.9),'p99':percentile(.99),'min':values[0],'max':values[-1]}

def array(blob):
    assert blob[:6]==b'\x93NUMPY'
    version=blob[6]
    h=10 if version==1 else 12
    length=int.from_bytes(blob[8:h],'little')
    header=ast.literal_eval(blob[h:h+length].decode())
    assert header['descr'] in ('<f8','=f8') and not header['fortran_order'],header
    data=blob[h+length:]
    return struct.unpack('<'+'d'*(len(data)//8),data)

connections={}
requests=[];itls=[]
for row in details:
    db=row['db_name']
    if db not in connections:
        paths=list(result.parents[2].rglob(db));assert len(paths)==1,paths
        connections[db]=sqlite3.connect('file:'+str(paths[0])+'?mode=ro',uri=True)
    blob=connections[db].execute('select arr_blob from numpy_store where id=?',(row['time_points']['__db_ref__'],)).fetchone()[0]
    times=array(blob);assert len(times)>2
    span=(times[-1]-times[1])*1000
    requests.append({'id':row['id'],'input_sha256':hashlib.sha256(row['input'].encode()).hexdigest(),
                     'ttft_ms':(times[1]-times[0])*1000,'span_ms':span,'tpot_ms':span/1023,
                     'e2e_ms':(times[-1]-times[0])*1000,'stream_events':len(times)-1})
    itls.extend((b-a)*1000 for a,b in zip(times[1:],times[2:]))

def counter(text,name):
    return sum(float(line.rsplit(' ',1)[1]) for line in text.splitlines()
               if re.match(r'^vllm:'+re.escape(name)+r'(?:\{| )',line))

warm=json.loads((R/'after-warmup-metrics.json').read_text())
server={}
for role,port in [('P','8100'),('D','8200')]:
    before=warm[port];after=completion['service_metrics'][port]
    def delta(name):return counter(after,name)-counter(before,name)
    summary={}
    for name in ['request_decode_time_seconds','request_prefill_time_seconds','request_queue_time_seconds',
                 'e2e_request_latency_seconds','time_to_first_token_seconds','inter_token_latency_seconds']:
        total=delta(name+'_sum');count=delta(name+'_count')
        summary[name]={'sum':total,'count':count,'mean':total/count if count else None}
    for key,name in [('drafts','spec_decode_num_drafts_total'),('draft_tokens','spec_decode_num_draft_tokens_total'),
                     ('accepted_tokens','spec_decode_num_accepted_tokens_total'),('preemptions','num_preemptions_total'),
                     ('generated_tokens','generation_tokens_total')]:summary[key]=delta(name)
    summary['acceptance']=summary['accepted_tokens']/summary['draft_tokens'] if summary['draft_tokens'] else None
    server[role]=summary
queries=counter(completion['service_metrics']['8100'],'prefix_cache_queries_total')-counter(warm['8100'],'prefix_cache_queries_total')
hits=counter(completion['service_metrics']['8100'],'prefix_cache_hits_total')-counter(warm['8100'],'prefix_cache_hits_total')
assert queries==16384000,(queries,hits)
assert abs(hits/queries-.9)<1e-8,(queries,hits)
duration=float(common['Benchmark Duration']['total'].split()[0])/1000
summary={'valid':True,'requests':128,'failed':0,'concurrency':32,'input_tokens_each':128000,
         'output_tokens_each':1024,'duration_seconds':duration,'output_tokens_per_second':131072/duration,
         'prefix_cache':{'queries':queries,'hits':hits,'hit_ratio':hits/queries},
         'client':{key:stats([x[key] for x in requests]) for key in ['ttft_ms','span_ms','tpot_ms','e2e_ms','stream_events']},
         'itl_ms':stats(itls),'server':server,'aisbench_latency':latency,
         'source':'CANN9.3 native operators plus modified HY4 decode paths',
         'comparability':'One modified run; historical CANN9.1 differences cannot be attributed to C8 alone.'}
(R/'decode-summary.json').write_text(json.dumps(summary,indent=2))
(R/'request-decode-metrics.json').write_text(json.dumps(requests,indent=2))
print(json.dumps(summary,indent=2))

