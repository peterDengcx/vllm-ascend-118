"""Own one modified serving run, its reservation, requests and cleanup."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

R=Path(__file__).resolve().parent
C='/workspace/hy4-cann93/'+R.name
CONTAINER='hy4-118-cann93-ops-1123'
TASK='hy4-prefix90-cann93-modified-run09-20261123'
sys.path.insert(0,'/home/d00943518/hy4/runs/hy4-118-tp4-dfly5-20261121-01/coord')
from host.mindie_npu_coordination import handle_request

OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))
children={}
logs=[]
owner=None
heartbeat_stop=threading.Event()
heartbeat_errors=[]

def save(name,value):
    path=R/name
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,default=str))
    tmp.replace(path)

def state(phase,**kw):
    value=dict(phase=phase,pid=os.getpid(),updated_at=time.time(),**kw)
    save('state.json',value)
    print(json.dumps(value),flush=True)

def endpoint(url,timeout=10):
    return OPENER.open(url,timeout=timeout)

def metrics():
    return {str(port):endpoint(f'http://127.0.0.1:{port}/metrics').read().decode()
            for port in (8100,8200)}

def launch(role):
    log=(R/(role+'.log')).open('x')
    logs.append(log)
    cmd=['bash',str(R/'run_aisbench.sh')] if role=='aisbench' else ['docker','exec','-e','HY4_MOONCAKE_LOCAL_COMM_RES_DIR='+C+'/communication-resources','-e','ASCEND_GLOBAL_RESOURCE_CONFIG='+json.dumps({'comm_resource_config.link_priority':'PEER2PEER'}),CONTAINER]
    if role in ('prefill','decode'):
        cmd+=['bash',C+'/start_'+role+'.sh']
    elif role=='proxy':
        cmd+=['python',C+'/launch.py','proxy']
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    children[role]=child
    save('host-processes.json',{k:{'pid':p.pid,'exit_code':p.poll()} for k,p in children.items()})
    return child

def check_children():
    failures={k:p.poll() for k,p in children.items() if k!='aisbench' and p.poll() is not None}
    if failures:raise RuntimeError('Service exited: '+repr(failures))
    if heartbeat_errors:raise RuntimeError('Lease heartbeat failed: '+repr(heartbeat_errors))

def wait_ready(url,timeout):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        check_children()
        try:
            with endpoint(url,3) as q:
                if q.status==200:return
        except Exception:pass
        time.sleep(3)
    raise TimeoutError(url)

def heartbeat():
    while not heartbeat_stop.wait(30):
        try:save('lease-heartbeat.json',handle_request({**owner,'action':'heartbeat','heartbeat_ttl_seconds':600}))
        except Exception as exc:heartbeat_errors.append(repr(exc));return

def interrupted(signum,_):
    raise InterruptedError('Signal '+str(signum))

signal.signal(signal.SIGTERM,interrupted)
signal.signal(signal.SIGINT,interrupted)
with (R/'run.lock').open('x') as f:f.write(str(os.getpid()))
try:
    state('acquiring_lease')
    submitted=handle_request({'action':'submit','task_id':TASK,'agent_id':'d00943518-hy4',
        'container_name':CONTAINER,'devices':list(range(8)),'estimated_duration_seconds':7200,
        'not_before':time.time(),'latest_start':time.time()+21600,'preemptible':False,
        'service_port':19383,'service_ports':[19383,8100,8200,13445,30000,30100,30101,30102,30103]})
    save('lease-submit.json',submitted)
    deadline=time.monotonic()+21600
    while True:
        acquired=handle_request({'action':'acquire','task_id':TASK,'grant_ttl_seconds':600})
        save('lease-acquire.json',acquired)
        if acquired.get('task',{}).get('state')=='granted':break
        state('waiting_for_devices',lease_status=acquired.get('status'),reason=acquired.get('reason'),task_state=acquired.get('task',{}).get('state'))
        if time.monotonic()>deadline:raise TimeoutError('Eight-card lease wait expired')
        time.sleep(20)
    owner={'task_id':TASK,'fence_token':acquired['task']['fence_token']}
    save('lease-preflight.json',handle_request({**owner,'action':'preflight','start_ttl_seconds':600}))
    active=handle_request({**owner,'action':'activate','pid':os.getpid(),'heartbeat_ttl_seconds':600})
    save('lease-active.json',active)
    assert active.get('task',{}).get('state')=='active',active
    threading.Thread(target=heartbeat,daemon=True).start()
    state('loading_services')
    launch('prefill');launch('decode')
    wait_ready('http://127.0.0.1:8100/health',2400)
    wait_ready('http://127.0.0.1:8200/health',2400)
    launch('proxy');wait_ready('http://127.0.0.1:19383/healthcheck',90)
    save('before-warmup-metrics.json',metrics())
    state('warming_common_prefix')
    started=time.time()
    request=urllib.request.Request('http://127.0.0.1:19383/v1/completions',data=(R/'warmup.json').read_bytes(),headers={'Content-Type':'application/json'})
    with OPENER.open(request,timeout=1200) as q:warmup=json.loads(q.read())
    save('warmup-response.json',warmup)
    assert warmup['usage']['prompt_tokens']==115200 and warmup['usage']['completion_tokens']==1,warmup
    save('warmup-status.json',dict(duration_seconds=time.time()-started,usage=warmup['usage']))
    save('after-warmup-metrics.json',metrics())
    state('aisbench_running')
    started=time.time();p=launch('aisbench')
    while p.poll() is None:
        check_children();time.sleep(3)
    save('aisbench-exit.json',dict(exit_code=p.returncode,started_at=started,ended_at=time.time()))
    assert p.returncode==0,'AISBench exit '+str(p.returncode)
    files=list((R/'aisbench-results/modified-run09').rglob('prefix90_128k_frozen.json'))
    assert len(files)==1,files
    common=json.loads(files[0].read_text())
    details=[json.loads(line) for line in files[0].with_name('prefix90_128k_frozen_details.jsonl').read_text().splitlines()]
    assert len(details)==len({x['id'] for x in details})==128
    assert all(x['success'] and not x['error_info'] and x['input_tokens']==128000 and x['output_tokens']==1024 for x in details)
    assert common['Success Requests']['total']==128 and common['Failed Requests']['total']==0
    save('completion.json',dict(completed=True,metrics=common,detail_count=len(details),result=str(files[0]),service_metrics=metrics()))
    state('completed',detail_count=len(details),result=str(files[0]))
except BaseException as exc:
    state('failed_or_interrupted',error=repr(exc))
    raise
finally:
    heartbeat_stop.set()
    for role in reversed(list(children)):
        info=R/(role+'.process.json')
        if info.exists():
            value=json.loads(info.read_text())
            script='import json,os,signal,pathlib; d='+repr(value)+'; p=d["pid"]; s=pathlib.Path("/proc")/str(p)/"stat"; exists=s.exists(); ok=exists and s.read_text().split()[21]==d["start_ticks"]; assert not ok or os.getpgid(p)==d["pgid"]==p; os.killpg(p,signal.SIGTERM) if ok else None'
            subprocess.run(['docker','exec',CONTAINER,'python','-c',script],capture_output=True)
    for child in children.values():
        try:child.wait(timeout=45)
        except subprocess.TimeoutExpired:pass
    subprocess.run(['docker','exec',CONTAINER,'python',C+'/cleanup_recorded_workers.py'],capture_output=True)
    if owner:
        released=handle_request({**owner,'action':'release','completion_confirmed':True,'free_samples':2})
        save('lease-released.json',released)
        print('LEASE_RELEASED',released.get('status'),flush=True)


