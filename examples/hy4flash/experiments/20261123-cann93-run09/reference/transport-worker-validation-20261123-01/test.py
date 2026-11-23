import json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,'/home/d00943518/hy4/runs/hy4-118-tp4-dfly5-20261121-01/coord')
from host.mindie_npu_coordination import handle_request
def call(n,a):
 d=handle_request(a);(R/n).write_text(json.dumps(d,indent=2,default=str));return d
task='hy4-full-mesh-worker-validation-'+str(time.time_ns())
call('lease-submit.json',{'action':'submit','task_id':task,'agent_id':'d00943518-hy4','container_name':'hy4-118-cann93-ops-1123','devices':list(range(8)),'estimated_duration_seconds':600,'not_before':time.time(),'latest_start':time.time()+180,'preemptible':False})
g=call('lease-acquire.json',{'action':'acquire','task_id':task,'grant_ttl_seconds':120});assert g.get('task',{}).get('state')=='granted',g
owner={'task_id':task,'fence_token':g['task']['fence_token']}
call('lease-preflight.json',{**owner,'action':'preflight','start_ttl_seconds':120});call('lease-active.json',{**owner,'action':'activate','pid':os.getpid(),'heartbeat_ttl_seconds':300})
try:
 children=[];logs=[]
 for phy in range(8):
  f=(R/(str(phy)+'.log')).open('x');logs.append(f)
  env=['-e','ASCEND_RT_VISIBLE_DEVICES='+('0,1,2,3' if phy<4 else '4,5,6,7'),'-e','ASCEND_LOCAL_COMM_RES={"version":"1.3"}','-e','ASCEND_GLOBAL_RESOURCE_CONFIG={"comm_resource_config.link_priority":"PEER2PEER"}','-e','HY4_MOONCAKE_LOCAL_COMM_RES_DIR=/workspace/hy4-cann93/transport-resource-export-20261123-01/resources','-e','ASCEND_TRANSFER_TIMEOUT=5000','-e','OMP_NUM_THREADS=1']
  shell='source /usr/local/Ascend/ascend-toolkit/set_env.sh && export PYTHONPATH=/workspace/hy4-cann93/src/vllm-src:/workspace/hy4-cann93/src/vllm-ascend-hy4-flash-2c123f517504fa551379f7f5c7f3c4ac0e8b5686:$PYTHONPATH && timeout --kill-after=5s 130s python -u /workspace/hy4-cann93/'+R.name+'/rank.py '+str(phy)
  children.append(subprocess.Popen(['docker','exec',*env,'hy4-118-cann93-ops-1123','bash','-c',shell],stdout=f,stderr=subprocess.STDOUT))
 codes=[p.wait(timeout=150) for p in children]
 for f in logs:f.close()
 assert codes==[0]*8,codes
 pairs=[]
 for phy in range(8):pairs+=json.loads((R/f'{phy}.done.json').read_text())['pairs']
 assert len(pairs)==len(set(map(tuple,pairs)))==32
 d={'success':True,'worker_count':8,'directional_pairs_passed':32,'payload_bytes':262144,'visibility':'P 0-3 / D 4-7','entrypoint':'patched GlobalTE used by formal Mooncake workers','pairs':pairs}
 (R/'completion.json').write_text(json.dumps(d,indent=2));print('VALIDATION_PASS 32/32 directional transfers',flush=True)
finally:call('lease-released.json',{**owner,'action':'release','completion_confirmed':True,'free_samples':2})
