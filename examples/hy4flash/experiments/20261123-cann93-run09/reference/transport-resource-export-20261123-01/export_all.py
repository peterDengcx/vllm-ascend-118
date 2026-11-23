import json,os,re,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,'/home/d00943518/hy4/runs/hy4-118-tp4-dfly5-20261121-01/coord')
from host.mindie_npu_coordination import handle_request
def call(n,a):
 d=handle_request(a);(R/n).write_text(json.dumps(d,indent=2,default=str));return d
task='hy4-export-resources-'+str(time.time_ns())
call('lease-submit.json',{'action':'submit','task_id':task,'agent_id':'d00943518-hy4','container_name':'hy4-118-cann93-ops-1123','devices':list(range(8)),'estimated_duration_seconds':600,'not_before':time.time(),'latest_start':time.time()+180,'preemptible':False})
g=call('lease-acquire.json',{'action':'acquire','task_id':task,'grant_ttl_seconds':120});assert g.get('task',{}).get('state')=='granted',g
owner={'task_id':task,'fence_token':g['task']['fence_token']}
call('lease-preflight.json',{**owner,'action':'preflight','start_ttl_seconds':120});call('lease-active.json',{**owner,'action':'activate','pid':os.getpid(),'heartbeat_ttl_seconds':300})
try:
 children=[];logs=[]
 for phy in range(8):
  f=(R/(str(phy)+'.log')).open('x');logs.append(f)
  cmd=['docker','exec','-e','ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','-e','ASCEND_LOCAL_COMM_RES={"version":"1.3"}','-e','ASCEND_GLOBAL_LOG_LEVEL=0','hy4-118-cann93-ops-1123','bash','-c','source /usr/local/Ascend/ascend-toolkit/set_env.sh && prlimit --nofile=1048576:1048576 -- timeout --kill-after=5s 90s python -u /workspace/hy4-cann93/'+R.name+'/export_rank.py '+str(phy)]
  children.append(subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT))
 codes=[p.wait(timeout=110) for p in children]
 for f in logs:f.close()
 assert codes==[0]*8,codes
 for phy in range(8):
  info=json.loads((R/(str(phy)+'.info.json')).read_text());pid=info['pid']
  out=subprocess.run(['docker','exec','hy4-118-cann93-ops-1123','bash','-c',f"grep -h 'LogEndpointList:' /root/ascend/log/debug/plog/plog-{pid}_*.log"],capture_output=True,text=True,check=True).stdout
  (R/(str(phy)+'.endpoints.log')).write_text(out)
  eps=[]
  for l in out.splitlines():
   m=re.search(r'protocol=(\w+), comm_id=(\w+), placement=(\w+), plane=([^,]*), dst_eid=([^,]*), net_instance_id=(\w+)',l)
   if not m:continue
   protocol,comm,placement,plane,dst,net=m.groups();e={'protocol':protocol,'comm_id':comm,'placement':placement}
   if plane:e['plane']=plane
   if dst:e['dst_eid']=dst
   eps.append(e)
  assert len(eps)==8,(phy,len(eps))
  assert sum(bool(e.get('dst_eid')) for e in eps)==7
  (R/'resources'/(str(phy)+'.json')).write_text(json.dumps({'version':'1.3','net_instance_id':net,'endpoint_list':eps},indent=2))
 (R/'completion.json').write_text(json.dumps({'success':True,'device_count':8,'endpoints_each':8,'direct_peers_each':7},indent=2));print('EXPORT_COMPLETE eight devices, seven direct peers each',flush=True)
finally:call('lease-released.json',{**owner,'action':'release','completion_confirmed':True,'free_samples':2})
