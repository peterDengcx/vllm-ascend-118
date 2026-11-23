import json,os,sys,time,traceback
from pathlib import Path
R=Path('/workspace/hy4-cann93/transport-worker-validation-20261123-01')
phy=int(sys.argv[1])
try:
 import torch,torch_npu
 from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
 logical=phy if phy<4 else phy-4;torch.npu.set_device(logical)
 x=torch.full((262144,),11+phy,dtype=torch.uint8,device='npu');y=torch.zeros_like(x);torch.npu.synchronize()
 e=global_te.get_transfer_engine('141.61.81.118',None)
 for t in [x,y]:assert e.register_memory(t.data_ptr(),t.numel())==0
 (R/f'{phy}.json').write_text(json.dumps({'port':e.get_rpc_port(),'ptr':x.data_ptr(),'pid':os.getpid(),'visible':os.environ['ASCEND_RT_VISIBLE_DEVICES'],'logical':logical,'resource':json.loads(os.environ['ASCEND_LOCAL_COMM_RES'])}))
 deadline=time.monotonic()+80
 pairs=[]
 for peer in (range(4,8) if phy<4 else range(4)):
  f=R/f'{peer}.json'
  while not f.exists():
   if time.monotonic()>deadline:raise TimeoutError('peer metadata timeout')
   time.sleep(.1)
  d=json.loads(f.read_text())
  ret=e.batch_transfer_sync_read('141.61.81.118:'+str(d['port']),[y.data_ptr()],[d['ptr']],[y.numel()])
  torch.npu.synchronize();assert ret==0 and torch.all(y==11+peer).item(),(phy,peer,ret)
  pairs.append([phy,peer]);print('PASS',phy,peer,flush=True)
 (R/f'{phy}.done.json').write_text(json.dumps({'success':True,'pairs':pairs}))
 while not all((R/f'{i}.done.json').exists() for i in range(8)):
  if time.monotonic()>deadline:raise TimeoutError('completion barrier timeout')
  time.sleep(.1)
 os._exit(0)
except BaseException:
 traceback.print_exc()
 (R/f'{phy}.error').write_text(traceback.format_exc());os._exit(1)
