import json,os,sys
from pathlib import Path
import torch,torch_npu
from mooncake.engine import TransferEngine
phy=int(sys.argv[1]);torch.npu.set_device(phy)
e=TransferEngine();ret=e.initialize('141.61.81.118','P2PHANDSHAKE','ascend','');assert ret==0,ret
r=Path('/workspace/hy4-cann93/transport-resource-export-20261123-01')
(r/(str(phy)+'.info.json')).write_text(json.dumps({'phy':phy,'pid':os.getpid(),'ret':ret}))
print('export-init',phy,os.getpid(),flush=True)
os._exit(0)
