import torch,torch_npu,sys
from types import SimpleNamespace
from vllm_ascend.model_executor.offloader.prefetch import _ModuleOffloader
from vllm.model_executor.offloader.prefetch import PrefetchOffloader
torch.cuda.current_stream=torch.npu.current_stream
torch.cuda.is_current_stream_capturing=torch.npu.is_current_stream_capturing
pin=sys.argv[1]=='pin'
cpu=torch.ones((512,512),pin_memory=pin)
gpu=torch.empty_like(cpu,device='npu:0');x=torch.ones_like(gpu)
stream=torch.npu.Stream()
obj=_ModuleOffloader.__new__(_ModuleOffloader)
obj._buffer_pool=True;obj.copy_stream=stream
obj._param_offloaders={'w':SimpleNamespace(_cpu_storage=cpu,_gpu_buffer=gpu)}
obj._copy_done_event=torch.npu.Event();obj._prefetch_in_capture=False;obj._event_valid_for_eager=False
loader=PrefetchOffloader.__new__(PrefetchOffloader);loader.copy_stream=stream;loader.module_offloaders=[obj]
_ModuleOffloader.start_onload_to_static(obj)
loader.sync_prev_onload();torch.npu.synchronize()
g=torch.npu.NPUGraph()
with torch.npu.graph(g):
 for i in range(4):
  obj.start_onload_to_static()
  loader._wait_for_layer(0)
  y=x+gpu
 obj.start_onload_to_static()
 loader.join_after_forward()
for value in (3,5,7):
 cpu.fill_(value)
 g.replay()
 torch.npu.synchronize()
 assert torch.all(y==value+1).item(), (value,y[0,0].item())
print('PASS capture/replay',sys.argv[1],flush=True)
