"""Prepare isolated profiling launch files without changing the pair experiment."""
import hashlib
import json
import os
import difflib
from pathlib import Path
import shutil
import subprocess

R = Path(__file__).resolve().parent
SOURCE = R.parent / 'hy4-prefix90-128k1k-latest-20261123-02'
assert SOURCE.is_dir()
for name in ['launch.py', 'supervise.reference.py', 'start_role.sh', 'launch_proxy.py']:
    target = R / name
    assert not target.exists(), target
    shutil.copy2(SOURCE/name, target)
text = (SOURCE/'start_proxy.sh').read_text()
text = text.replace('run_dir='+str(SOURCE), 'run_dir='+str(R))
text = text.replace(str(SOURCE/'baseline/overlay'), str(SOURCE/'candidate/overlay'))
(R/'start_proxy.sh').write_text(text)
(R/'candidate').mkdir(exist_ok=False)
# A task-local admission control caps cache reads without changing prompt text.
# Hard links preserve the original runtime; replace only this one linked file.
overlay = R/'prefill-overlay'
original = SOURCE/'candidate/overlay'
shutil.copytree(original, overlay, copy_function=os.link, symlinks=True)
request = overlay/'vllm/v1/request.py'
original_hash = hashlib.sha256((original/'vllm/v1/request.py').read_bytes()).hexdigest()
text = request.read_text()
needle = 'limits = (self.num_tokens - 1, self.prefix_cache_hit_limit, bound)'
assert text.count(needle) == 1
changed = text.replace(needle, '# Profiling load control: reuse exactly the prewarmed 90% prefix.\n        limits = (self.num_tokens - 1, self.prefix_cache_hit_limit, bound, 115200)')
tmp = request.with_suffix('.profile-control.tmp')
tmp.write_text(changed)
tmp.replace(request)
assert hashlib.sha256((original/'vllm/v1/request.py').read_bytes()).hexdigest() == original_hash
(R/'prefix-cache-control.patch').write_text(''.join(difflib.unified_diff(text.splitlines(True), changed.splitlines(True), fromfile='candidate/vllm/v1/request.py', tofile='profile-prefill/vllm/v1/request.py')))
for role in ['prefill', 'decode']:
    config = dict(profiler='torch', torch_profiler_dir=str(R/'prof'/role),
                  torch_profiler_with_stack=False, torch_profiler_with_memory=False,
                  torch_profiler_record_shapes=False, ignore_frontend=True,
                  delay_iterations=0, max_iterations=0)
    text = (SOURCE/'candidate'/f'start_{role}.sh').read_text()
    if role == 'prefill':
        text = text.replace(str(original), str(overlay))
    marker = '  --host 127.0.0.1'
    assert marker in text and '--profiler-config' not in text
    text = text.replace(marker, "  --profiler-config '"+json.dumps(config,separators=(',',':'))+"' \\\n"+marker, 1)
    text = text.replace('prefix-latest02-', 'profile-c8n8-')
    target = R/'candidate'/f'start_{role}.sh'
    target.write_text(text)
    subprocess.run(['bash', '-n', str(target)], check=True)
    (R/'prof'/role).mkdir(parents=True)
text = (SOURCE/'deploy.py').read_text()
text = text.replace("TASK='hy4-prefix90-latest02-'", "TASK='hy4-profile-c8n8-'")
start = text.index('old=oldstates()')
end = text.index("state('waiting_for_owned_services_to_stop')", start)
text = text[:start] + "old=oldstates()\nif old: raise RuntimeError('Active reservation; profiling will not stop it: '+repr(old))\n" + text[end:]
(R/'deploy.py').write_text(text)
constraints = json.loads((SOURCE/'constraints.json').read_text())
constraints.update(experiment=R.name, requests=8, concurrency=8, unique_inputs=1,
                   cases=['candidate-profile-only'], profiling=True,
                   phases=['prefill','decode'], output_tokens=1024,
                   prefix_warmup_tokens=115200,
                   actual_cache_hit_rate='measure; identical repeated inputs can reuse tail',
                   source_experiment=str(SOURCE), status='prepared',
                   execution_order=['wait for successful matching pair and lease release',
                                    'candidate cold start', 'common-prefix warmup',
                                    'start P/D profiler', 'C8/N8 single-input replay',
                                    'stop and parse', 'validate eight workers', 'package and download'])
constraints['actual_cache_hit_rate'] = 'strict 90%; P-side task-local cache-read cap of 115200 tokens'
constraints['cache_control_only_change'] = 'prefill-overlay/vllm/v1/request.py:get_max_cache_hit_length'
(R/'constraints.json').write_text(json.dumps(constraints,indent=2))
hashes={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in R.rglob('*') if f.is_file() and 'prefill-overlay' not in f.parts}
(R/'preparation.json').write_text(json.dumps(dict(files=hashes, source=str(SOURCE),
                                                model_code_modified=False,
                                                profiler='native Level1 + PipeUtilization'),indent=2))
for f in R.glob('*.py'):
    compile(f.read_text(), str(f), 'exec')
print(json.dumps(dict(prepared=str(R), files=len(hashes))))
