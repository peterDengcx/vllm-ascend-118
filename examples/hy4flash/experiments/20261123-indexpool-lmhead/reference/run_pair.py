"""Run latest-branch candidate first, then a fresh matching baseline."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

R = Path(__file__).resolve().parent
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
child = None
current_attempt = None

def save(name, value):
    target = R / name
    temp = target.with_suffix(target.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(target)

def state(phase, **kw):
    item = dict(phase=phase, pid=os.getpid(), updated_at=time.time(), **kw)
    save('pair-state.json', item)
    print(json.dumps(item), flush=True)

def metrics():
    return {str(port): OPENER.open(f'http://127.0.0.1:{port}/metrics', timeout=15).read().decode()
            for port in (8100, 8200)}

def run(command, log_name):
    global child
    with (R / log_name).open('x') as log_file:
        child = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        code = child.wait()
    child = None
    return code

def stop_signal(signum, _frame):
    raise InterruptedError(f'Controller received signal {signum}')

signal.signal(signal.SIGTERM, stop_signal)
signal.signal(signal.SIGINT, stop_signal)
with (R / 'pair-driver.run03.lock').open('x') as lock:
    lock.write(str(os.getpid()))
try:
    payload_hash = hashlib.sha256((R / 'prefix90_128k_frozen.jsonl').read_bytes()).hexdigest()
    assert payload_hash == '8b46b164426cf173c2315bc20e270c15cb2cab42b12a493d65472cf2d420b9a2'
    for arm in ('candidate', 'baseline'):
        current_attempt = 'candidate-run03' if arm == 'candidate' else 'baseline-run01'
        assert not (R / f'aisbench-{arm}.log').exists()
        state('deploying', arm=arm, attempt=current_attempt)
        code = run(['/usr/bin/python3', '-u', str(R / 'deploy.py'), arm, current_attempt],
                   f'deployment.{current_attempt}.driver.log')
        assert code == 0, f'Deployment exit {code}'
        deployment = json.loads((R / 'deployment-progress.json').read_text())
        assert deployment['phase'] == 'healthy' and deployment['arm'] == arm
        assert (R / 'active-arm.txt').read_text().strip() == arm
        text = (R / f'decode.{current_attempt}.log').read_text(errors='replace')
        cached = text.count('HY4_CACHE_FP32_LM_HEAD prepared shared FP32 weight:')
        graphs = text.count('HY4_INDEXPOOL_CONTEXT_AWARE captured')
        assert (cached >= 4 and graphs >= 4) if arm == 'candidate' else (cached == 0 and graphs == 0)
        save(f'{arm}-before-warmup-metrics.json', metrics())
        state('warming_common_prefix', arm=arm, attempt=current_attempt)
        started = time.time()
        request = urllib.request.Request('http://127.0.0.1:19383/v1/completions',
                                        data=(R / 'warmup.json').read_bytes(),
                                        headers={'Content-Type': 'application/json'})
        with OPENER.open(request, timeout=600) as response:
            result = json.loads(response.read())
        save(f'warmup.{current_attempt}.response.json', result)
        assert result['usage']['prompt_tokens'] == 115200 and result['usage']['completion_tokens'] == 1
        save(f'{arm}-after-warmup-metrics.json', metrics())
        save(f'{arm}-warmup-status.json', dict(success=True, duration_seconds=time.time()-started,
                                              usage=result['usage']))
        state('aisbench_running', arm=arm, attempt=current_attempt)
        started = time.time()
        code = run(['bash', str(R / 'run_aisbench.sh'), arm], f'aisbench.{current_attempt}.driver.log')
        save(f'{arm}-aisbench-exit.json', dict(exit_code=code, started_at=started, ended_at=time.time()))
        assert code == 0, f'AISBench exit {code}'
        files = list((R / 'aisbench-results' / arm).rglob('prefix90_128k_frozen.json'))
        assert len(files) == 1, files
        common = json.loads(files[0].read_text())
        with files[0].with_name('prefix90_128k_frozen_details.jsonl').open() as stream:
            details = [json.loads(line) for line in stream]
        assert len(details) == len({x['id'] for x in details}) == 128
        assert all(x['success'] and not x['error_info'] for x in details)
        assert all(x['input_tokens'] == 128000 and x['output_tokens'] == 1024 for x in details)
        assert common['Success Requests']['total'] == 128 and common['Failed Requests']['total'] == 0
        save(f'{arm}-completion.json', dict(completed=True, aisbench_exit_code=code,
                                           metrics=common, result=str(files[0]), detail_count=len(details),
                                           service_metrics=metrics(), verified_at=time.time()))
        manifest = json.loads((R / 'experiment-manifest.json').read_text())
        manifest['files'].update({f'{arm}_aisbench_result': str(files[0].parents[2]),
                                  f'{arm}_aisbench_log': str(R / f'aisbench-{arm}.log'),
                                  f'{arm}_prefill_log': str(R / f'prefill.{current_attempt}.log'),
                                  f'{arm}_decode_log': str(R / f'decode.{current_attempt}.log'),
                                  f'{arm}_proxy_log': str(R / f'proxy.{current_attempt}.log')})
        save('experiment-manifest.json', manifest)
        state('arm_completed', arm=arm, detail_count=len(details), metrics=common)
    state('completed', execution_order=['candidate', 'baseline'])
except BaseException as exc:
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
    state('failed_or_interrupted', arm=current_attempt, error=repr(exc))
    raise
finally:
    if current_attempt is not None:
        (R / f'stop.{current_attempt}.requested').touch()
