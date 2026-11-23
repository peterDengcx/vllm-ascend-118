"""Detached, task-owned candidate profiling after the matching pair completes."""
import concurrent.futures as cf
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import threading
import time
import urllib.request

R = Path(__file__).resolve().parent
SOURCE = R.parent / 'hy4-prefix90-128k1k-latest-20261123-02'
ATTEMPT = 'profile-run02'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def save(name, data):
    dest = R / name
    tmp = dest.with_suffix(dest.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(dest)


def state(phase, **extra):
    data = dict(phase=phase, pid=os.getpid(), updated_at=time.time(), **extra)
    save('profile-state.json', data)
    print(json.dumps(data), flush=True)


def post(port, route, data=None, timeout=1800):
    request = urllib.request.Request(
        f'http://127.0.0.1:{port}/{route}',
        data=json.dumps(data or {}).encode(),
        headers={'Content-Type': 'application/json'}, method='POST')
    with OPENER.open(request, timeout=timeout) as response:
        text = response.read().decode()
        return json.loads(text) if text else {'http_status': response.status}


def metrics():
    return {str(port): OPENER.open(f'http://127.0.0.1:{port}/metrics', timeout=30).read().decode()
            for port in (8100, 8200)}


def capture_control(route):
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        tasks = {port: pool.submit(post, port, route) for port in (8100, 8200)}
        return {str(port): future.result() for port, future in tasks.items()}


def collect_requests(prompt):
    barrier = threading.Barrier(8)
    lock = threading.Lock()
    active = 0
    peak = 0
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

    def one(index):
        nonlocal active, peak
        payload = dict(model='hy4-flash', prompt=prompt, max_tokens=1024,
                       temperature=0, ignore_eos=True, stream=True,
                       stream_options={'include_usage': True})
        request = urllib.request.Request('http://127.0.0.1:19383/v1/completions',
                                         data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json',
                                                  'X-Request-Id': f'profile-c8n8-{index}'})
        barrier.wait(timeout=60)
        started = time.time()
        first = None
        usage = None
        request_id = None
        chunks = 0
        output = []
        done = False
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            with OPENER.open(request, timeout=1800) as response:
                assert response.status == 200
                for line in response:
                    line = line.decode().strip()
                    if not line.startswith('data:'):
                        continue
                    text = line[5:].strip()
                    if text == '[DONE]':
                        done = True
                        break
                    data = json.loads(text)
                    assert 'error' not in data, data
                    request_id = data.get('id', request_id)
                    if data.get('usage'):
                        usage = data['usage']
                    for choice in data.get('choices', []):
                        token_text = choice.get('text', '')
                        if token_text:
                            first = first or time.time()
                            chunks += 1
                            output.append(token_text)
            assert done and usage, (done, usage)
            assert usage['prompt_tokens'] == 128000 and usage['completion_tokens'] == 1024, usage
            result = dict(index=index, success=True, id=request_id, input_sha256=prompt_hash,
                          usage=usage, started_at=started, ended_at=time.time(),
                          ttft_seconds=None if first is None else first-started,
                          chunks=chunks, output=''.join(output))
            save(f'request-{index:02d}.json', result)
            return result
        finally:
            with lock:
                active -= 1

    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, range(8)))
    assert peak == 8 and len({x['id'] for x in results}) == 8
    save('requests-completion.json', dict(requests=8, concurrency=8, max_client_inflight=peak,
                                         input_unique_count=1, input_sha256=prompt_hash,
                                         success=8, failed=0, results=results))


def validate():
    roots = sorted((R / 'prof').rglob('*_ascend_pt'))
    assert len(roots) == 8, [str(x) for x in roots]
    checks = []
    for root in roots:
        info = list(root.glob('profiler_info*.json'))
        assert info, f'Profiler did not stop: {root}'
        out = root / 'ASCEND_PROFILER_OUTPUT'
        needed = ['kernel_details.csv', 'op_statistic.csv', 'api_statistic.csv', 'trace_view.json']
        assert all((out / name).is_file() for name in needed), str(out)
        with (out / 'kernel_details.csv').open() as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            rows = sum(1 for _ in reader)
        assert len(columns) >= 40 and rows > 0, (root, len(columns), rows)
        with (out / 'trace_view.json').open() as stream:
            trace = json.load(stream)
        events = len(trace.get('traceEvents', [])) if isinstance(trace, dict) else len(trace)
        assert events > 0
        del trace
        checks.append(dict(path=str(root.relative_to(R)), kernel_rows=rows,
                           kernel_columns=len(columns), trace_events=events,
                           profiler_info=[x.name for x in info], valid=True))
    assert sum('/prefill/' in str(x) for x in roots) == 4
    assert sum('/decode/' in str(x) for x in roots) == 4
    save('validation.json', dict(valid=True, workers=8, profiles=checks))
    return checks


with (R / 'driver.lock').open('x') as lock_file:
    lock_file.write(str(os.getpid()))
started_profile = False
try:
    state('waiting_for_matching_pair')
    deadline = time.monotonic() + 7200
    while True:
        pair = json.loads((SOURCE / 'pair-state.json').read_text())
        if pair['phase'] == 'failed_or_interrupted':
            raise RuntimeError('Dependency did not complete: ' + repr(pair))
        if pair['phase'] == 'completed' and (SOURCE / 'lease-released.baseline-run01.json').exists():
            for arm in ('baseline', 'candidate'):
                item = json.loads((SOURCE / f'{arm}-completion.json').read_text())
                assert item['completed'] and item['aisbench_exit_code'] == 0 and item['detail_count'] == 128
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Waiting for matching pair')
        time.sleep(5)
    state('deploying_candidate')
    with (R / 'deployment.driver.log').open('x') as log:
        subprocess.run(['/usr/bin/python3', '-u', str(R/'deploy.py'), 'candidate', ATTEMPT],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3000)
    text = (R / f'decode.{ATTEMPT}.log').read_text(errors='replace')
    assert text.count('HY4_CACHE_FP32_LM_HEAD prepared shared FP32 weight:') >= 4
    assert text.count('HY4_INDEXPOOL_CONTEXT_AWARE captured') >= 4
    save('before-warmup-metrics.json', metrics())
    state('warming_common_prefix')
    warmup = json.loads((SOURCE/'warmup.json').read_text())
    response = post(19383, 'v1/completions', warmup)
    assert response['usage']['prompt_tokens'] == 115200 and response['usage']['completion_tokens'] == 1
    save('warmup.response.json', response)
    save('before-profile-metrics.json', metrics())
    with (SOURCE / 'prefix90_128k_frozen.jsonl').open() as stream:
        prompt = json.loads(stream.readline())['question']
    save('input-metadata.json', dict(unique_inputs=1, input_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                                    requests=8, concurrency=8, input_tokens=128000, output_tokens=1024,
                                    shared_prefix_tokens=115200))
    state('starting_profiler')
    started_profile = True
    save('start-profile.json', capture_control('start_profile'))
    state('profiling_c8n8')
    collect_requests(prompt)
    save('after-profile-metrics.json', metrics())
    state('stopping_and_parsing_profiler')
    save('stop-profile.json', capture_control('stop_profile'))
    started_profile = False
    # vLLM workers are daemons: run the framework parser in an independent process.
    (R / f'stop.{ATTEMPT}.requested').touch()
    state('offline_parse_pending')
    command = ['docker', 'exec', 'hy4-118-pd-dfly5-1121', 'bash', '-lc',
               'source /usr/local/Ascend/ascend-toolkit/set_env.sh; '
               'source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh; '
               'python -u '+str(R/'finalize_profile.py')]
    with (R/'offline-parse.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=7200)
    assert json.loads((R/'profile-state.json').read_text())['phase'] == 'completed'
except BaseException as exc:
    if started_profile:
        try:
            save('stop-profile.after-error.json', capture_control('stop_profile'))
        except BaseException as stop_error:
            save('stop-profile-error.json', dict(error=repr(stop_error)))
    state('failed', error=repr(exc))
    raise
finally:
    if (R / 'current-deployment.json').exists():
        (R / f'stop.{ATTEMPT}.requested').touch()
