"""Parse collected daemon-worker traces from a non-daemon process and package."""
import csv
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import time
import subprocess
import sys

R = Path(__file__).resolve().parent


def save(name, data):
    target = R / name
    tmp = target.with_suffix(target.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(target)


def state(phase, **kw):
    item = dict(phase=phase, pid=os.getpid(), updated_at=time.time(), **kw)
    save('profile-state.json', item)
    print(json.dumps(item), flush=True)


def counter(text, name):
    return sum(float(line.rsplit(' ',1)[1]) for line in text.splitlines()
               if re.match(r'^vllm:'+re.escape(name)+r'(?:\{| )',line))


def parse_one(root):
    code = 'from torch_npu.profiler.profiler import analyse; import sys; analyse(profiler_path=sys.argv[1])'
    label = root.parent.name+'-'+root.name
    with (R/('offline-parse-'+label+'.log')).open('x') as log:
        subprocess.run([sys.executable, '-u', '-c', code, str(root)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=7200)


if __name__ == '__main__':
    previous = json.loads((R/'profile-state.json').read_text())
    if previous['phase'] == 'failed':
        save('profile-state.online-parse-failed.json', previous)
    try:
        roots = sorted((R/'prof').rglob('*_ascend_pt'))
        assert len(roots) == 8
        parsed = 0
        state('offline_parsing', parsed=0, workers=8, parallel_workers=4)
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(parse_one, root): root for root in roots}
            for future in cf.as_completed(futures):
                future.result()
                parsed += 1
                state('offline_parsing', parsed=parsed, workers=8, parallel_workers=4,
                      path=str(futures[future]))
        checks = []
        for i, root in enumerate(roots):
            info = list(root.glob('profiler_info*.json'))
            assert info, 'Missing stop metadata: '+str(root)
            state('validating', worker=i+1, workers=8, path=str(root))
            out = root/'ASCEND_PROFILER_OUTPUT'
            for name in ['kernel_details.csv','op_statistic.csv','api_statistic.csv','trace_view.json']:
                assert (out/name).is_file(), str(out/name)
            with (out/'kernel_details.csv').open() as stream:
                reader = csv.DictReader(stream)
                columns = reader.fieldnames
                rows = sum(1 for _ in reader)
            assert len(columns) >= 40 and rows > 0, (root, len(columns), rows)
            with (out/'trace_view.json').open() as stream:
                trace = json.load(stream)
            events = len(trace.get('traceEvents',[])) if isinstance(trace,dict) else len(trace)
            assert events > 0
            del trace
            checks.append(dict(path=str(root.relative_to(R)), valid=True, kernel_rows=rows,
                               kernel_columns=len(columns), trace_events=events,
                               profiler_info=[x.name for x in info]))
            save('validation-progress.json',dict(checked=len(checks),workers=8,profiles=checks))
        assert sum('/prefill/' in str(x) for x in roots) == 4
        assert sum('/decode/' in str(x) for x in roots) == 4
        save('validation.json',dict(valid=True, workers=8, profiles=checks,
                                    parsing='offline non-daemon torch_npu.profiler.profiler.analyse'))
        before=json.loads((R/'before-profile-metrics.json').read_text())['8100']
        after=json.loads((R/'after-profile-metrics.json').read_text())['8100']
        queries=counter(after,'prefix_cache_queries_total')-counter(before,'prefix_cache_queries_total')
        hits=counter(after,'prefix_cache_hits_total')-counter(before,'prefix_cache_hits_total')
        assert queries==8*128000
        assert hits==8*115200, (queries,hits)
        cache=dict(formal_queries=queries,formal_hits=hits,formal_hit_ratio=hits/queries,
                   prefix_warmup_tokens=115200,configured_prefix_ratio=0.9,
                   note='One identical input replayed eight times; P cache reads capped at 115200 tokens.')
        save('cache-result.json',cache)
        request=json.loads((R/'requests-completion.json').read_text())
        assert request['success']==8 and request['failed']==0 and request['input_unique_count']==1
        report=(R/'report.md').read_text()
        report += '\n## 完成结果\n\n'
        report += '- 正式请求：8 成功、0 失败；客户端最大在途请求 8；仅一条唯一输入。\n'
        report += '- 每请求实际输入/输出：128000/1024 token；两端 stop_profile 均返回 HTTP 200。\n'
        report += '- 8 个 worker 均有 stop 元数据、完整 CSV 和合法 JSON 时间线；每份 kernel_details 至少 40 列。\n'
        report += f'- 正式 P 端缓存 queries={queries:.0f}，hits={hits:.0f}，实测命中率={100*hits/queries:.4f}%。\n'
        report += '- 通过仅作用于 P 端的任务内缓存读取上限，单一输入重复 8 次仍严格保持实际 90% 命中；原始模型计算与两项优化代码保持原样。\n'
        report += '- 在线解析被 daemon worker 限制阻止；本轮使用非 daemon 进程离线解析同一份原始数据，未额外发正式请求。内置 profiler 有 RECORD 收尾警告，已通过逐 worker 产物完整性检查；原始日志保留。\n'
        (R/'report.md').write_text(report)
        state('packaging', workers=8)
        archive=R/'hy4-candidate-c8n8-128k1k-profiling.tar.gz'
        assert not archive.exists()
        with tarfile.open(archive,'x:gz',compresslevel=3) as tar:
            for item in sorted(R.iterdir()):
                if item==archive or item.name=='prefill-overlay' or item.name.endswith('.tmp'):
                    continue
                tar.add(item,arcname=R.name+'/'+item.name)
        digest=hashlib.sha256()
        with archive.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):
                digest.update(block)
        save('artifact.json',dict(path=str(archive),bytes=archive.stat().st_size,sha256=digest.hexdigest()))
        state('completed',archive=str(archive),bytes=archive.stat().st_size,workers=8)
    except BaseException as exc:
        state('offline_parse_failed',error=repr(exc))
        raise
