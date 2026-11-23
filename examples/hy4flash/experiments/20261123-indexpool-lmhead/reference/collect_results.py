"""Validate retained AISBench results and calculate comparable service metrics."""
import csv
import hashlib
import json
from pathlib import Path
import re

R = Path(__file__).resolve().parent
def number(value):
    return float(str(value).split()[0])

def counter(text, name):
    return sum(float(line.rsplit(' ', 1)[1]) for line in text.splitlines()
               if re.match(r'^vllm:' + re.escape(name) + r'(?:\{| )', line))

def collect(arm, attempt):
    root = R / 'aisbench-results' / arm
    candidates = list(root.rglob('prefix90_128k_frozen.json'))
    assert len(candidates) == 1, candidates
    path = candidates[0]
    common = json.loads(path.read_text())
    with path.with_suffix('.csv').open() as stream:
        latency = {row['Performance Parameters']: row for row in csv.DictReader(stream)}
    with path.with_name(path.stem + '_details.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 128 and len({row['id'] for row in rows}) == 128
    assert all(row['success'] and not row['error_info'] for row in rows)
    assert all(row['input_tokens'] == 128000 and row['output_tokens'] == 1024 for row in rows)
    assert common['Total Requests']['total'] == common['Success Requests']['total'] == 128
    assert common['Failed Requests']['total'] == 0
    assert common['Total Input Tokens']['total'] == 16384000
    assert common['Total Generated Tokens']['total'] == 131072
    assert common['Max Concurrency']['total'] == 32
    completion = json.loads((R / f'{arm}-completion.json').read_text())
    assert completion['completed'] and completion['aisbench_exit_code'] == 0
    warmup = json.loads((R / f'warmup.{attempt}.response.json').read_text())
    assert warmup['usage']['prompt_tokens'] == 115200 and warmup['usage']['completion_tokens'] == 1
    pmetrics = completion['service_metrics']['8100']
    total_queries = counter(pmetrics, 'prefix_cache_queries_total')
    total_hits = counter(pmetrics, 'prefix_cache_hits_total')
    formal_queries = total_queries - warmup['usage']['prompt_tokens']
    assert formal_queries == 16384000, (arm, total_queries)
    duration = number(common['Benchmark Duration']['total']) / 1000
    output_tps = 131072 / duration
    input_hashes = {str(row['id']): hashlib.sha256(row['input'].encode()).hexdigest() for row in rows}
    summary = dict(attempt=attempt, result_directory=str(path.parents[2]), exit_code=0,
                   validated_requests=len(rows), common_metrics=common, latency=latency,
                   duration_seconds=duration, service_output_tokens_per_second=output_tps,
                   output_tokens_per_second_per_total_card=output_tps / 8,
                   output_tokens_per_second_per_decode_card=output_tps / 4,
                   mean_decode_span_ms=number(latency['E2EL']['Average']) - number(latency['TTFT']['Average']),
                   prefix_cache=dict(total_queries=total_queries, total_hits=total_hits,
                                     warmup_queries=115200, formal_queries=formal_queries,
                                     formal_hit_ratio=total_hits / formal_queries),
                   input_hashes=input_hashes)
    return summary

result = {arm: collect(arm, 'candidate-run03' if arm == 'candidate' else 'baseline-run01') for arm in ('candidate', 'baseline') if (R / (arm + '-completion.json')).exists()}
if 'candidate' in result and 'baseline' in result:
    a, b = result['baseline'], result['candidate']
    assert a['input_hashes'] == b['input_hashes'], 'Formal inputs differ'
    result['inputs_identical'] = True
    result['output_throughput_change_percent'] = 100 * (b['service_output_tokens_per_second'] / a['service_output_tokens_per_second'] - 1)
    result['latency_change_percent'] = {
        metric: 100 * (number(b['latency'][metric]['Average']) / number(a['latency'][metric]['Average']) - 1)
        for metric in ('TTFT', 'TPOT', 'ITL', 'E2EL')
    }
(R / 'comparison.json').write_text(json.dumps(result, indent=2))
print(json.dumps({arm: {k: v for k, v in item.items() if k not in ('input_hashes', 'latency')}
                  if isinstance(item, dict) and 'attempt' in item else item
                  for arm, item in result.items()}, indent=2))
