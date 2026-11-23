import pathlib,re,os,signal,json
r=pathlib.Path(__file__).parent
known=set()
for role in ('prefill','decode'):
 p=r/(role+'.log')
 if p.exists():known.update(int(x) for x in re.findall(r'pid=(\d+)',p.read_text(errors='replace')))
starts=[int(json.loads(p.read_text())['start_ticks']) for p in r.glob('*.process.json')]
minimum=min(starts) if starts else 0
killed=[]
for pid in known:
 if pid<=1 or pid==os.getpid():continue
 p=pathlib.Path('/proc')/str(pid)
 try:
  stat=(p/'stat').read_text().split();cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if int(stat[21])>=minimum and ('VLLM::' in cmd or 'multiprocessing' in cmd or '/bin/vllm ' in cmd):
   os.kill(pid,signal.SIGKILL);killed.append(pid)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
(r/'cleanup-recorded-workers.json').write_text(json.dumps(killed))
print('cleaned_recorded_workers',killed)
