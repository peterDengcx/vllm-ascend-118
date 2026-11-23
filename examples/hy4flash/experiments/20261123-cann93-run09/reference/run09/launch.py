import json,os,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent
role=sys.argv[1]
if os.getpgrp()!=os.getpid():
    os.setsid()
(r/(role+'.process.json')).write_text(json.dumps({'pid':os.getpid(),'pgid':os.getpgrp(),'start_ticks':Path('/proc/self/stat').read_text().split()[21]}))
if role in ('prefill','decode'):
    command=['bash',str(r/('serve_'+role+'.sh'))]
elif role=='proxy':
    command=[sys.executable,str(r/'proxy.py'),*json.loads((r/'proxy-args.json').read_text())]
else:
    command=[sys.executable,'-m','ais_bench.benchmark.cli.main',*sys.argv[2:]]
os.execvp(command[0],command)
