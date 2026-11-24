"""Project isolated event times onto traced eligible grid counts, not wall time."""
import csv, sys
from collections import defaultdict
from pathlib import Path
p = Path(sys.argv[1]); events = list(csv.DictReader((p/sys.argv[2]).open()))
t = {(int(r['m']), int(r['n']), int(r['k']), r['arm']): float(r['us']) for r in events}
total = defaultdict(lambda: [0, 0, 0, 0]); arms = ('visit','q6visit')
if len(sys.argv)>3: arms = (sys.argv[3],sys.argv[4])
for r in csv.DictReader((p/'request.csv').open()):
    name = r['kernel']; inst = r['instantiation']; n = 8*int(r['gridX']); y = int(r['gridY']); k = int(r['dynamicSharedMemory'])//4; ms = float(r['ms']); count=int(r['launches']); a=arms[0]; m=[y]; replace=False
    if name=='gguf_packed_gemv' and '<(int)8,' in inst and y in (7,8,63,64): replace=(y,n,k,a) in t
    if name=='inkling_q8_reuse_pair':
        k//=2; m=[2*y-1,2*y]; replace=True
    if name=='gguf_exact_expert_tile' and '<(int)14,' in inst and n==3072 and k==4608 and y in (4,32):
        k//=2; m=[8] if y==4 else [63,64]; a=arms[1]; replace=True
    times=[t[i,n,k,a]*count/1000 for i in m] if replace else [ms]; s=total[name]; s[0]+=ms; s[1]+=min(times); s[2]+=max(times); s[3]+=count if replace else 0
for name,s in total.items(): print(name,*(round(x,3) for x in s))
print('TOTAL',*(round(sum(s[i] for s in total.values()),3) for i in range(4)))
print('arms',len(events),'visit Q8/Q6',*[sum(r['arm']==a for r in events) for a in arms]); print('compared candidate words including padded',sum(int(r['exact_words'])*(2 if r['arm']==arms[0] else 1) for r in events if r['arm'] in arms))
