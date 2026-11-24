"""Apply production admission to prefill-only grids. Row counts are ambiguous for tiled grids."""
import collections,csv,re,sys
from pathlib import Path
p=Path(sys.argv[1]); ev=list(csv.DictReader((p/sys.argv[2]).open())); times={(r['arm'],int(r['m']),int(r['n']),int(r['k'])):float(r['us']) for r in ev}; summary=collections.defaultdict(lambda:[0.,0.,0.,0]); aggregate=collections.defaultdict(float)
tuned=any(r['arm']=='c8_6' for r in ev);panels={};active=None
for r in ev:
    if r['arm'] in ('iqvisit','iq1visit','iq2svisit','nvvisit'):active=r['arm']
    if r['arm']=='panel_each':panels[active,int(r['m'])]=float(r['us'])
for r in csv.DictReader((p/'request.csv').open()):
    name=r['kernel']; numbers=list(map(int,re.findall(r'\(int\)(\d+)',r['instantiation']))); cid=numbers[0] if numbers else -1; y=int(r['gridY']); n=int(r['gridX'])*8;k=int(r['dynamicSharedMemory'])//4; cnt=int(r['launches']);ms=float(r['ms']);values=[];weight=cnt;arm='off'
    if name=='gguf_packed_gemv':
        if cid in (8,12,13,14,18) and y in (6,7,8,63,64) and not(n==384 and y<=8 and cid!=8) or cid in (17,19,22) and 5<=y<=15:
            v=6 if tuned else 1 if n<=1024 and y<=8 else 2;arm=f'c{cid}_{v}';values=[times[arm,y,n,k]]
    elif name=='inkling_q8_reuse_pair':
        arm='visit'; values=[times[arm,m,n,k//2] for m in (2*y-1,2*y)]
    elif name=='gguf_exact_expert_tile':
        t=numbers[3]; k//=t
        if cid==14: arm='q6visit';values=[times[arm,m,n,k] for m in ([8] if y==4 else [63,64])]
        else:
            rows=list(range(max(2 if cid!=42 or t==2 else 4,t*(y-1)+1),t*y+1)); rows=[m for m in rows if not(cid==42 and t==2 and m>=4)]
            def allow(m):
                if 'expert_tuned' in sys.argv[4:]:return 4<=m<=(16 if cid==16 else 32)
                return 5<=m<=16 if cid==16 else 5<=m<=8 or 13<=m<=32
            if all(map(allow,rows)): arm='expert_visit';values=[times[arm,m,n,k] for m in rows]
    elif name=='gguf_wmma_kernel':
        arm={17:'iqvisit',19:'iq1visit',22:'iq2svisit',40:'nvvisit'}[cid];n*=4;k=1152 if cid==40 else 2304
        values=[times[arm,32*y,n,k]] # at full tile; no skipped MMA on partial tiles
        if len(sys.argv)>4 and sys.argv[4]=='panels' and y>1 and cid!=19:
            values=[panels[arm,32*y]] # scenario: sufficient aligned scratch for every eligible call
    elif name=='dense_gemv_k' and n==2304 and k==64 and y in ((6,7,8,21,63,64) if tuned else (6,7,8,21)):
        weight=cnt//2;assert weight*2==cnt; arm='heads_pair';values=[times[arm,y,2*n,k]]
    elif name=='dense_gemv_k' and ('down_selected',y,n,k) in times:
        arm='down_selected';values=[times[arm,y,n,k]]
    low=min(values)*weight/1000 if values else ms;high=max(values)*weight/1000 if values else ms
    if len(sys.argv)>3 and name=='inkling_q8_reuse_pair' and n==1024 and y==4:
        # Optional upper-admission scenario: consecutive same-stream eligible
        # calls must also belong to one host group and pass alias checks.
        pairs=int(sys.argv[3]);assert 2*pairs<=cnt
        possibilities=[((cnt-2*pairs)*times[arm,m,n,k//2]+pairs*times['qkv_group',m,2*n,k//2])/1000 for m in (7,8)]
        low=min(possibilities);high=max(possibilities)
    s=summary[name];s[0]+=ms;s[1]+=low;s[2]+=high;s[3]+=cnt if values else 0; aggregate[(name,str(cid),arm)]+=high
print('kernel observed_ms projected_low_ms projected_high_ms replaced_calls');
for n,s in summary.items(): print(n,*[round(x,3) for x in s]);
print('TOTAL',*[round(sum(s[i] for s in summary.values()),3) for i in range(4)]);print('Largest projected costs (upper endpoint by codec):');print(sorted(aggregate.items(),key=lambda x:-x[1])[:16]);
for prefix in ('c','heads_pair','visit','q6visit','expert_visit','iqvisit','iq1visit','iq2svisit','nvvisit'):
    arms=[r for r in ev if (r['arm'].startswith(prefix) if prefix=='c' else r['arm']==prefix) and int(r['exact_words'])];print(prefix,len(arms),sum(int(r['exact_words'])*(2 if prefix=='visit' else 1) for r in arms))
