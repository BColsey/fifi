"""Make a matched off/on table; effective rates use logical, not HW bytes."""
import csv, sys
from pathlib import Path
src=Path(sys.argv[1]); rows=list(csv.DictReader(src.open())); output=Path(sys.argv[2]); index={}; active={}; pairs=[]
tuned=any(r['arm']=='c8_6' for r in rows);codec_name={}
for r in rows:
    m,n,k=map(int,(r['m'],r['n'],r['k']));shape=(m,n,k);arm=r['arm'];index[shape,arm]=r
    if arm=='wmma':active[shape]=r # the same shape has multiple codecs
    off=None
    if arm=='visit': off=index.get((shape,'pair'),index[shape,'packed'])
    elif arm=='q6visit':off=index[shape,'q6tile']
    elif arm in ('iqvisit','iq1visit','iq2svisit','nvvisit'):off=active[shape];codec_name[shape]=arm
    elif arm=='down_selected':off=index[shape,'dense']
    elif arm=='panel_each' and m>32 and codec_name[shape]!='iq1visit':off=active[shape];r=dict(r,arm='panel_each_'+codec_name[shape])
    elif arm=='expert_visit':off=index[shape,'expert_off']
    elif arm=='heads_pair':off=index[shape,'heads_off']
    elif arm=='qkv_group':off=index[shape,'qkv_serial']
    elif arm.startswith('c') and arm.endswith(('_1','_2','_6')):
        variant=6 if tuned else 1 if n<=1024 and m<=8 else 2
        if arm.endswith('_'+str(variant)):off=index[shape,arm[:-1]+'0']
    if off:pairs.append((r,off))
cols=['M','N','K','off','on','off_bytes','on_bytes','off_us','on_us','off_GBps','on_GBps','off_TFLOPs','on_TFLOPs','exact_output_words']
with output.open('w') as f:
    w=csv.writer(f);w.writerow(cols)
    for r,off in pairs:
        values=[off['arm'],r['arm']]
        for key in ['logical_weight_bytes','us','logical_weight_GBps','TFLOPs']:values.extend((off[key],r[key]))
        w.writerow([r['m'],r['n'],r['k'],*values,r['exact_words']])
print('table rows',len(pairs),'file',output)
