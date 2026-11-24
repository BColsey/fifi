"""Pool tail shapes before selection. Require all training tail bins improve by >=3%.
Report time ratios on later seeds independently; do not pick the tile on held-out runs.
"""
import csv,sys,pathlib,collections,statistics as st
p=pathlib.Path(sys.argv[1]);data=collections.defaultdict(lambda:collections.defaultdict(list));nwords=0
for seed in (345,731,913,2027):
 rows=list(csv.DictReader((p/f'extra-tune-{seed}.csv').open()));assert len(rows)==80*12
 for r in rows:
  key=tuple(int(r[i]) for i in ('codec','m','n','k'));data[key][r['tile']].append(float(r['us']));nwords+=int(r['exact'])
print('words exact',nwords);bins=collections.defaultdict(list)
for c,m,n,k in data:bins[c,n,k,0 if m<=8 else 1 if m<=12 else 2].append((c,m,n,k))
res=[]
for (c,n,k,b),shapes in sorted(bins.items()):
 def ratio(s,a,sl):return st.mean(data[s][a][sl])/st.mean(data[s]['deployed'][sl]);
 options=[a for a in data[shapes[0]] if a!='deployed' and max(ratio(s,a,slice(2)) for s in shapes)<.97];tile=min(options,key=lambda a:st.mean(ratio(s,a,slice(2)) for s in shapes)) if options else 'deployed';
 tr=st.mean(ratio(s,tile,slice(2)) for s in shapes);te=max(ratio(s,tile,slice(2,None)) for s in shapes);print(c,n,k,b,tile,round(tr,3),round(te,3));res.append([c,n,k,b,tile,tr,te]);
with (p/'extra-tune-selection.csv').open('w') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['codec','n','k','m_bin','tile','train_mean_ratio','test_max_ratio']);w.writerows(res)
