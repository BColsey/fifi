"""Pool nearby row counts before selecting; retain baseline if any training bin loses."""
import csv,sys,pathlib,collections,statistics as st
p=pathlib.Path(sys.argv[1]);data=collections.defaultdict(lambda:collections.defaultdict(list));words=0
for seed in (345,731,913,2027):
 rs=list(csv.DictReader((p/f'expert-tune-{seed}.csv').open()));assert len(rs)==46*13
 for r in rs:data[int(r['codec']),int(r['m'])][r['tile']].append(float(r['us']));words+=int(r['exact'])
print('exact words',words);rows=[]
for c in (16,42):
 for lo,hi in ((2,3),(4,4),(5,8),(9,12),(13,16),(17,24),(25,32)):
  if c==16 and lo>16:continue
  shapes=[(c,m) for m in range(lo,hi+1)];base='deployed'
  def ratios(a,sl):return [st.mean(data[s][a][sl])/st.mean(data[s][base][sl]) for s in shapes]
  options=[a for a in data[shapes[0]] if a!=base and max(ratios(a,slice(2)))<.97];win=min(options,key=lambda a:st.mean(ratios(a,slice(2)))) if options else base
  tr=st.mean(ratios(win,slice(2)));te=max(ratios(win,slice(2,None)));print(c,lo,hi,win,round(tr,3),round(te,3));rows.append((c,lo,hi,win,tr,te))
with (p/'expert-tune-selection.csv').open('w') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(('codec','lo','hi','tile','train_mean_ratio','test_max_ratio'));w.writerows(rows)
