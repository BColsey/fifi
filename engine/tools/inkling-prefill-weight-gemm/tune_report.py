"""Select on two randomized trials, evaluate on two unseen input/order seeds."""
import csv,pathlib,sys,statistics,collections
p=pathlib.Path(sys.argv[1]);v=collections.defaultdict(dict); counts=[]
for seed in ('345','731','913','2027'):
 rows=list(csv.DictReader((p/f'tune-{seed}.csv').open()));assert len(rows)==8*97;counts.append(sum(int(r['exact']) for r in rows))
 for r in rows:v[(r['codec'],r['m'],r['n'],r['k'])].setdefault(r['tile'],[]).append(float(r['us']))
print('exact comparisons per seed:',counts)
for shape,arms in v.items():
 base=arms.pop('deployed');ranked=sorted(arms,key=lambda k:statistics.mean(arms[k][:2]));w=ranked[0];print(*shape,'selected BM:R:BK:pipe',w,'train',base[:2],arms[w][:2],'test',base[2:],arms[w][2:],'ratio',round(statistics.mean(arms[w][2:])/statistics.mean(base[2:]),3))
