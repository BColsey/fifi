import csv, re, sys
for row in csv.DictReader(open(sys.argv[1])):
    if row['kernel']!='gguf_packed_gemv' or int(row['gridY'])==1: continue
    codec,block,size=map(int,re.findall(r'\(int\)(\d+)',row['instantiation'])[:3]); print(codec,block,size,int(row['gridX'])*8,int(row['dynamicSharedMemory'])//4,row['gridY'])
