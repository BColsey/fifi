"""Extract one real matrix per benchmark geometry (first expert in 3-D stacks)."""
import hashlib
import importlib.util
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location('q8', Path(__file__).parents[1] / 'inkling-q8-row-reuse/extract.py')
q8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q8)
# (codec, K, N): GGUF block elements and bytes
wanted = {(14,2304,3072):(256,210), (17,2304,2304):(256,74),
          (19,2304,2304):(256,50), (22,2304,2304):(256,82), (40,1152,2304):(64,36),
          (16,2304,2304):(256,66), (42,1152,2304):(64,18),
          (0,64,2304):(1,4), (0,4608,64):(1,4)}
for codec, block, size, shapes in [(8,32,34,[(2304,384)]), (12,256,144,[(2304,n) for n in (384,1024,3072)]), (13,256,176,[(2304,n) for n in (384,1024,3072)]+[(3072,2304)]), (14,256,210,[(2304,n) for n in (384,1024)]+[(3072,2304)]), (18,256,98,[(2304,3072)])]:
    for k,n in shapes: wanted[codec,k,n] = block,size
with open(sys.argv[1],'rb') as f:
    r = q8.Reader(f)
    assert r.read(4) == b'GGUF' and r.number('I') == 3
    count, metadata = r.number('Q'), r.number('Q')
    align = q8.alignment(r, metadata)
    found = {}
    for _ in range(count):
        name, rank = r.name(), r.number('I')
        dims = tuple(r.number('Q') for _ in range(rank))
        kind, offset = r.number('I'), r.number('Q')
        key = (kind, *dims[:2])
        if name == 'blk.6.loop_attn_gain.weight':
            gain = (name, offset, dims)
        if key in wanted and key not in found:
            found[key] = (name, offset, dims)
    start = (f.tell()+align-1)//align*align
    for key, (name, offset, dims) in found.items():
        block, size = wanted[key]
        kind, k, n = key
        f.seek(start+offset)
        data = r.read(k*n//block*size)
        (Path(sys.argv[2])/f'extra-{kind}-{k}-{n}.bin').write_bytes(data)
        print(key,name,dims,start+offset,len(data),hashlib.sha256(data).hexdigest())
        if kind in (16,17,19,22,40,42) and len(dims)==3:
            f.seek(start+offset)
            bank=r.read(8*len(data))
            (Path(sys.argv[2])/f'bank-{kind}.bin').write_bytes(bank)
            print('expert group',kind,len(bank),hashlib.sha256(bank).hexdigest())
    name, offset, dims = gain
    assert dims == (64,2304)
    f.seek(start+offset)
    data = r.read(64*2304*4)
    (Path(sys.argv[2])/'gain.bin').write_bytes(data)
    print(name, dims, start+offset, len(data), hashlib.sha256(data).hexdigest())
    print('missing',wanted.keys()-found.keys())
