"""Print byte offsets of decode weight matrices in a single GGUF v3 file."""
import struct
import sys

f = open(sys.argv[1], 'rb')

def get(fmt):
    return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]

def string():
    return f.read(get('Q')).decode()

def skip(t, n=1):
    if t == 8:
        for _ in range(n):
            size = get('Q')
            f.seek(size, 1)
    elif t == 9:
        for _ in range(n):
            skip(get('I'), get('Q'))
    else:
        f.seek([1,1,2,2,4,4,4,1,0,0,8,8,8][t] * n, 1)

assert f.read(4) == b'GGUF' and get('I') == 3
nt, nk = get('Q'), get('Q')
align = 32
for _ in range(nk):
    key, t = string(), get('I')
    if key == 'general.alignment':
        assert t == 4
        align = get('I')
    else:
        skip(t)
tensors = []
for _ in range(nt):
    name = string()
    dims = [get('Q') for _ in range(get('I'))]
    t, offset = get('I'), get('Q')
    if len(dims) >= 2:
        tensors.append((name,t,*dims[:2],dims[2] if len(dims)>2 else 1,offset))
base = (f.tell() + align - 1) // align * align
for *info, offset in tensors:
    print(*info, base + offset)
