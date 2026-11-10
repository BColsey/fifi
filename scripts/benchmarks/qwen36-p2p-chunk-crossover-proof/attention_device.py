#!/usr/bin/env python3
"""Guarded private-server P2P prefill/decode A/B; called only by the pair runner."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parent
TASK=ROOT.parent
MEASURE=TASK/'measurement'
PEER_ROOT=TASK/'peer-check'
spec=importlib.util.spec_from_file_location('peer_evidence',PEER_ROOT/'attention_device.py')
peer=importlib.util.module_from_spec(spec);spec.loader.exec_module(peer)
spec2=importlib.util.spec_from_file_location('compare_prefill',MEASURE/'compare_prefill.py')
compare=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(compare)
LOADED_COMPARATOR_SHA=hashlib.sha256((MEASURE/'compare_prefill.py').read_bytes()).hexdigest()
LOCK=peer.LOCK
DEVICE=peer.DEVICE
PAIR_VISIBLE=peer.PAIR_VISIBLE
CONFIG=Path('/home/ben/fifi-qwen36/qwen36-p2p-256k.json')
MODEL=Path('/home/ben/fifi-scratch/a81ebdf26a2a/qwen36-local/model/Qwen3.6-35B-A3B-Q4_K_M.gguf')
PORT=18837
URL=f'http://127.0.0.1:{PORT}'
KEY_ENV='FIFI_BENCH_EPHEMERAL_KEY'
DRIVERS=('scripts/bench_p2p_prefill.py','scripts/bench_p2p_decode.py','compare_prefill.py')
CHUNKS={'control':256,'candidate':512}
HOST_REQUESTED=[1395864370,3489660928]
HOST_SELECTED=[1434451968,3548381184]
PARITY_ROOT=Path('/home/ben/fifi-scratch/76dec8a61d04/p2p-chunk512-validation/device-check')
REFERENCE_ROOT=TASK/'placement-check'
SMOKE_ROOT=TASK/'chunk-size-check'


def sha(path):return peer.sha(path)


def private_config():
    value=json.loads(CONFIG.read_text())
    engines=value.get('engines')
    if (not isinstance(engines,list) or len(engines)!=1 or engines[0].get('model')!='fifi-q36'
            or engines[0].get('family')!='qwen3_6_35b_a3b'
            or Path(engines[0].get('model_dir','')).resolve()!=MODEL.parent.resolve()):
        raise ValueError('unexpected model or engine configuration')
    q=engines[0].get('qwen36',{})
    if q.get('context_limit')!=262144 or q.get('kv_dtype')!='q8' or q.get('cuda_devices')!=[0,1]:
        raise ValueError('unexpected context, KV mode or CUDA devices')
    value.update({'listen':f'127.0.0.1:{PORT}','api_key_env':KEY_ENV,
                  'prefix_reuse':False,'sample_trace':False,'web_dir':None})
    return value


def config_bytes():return (json.dumps(private_config(),sort_keys=True,indent=2)+'\n').encode()


def check_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',PORT))


def model_fingerprint():
    x=json.loads((MEASURE/'model-fingerprint.json').read_text())
    stat=MODEL.stat()
    if (not MODEL.is_file() or x.get('resolved')!=str(MODEL.resolve()) or x.get('size')!=stat.st_size
            or x.get('inode')!=stat.st_ino or x.get('mtime_ns')!=stat.st_mtime_ns
            or not re.fullmatch(r'[0-9a-f]{64}',x.get('sha256',''))):
        raise ValueError('model changed since its hash was recorded')
    return x


def verify_drivers():
    manifest=json.loads((MEASURE/'driver-manifest.json').read_text())
    if set(manifest)!=set(DRIVERS):raise ValueError('driver manifest has unexpected entries')
    if manifest['compare_prefill.py']!=LOADED_COMPARATOR_SHA:
        raise ValueError('loaded comparator differs from the recorded file')
    for name in DRIVERS:
        path=MEASURE/name
        if path.is_symlink() or not path.is_file() or sha(path)!=manifest[name]:
            raise ValueError('measurement driver changed: '+name)
    return manifest


def validate_peer_proof(path,info):
    campaign=Path(path)
    if (not campaign.is_absolute() or campaign.is_symlink() or campaign.parent.resolve()!=PEER_ROOT.resolve()
            or not re.fullmatch(r'pair-reserved-[a-z0-9_]+',campaign.name)):
        raise ValueError('invalid peer correctness campaign')
    required=[campaign/'complete.json',campaign/'restore.json',campaign/'attention/summary.json',campaign/'guard.jsonl']
    if any(p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(campaign.resolve()) for p in required):
        raise ValueError('missing peer correctness artifacts')
    complete,restore,summary=(json.loads(p.read_text()) for p in required[:3])
    outcomes=summary.get('outcomes',[])
    if (complete.get('restored') is not True or complete.get('result_type')!='p2p_diagnostic_matrix'
            or restore.get('restored') is not True or summary.get('mode')!='full' or summary.get('inputs')!=info
            or len(outcomes)!=len(peer.CASES) or summary.get('tests_failed')):
        raise ValueError('peer correctness receipt is incomplete or differs from build inputs')
    for item,case in zip(outcomes,peer.CASES,strict=True):
        label,arm,name,rows,stage=case
        if (item.get('case')!=label or item.get('arm')!=arm or item.get('test')!=name
                or item.get('passed') is not True or item.get('exit')!=0):
            raise ValueError('peer case failed or was replaced')
        log=campaign/'attention'/item.get('log','')
        if log.is_symlink() or not log.is_file() or not log.resolve().is_relative_to(campaign.resolve()) or sha(log)!=item.get('sha256'):
            raise ValueError('peer log missing or changed')
        peer.validate(name,log.read_text(),rows,stage)
    samples=[json.loads(line) for line in required[3].read_text().splitlines()]
    if not samples or any(s['builders'] or s['other_jobs'] or s['unexpected'] or not s['states_ok']
                          or s['clients'] or s['listeners'] or any(s['non_target_utilization'].values()) for s in samples):
        raise ValueError('peer correctness guard was not clear')
    return {'summary_sha256':sha(required[2]),'guard_sha256':sha(required[3]),'guard_samples':len(samples)}


def validate_512_parity(path):
    campaign=Path(path)
    if (not campaign.is_absolute() or campaign.is_symlink() or campaign.parent.resolve()!=PARITY_ROOT.resolve()
            or not re.fullmatch(r'pair-reserved-[a-z0-9_]+',campaign.name)):
        raise ValueError('invalid 512-row parity campaign')
    files=[campaign/'complete.json',campaign/'restore.json',campaign/'attention/summary.json',
           campaign/'attention/00.log',campaign/'guard.jsonl']
    if any(p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(campaign.resolve()) for p in files):
        raise ValueError('missing 512-row parity artifact')
    complete,restore,summary=(json.loads(p.read_text()) for p in files[:3]);log=files[3].read_text()
    expected='family::qwen36::cuda::card_chunk_device_tests::max_card_chunk_matches_the_per_token_path_on_physical_peers'
    if (complete.get('restored') is not True or restore.get('restored') is not True or summary.get('mode')!='full'
            or summary.get('inputs',{}).get('commit')!='7142c14bf321238e9d5850a7afa640f7e26a6fb8'
            or summary.get('inputs',{}).get('binary_sha256')!='99378859c009c91a3df8d9085e5019e60c13fdb1dc572b944f4ef40d11a1ade0'
            or summary.get('inputs',{}).get('gpu_uuids')!=PAIR_VISIBLE.split(',')
            or summary.get('tests_passed')!=[expected] or summary.get('tests_failed')
            or len(summary.get('outcomes',[]))!=1 or summary['outcomes'][0].get('passed') is not True
            or summary['outcomes'][0].get('sha256')!=sha(files[3])
            or 'CUDA devices visible: 2' not in log
            or '512 rows, winner' not in log or 'test result: ok. 1 passed; 0 failed; 0 ignored;' not in log
            or any(f'card {card} KV8 staging: 2048 rows' not in log for card in (0,1))):
        raise ValueError('512-row physical-peer correctness proof is incomplete')
    samples=[json.loads(l) for l in files[4].read_text().splitlines()]
    if not samples or any(s['builders'] or s['other_jobs'] or s['unexpected'] or not s['states_ok']
            or s['clients'] or s['listeners'] or any(s['non_target_utilization'].values()) for s in samples):
        raise ValueError('512-row parity guard was not clear')
    source=Path('/home/ben/fifi-scratch/76dec8a61d04/p2p-chunk512-validation/wt')
    changed=subprocess.check_output(['git','-C',str(source),'diff','--name-only',
             '76dec8a61d040efa97beefe273bed8dc9e68ff57','7142c14bf321238e9d5850a7afa640f7e26a6fb8'],text=True,timeout=10).splitlines()
    if set(changed)!={'engine/src/family/qwen36/cuda/card_chunk_device_tests.rs','scripts/qual_facts.json'}:
        raise ValueError('512-row parity source differs from server by more than test files')
    hashes={str(p.relative_to(campaign)):sha(p) for p in files}
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()


def validate_reference(path,info):
    campaign=Path(path)
    if (not campaign.is_absolute() or campaign.is_symlink() or campaign.parent.resolve()!=REFERENCE_ROOT.resolve()
            or not re.fullmatch(r'pair-reserved-[a-z0-9_]+',campaign.name)):
        raise ValueError('invalid 256-row memory reference')
    files=[campaign/'complete.json',campaign/'restore.json',campaign/'attention/summary.json',
           campaign/'guard.jsonl',campaign/'attention/control-engine.log']
    if any(p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(campaign.resolve()) for p in files):
        raise ValueError('missing 256-row reference artifact')
    complete,restore,summary=(json.loads(p.read_text()) for p in files[:3]);inputs=summary.get('inputs',{})
    arm=summary.get('arms',{}).get('control',{});free=arm.get('free_mib_after_warmup',{})
    if (complete.get('restored') is not True or restore.get('restored') is not True or summary.get('mode')!='full'
            or inputs.get('peer')!=info['peer'] or inputs.get('model')!=info['model']
            or inputs.get('private_config_sha256')!=info['private_config_sha256']
            or inputs.get('placements',{}).get('control',{}).get('requested')!=HOST_REQUESTED
            or arm.get('server_log_sha256')!=sha(files[4])
            or set(free)!=set(PAIR_VISIBLE.split(',')) or any(v < 1024+256 for v in free.values())):
        raise ValueError('256-row reference does not leave the declared 256 MiB plus 1 GiB floor')
    check_path(files[4].read_text(),'control')
    samples=[json.loads(l) for l in files[3].read_text().splitlines()]
    if not samples or any(s['builders'] or s['other_jobs'] or s['unexpected'] or not s['states_ok']
            or s['clients'] or s['listeners'] or any(s['non_target_utilization'].values())
            or set(s['pair_free_mib'])!=set(PAIR_VISIBLE.split(','))
            or any(v < 1024 for v in s['pair_free_mib'].values()) for s in samples):
        raise ValueError('256-row reference guard was not clear')
    hashes={str(p.relative_to(campaign)):sha(p) for p in files}
    return {'digest_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
            'free_mib_after_warmup':free,'declared_extra_mib_per_card':256,
            'note':'256 MiB is a planning allowance, not a measured upper bound for the 512-row allocation; the new server must also pass its own admission and the 1 GiB live guard.'}


def preflight():
    info=peer.preflight()
    proof=validate_peer_proof(os.environ.get('FIFI_PEER_PROOF',''),info)
    drivers=verify_drivers();model=model_fingerprint();config=config_bytes();check_port()
    base={'peer':info,'peer_proof':proof,'drivers':drivers,'model':model,
            'private_config_sha256':hashlib.sha256(config).hexdigest(),'chunk_rows':256,
            'scope':'Private served-model smoke or matched A/B; not long-context qualification.'}
    return {**base,'chunk_rows_by_arm':CHUNKS,'host_requested':HOST_REQUESTED,'host_selected':HOST_SELECTED,
            'parity_512_sha256':validate_512_parity(os.environ.get('FIFI_512_PROOF','')),
            'memory_reference':validate_reference(os.environ.get('FIFI_256_REFERENCE',''),base)}


def validate_smoke_receipt(path,info):
    campaign=Path(path)
    if (not campaign.is_absolute() or campaign.is_symlink() or campaign.parent.resolve()!=SMOKE_ROOT.resolve()
            or not re.fullmatch(r'pair-reserved-[a-z0-9_]+',campaign.name)):
        raise ValueError('invalid benchmark smoke campaign')
    files=[campaign/'complete.json',campaign/'restore.json',campaign/'attention/summary.json',
           campaign/'attention/candidate-engine.log',campaign/'attention/candidate-warmup.json',campaign/'guard.jsonl']
    if any(p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(campaign.resolve()) for p in files):
        raise ValueError('missing benchmark smoke artifact')
    complete,restore,summary=(json.loads(p.read_text()) for p in files[:3])
    if (complete.get('restored') is not True or complete.get('result_type')!='p2p_correctness_tests'
            or restore.get('restored') is not True or summary.get('mode')!='smoke' or summary.get('inputs')!=info
            or summary.get('candidate_warmup') is not True):
        raise ValueError('benchmark smoke does not match inputs and restoration')
    arm=summary.get('arms',{}).get('candidate',{})
    if arm.get('server_log_sha256')!=sha(files[3]) or arm.get('warmup_sha256')!=sha(files[4]):
        raise ValueError('benchmark smoke log or sample changed')
    check_path(files[3].read_text(),'candidate')
    compare.summarize(json.loads(files[4].read_text()),files[3].read_text(),min_samples=1)
    samples=[json.loads(line) for line in files[5].read_text().splitlines()]
    if (not samples or any(s['builders'] or s['other_jobs'] or s['unexpected'] or not s['states_ok']
            or s['clients'] or s['listeners'] or any(s['non_target_utilization'].values())
            or set(s['pair_free_mib'])!=set(PAIR_VISIBLE.split(','))
            or any(v < 1024 for v in s['pair_free_mib'].values()) for s in samples)):
        raise ValueError('placement smoke guard or VRAM floor was not clear')
    # Bind all validated artifacts, not just the generic completion marker.
    hashes={str(p.relative_to(campaign)):sha(p) for p in files}
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()


def check_path(text,arm):
    if arm not in CHUNKS:raise ValueError('unknown benchmark arm')
    if f'q36 cuda: chunked prefill engaged at {CHUNKS[arm]} rows' not in text or 'chunked prefill refused:' in text:
        raise ValueError('requested chunk path was not admitted')
    if any(f'card {c} KV8 staging: 32768 rows' not in text for c in (0,1)):
        raise ValueError('both placements must admit KV8 staging on both cards')
    selected=[int(n) for n in re.findall(r'cuda mapped experts: placing (\d+) bytes in host memory',text)]
    if selected!=HOST_SELECTED:
        raise ValueError('selected host expert bytes differ from the placement plan')


def ready(key):
    req=urllib.request.Request(URL+'/v1/system',headers={'Authorization':'Bearer '+key})
    with urllib.request.urlopen(req,timeout=2) as response:data=json.load(response)
    models=[e for e in data.get('engines',[]) if e.get('model')=='fifi-q36']
    return len(models)==1 and models[0].get('state')=='ready' and models[0].get('context_limit')==262144


def pair_processes():
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=5)
    result=set()
    for line in out.splitlines():
        if not line.strip():continue
        parts=line.split(',')
        if len(parts)!=2:raise ValueError('malformed GPU process record')
        gpu,pid=parts[0].strip(),int(parts[1])
        if gpu in set(PAIR_VISIBLE.split(',')):result.add((gpu,pid))
    return result


def memory_free():
    out=subprocess.check_output(['nvidia-smi','-i',PAIR_VISIBLE,'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=5)
    rows={p[0].strip():int(p[1]) for line in out.splitlines() if len(p:=line.split(','))==2}
    if set(rows)!=set(PAIR_VISIBLE.split(',')) or any(x<1024 for x in rows.values()):
        raise RuntimeError('private engine is below the 1 GiB per-card free-VRAM floor')
    return rows


def wait_ready(process,key,binary):
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        if process.poll() is not None:raise RuntimeError('private server exited before ready')
        try:
            if ready(key):
                if Path(f'/proc/{process.pid}/exe').resolve()!=Path(binary).resolve():
                    raise RuntimeError('private server executable changed')
                if pair_processes()!={(gpu,process.pid) for gpu in PAIR_VISIBLE.split(',')}:
                    raise RuntimeError('private server does not own the GPU pair')
                return memory_free()
        except (OSError,urllib.error.URLError,TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError('private server did not become ready')


def stop_server(process):
    if process.poll() is None:
        process.terminate()
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill();process.wait(timeout=15)
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        if not pair_processes():return
        time.sleep(1)
    raise RuntimeError('private server did not release both GPUs')


def driver(script,args,out,env,timeout=1200):
    cmd=[sys.executable,str(MEASURE/'scripts'/script),*args,'--url',URL,'--model','fifi-q36',
         '--expected-context','262144','--api-key-env',KEY_ENV,'--timeout','600']
    with out.open('x') as log:
        subprocess.run(cmd,env=env,cwd=MEASURE,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,check=True)


def env_for(key,arm='control'):
    if arm not in CHUNKS:raise ValueError('unknown chunk arm')
    env={k:v for k,v in os.environ.items() if not k.startswith(('FIFI_','Q36_','INFERENCE_','ENGINESRV_'))}
    first,second=HOST_REQUESTED
    env.update({KEY_ENV:key,'CUDA_VISIBLE_DEVICES':PAIR_VISIBLE,'Q36_P2P_HOST_EXPERT_BYTES_0':str(first),
                'Q36_P2P_HOST_EXPERT_BYTES':str(second),'Q36_P2P_SYNC_BROADCAST':'1',
                'Q36_P2P_SKIP_PREFILL_HEAD':'1','Q36_P2P_CHUNK':str(CHUNKS[arm]),'ENGINESRV_CACHE_USAGE':'1',
                'INFERENCE_CUDA_REQUIRE_DEVICE':'1'})
    return env


def decode_comparison(a,alog,b,blog):
    def signature(data,log):
        samples=data.get('samples');records=compare.records(log)
        if not isinstance(samples,list) or len(samples)<3:raise ValueError('insufficient decode samples')
        values=[];rates=[];seen=set()
        for x in samples:
            request=x.get('request_id');row=records.get(request);rate=x.get('decode_tokens_per_second')
            if (not isinstance(request,str) or not request or request in seen or row is None
                    or row.get('status')!=200 or row.get('outcome')!='ok'
                    or type(row.get('reused_tokens')) is not int or row['reused_tokens']!=0
                    or type(x.get('prompt_tokens')) is not int or x['prompt_tokens']<1
                    or type(x.get('completion_tokens')) is not int or x['completion_tokens']<64
                    or row.get('prompt_tokens')!=x['prompt_tokens'] or row.get('completion_tokens')!=x['completion_tokens']
                    or not compare.positive_number(rate) or row.get('decode_tokens_per_second')!=rate
                    or not re.fullmatch(r'[0-9a-f]{64}',x.get('answer_sha256',''))):
                raise ValueError('decode sample does not match successful cold server telemetry')
            seen.add(request);rates.append(rate)
            values.append((x['prompt_tokens'],x['completion_tokens'],x['answer_sha256'],x.get('finish_reason')))
        if len(set(values))!=1:raise ValueError('decode results vary within an arm')
        return values[0],statistics.median(rates)
    payload=a.get('payload_sha256')
    if not isinstance(payload,str) or not re.fullmatch(r'[0-9a-f]{64}',payload) or payload!=b.get('payload_sha256'):
        raise ValueError('decode payloads differ or lack a valid hash')
    sa,ca=signature(a,alog);sb,cb=signature(b,blog)
    if sa!=sb:raise ValueError('decode arms have different outputs')
    return {'control_median_tokens_per_second':ca,'candidate_median_tokens_per_second':cb,'speedup':cb/ca}


def run_arm(out,arm,info,key,full):
    item=info['peer']['candidate'];binary=item['server_binary']
    if sha(binary)!=item['server_binary_sha256']:raise RuntimeError('server binary changed')
    config=out/'private-config.json';env=env_for(key,arm);logpath=out/f'{arm}-engine.log'
    with logpath.open('x') as log:
        process=subprocess.Popen([binary,'--config',str(config)],cwd=Path(item['source'])/'engine',env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            free=wait_ready(process,key,binary)
            warm=out/f'{arm}-warmup.json'
            driver('bench_p2p_prefill.py',['--repeat','8','--samples','1','--output',str(warm)],out/f'{arm}-warmup.log',env)
            text=logpath.read_text();check_path(text,arm)
            warm_result=compare.summarize(json.loads(warm.read_text()),text,min_samples=1)
            result={'warmup':warm_result,'warmup_sha256':sha(warm),'free_mib_after_load':free,
                    'free_mib_after_warmup':memory_free(),'server_log_sha256':None}
            if full:
                for label,repeat in [('short',100),('long',850)]:
                    target=out/f'{arm}-{label}.json'
                    driver('bench_p2p_prefill.py',['--repeat',str(repeat),'--samples','3','--output',str(target)],out/f'{arm}-{label}.log',env)
                    compare.summarize(json.loads(target.read_text()),logpath.read_text(),min_samples=3)
                target=out/f'{arm}-decode.json'
                driver('bench_p2p_decode.py',['--repeat','0','--samples','3','--server-log',str(logpath),'--output',str(target)],out/f'{arm}-decode.log',env)
            result['free_mib_after_requests']=memory_free()
            return result
        finally:
            stop_server(process)


def run(out,mode):
    actual,expected=os.fstat(9),LOCK.stat()
    if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):raise RuntimeError('shared lease identity mismatch')
    fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=PAIR_VISIBLE or os.environ.get('INFERENCE_CUDA_REQUIRE_DEVICE')!='1':
        raise RuntimeError('device pair or required-device setting mismatch')
    info=preflight();out.mkdir(parents=True,exist_ok=False)
    config=out/'private-config.json';config.write_bytes(config_bytes());config.chmod(0o600)
    if sha(config)!=info['private_config_sha256']:raise RuntimeError('private configuration changed')
    key=secrets.token_urlsafe(32)
    arms=('candidate',) if mode=='smoke' else ('candidate','control')
    results={}
    for arm in arms:results[arm]=run_arm(out,arm,info,key,mode=='full')
    comparisons={}
    if mode=='full':
        for label in ('short','long'):
            comparisons[label]=compare.compare(json.loads((out/f'control-{label}.json').read_text()),(out/'control-engine.log').read_text(),
                         json.loads((out/f'candidate-{label}.json').read_text()),(out/'candidate-engine.log').read_text())
        comparisons['decode']=decode_comparison(json.loads((out/'control-decode.json').read_text()),(out/'control-engine.log').read_text(),
                         json.loads((out/'candidate-decode.json').read_text()),(out/'candidate-engine.log').read_text())
    if verify_drivers()!=info['drivers'] or model_fingerprint()!=info['model'] or hashlib.sha256(config_bytes()).hexdigest()!=info['private_config_sha256']:
        raise RuntimeError('benchmark inputs changed during run')
    for arm in arms:results[arm]['server_log_sha256']=sha(out/f'{arm}-engine.log')
    summary={'inputs':info,'mode':mode,'candidate_warmup':True,'arms':results,'comparisons':comparisons,
             'limitations':['Both arms use the same staged server binary and differ only in chunk rows.',
                            'Experiment disables prefix reuse.','Server prefill timing includes first-token generation; client timing includes complete reply.',
                            'This A/B order is candidate then control; assess alongside the control-first run.',
                            'This is not 128k/256k prompt, coding-quality, or production qualification evidence.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1] not in ('smoke','full'):raise SystemExit('usage: attention_device.py smoke|full OUTPUT')
    run(Path(sys.argv[2]),sys.argv[1])
