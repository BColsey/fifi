#!/usr/bin/env python3
"""Exact-source staged and unstaged P2P correctness checks under the pair guard."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
LOCK = Path('/home/ben/fifi-qwen36/gpu.lock')
DEVICE = 'GPU-282e266a-24ac-6fa8-a16d-4281e52b4c0a'
PAIR_VISIBLE = 'GPU-704fef60-5280-9b88-fa3d-d11135caf576,GPU-282e266a-24ac-6fa8-a16d-4281e52b4c0a'
CANDIDATE = Path('/home/ben/fifi-scratch/c01bfb61a695/p2p-staging-fixed')
CONTROL = Path('/home/ben/fifi-scratch/76dec8a61d04/p2p-staging-matched-control')
CANDIDATE_COMMIT = '76dec8a61d040efa97beefe273bed8dc9e68ff57'
CONTROL_COMMIT = 'c2df0bd2a8220fe20f39235e5015a96dd8080a79'
CARD = 'family::qwen36::cuda::card_chunk_device_tests::'
ARM = 'kernels::cuda::prefill::attn_prefill::tests::attn_prefill_arms_each_physical_device_before_launch'
BASE_CARD = CARD+'card_chunk_matches_the_per_token_path_on_physical_peers'
LARGE_CARD = CARD+'large_card_chunk_matches_the_per_token_path_on_physical_peers'
SEGMENTED_CARD = CARD+'segmented_card_chunk_matches_the_per_token_path_on_physical_peers'
HOST_COMMON = [
 'family::qwen36::cuda::card_prefill::config_tests::chunk_setting_is_explicit_and_bounded',
 'family::qwen36::cuda::card_prefill::config_tests::mailbox_covers_both_chunk_sums_and_eager_decode',
 'family::qwen36::cuda::card_prefill::config_tests::chunk_admission_preserves_the_free_vram_floor',
 'family::qwen36::cuda::prefill_plan::card_tests::card_chunk_arenas_are_narrower_and_complete',
 'family::qwen36::cuda::prefill_plan::card_tests::card_combine_output_is_disjoint_from_its_inputs',
 'family::qwen36::cuda::prefill_tests::zero_row_is_optional_only_for_native_card_views',
 'family::qwen36::settings::tests::the_pin_holds_the_digest_of_each_tokenizer_file',
 'family::qwen36::settings::tests::tokenizer_pin_selection_keeps_dense_and_moe_bundles_separate',
]
HOST_CANDIDATE = HOST_COMMON[:3]+['family::qwen36::cuda::card_prefill::config_tests::card_staging_respects_shard_geometry_and_live_floor']+HOST_COMMON[3:]
CANDIDATE_TESTS = [BASE_CARD, LARGE_CARD, SEGMENTED_CARD, ARM]
CONTROL_TESTS = [BASE_CARD, LARGE_CARD, ARM]
SMOKE_CASE = ('candidate_staged8', 'candidate', BASE_CARD, 8, 256)
CASES = (
 SMOKE_CASE,
 ('candidate_staged96', 'candidate', LARGE_CARD, 96, 256),
 ('candidate_segmented96', 'candidate', SEGMENTED_CARD, 96, 32),
 ('control_unstaged8', 'control', BASE_CARD, 8, None),
 ('control_unstaged96', 'control', LARGE_CARD, 96, None),
 ('candidate_device_arm', 'candidate', ARM, None, None),
 ('control_device_arm', 'control', ARM, None, None),
)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def artifact_events(path):
    events=[]
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if [x.get('success') for x in events if x.get('reason')=='build-finished'] != [True]:
        raise ValueError('Cargo build did not finish successfully')
    return events


def checked_file(path, parent, digest, pattern=None, executable=False):
    path=Path(path)
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.parent.resolve()!=parent.resolve() or sha(path)!=digest
            or (pattern is not None and not re.fullmatch(pattern,path.name))
            or (executable and not os.access(path,os.X_OK))):
        raise ValueError('artifact path, type or hash differs from receipt')
    return path


def verify_arm(root, commit, receipt_path, host_tests, peer_tests):
    source=root/'wt'
    def git(*args):
        return subprocess.check_output(['git','-C',str(source),*args],text=True,timeout=10).strip()
    if git('rev-parse','HEAD')!=commit or git('status','--porcelain','--untracked-files=all'):
        raise ValueError('benchmark source is dirty or differs from pinned commit')
    receipt=Path(receipt_path)
    if (not receipt.is_absolute() or receipt.is_symlink() or not receipt.is_file()
            or receipt.parent.resolve()!=(root/'logs').resolve()
            or not re.fullmatch(r'verified-build-[0-9]+\.json',receipt.name)):
        raise ValueError('missing exact-source build receipt')
    stamp=json.loads(receipt.read_text())
    if (stamp.get('commit')!=commit or stamp.get('target')!='native-cuda linux sm86 release'
            or stamp.get('device_tests_run') is not False or stamp.get('physical_peer_tests_enumerated')!=peer_tests):
        raise ValueError('build receipt identity or peer test list mismatch')
    results=stamp.get('host_results',[])
    if len(results)!=len(host_tests):
        raise ValueError('host test count mismatch')
    for name,result in zip(host_tests,results,strict=True):
        if result.get('name')!=name or result.get('exit')!=0 or result.get('pass') is not True:
            raise ValueError('host test did not pass')
        log=checked_file(result.get('log',''),root/'logs',result.get('sha256'))
        if 'test result: ok. 1 passed; 0 failed; 0 ignored;' not in log.read_text():
            raise ValueError('host test log is missing its pass')
    test=checked_file(stamp.get('test_binary',''),root/'target/release/deps',stamp.get('test_binary_sha256'),r'inference_engine-[0-9a-f]+',True)
    server=checked_file(stamp.get('server_binary',''),root/'target/release',stamp.get('server_binary_sha256'),r'enginesrv',True)
    for binary,log_key,hash_key,target,test_profile in (
        (test,'test_build_log','test_build_log_sha256','inference_engine',True),
        (server,'server_build_log','server_build_log_sha256','enginesrv',False),
    ):
        log=checked_file(stamp.get(log_key,''),root/'logs',stamp.get(hash_key))
        matches=[e for e in artifact_events(log) if e.get('reason')=='compiler-artifact'
                 and e.get('executable')==str(binary) and e.get('target',{}).get('name')==target
                 and e.get('profile',{}).get('test') is test_profile and e.get('fresh') is False
                 and e.get('manifest_path')==str(source/'engine/Cargo.toml')
                 and 'native-cuda' in e.get('features',[])]
        if len(matches)!=1:
            raise ValueError('Cargo log lacks exact-source native-CUDA artifact')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'}
    listing=subprocess.check_output([str(test),'--list'],cwd=source/'engine',env=env,text=True,timeout=60).splitlines()
    ignored=subprocess.check_output([str(test),'--list','--ignored'],cwd=source/'engine',env=env,text=True,timeout=60).splitlines()
    if any(listing.count(n+': test')!=1 or ignored.count(n+': test')!=1 for n in peer_tests):
        raise ValueError('named ignored peer test missing or duplicated')
    cuobjdump=shutil.which('cuobjdump')
    if not cuobjdump:
        raise ValueError('cuobjdump unavailable')
    for binary in (test,server):
        if '.sm_86.cubin' not in subprocess.check_output([cuobjdump,'-all','-lelf',str(binary)],text=True,timeout=120):
            raise ValueError('binary lacks native sm86 image')
    if shutil.disk_usage(root).free < 1_000_000_000:
        raise ValueError('insufficient scratch disk')
    return {'commit':commit,'source':str(source),'test_binary':str(test),'test_binary_sha256':sha(test),
            'server_binary':str(server),'server_binary_sha256':sha(server),'receipt_sha256':sha(receipt),
            'test_build_log_sha256':stamp['test_build_log_sha256'],'server_build_log_sha256':stamp['server_build_log_sha256'],
            'host_tests':len(host_tests),'peer_tests':peer_tests}


def preflight():
    candidate=verify_arm(CANDIDATE,CANDIDATE_COMMIT,os.environ.get('FIFI_P2P_BUILD_RECEIPT',''),HOST_CANDIDATE,CANDIDATE_TESTS)
    control=verify_arm(CONTROL,CONTROL_COMMIT,os.environ.get('FIFI_CONTROL_BUILD_RECEIPT',''),HOST_COMMON,CONTROL_TESTS)
    return {'candidate':candidate,'control':control,'gpu_uuids':PAIR_VISIBLE.split(','),'arch':'sm86',
            'scope':'Synthetic P2P correctness only; no served-model or performance claim.'}


def validate(name, out, rows, stage):
    if ('SKIPPED' in out or not re.search(r'CUDA devices visible: 2\b',out)
            or not re.search(r'^test result: ok\. 1 passed; 0 failed; 0 ignored;',out,re.M)
            or not re.search(r'^test '+re.escape(name)+r' \.\.\. ',out,re.M)):
        raise ValueError('test missing, skipped, failed, or did not see two GPUs')
    if name==ARM:
        if re.findall(r'completed device ([01])',out)!=['0','1','0']:
            raise ValueError('device arm test did not switch 0, 1, 0')
        return
    if f'{rows} rows, winner' not in out:
        raise ValueError('peer fixture did not report the expected comparison')
    if stage is None:
        if 'KV8 staging' in out:
            raise ValueError('unstaged control unexpectedly mentioned staging')
    elif any(f'card {card} KV8 staging: {stage} rows' not in out for card in (0,1)):
        raise ValueError('both cards did not admit the requested staging')


def validate_smoke_receipt(path, info):
    campaign=Path(path)
    if (not campaign.is_absolute() or campaign.is_symlink() or campaign.parent.resolve()!=ROOT.resolve()
            or not re.fullmatch(r'pair-reserved-[a-z0-9_]+',campaign.name)):
        raise ValueError('invalid smoke campaign')
    files=[campaign/'complete.json',campaign/'restore.json',campaign/'attention/summary.json',campaign/'attention/00.log']
    if any(p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(campaign.resolve()) for p in files):
        raise ValueError('missing or unsafe smoke artifact')
    complete,restore,summary=(json.loads(p.read_text()) for p in files[:3])
    if (complete.get('restored') is not True or complete.get('result_type')!='p2p_correctness_tests'
            or restore.get('restored') is not True or summary.get('mode')!='smoke'
            or summary.get('inputs')!=info or summary.get('outcomes',[{}])[0].get('case')!=SMOKE_CASE[0]
            or summary.get('outcomes',[{}])[0].get('passed') is not True):
        raise ValueError('smoke does not prove matching inputs and restored service')
    validate(SMOKE_CASE[2],files[3].read_text(),SMOKE_CASE[3],SMOKE_CASE[4])
    return sha(files[0])


def run(out,mode):
    actual,expected=os.fstat(9),LOCK.stat()
    if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):
        raise RuntimeError('shared lease identity mismatch')
    fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=PAIR_VISIBLE or os.environ.get('INFERENCE_CUDA_REQUIRE_DEVICE')!='1':
        raise RuntimeError('device pair or required-device setting mismatch')
    info=preflight()
    out.mkdir(parents=True,exist_ok=False)
    cases=(SMOKE_CASE,) if mode=='smoke' else CASES
    outcomes=[]
    for index,(label,arm,name,rows,stage) in enumerate(cases):
        item=info[arm];binary=item['test_binary']
        if sha(binary)!=item['test_binary_sha256']:
            raise RuntimeError('binary changed during campaign')
        env=os.environ.copy();env.pop('FIFI_CUDA_ATTN_PREFILL_ADAPT',None)
        cmd=[binary,'--exact',name,'--ignored','--test-threads=1','--nocapture']
        result=subprocess.run(cmd,cwd=Path(item['source'])/'engine',env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=900)
        path=out/f'{index:02d}.log';path.write_text(result.stdout)
        if result.returncode:
            raise RuntimeError('required device case failed: '+label)
        validate(name,result.stdout,rows,stage)
        outcomes.append({'case':label,'arm':arm,'test':name,'rows':rows,'stage_rows':stage,'exit':result.returncode,
                         'passed':True,'log':path.name,'sha256':sha(path)})
    (out/'summary.json').write_text(json.dumps({'inputs':info,'mode':mode,'tests_passed':[r['test'] for r in outcomes],
          'tests_failed':[],'outcomes':outcomes,'scope':'Synthetic correctness, no throughput or long-context claim.'},indent=2)+'\n')

if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1] not in ('smoke','full'):
        raise SystemExit('usage: attention_device.py smoke|full OUTPUT')
    run(Path(sys.argv[2]),sys.argv[1])
