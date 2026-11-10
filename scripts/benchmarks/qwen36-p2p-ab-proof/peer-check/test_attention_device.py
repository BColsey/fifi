import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import attention_device as d
import reserve_pair_attention as r


def fake_output(name,rows,stage):
    out=f'running 1 test\ntest {name} ... CUDA devices visible: 2\n'
    if name==d.ARM:
        out+='completed device 0\ncompleted device 1\ncompleted device 0\n'
    else:
        if stage is not None:
            out+='\n'.join(f'q36 cuda: card {c} KV8 staging: {stage} rows, 1 bytes' for c in (0,1))+'\n'
        out+=f'card_chunk_matches_the_per_token_path: {rows} rows, winner 42\n'
    return out+'ok\ntest result: ok. 1 passed; 0 failed; 0 ignored;\n'

class PairChecks(unittest.TestCase):
    def test_validate_requires_exact_devices_stage_and_row_output(self):
        good=fake_output(d.BASE_CARD,8,256);d.validate(d.BASE_CARD,good,8,256)
        for bad in (good.replace('visible: 2','visible: 1'),good.replace('1 passed','0 passed'),
                    good.replace('card 1 KV8 staging','card 2 KV8 staging'),good.replace('8 rows, winner','9 rows, winner'),
                    good+'SKIPPED'):
            with self.assertRaises(ValueError):d.validate(d.BASE_CARD,bad,8,256)
        with self.assertRaises(ValueError):d.validate(d.BASE_CARD,good,8,None)
        arm=fake_output(d.ARM,None,None);d.validate(d.ARM,arm,None,None)
        with self.assertRaises(ValueError):d.validate(d.ARM,arm.replace('completed device 1','completed device 0'),None,None)

    def test_smoke_and_full_run_both_pinned_binaries_with_gate_unset(self):
        for mode in ('smoke','full'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
                root=Path(td);candidate=root/'candidate';control=root/'control';candidate.write_bytes(b'a');control.write_bytes(b'b')
                info={'candidate':{'test_binary':str(candidate),'test_binary_sha256':d.sha(candidate),'source':str(root)},
                      'control':{'test_binary':str(control),'test_binary_sha256':d.sha(control),'source':str(root)}}
                identity=SimpleNamespace(st_dev=1,st_ino=2);lock=Mock();lock.stat.return_value=identity;calls=[]
                def execute(cmd,**kwargs):
                    name=cmd[2];arm='candidate' if cmd[0]==str(candidate) else 'control'
                    match=next(c for c in d.CASES if c[1]==arm and c[2]==name)
                    calls.append((arm,name,kwargs['env'].get('FIFI_CUDA_ATTN_PREFILL_ADAPT'),'--ignored' in cmd))
                    return SimpleNamespace(returncode=0,stdout=fake_output(name,match[3],match[4]))
                with patch.object(d,'LOCK',lock),patch.object(d.os,'fstat',return_value=identity), \
                     patch.object(d.fcntl,'flock'),patch.object(d,'preflight',return_value=info), \
                     patch.object(d.subprocess,'run',side_effect=execute), \
                     patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':d.PAIR_VISIBLE,'INFERENCE_CUDA_REQUIRE_DEVICE':'1',
                                            'FIFI_CUDA_ATTN_PREFILL_ADAPT':'1'}):
                    d.run(root/'out',mode)
                expected=d.CASES[:1] if mode=='smoke' else d.CASES
                self.assertEqual(calls,[(c[1],c[2],None,True) for c in expected])
                self.assertTrue(all(x['passed'] for x in json.loads((root/'out/summary.json').read_text())['outcomes']))

    def test_failed_case_never_creates_successful_summary(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            root=Path(td);binary=root/'candidate';binary.write_bytes(b'a')
            info={'candidate':{'test_binary':str(binary),'test_binary_sha256':d.sha(binary),'source':str(root)}}
            identity=SimpleNamespace(st_dev=1,st_ino=2);lock=Mock();lock.stat.return_value=identity
            with patch.object(d,'LOCK',lock),patch.object(d.os,'fstat',return_value=identity), \
                 patch.object(d.fcntl,'flock'),patch.object(d,'preflight',return_value=info), \
                 patch.object(d.subprocess,'run',return_value=SimpleNamespace(returncode=101,stdout='failed')), \
                 patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':d.PAIR_VISIBLE,'INFERENCE_CUDA_REQUIRE_DEVICE':'1'}):
                with self.assertRaisesRegex(RuntimeError,'required device case failed'):d.run(root/'out','smoke')
            self.assertFalse((root/'out/summary.json').exists())

    def test_smoke_receipt_binds_inputs_and_restoration(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            root=Path(td);campaign=root/'pair-reserved-abc';(campaign/'attention').mkdir(parents=True);info={'candidate':'pinned'}
            (campaign/'complete.json').write_text(json.dumps({'restored':True,'result_type':'p2p_correctness_tests'}))
            (campaign/'restore.json').write_text(json.dumps({'restored':True}))
            (campaign/'attention/summary.json').write_text(json.dumps({'mode':'smoke','outcomes':[{'case':d.SMOKE_CASE[0],'passed':True}],'inputs':info}))
            log=campaign/'attention/00.log';log.write_text(fake_output(d.BASE_CARD,8,256))
            with patch.object(d,'ROOT',root):
                self.assertEqual(len(d.validate_smoke_receipt(str(campaign),info)),64)
                with self.assertRaises(ValueError):d.validate_smoke_receipt(str(campaign),{'candidate':'other'})
                log.write_text(fake_output(d.BASE_CARD,8,None))
                with self.assertRaises(ValueError):d.validate_smoke_receipt(str(campaign),info)

    def test_both_peer_cards_must_belong_to_owned_scope(self):
        scope=Mock();scope.contains.side_effect=lambda pid:pid==100
        for gpu in r.PAIR:
            self.assertTrue(r.in_pair_scope(gpu,100,scope));self.assertFalse(r.in_pair_scope(gpu,200,scope))
        self.assertFalse(r.in_pair_scope(r.OTHER,100,scope))

    def test_verify_arm_pins_source_artifacts_host_logs_and_test_listing(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            root=Path(td);source=root/'wt';(source/'engine').mkdir(parents=True);(root/'logs').mkdir()
            (root/'target/release/deps').mkdir(parents=True);(source/'engine/Cargo.toml').write_text('fixture')
            test=root/'target/release/deps/inference_engine-abc';server=root/'target/release/enginesrv'
            test.write_bytes(b'test');server.write_bytes(b'server');test.chmod(0o755);server.chmod(0o755)
            host=root/'logs/host.log';host.write_text('test result: ok. 1 passed; 0 failed; 0 ignored;\n')
            def event(binary,target,is_test):
                return {'reason':'compiler-artifact','executable':str(binary),'target':{'name':target},
                        'profile':{'test':is_test},'fresh':False,'manifest_path':str(source/'engine/Cargo.toml'),
                        'features':['native-cuda']}
            tlog=root/'logs/test.jsonl';slog=root/'logs/server.jsonl'
            tlog.write_text(json.dumps(event(test,'inference_engine',True))+'\n'+json.dumps({'reason':'build-finished','success':True})+'\n')
            slog.write_text(json.dumps(event(server,'enginesrv',False))+'\n'+json.dumps({'reason':'build-finished','success':True})+'\n')
            receipt=root/'logs/verified-build-1.json';receipt.write_text(json.dumps({
                'commit':'abc','target':'native-cuda linux sm86 release','device_tests_run':False,'physical_peer_tests_enumerated':['peer'],
                'host_results':[{'name':'host','exit':0,'pass':True,'log':str(host),'sha256':d.sha(host)}],
                'test_binary':str(test),'test_binary_sha256':d.sha(test),'server_binary':str(server),'server_binary_sha256':d.sha(server),
                'test_build_log':str(tlog),'test_build_log_sha256':d.sha(tlog),'server_build_log':str(slog),'server_build_log_sha256':d.sha(slog)}))
            def command(cmd,**kwargs):
                if cmd[0]=='git':return 'abc\n' if cmd[-1]=='HEAD' else ''
                if cmd[0]=='cuobjdump':return '.sm_86.cubin'
                return 'peer: test\n'
            with patch.object(d.subprocess,'check_output',side_effect=command),patch.object(d.shutil,'which',return_value='cuobjdump'):
                result=d.verify_arm(root,'abc',str(receipt),['host'],['peer']);self.assertEqual(result['host_tests'],1)
                server.write_bytes(b'changed')
                with self.assertRaises(ValueError):d.verify_arm(root,'abc',str(receipt),['host'],['peer'])

if __name__=='__main__':unittest.main(verbosity=2)
