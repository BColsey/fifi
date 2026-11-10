import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import attention_device as b


def line(request='id',rate=20):
    return {'request_id':request,'endpoint':'/v1/chat/completions','status':200,'outcome':'ok',
            'prompt_tokens':25,'completion_tokens':128,'reused_tokens':0,
            'prefill_ms':10,'decode_tokens_per_second':rate}


def decode(rate=20):
    samples=[];logs=[]
    for i in range(3):
        rid='id'+str(i);samples.append({'request_id':rid,'prompt_tokens':25,'completion_tokens':128,
                     'answer_sha256':'a'*64,'finish_reason':'length','decode_tokens_per_second':rate})
        logs.append(json.dumps(line(rid,rate)))
    return {'payload_sha256':'b'*64,'samples':samples},'\n'.join(logs)

class BenchmarkSafetyTests(unittest.TestCase):
    def test_private_config_preserves_model_and_disables_reuse_and_public_ui(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);model=root/'model';model.mkdir();gguf=model/'weights';gguf.write_bytes(b'x')
            config=root/'config.json';base={'listen':'0.0.0.0:1','api_key_env':'OLD','prefix_reuse':True,'sample_trace':True,
                  'web_dir':'/web','engines':[{'model':'fifi-q36','family':'qwen3_6_35b_a3b','model_dir':str(model),
                  'qwen36':{'context_limit':262144,'kv_dtype':'q8','cuda_devices':[0,1]}}]};config.write_text(json.dumps(base))
            with patch.object(b,'CONFIG',config),patch.object(b,'MODEL',gguf):
                x=b.private_config();self.assertEqual(x['listen'],f'127.0.0.1:{b.PORT}')
                self.assertFalse(x['prefix_reuse']);self.assertFalse(x['sample_trace']);self.assertIsNone(x['web_dir'])
                self.assertEqual(x['api_key_env'],b.KEY_ENV);self.assertEqual(x['engines'],base['engines'])
                base['engines'][0]['qwen36']['context_limit']=4096;config.write_text(json.dumps(base))
                with self.assertRaises(ValueError):b.private_config()

    def test_path_checks_reject_eager_fallback_or_missing_card_stage(self):
        chunk=lambda arm:f'q36 cuda: chunked prefill engaged at {b.CHUNKS[arm]} rows\n'
        stage=''.join(f'q36 cuda: card {i} KV8 staging: 32768 rows, 33554432 bytes\n' for i in (0,1))
        lines=''.join(f'cuda mapped experts: placing {n} bytes in host memory\n' for n in b.HOST_SELECTED)
        b.check_path(chunk('candidate')+stage+lines,'candidate');b.check_path(chunk('control')+stage+lines,'control')
        for text,arm in [(chunk('candidate')+lines,'candidate'),(stage+lines,'candidate'),
                         (chunk('candidate')+stage+lines,'control'),
                         (chunk('control')+stage+lines+'chunked prefill refused: bad','control')]:
            with self.assertRaises(ValueError):b.check_path(text,arm)

    def test_chunk_arms_differ_only_in_chunk_size(self):
        a=b.env_for('ephemeral','control');c=b.env_for('ephemeral','candidate')
        changed={key for key in a if a[key]!=c[key]}
        self.assertEqual(changed,{'Q36_P2P_CHUNK'})
        self.assertEqual(a['Q36_P2P_CHUNK'],'256');self.assertEqual(c['Q36_P2P_CHUNK'],'512')
        with self.assertRaises(ValueError):b.env_for('ephemeral','unknown')

    def test_arms_launch_the_same_binary_with_their_own_chunk_sizes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);binary=root/'same-server';binary.write_bytes(b'server');(root/'engine').mkdir()
            (root/'private-config.json').write_text('{}')
            item={'source':str(root),'server_binary':str(binary),'server_binary_sha256':b.sha(binary)}
            info={'peer':{'candidate':item,'control':{'server_binary':'must-not-run'}}}
            def driver(_script,args,_out,_env):Path(args[args.index('--output')+1]).write_text('{}')
            process=Mock()
            with patch.object(b.subprocess,'Popen',return_value=process) as spawn, \
                 patch.object(b,'wait_ready',return_value={}), patch.object(b,'stop_server'), \
                 patch.object(b,'memory_free',return_value={}), patch.object(b,'check_path'), \
                 patch.object(b,'driver',side_effect=driver), patch.object(b.compare,'summarize',return_value={}):
                b.run_arm(root,'control',info,'key',False)
                b.run_arm(root,'candidate',info,'key',False)
            first,second=spawn.call_args_list
            self.assertEqual(first.args[0][0],str(binary));self.assertEqual(second.args[0][0],str(binary))
            self.assertEqual(first.kwargs['env']['Q36_P2P_HOST_EXPERT_BYTES_0'],'1395864370')
            self.assertEqual(second.kwargs['env']['Q36_P2P_HOST_EXPERT_BYTES_0'],'1395864370')
            self.assertEqual(second.kwargs['env']['Q36_P2P_HOST_EXPERT_BYTES'],'3489660928')
            self.assertEqual(first.kwargs['env']['Q36_P2P_CHUNK'],'256')
            self.assertEqual(second.kwargs['env']['Q36_P2P_CHUNK'],'512')

    def test_full_crossover_starts_with_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);lock=Mock();stat=Mock(st_dev=1,st_ino=2);lock.stat.return_value=stat
            config=b'config';info={'private_config_sha256':hashlib.sha256(config).hexdigest()}
            with patch.object(b,'LOCK',lock),patch.object(b.os,'fstat',return_value=stat),patch.object(b.fcntl,'flock'), \
                 patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':b.PAIR_VISIBLE,'INFERENCE_CUDA_REQUIRE_DEVICE':'1'}), \
                 patch.object(b,'preflight',return_value=info),patch.object(b,'config_bytes',return_value=config), \
                 patch.object(b,'run_arm',side_effect=RuntimeError('stop after first arm')) as run_arm:
                with self.assertRaisesRegex(RuntimeError,'stop after first arm'):b.run(root/'out','full')
            self.assertEqual(run_arm.call_args.args[1],'candidate')

    def test_private_environment_excludes_old_flags_and_keys(self):
        with patch.dict(os.environ,{'PATH':'/bin','FIFI_OLD_KEY':'old','Q36_P2P_PREFIX':'1',
                                    'FIFI_CUDA_ATTN_PREFILL_ADAPT':'1','ENGINESRV_CACHE_USAGE':'0'},clear=True):
            x=b.env_for('ephemeral');self.assertEqual(x[b.KEY_ENV],'ephemeral')
            self.assertNotIn('FIFI_OLD_KEY',x);self.assertNotIn('Q36_P2P_PREFIX',x)
            self.assertNotIn('FIFI_CUDA_ATTN_PREFILL_ADAPT',x);self.assertEqual(x['Q36_P2P_CHUNK'],'256')
            self.assertEqual(x['ENGINESRV_CACHE_USAGE'],'1');self.assertEqual(x['INFERENCE_CUDA_REQUIRE_DEVICE'],'1')

    def test_decode_comparison_joins_unique_cold_server_rows(self):
        a,alog=decode(20);c,clog=decode(25);result=b.decode_comparison(a,alog,c,clog)
        self.assertEqual(result['speedup'],1.25)
        changed=copy.deepcopy(c);changed['samples'][0]['answer_sha256']='c'*64
        with self.assertRaises(ValueError):b.decode_comparison(a,alog,changed,clog)
        with self.assertRaises(ValueError):b.decode_comparison(a,alog,c,clog.replace('"reused_tokens": 0','"reused_tokens": 1'))
        changed=copy.deepcopy(c);changed['samples'][1]['request_id']=changed['samples'][0]['request_id']
        with self.assertRaises(ValueError):b.decode_comparison(a,alog,changed,clog)
        changed=copy.deepcopy(c);changed['samples'][0]['decode_tokens_per_second']=999
        with self.assertRaises(ValueError):b.decode_comparison(a,alog,changed,clog)

    def test_gpu_process_parser_refuses_malformed_rows(self):
        with patch.object(b.subprocess,'check_output',return_value='bad,row,extra\n'):
            with self.assertRaises(ValueError):b.pair_processes()
        gpu=b.PAIR_VISIBLE.split(',')[0]
        with patch.object(b.subprocess,'check_output',return_value=f'{gpu}, 123\n'):
            self.assertEqual(b.pair_processes(),{(gpu,123)})

    def test_driver_manifest_rejects_changed_measurement_code(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'scripts').mkdir()
            for name in b.DRIVERS:(root/name).write_text(name)
            manifest={name:b.sha(root/name) for name in b.DRIVERS};(root/'driver-manifest.json').write_text(json.dumps(manifest))
            with patch.object(b,'MEASURE',root),patch.object(b,'LOADED_COMPARATOR_SHA',manifest['compare_prefill.py']):
                self.assertEqual(b.verify_drivers(),manifest)
                (root/'scripts/bench_p2p_decode.py').write_text('modified')
                with self.assertRaises(ValueError):b.verify_drivers()

    def test_model_fingerprint_detects_replacement_or_modification(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);model=root/'weights';model.write_bytes(b'weights')
            stat=model.stat();fingerprint={'resolved':str(model.resolve()),'size':stat.st_size,'inode':stat.st_ino,
                                           'mtime_ns':stat.st_mtime_ns,'sha256':hashlib.sha256(b'weights').hexdigest()}
            (root/'model-fingerprint.json').write_text(json.dumps(fingerprint))
            with patch.object(b,'MEASURE',root),patch.object(b,'MODEL',model):
                self.assertEqual(b.model_fingerprint(),fingerprint)
                model.write_bytes(b'new-weights')
                with self.assertRaises(ValueError):b.model_fingerprint()

    def test_smoke_receipt_binds_the_server_log_and_cold_request(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);campaign=root/'pair-reserved-abc';(campaign/'attention').mkdir(parents=True);info={'pin':'abc'}
            request='warm';server=line(request);server.update({'prompt_tokens':25,'completion_tokens':2,'prefill_ms':10})
            log=campaign/'attention/candidate-engine.log';log.write_text('q36 cuda: chunked prefill engaged at 512 rows\n'+
                ''.join(f'q36 cuda: card {i} KV8 staging: 32768 rows\n' for i in (0,1))+
                ''.join(f'cuda mapped experts: placing {n} bytes in host memory\n' for n in b.HOST_SELECTED)+json.dumps(server)+'\n')
            warm=campaign/'attention/candidate-warmup.json';warm.write_text(json.dumps({'samples':[{'request_id':request,
                'prompt_tokens':25,'completion_tokens':2,'reused_tokens':0,'reuse_class':'cold','seconds':1.0}]}))
            (campaign/'complete.json').write_text(json.dumps({'restored':True,'result_type':'p2p_correctness_tests'}))
            (campaign/'restore.json').write_text(json.dumps({'restored':True}))
            guard={'builders':[],'other_jobs':[],'unexpected':[],'states_ok':True,'clients':False,'listeners':False,
                   'non_target_utilization':{'other':0},'pair_free_mib':{g:1024 for g in b.PAIR_VISIBLE.split(',')}}
            (campaign/'guard.jsonl').write_text(json.dumps(guard)+'\n')
            (campaign/'attention/summary.json').write_text(json.dumps({'mode':'smoke','inputs':info,'candidate_warmup':True,
                'arms':{'candidate':{'server_log_sha256':b.sha(log),'warmup_sha256':b.sha(warm)}}}))
            with patch.object(b,'SMOKE_ROOT',root):
                self.assertEqual(len(b.validate_smoke_receipt(str(campaign),info)),64)
                guard['pair_free_mib'][b.PAIR_VISIBLE.split(',')[0]]=1023
                (campaign/'guard.jsonl').write_text(json.dumps(guard)+'\n')
                with self.assertRaises(ValueError):b.validate_smoke_receipt(str(campaign),info)
                guard['pair_free_mib'][b.PAIR_VISIBLE.split(',')[0]]=1024
                (campaign/'guard.jsonl').write_text(json.dumps(guard)+'\n')
                log.write_text(log.read_text()+'modified')
                with self.assertRaises(ValueError):b.validate_smoke_receipt(str(campaign),info)

    def test_private_server_is_stopped_when_warmup_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);binary=root/'server';binary.write_bytes(b'server');(root/'engine').mkdir();(root/'private-config.json').write_text('{}')
            info={'peer':{'candidate':{'source':str(root),'server_binary':str(binary),'server_binary_sha256':b.sha(binary)}}}
            process=Mock(pid=10)
            with patch.object(b.subprocess,'Popen',return_value=process),patch.object(b,'wait_ready',return_value={'gpu':2000}), \
                 patch.object(b,'driver',side_effect=RuntimeError('warmup failed')),patch.object(b,'stop_server') as stop:
                with self.assertRaisesRegex(RuntimeError,'warmup failed'):b.run_arm(root,'candidate',info,'secret',False)
                stop.assert_called_once_with(process)

    def test_peer_evidence_requires_all_cases_matching_inputs_and_clean_guard(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);campaign=root/'pair-reserved-abc';(campaign/'attention').mkdir(parents=True)
            info={'candidate':'pinned'};outcomes=[]
            for i,(label,arm,name,rows,stage) in enumerate(b.peer.CASES):
                text=f'test {name} ... CUDA devices visible: 2\n'
                if name==b.peer.ARM:text+='completed device 0\ncompleted device 1\ncompleted device 0\n'
                else:
                    if stage is not None:text+='\n'.join(f'card {card} KV8 staging: {stage} rows' for card in (0,1))+'\n'
                    text+=f'{rows} rows, winner 3\n'
                text+='test result: ok. 1 passed; 0 failed; 0 ignored;\n'
                p=campaign/'attention'/f'{i:02}.log';p.write_text(text)
                outcomes.append({'case':label,'arm':arm,'test':name,'passed':True,'exit':0,'log':p.name,'sha256':b.sha(p)})
            (campaign/'complete.json').write_text(json.dumps({'restored':True,'result_type':'p2p_diagnostic_matrix'}))
            (campaign/'restore.json').write_text(json.dumps({'restored':True}))
            (campaign/'attention/summary.json').write_text(json.dumps({'mode':'full','inputs':info,'tests_failed':[],'outcomes':outcomes}))
            guard={'builders':[],'other_jobs':[],'unexpected':[],'states_ok':True,'clients':False,'listeners':False,'non_target_utilization':{'g':0}}
            (campaign/'guard.jsonl').write_text(json.dumps(guard)+'\n')
            with patch.object(b,'PEER_ROOT',root):
                self.assertEqual(b.validate_peer_proof(str(campaign),info)['guard_samples'],1)
                (campaign/'attention/00.log').write_text('changed')
                with self.assertRaises(ValueError):b.validate_peer_proof(str(campaign),info)

if __name__=='__main__':unittest.main(verbosity=2)
