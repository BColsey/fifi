import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import attention_device as a


def output(name,rows,stage):
 return ('CUDA devices visible: 2\n'+''.join(f'card {card} KV8 staging: {stage} rows\n' for card in (0,1))+
         f'test {name} ... card_chunk_matches_the_per_token_path: {rows} rows, winner 1\n'+
         'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n')


class ChunkRunnerTests(unittest.TestCase):
 def test_validation_requires_exact_peer_count_staging_and_row_result(self):
  for index,(rows,stage) in enumerate(((8,256),(512,2048))):
   name=a.TESTS[index][0];good=output(name,rows,stage);a.validate(name,good)
   for bad in (good.replace('CUDA devices visible: 2','CUDA devices visible: 1'),
               good.replace(f'card 1 KV8 staging: {stage} rows',''),
               good.replace(f'{rows} rows, winner','wrong rows, winner'),
               good+'SKIPPED\n'):
    with self.assertRaises(ValueError):a.validate(name,bad)

 def test_smoke_and_full_select_exact_tests_and_one_thread(self):
  for mode,index in (('smoke',0),('full',1)):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
    root=Path(td);binary=root/'binary';binary.write_bytes(b'binary');lock=root/'lock';lock.write_bytes(b'lock')
    info={'binary':str(binary),'binary_sha256':a.sha(binary)};out=root/'out'
    stat=MagicMock();stat.st_dev=1;stat.st_ino=2
    rows,stage=(8,256) if index==0 else (512,2048)
    result=MagicMock(returncode=0,stdout=output(a.TESTS[index][0],rows,stage))
    with patch.object(a,'LOCK',lock),patch.object(a.os,'fstat',return_value=stat),patch.object(Path,'stat',return_value=stat), \
         patch.object(a.fcntl,'flock'),patch.object(a,'preflight',return_value=info), \
         patch.dict(a.os.environ,{'CUDA_VISIBLE_DEVICES':a.PAIR_VISIBLE,'INFERENCE_CUDA_REQUIRE_DEVICE':'1'}), \
         patch.object(a.subprocess,'run',return_value=result) as call:
     a.run(out,mode)
    cmd=call.call_args.args[0]
    self.assertEqual(cmd[2],a.TESTS[index][0]);self.assertIn('--ignored',cmd);self.assertIn('--test-threads=1',cmd)
    self.assertEqual(json.loads((out/'summary.json').read_text())['tests_passed'],[a.TESTS[index][0]])

 def test_smoke_receipt_requires_matching_inputs_and_restoration(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);campaign=root/'pair-reserved-abc';out=campaign/'attention';out.mkdir(parents=True)
   name=a.TESTS[0][0];info={'binary_sha256':'pinned'}
   (campaign/'complete.json').write_text(json.dumps({'restored':True,'result_type':'p2p_correctness_tests'}))
   (campaign/'restore.json').write_text(json.dumps({'restored':True}))
   (out/'00.log').write_text(output(name,8,256))
   (out/'summary.json').write_text(json.dumps({'mode':'smoke','tests_passed':[name],'inputs':info,
                                                'outcomes':[{'sha256':a.sha(out/'00.log')}]}))
   (campaign/'guard.jsonl').write_text(json.dumps({'builders':[],'other_jobs':[],'unexpected':[],
     'states_ok':True,'clients':False,'listeners':False,'non_target_utilization':{'other':0}})+'\n')
   with patch.object(a,'ROOT',root):
    self.assertEqual(len(a.validate_smoke_receipt(str(campaign),info)),64)
    with self.assertRaises(ValueError):a.validate_smoke_receipt(str(campaign),{'binary_sha256':'changed'})
    (campaign/'restore.json').write_text(json.dumps({'restored':False}))
    with self.assertRaises(ValueError):a.validate_smoke_receipt(str(campaign),info)

if __name__=='__main__':unittest.main()
