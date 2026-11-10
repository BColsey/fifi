import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import attention_device as b


class ChunkProofTests(unittest.TestCase):
 def test_parity_proof_binds_test_log_and_source(self):
  original=b.PARITY_ROOT/'pair-reserved-y21enrnp'
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);candidate=root/original.name;shutil.copytree(original,candidate)
   with patch.object(b,'PARITY_ROOT',root):
    self.assertEqual(len(b.validate_512_parity(str(candidate))),64)
    log=candidate/'attention/00.log';log.write_text(log.read_text()+'changed')
    with self.assertRaises(ValueError):b.validate_512_parity(str(candidate))

 def test_memory_reference_requires_headroom_and_unchanged_server_log(self):
  original=b.REFERENCE_ROOT/'pair-reserved-i9yy01hl'
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);candidate=root/original.name;shutil.copytree(original,candidate)
   summary_path=candidate/'attention/summary.json';summary=json.loads(summary_path.read_text())
   info={k:summary['inputs'][k] for k in ('peer','model','private_config_sha256')}
   with patch.object(b,'REFERENCE_ROOT',root):
    result=b.validate_reference(str(candidate),info)
    self.assertEqual(result['declared_extra_mib_per_card'],256)
    summary['arms']['control']['free_mib_after_warmup'][b.PAIR_VISIBLE.split(',')[0]]=1279
    summary_path.write_text(json.dumps(summary))
    with self.assertRaises(ValueError):b.validate_reference(str(candidate),info)
    summary['arms']['control']['free_mib_after_warmup'][b.PAIR_VISIBLE.split(',')[0]]=1400
    summary_path.write_text(json.dumps(summary))
    log=candidate/'attention/control-engine.log';log.write_text(log.read_text()+'modified')
    with self.assertRaises(ValueError):b.validate_reference(str(candidate),info)

if __name__=='__main__':unittest.main()
