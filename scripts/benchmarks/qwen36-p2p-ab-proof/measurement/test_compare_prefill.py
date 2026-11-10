import copy
import json
import unittest
import compare_prefill as c


def case(rate_ms=1000):
    samples=[]; lines=[]
    for i in range(3):
        request='chat-'+str(i)
        samples.append({'request_id':request,'prompt_tokens':1000,'completion_tokens':2,
                        'reused_tokens':0,'reuse_class':'cold','seconds':2.0})
        lines.append(json.dumps({'request_id':request,'endpoint':'/v1/chat/completions',
                                 'status':200,'outcome':'ok','prompt_tokens':1000,
                                 'completion_tokens':2,'reused_tokens':0,'prefill_ms':rate_ms,
                                 'decode_tokens_per_second':20.0}))
    return {'payload_sha256':'a'*64,'samples':samples},'startup\n'+'\n'.join(lines)

class CompareTest(unittest.TestCase):
    def test_joins_by_id_and_compares_server_and_wall_rates(self):
        a,alog=case(1000);b,blog=case(500)
        result=c.compare(a,alog,b,blog)
        self.assertEqual(result['started_to_first_token_speedup'],2)
        self.assertEqual(result['control']['median_complete_reply_prompt_tokens_per_second'],500)
        self.assertEqual(result['candidate']['median_started_to_first_token_prompt_tokens_per_second'],2000)

    def test_rejects_missing_unknown_or_warm_reuse(self):
        a,alog=case()
        for value in (None,1,False):
            b=copy.deepcopy(a);b['samples'][0]['reused_tokens']=value
            with self.assertRaises(ValueError):c.compare(a,alog,b,alog)
        with self.assertRaises(ValueError):c.summarize(a,alog.replace('"reused_tokens": 0','"reused_tokens": 5'))

    def test_rejects_missing_duplicate_or_wrong_identity(self):
        a,alog=case();b=copy.deepcopy(a);b['samples'][1]['request_id']=b['samples'][0]['request_id']
        with self.assertRaises(ValueError):c.compare(a,alog,b,alog)
        with self.assertRaises(ValueError):c.summarize(a,alog+'\n'+alog.splitlines()[-1])
        with self.assertRaises(ValueError):c.summarize(a,'')

    def test_rejects_mismatched_or_invalid_timings_and_tokens(self):
        a,alog=case();b=copy.deepcopy(a);b['samples'][0]['prompt_tokens']=999
        with self.assertRaises(ValueError):c.compare(a,alog,b,alog)
        for value in ('null','0','-1','true'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                c.summarize(a,alog.replace('"prefill_ms": 1000','"prefill_ms": '+value))

    def test_rejects_payload_and_sample_count_mismatches(self):
        a,alog=case();b=copy.deepcopy(a);b['payload_sha256']='different'
        with self.assertRaises(ValueError):c.compare(a,alog,b,alog)
        b=copy.deepcopy(a);b['samples'].pop()
        with self.assertRaises(ValueError):c.compare(a,alog,b,alog)

if __name__=='__main__':unittest.main()
