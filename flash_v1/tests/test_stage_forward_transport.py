import json
from pathlib import Path
import tempfile
import unittest
from flash_v1.scripts.v2_transport import EvidencePublisher


class Client:
    def __init__(self):self.lists={};self.values={}
    def rpush(self,key,*rows):self.lists.setdefault(key,[]).extend(rows)
    def set(self,key,value):self.values[key]=value


class TransportTests(unittest.TestCase):
    def test_independent_aux_stream_preserves_original_completion(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d);client=Client()
            for r in range(8):
                (output/f'evidence_rank{r}.jsonl').write_text(json.dumps(dict(rank=r,iteration=8,record_type='iteration_complete'))+'\n')
                (output/f'stage_forward_aux_rank{r}.jsonl').write_text(json.dumps(dict(rank=r,iteration=8,record_type='stage_forward_aux'))+'\n')
            original=EvidencePublisher(output,0,client,publish_completion=True)
            aux=EvidencePublisher(output,0,client,filename='stage_forward_aux_rank{rank}.jsonl',key_prefix='online_stage_forward_aux_node',completed_record='stage_forward_aux',completion_prefix='online_stage_forward_drained_node',publish_completion=True)
            self.assertEqual(aux.publish(),8)
            self.assertNotIn('online_drained_node0',client.values)
            self.assertEqual(original.publish(),8)
            self.assertEqual(len(client.lists['online_evidence_node0']),8)
            self.assertEqual(len(client.lists['online_stage_forward_aux_node0']),8)
            self.assertEqual(original.publish(),0);self.assertEqual(aux.publish(),0)
    def test_partial_aux_record_is_retained_until_newline(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d,'stage_forward_aux_rank0.jsonl');line=json.dumps(dict(rank=0,iteration=1,record_type='stage_forward_aux'))
            p.write_text(line);client=Client();aux=EvidencePublisher(Path(d),0,client,filename='stage_forward_aux_rank{rank}.jsonl',key_prefix='aux')
            self.assertEqual(aux.publish(),0)
            with p.open('a') as stream:stream.write('\n')
            self.assertEqual(aux.publish(),1);self.assertEqual(aux.publish(),0)

if __name__=='__main__':unittest.main()
