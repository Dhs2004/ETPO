"""Exercise the real collector/data path with scripted actions, not an LLM benchmark."""
import os
import unittest
import numpy as np
import torch
from transformers import AutoTokenizer
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from agent_system.multi_turn_rollout import TrajectoryCollector
from etpo.data import build_teacher_batch, EMPTY_SKILL
from etpo.skills import SkillBank
from etpo.core import select_successful_rows
from test_etpo import configuration, arr, ROOT


class ScriptedEnvironment:
    meta_mode='skillrise'
    task_mode='cross'
    carry_mode='skill'  # ETPO must override this and never call curate.
    num_attempts=3
    num_processes=1
    max_turns=1
    def observation(self):
        text='Agent.\n\n## Current Skill Document\nGuidance\n'+EMPTY_SKILL+'\n\nInspect the cabinet.'
        return {'text':[text],'image':None,'anchor':['cabinet']}
    def reset(self):
        self.pos=0
        self.row_task_type=['pick_and_place_simple']
        return self.observation(),[{'won':False}]
    def advance(self):
        self.pos+=1
        return self.observation(),[{'won':False}]
    def step(self,actions,phase):
        assert phase=='play'
        won=self.pos!=1
        return self.observation(),np.array([10. if won else 0.]),np.array([True]),[{'won':won,'is_action_valid':True}]
    def success_evaluator(self,**kwargs):
        return {'success_rate':np.array([2/3])}


class ScriptedGenerator:
    def __init__(self,tokenizer):self.tokenizer=tokenizer
    def generate_sequences_agent(self,b):
        ids=self.tokenizer.encode('<action>look</action>',add_special_tokens=False)+[self.tokenizer.eos_token_id]
        response=torch.full((1,32),self.tokenizer.pad_token_id,dtype=torch.long)
        response[0,:len(ids)]=torch.tensor(ids)
        rm=torch.zeros_like(response);rm[0,:len(ids)]=1
        prompts=b.batch['input_ids']
        mask=torch.cat([b.batch['attention_mask'],rm],1)
        n={k:v for k,v in b.non_tensor_batch.items() if k!='raw_prompt_ids'}
        return DataProto.from_dict(tensors={'prompts':prompts,'responses':response,
            'input_ids':torch.cat([prompts,response],1),'attention_mask':mask,
            'position_ids':compute_position_id_with_mask(mask)},non_tensors=n)


class RolloutTests(unittest.TestCase):
    def test_real_collector_and_teacher_context(self):
        t=AutoTokenizer.from_pretrained(os.environ.get('SKILLRISE_MODEL_PATH', str(ROOT / '.runtime/models/Qwen3-4B')),local_files_only=True)
        c=configuration();c.env.rollout.n=1;c.data.max_prompt_length=256;c.trainer.rollout_data_dir=None
        base=DataProto.from_dict(tensors={'input_ids':torch.ones(1,1,dtype=torch.long)},non_tensors={
            'raw_prompt':arr([[{'role':'user','content':''}]]),'data_source':arr(['text'])})
        collector=TrajectoryCollector(c,t)
        data=collector.multi_turn_loop(base,ScriptedGenerator(t),ScriptedEnvironment())
        self.assertEqual(len(data),3)
        self.assertEqual(list(data.non_tensor_batch['phase']),['play']*3)
        self.assertEqual(list(data.non_tensor_batch['etpo_env_won']),[True,False,True])
        self.assertEqual(list(data.non_tensor_batch['traj_idx']),[0,1,2])
        for prompt in data.non_tensor_batch['raw_prompt']:
            self.assertNotIn('Current Skill Document',str(prompt))
        rows,_=select_successful_rows(data)
        self.assertEqual(set(rows),{0,2})
        data.batch['response_mask']=data.batch['attention_mask'][:,-32:]
        teacher=build_teacher_batch(data,t,SkillBank(ROOT/'skills','alfworld'),c)
        self.assertTrue(torch.equal(teacher.batch['responses'],data.batch['responses']))
        self.assertIn('GENERAL SKILLS',t.decode(teacher.batch['prompts'][0]))

if __name__=='__main__':unittest.main()
