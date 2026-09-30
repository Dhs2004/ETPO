import copy
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from etpo.config import validate
from etpo.core import filtered_distillation, masked_sft_sum, select_successful_rows
from etpo.data import build_teacher_batch, supervised_subset, student_observation, EMPTY_SKILL
from etpo.skills import SkillBank
from etpo.training import evolve_and_score

ROOT = Path(__file__).resolve().parents[2]


def configuration():
    c = OmegaConf.load(ROOT/'verl/trainer/config/ppo_trainer.yaml')
    c.etpo.enabled = True
    c.etpo.skill_root = str(ROOT/'skills')
    c.etpo.confidence_min = 1e-8
    c.etpo.confidence_max = 1.0
    c.etpo.max_abs_log_ratio = 10
    c.data.return_raw_chat = True
    c.data.truncation = 'error'
    c.algorithm.adv_estimator = 'skillrise'
    c.env.env_name = 'skillrise_alfworld/AlfredTWEnv'
    c.trainer.n_gpus_per_node = 1
    c.trainer.nnodes = 1
    c.actor_rollout_ref.actor.use_torch_compile = False
    c.actor_rollout_ref.actor.ppo_mini_batch_size = 2
    c.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
    c.actor_rollout_ref.actor.entropy_coeff = 0.0
    c.actor_rollout_ref.actor.optim.lr = 0.001
    validate(c)
    return c


class TinyTokenizer:
    pad_token_id = 0
    def encode(self, text, **kwargs):
        return [ord(x) % 63 + 1 for x in text]
    def apply_chat_template(self, messages, **kwargs):
        return ''.join(m['role']+':'+m['content']+'\n' for m in messages)+'assistant:'


class TinyBank:
    def get(self, task_type):
        return 'Inspect the target before acting.'


def arr(items):
    result = np.empty(len(items), dtype=object)
    result[:] = items
    return result


def make_batch():
    tokenizer = TinyTokenizer()
    messages = [[{'role':'user','content':x}] for x in ['task A','task B']]
    ids = [tokenizer.encode(tokenizer.apply_chat_template(m)) for m in messages]
    prompts = torch.tensor(ids)
    responses = torch.tensor([[12,13,0],[15,16,17]])
    mask = torch.cat([torch.ones_like(prompts),torch.tensor([[1,1,0],[1,1,1]])],1)
    batch = DataProto.from_dict(tensors={
        'prompts':prompts, 'responses':responses,
        'input_ids':torch.cat([prompts,responses],1),
        'attention_mask':mask, 'position_ids':compute_position_id_with_mask(mask),
        'response_mask':mask[:,-3:], 'advantages':torch.tensor([[1.,1.,0.],[-1.,-1.,-1.]])},
        non_tensors={
        'raw_prompt':arr(messages),'etpo_task_type':arr(['pick_and_place_simple']*2),
        'traj_uid':arr(['a','b']),'traj_idx':arr([0,0]),'turn_idx':arr([0,0]),
        'phase':arr(['play','play']),'etpo_env_won':arr([True,False]),
        'rewards':arr([10.,0.]),'is_action_valid':arr([True,True])},
        meta_info={'temperature':1.0})
    return batch,tokenizer


class TinyCausalLM(torch.nn.Module):
    """Actual trainable causal context model; no mocked gradients or logprobs."""
    def __init__(self):
        super().__init__()
        self.embed=torch.nn.Embedding(65,16)
        self.head=torch.nn.Linear(16,65)
    def forward(self,input_ids,attention_mask,position_ids,**kwargs):
        h=self.embed(input_ids)
        weight=attention_mask.unsqueeze(-1).to(h.dtype)
        h=(h*weight).cumsum(1)/weight.cumsum(1).clamp_min(1)
        return SimpleNamespace(logits=self.head(h))


class LocalWorker:
    def __init__(self,c):
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        self.model=TinyCausalLM()
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=.005,weight_decay=0)
        self.actor=DataParallelPPOActor(c.actor_rollout_ref.actor,self.model,self.optimizer)
        self.calls=[]
    def compute_log_prob(self,batch):
        self.calls.append('score')
        data=copy.deepcopy(batch)
        data.meta_info.update(micro_batch_size=1,temperature=1.,use_dynamic_bsz=False)
        logp,_=self.actor.compute_log_prob(data)
        return DataProto.from_dict(tensors={'old_log_probs':logp})
    def update_etpo_teacher(self,batch):
        self.calls.append('sft')
        return DataProto(meta_info={'metrics':self.actor.update_etpo_teacher(batch)})


class ETPOTests(unittest.TestCase):
    def test_skill_sources_and_alias(self):
        for env in ['alfworld','webshop','sciworld']:
            bank=SkillBank(ROOT/'skills',env)
            self.assertTrue(bank.get('unknown').strip())
        bank=SkillBank(ROOT/'skills','alfworld')
        self.assertEqual(bank.get('pick_and_place_simple'),bank.get('pick_and_place'))
        self.assertGreater(len(bank.get('pick_and_place')),len(bank.get('unknown')))

    def test_filter_signed_credit_and_nonfinite(self):
        old=torch.log(torch.tensor([[.1,.8,.2,.2,.2,.1,.1,.1]]))
        teacher=torch.tensor([[math.log(.4),math.log(.2),float('nan'),float('inf'),-100.,0.,-1.,-1.]],requires_grad=True)
        mask=torch.tensor([[1,1,1,1,1,1,0,1]])
        credit,keep=filtered_distillation(teacher,old,mask,confidence_min=.01,confidence_max=.9,max_abs_log_ratio=2.)
        self.assertTrue(credit[0,0]>0 and credit[0,1]<0)
        self.assertTrue(torch.isfinite(credit).all())
        self.assertEqual(credit[0,2:7].abs().sum(),0)
        self.assertFalse(credit.requires_grad)
        self.assertIsNone(teacher.grad)

    def test_all_rejected_loss_is_zero(self):
        old=torch.full((2,3),-2.)
        credit,_=filtered_distillation(torch.full((2,3),float('nan')),old,torch.ones_like(old),confidence_min=.1,confidence_max=.9,max_abs_log_ratio=2.)
        from verl.trainer.ppo.core_algos import compute_policy_loss
        now=old.clone().requires_grad_()
        loss,*_=compute_policy_loss(old,now,credit,torch.ones_like(old),.2)
        loss.backward()
        self.assertEqual(loss.item(),0)
        self.assertEqual(now.grad.abs().sum().item(),0)

    def test_success_dedup_position_and_raw_reward(self):
        b,_=make_batch()
        b=DataProto.concat([b.select_idxs([0]),b.select_idxs([0]),b.select_idxs([0]),b.select_idxs([1])])
        b.non_tensor_batch['traj_idx']=arr([0,0,1,0])
        b.non_tensor_batch['etpo_env_won']=arr([True,True,False,False])
        b.non_tensor_batch['rewards']=arr([10.,10.,100.,100.])
        selected,m=select_successful_rows(b)
        self.assertEqual(selected,[0])
        self.assertEqual(m['successful_tasks'] if 'successful_tasks' in m else m['etpo/successful_tasks'],1)
        b.non_tensor_batch['is_action_valid'][0]=False
        self.assertEqual(select_successful_rows(b)[0],[])

    def test_response_alignment_context_guard_and_padding(self):
        b,t=make_batch();c=configuration()
        teacher=build_teacher_batch(b,t,TinyBank(),c)
        self.assertTrue(torch.equal(teacher.batch['responses'],b.batch['responses']))
        self.assertTrue(torch.equal(teacher.batch['response_mask'],b.batch['response_mask']))
        self.assertGreater(teacher.batch['prompts'].shape[1],b.batch['prompts'].shape[1])
        supervised=supervised_subset(teacher,[0],4)
        self.assertEqual(len(supervised),4)
        self.assertEqual(supervised.batch['sft_mask'][1:].sum(),0)
        self.assertEqual(supervised.meta_info['sft_global_tokens'],2)
        self.assertEqual(teacher.batch['response_mask'][0].sum(),2)
        b.batch['input_ids'][0,0]=0
        with self.assertRaisesRegex(ValueError,'differs'):build_teacher_batch(b,t,TinyBank(),c)

    def test_global_sft_normalization(self):
        logp=torch.tensor([[-1.,-2.],[-3.,-4.],[-5.,-6.],[-7.,-8.]],requires_grad=True)
        mask=torch.tensor([[1.,1.],[1.,0.],[0.,0.],[1.,1.]])
        expected=masked_sft_sum(logp,mask)/mask.sum()
        distributed=sum(masked_sft_sum(lp,m)*2/mask.sum() for lp,m in zip(logp.chunk(2),mask.chunk(2)))/2
        self.assertTrue(torch.allclose(expected,distributed))
        with self.assertRaises(FloatingPointError):masked_sft_sum(torch.tensor([[float('nan')]]),torch.ones(1,1))

    def test_student_has_no_skill(self):
        text='Agent\n\n## Current Skill Document\nGuidance\n'+EMPTY_SKILL+'\n\nTask goal remains.'
        self.assertEqual(student_observation(text),'Agent\n\nTask goal remains.')

    def test_real_optimizer_closed_loop_and_checkpoint(self):
        torch.manual_seed(2)
        b,t=make_batch();c=configuration();w=LocalWorker(c)
        b.batch['old_log_probs']=w.compute_log_prob(b).batch['old_log_probs']
        old=b.batch['old_log_probs'].clone(); reward_adv=b.batch['advantages'].clone()
        before=[p.detach().clone() for p in w.model.parameters()]
        w.calls=[]
        batch,metrics=evolve_and_score(b,w,t,TinyBank(),c,1)
        self.assertEqual(w.calls,['sft','score'])
        self.assertTrue(any(not torch.equal(a,p) for a,p in zip(before,w.model.parameters())))
        self.assertTrue(torch.equal(old,batch.batch['old_log_probs']))
        self.assertTrue(torch.equal(reward_adv,batch.batch['advantages']))
        self.assertFalse(batch.batch['etpo_distill_advantages'].requires_grad)
        self.assertEqual(w.optimizer.param_groups[0]['lr'],.005)
        trained=[p.detach().clone() for p in w.model.parameters()]
        result=w.actor.update_policy(batch)
        self.assertIn('etpo/distill_loss',result)
        self.assertTrue(any(not torch.equal(a,p) for a,p in zip(trained,w.model.parameters())))
        # Shared teacher/student require one model + optimizer checkpoint.
        state=copy.deepcopy(w.model.state_dict());opt=copy.deepcopy(w.optimizer.state_dict())
        restored=LocalWorker(c);restored.model.load_state_dict(state);restored.optimizer.load_state_dict(opt)
        self.assertTrue(torch.equal(w.compute_log_prob(batch).batch['old_log_probs'],restored.compute_log_prob(batch).batch['old_log_probs']))

    def test_no_success_skips_sft(self):
        b,t=make_batch();c=configuration();w=LocalWorker(c)
        b.batch['old_log_probs']=w.compute_log_prob(b).batch['old_log_probs']
        b.non_tensor_batch['etpo_env_won']=arr([False,False])
        w.calls=[]
        _,m=evolve_and_score(b,w,t,TinyBank(),c,0)
        self.assertEqual(w.calls,['score']);self.assertEqual(m['etpo/sft_skipped'],1)

    def test_real_qwen_optimizer_path(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        torch.manual_seed(7)
        c=configuration();b,t=make_batch();w=LocalWorker(c)
        mc=Qwen3Config(vocab_size=65,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                      num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=2048)
        mc._attn_implementation='eager'
        w.model=Qwen3ForCausalLM(mc)
        w.optimizer=torch.optim.AdamW(w.model.parameters(),lr=.0001)
        w.actor=DataParallelPPOActor(c.actor_rollout_ref.actor,w.model,w.optimizer)
        b.batch['old_log_probs']=w.compute_log_prob(b).batch['old_log_probs']
        b,metrics=evolve_and_score(b,w,t,TinyBank(),c,0)
        result=w.actor.update_policy(b)
        self.assertTrue(all(torch.isfinite(p).all() for p in w.model.parameters()))
        self.assertIn('etpo/sft_loss',metrics)
        self.assertIn('etpo/distill_loss',result)

    def test_skill_snapshot_resume_guard(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            bank=SkillBank(ROOT/'skills','webshop')
            path=bank.snapshot(directory)
            self.assertTrue(path.exists())
            bank.snapshot(directory)
            bank.skills['general_skills'] += ' changed'
            with self.assertRaisesRegex(ValueError,'changed'):bank.snapshot(directory)

    def test_unsupported_configuration_rejected(self):
        c=configuration();c.etpo.teacher_mode='independent'
        with self.assertRaisesRegex(ValueError,'shared'):validate(c)
        c=configuration();c.actor_rollout_ref.rollout.temperature=.7
        with self.assertRaisesRegex(ValueError,'temperature'):validate(c)
        c=configuration();c.data.truncation='left'
        with self.assertRaisesRegex(ValueError,'truncation'):validate(c)

if __name__=='__main__':unittest.main()
