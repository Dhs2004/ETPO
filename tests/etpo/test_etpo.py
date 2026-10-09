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
from etpo.core import reward_aligned_credit, masked_sft_sum, select_successful_rows
from etpo.buffer import SuccessBuffer
from etpo.data import build_teacher_batch, supervised_subset, student_observation, EMPTY_SKILL
from etpo.skills import SkillBank
from etpo.training import evolve_and_score

ROOT = Path(__file__).resolve().parents[2]


def configuration():
    c = OmegaConf.load(ROOT/'verl/trainer/config/ppo_trainer.yaml')
    c.etpo.enabled = True
    c.etpo.skill_root = str(ROOT/'skills')
    c.etpo.teacher_update_batch_size = 1
    c.etpo.teacher_lr = .0005
    c.data.return_raw_chat = True
    c.data.truncation = 'error'
    c.algorithm.adv_estimator = 'grpo'
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
        'uid':arr(['group','group']), 'traj_uid':arr(['a','b']),'traj_idx':arr([0,0]),'turn_idx':arr([0,0]),
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
        self.initialize_teacher(c)
        self.calls=[]
    def initialize_teacher(self,c):
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        from etpo.teacher import initialize_teacher_from_student
        self.teacher_model=copy.deepcopy(self.model)
        initialize_teacher_from_student(self.model,self.teacher_model)
        self.teacher_optimizer=torch.optim.AdamW(self.teacher_model.parameters(),lr=c.etpo.teacher_lr,weight_decay=0)
        self.teacher_actor=DataParallelPPOActor(c.actor_rollout_ref.actor,self.teacher_model,self.teacher_optimizer)
    def compute_etpo_teacher_log_prob(self,batch):
        self.calls.append('score')
        data=copy.deepcopy(batch)
        data.meta_info.update(micro_batch_size=1,temperature=1.,use_dynamic_bsz=False)
        logp,_=self.teacher_actor.compute_log_prob(data)
        return DataProto.from_dict(tensors={'old_log_probs':logp})
    def compute_log_prob(self,batch):
        self.calls.append('score')
        data=copy.deepcopy(batch)
        data.meta_info.update(micro_batch_size=1,temperature=1.,use_dynamic_bsz=False)
        logp,_=self.actor.compute_log_prob(data)
        return DataProto.from_dict(tensors={'old_log_probs':logp})
    def update_etpo_teacher(self,batch):
        self.calls.append('sft')
        return DataProto(meta_info={'metrics':self.teacher_actor.update_etpo_teacher(batch)})


class ETPOTests(unittest.TestCase):
    def test_teacher_initialization_and_student_update_isolation(self):
        from etpo.teacher import initialize_teacher_from_student
        c=configuration();w=LocalWorker(c)
        for student,teacher in zip(w.model.parameters(),w.teacher_model.parameters()):
            self.assertTrue(torch.equal(student,teacher))
            self.assertNotEqual(student.data_ptr(),teacher.data_ptr())
        with self.assertRaisesRegex(ValueError,'distinct'):
            initialize_teacher_from_student(w.model,w.model)
        with self.assertRaisesRegex(ValueError,'storage'):
            initialize_teacher_from_student(w.model,copy.copy(w.model))
        b,t=make_batch();b.batch['old_log_probs']=w.compute_log_prob(b).batch['old_log_probs']
        before=[p.detach().clone() for p in w.teacher_model.parameters()]
        w.actor.update_policy(b)
        self.assertTrue(all(torch.equal(a,p) for a,p in zip(before,w.teacher_model.parameters())))
        self.assertEqual(len(w.teacher_optimizer.state),0)

    def test_skill_sources_and_alias(self):
        for env in ['alfworld','webshop','sciworld']:
            bank=SkillBank(ROOT/'skills',env)
            self.assertTrue(bank.get('unknown').strip())
        bank=SkillBank(ROOT/'skills','alfworld')
        self.assertEqual(bank.get('pick_and_place_simple'),bank.get('pick_and_place'))
        self.assertGreater(len(bank.get('pick_and_place')),len(bank.get('unknown')))

    def test_credit_matches_gap_grpo_and_preserves_sign(self):
        reward=torch.tensor([[2.,2.,-2.,-2.,0.,0.]])
        old=torch.full_like(reward,-3.)
        teacher=old+torch.tensor([[1.,-1.,1.,-1.,1.,-1.]])
        adjusted,metrics=reward_aligned_credit(teacher,old,torch.ones_like(old),reward)
        expected=reward*(.5+torch.sigmoid(5*reward.sign()*(teacher-old)))
        self.assertTrue(torch.equal(adjusted,expected))
        self.assertTrue(torch.equal(adjusted.sign(),reward.sign()))
        self.assertTrue((adjusted.abs()>=.5*reward.abs()).all())
        self.assertTrue((adjusted.abs()<=1.5*reward.abs()).all())
        self.assertFalse(adjusted.requires_grad)
        self.assertGreater(adjusted[0,0],reward[0,0])
        self.assertLess(adjusted[0,1],reward[0,1])
        self.assertGreater(adjusted[0,2],reward[0,2])
        self.assertLess(adjusted[0,3],reward[0,3])
        for ratio in (.5,1.,1.5):
            from verl.trainer.ppo.core_algos import compute_policy_loss
            now=(old+math.log(ratio)).requires_grad_()
            loss,*_=compute_policy_loss(old,now,adjusted,torch.ones_like(old),.2)
            loss.backward()
            self.assertTrue((now.grad*reward<=0).all())

    def test_top20_gap_exclusion_is_global_signed_and_keeps_reward(self):
        # Ten valid response tokens, two padding tokens with extreme scores.
        old=torch.full((2,6),-200.)
        gap=torch.tensor([[10.,9.,8.,7.,6.,1000.],[-100.,1.,2.,3.,4.,1000.]])
        mask=torch.tensor([[1.,1.,1.,1.,1.,0.],[1.,1.,1.,1.,1.,0.]])
        reward=torch.tensor([[1.,-1.,1.,1.,0.,0.],[-1.,1.,1.,-1.,1.,0.]])
        full,_=reward_aligned_credit(old+gap,old,mask,reward)
        adjusted,metrics=reward_aligned_credit(old+gap,old,mask,reward,exclude_top20_gap_tokens=True)
        self.assertEqual(metrics['etpo/excluded_gap_tokens'],2)
        self.assertEqual(adjusted[0,0].item(),1.)
        self.assertEqual(adjusted[0,1].item(),-1.)
        self.assertNotEqual(full[0,0].item(),1.)
        self.assertNotEqual(full[0,1].item(),-1.)
        self.assertTrue(torch.equal(adjusted[1],full[1]))  # -100 not selected by abs gap
        self.assertTrue(torch.equal(adjusted[0,2:],full[0,2:]))
        self.assertTrue(torch.equal(adjusted.sign(),reward.sign()))

    def test_top20_ties_small_batch_and_all_padding(self):
        old=torch.full((1,5),-3.);reward=torch.ones_like(old)
        adjusted,m=reward_aligned_credit(old+1,old,reward,reward,exclude_top20_gap_tokens=True)
        self.assertEqual(m['etpo/excluded_gap_tokens'],1)
        self.assertEqual(adjusted[0,0].item(),1.)
        self.assertTrue((adjusted[0,1:]>1).all())
        _,m=reward_aligned_credit(old[:,:4]+1,old[:,:4],reward[:,:4],reward[:,:4],exclude_top20_gap_tokens=True)
        self.assertEqual(m['etpo/excluded_gap_tokens'],0)
        adjusted,m=reward_aligned_credit(old,old,torch.zeros_like(old),reward,exclude_top20_gap_tokens=True)
        self.assertEqual(adjusted.sum(),0);self.assertEqual(m['etpo/excluded_gap_tokens'],0)

    def test_grpo_task_groups_and_advantages(self):
        from etpo.data import prepare_grpo_groups
        from verl.trainer.ppo.ray_trainer import compute_advantage
        mask=torch.ones(4,2)
        batch=DataProto.from_dict(tensors={'response_mask':mask,
            'token_level_rewards':torch.tensor([[0.,10.],[0.,0.],[0.,0.],[0.,10.]])},
            non_tensors={'uid':arr(['group']*4),'traj_uid':arr(['a','b','a','b']),
                         'traj_idx':arr([0,0,1,1])})
        prepare_grpo_groups(batch)
        out=compute_advantage(batch,adv_estimator='grpo')
        expected=torch.tensor([1.,-1.,-1.,1.])*(5/(math.sqrt(50)+1e-6))
        self.assertTrue(torch.allclose(out.batch['advantages'],expected[:,None].expand(4,2)))
        self.assertEqual(list(out.non_tensor_batch['traj_uid']),['a','b','a','b'])

    def test_sft_default_lr_and_grpo_config(self):
        c=configuration()
        raw=OmegaConf.load(ROOT/'verl/trainer/config/ppo_trainer.yaml')
        sft=OmegaConf.load(ROOT/'verl/trainer/config/sft_trainer.yaml')
        self.assertEqual(float(raw.etpo.teacher_lr),float(sft.optim.lr))
        c.algorithm.adv_estimator='skillrise'
        with self.assertRaisesRegex(ValueError,'GRPO'): validate(c)
        c=configuration();c.etpo.exclude_top20_gap_tokens='true'
        with self.assertRaisesRegex(ValueError,'boolean'): validate(c)

    def test_credit_padding_zero_gap_and_validation(self):
        old=torch.tensor([[-2.,float('nan')]])
        mask=torch.tensor([[1.,0.]])
        reward=torch.tensor([[2.,float('nan')]])
        adjusted,_=reward_aligned_credit(old,old,mask,reward)
        self.assertEqual(adjusted.tolist(),[[2.,0.]])
        with self.assertRaisesRegex(ValueError,'finite'):
            reward_aligned_credit(old,old,torch.ones_like(mask),reward)
        with self.assertRaisesRegex(ValueError,'token-aligned'):
            reward_aligned_credit(old,old,mask,torch.ones(1,1))
        with self.assertRaisesRegex(ValueError,'gate_beta'):
            reward_aligned_credit(old,old,mask,reward,gate_beta=-1)

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
        teacher_before=[p.detach().clone() for p in w.teacher_model.parameters()]
        student_optimizer_before=copy.deepcopy(w.optimizer.state_dict())
        w.calls=[]
        batch,metrics=evolve_and_score(b,w,t,TinyBank(),c,1,SuccessBuffer())
        self.assertEqual(w.calls,['sft','score'])
        self.assertTrue(all(torch.equal(a,p) for a,p in zip(before,w.model.parameters())))
        self.assertTrue(any(not torch.equal(a,p) for a,p in zip(teacher_before,w.teacher_model.parameters())))
        self.assertEqual(student_optimizer_before,w.optimizer.state_dict())
        self.assertTrue(torch.equal(old,batch.batch['old_log_probs']))
        adjusted=batch.batch['advantages']
        self.assertTrue(torch.equal(adjusted.sign(),reward_adv.sign()))
        self.assertTrue((adjusted.abs()>=.5*reward_adv.abs()).all())
        self.assertTrue((adjusted.abs()<=1.5*reward_adv.abs()).all())
        self.assertFalse(adjusted.requires_grad)
        self.assertNotIn('etpo_distill_advantages',batch.batch)
        self.assertEqual(w.optimizer.param_groups[0]['lr'],.005)
        trained=[p.detach().clone() for p in w.model.parameters()]
        teacher_after_sft=[p.detach().clone() for p in w.teacher_model.parameters()]
        result=w.actor.update_policy(batch)
        self.assertTrue(all(torch.equal(a,p) for a,p in zip(teacher_after_sft,w.teacher_model.parameters())))
        self.assertNotIn('etpo/distill_loss',result)
        self.assertTrue(any(not torch.equal(a,p) for a,p in zip(trained,w.model.parameters())))
        # Independent teacher and student retain separate weights and optimizer states.
        state=copy.deepcopy(w.model.state_dict());opt=copy.deepcopy(w.optimizer.state_dict())
        restored=LocalWorker(c);restored.model.load_state_dict(state);restored.optimizer.load_state_dict(opt)
        restored.teacher_model.load_state_dict(w.teacher_model.state_dict())
        restored.teacher_optimizer.load_state_dict(w.teacher_optimizer.state_dict())
        self.assertTrue(torch.equal(w.compute_log_prob(batch).batch['old_log_probs'],restored.compute_log_prob(batch).batch['old_log_probs']))
        self.assertTrue(torch.equal(w.compute_etpo_teacher_log_prob(batch).batch['old_log_probs'],restored.compute_etpo_teacher_log_prob(batch).batch['old_log_probs']))

    def test_no_success_skips_sft(self):
        b,t=make_batch();c=configuration();w=LocalWorker(c)
        b.batch['old_log_probs']=w.compute_log_prob(b).batch['old_log_probs']
        b.non_tensor_batch['etpo_env_won']=arr([False,False])
        w.calls=[]
        _,m=evolve_and_score(b,w,t,TinyBank(),c,0,SuccessBuffer())
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
        w.initialize_teacher(c)
        b.batch['old_log_probs']=w.compute_log_prob(b).batch['old_log_probs']
        b,metrics=evolve_and_score(b,w,t,TinyBank(),c,0,SuccessBuffer())
        result=w.actor.update_policy(b)
        self.assertTrue(all(torch.isfinite(p).all() for p in w.model.parameters()))
        self.assertIn('etpo/sft_loss',metrics)
        self.assertNotIn('etpo/distill_loss',result)

    def test_buffer_threshold_across_steps_and_resume(self):
        import tempfile
        c=configuration();c.etpo.teacher_update_batch_size=3
        w=LocalWorker(c);buffer=SuccessBuffer()
        for step in range(2):
            b,t=make_batch()
            b.batch['old_log_probs']=w.compute_log_prob(b).batch['old_log_probs']
            before=[p.detach().clone() for p in w.model.parameters()]
            w.calls=[]
            _,m=evolve_and_score(b,w,t,TinyBank(),c,step,buffer)
            self.assertEqual(w.calls,['score'])
            self.assertEqual(m['etpo/teacher_updates'],0)
            self.assertEqual(len(buffer),step+1)
            self.assertTrue(all(torch.equal(a,p) for a,p in zip(before,w.model.parameters())))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'buffer.pt';buffer.save(path)
            restored=SuccessBuffer();restored.load(path)
            self.assertEqual(len(restored),2)
            self.assertTrue(torch.equal(buffer.batch(2,0).batch['input_ids'],restored.batch(2,0).batch['input_ids']))
        # Two successes added: consume exactly three and retain one.
        b,t=make_batch();b.non_tensor_batch['etpo_env_won']=arr([True,True]);b.non_tensor_batch['rewards']=arr([10.,10.])
        b.batch['old_log_probs']=w.compute_log_prob(b).batch['old_log_probs'];w.calls=[]
        _,m=evolve_and_score(b,w,t,TinyBank(),c,2,restored)
        self.assertEqual(w.calls,['sft','score'])
        self.assertEqual(m['etpo/sft_tasks'],3)
        self.assertEqual(len(restored),1)
        # Exactly two new successes complete another batch.
        b,t=make_batch();b.non_tensor_batch['etpo_env_won']=arr([True,True]);b.non_tensor_batch['rewards']=arr([10.,10.])
        b.batch['old_log_probs']=w.compute_log_prob(b).batch['old_log_probs'];w.calls=[]
        _,m=evolve_and_score(b,w,t,TinyBank(),c,3,restored)
        self.assertEqual(w.calls,['sft','score']);self.assertEqual(len(restored),0)

    def test_buffer_repadding_and_multiple_updates(self):
        b,t=make_batch();c=configuration();teacher=build_teacher_batch(b,t,TinyBank(),c)
        buffer=SuccessBuffer()
        buffer.add(teacher,b,[0,1],raw_weights={('a',0):1.,('b',0):1.})
        combined=buffer.batch(2,t.pad_token_id)
        self.assertEqual(combined.batch['response_mask'].sum().item(),5)
        self.assertEqual(combined.batch['responses'][0,:2].tolist(),[12,13])
        self.assertEqual(combined.batch['responses'][1,:3].tolist(),[15,16,17])
        w=LocalWorker(c);b.batch['old_log_probs']=w.compute_log_prob(b).batch['old_log_probs'];w.calls=[]
        _,m=evolve_and_score(b,w,t,TinyBank(),c,1,buffer)
        self.assertEqual(w.calls,['sft','sft','sft','score'])
        self.assertEqual(m['etpo/teacher_updates'],3)
        self.assertEqual(len(buffer),0)

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
        c=configuration();c.etpo.teacher_mode='shared'
        with self.assertRaisesRegex(ValueError,'independent'):validate(c)
        c=configuration();c.actor_rollout_ref.rollout.temperature=.7
        with self.assertRaisesRegex(ValueError,'temperature'):validate(c)
        c=configuration();c.data.truncation='left'
        with self.assertRaisesRegex(ValueError,'truncation'):validate(c)

if __name__=='__main__':unittest.main()
