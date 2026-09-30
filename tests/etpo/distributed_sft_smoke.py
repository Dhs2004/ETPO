"""Two-rank CPU DDP check: one success rank, one zero-loss padding rank.

Run with torchrun --standalone --nproc_per_node=2; verifies the production SFT
optimizer routine against an unsharded reference, without claiming GPU/FSDP proof.
"""
import copy
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from test_etpo import configuration, make_batch, TinyBank, TinyCausalLM, ROOT
from etpo.data import build_teacher_batch, supervised_subset
from verl.workers.actor.dp_actor import DataParallelPPOActor


def main():
    dist.init_process_group('gloo')
    try:
        rank=dist.get_rank()
        torch.manual_seed(19)
        config=configuration()
        model=TinyCausalLM()
        initial=copy.deepcopy(model.state_dict())
        wrapped=DistributedDataParallel(model)
        optimizer=torch.optim.SGD(wrapped.parameters(),lr=.01)
        actor=DataParallelPPOActor(config.actor_rollout_ref.actor,wrapped,optimizer)
        batch,tokenizer=make_batch()
        teacher=build_teacher_batch(batch,tokenizer,TinyBank(),config)
        data=supervised_subset(teacher,[0],2)
        actor.update_etpo_teacher(data.select_idxs([rank]))
        vector=torch.cat([p.detach().flatten() for p in model.parameters()])
        gathered=[torch.empty_like(vector) for _ in range(2)]
        dist.all_gather(gathered,vector)
        assert torch.equal(gathered[0],gathered[1]), 'Ranks diverged after SFT'
        if rank==0:
            reference=TinyCausalLM();reference.load_state_dict(initial)
            opt=torch.optim.SGD(reference.parameters(),lr=.01)
            single=DataParallelPPOActor(config.actor_rollout_ref.actor,reference,opt)
            single.update_etpo_teacher(supervised_subset(teacher,[0],1))
            ref=torch.cat([p.detach().flatten() for p in reference.parameters()])
            error=float((vector-ref).abs().max())
            assert torch.allclose(vector,ref,atol=2e-6,rtol=2e-6), error
            result={'backend':'gloo','ranks':2,'padding_rank_has_zero_supervision':True,
                    'ranks_identical':True,'max_difference_from_single_rank':error,
                    'scope':'CPU DDP SFT normalization; GPU FSDP not tested'}
            out=ROOT/'logs/distributed-sft.json'
            out.write_text(json.dumps(result,indent=2))
            print(json.dumps(result))
    finally:
        dist.destroy_process_group()

if __name__=='__main__':main()
