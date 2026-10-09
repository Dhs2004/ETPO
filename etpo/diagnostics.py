"""Detached diagnostics; never modifies objectives or sampling decisions."""
import json,math,os
from pathlib import Path
from collections import defaultdict
import torch


def distribution(values, prefix):
    x=torch.as_tensor(values,dtype=torch.float64).flatten().detach().cpu()
    result={prefix+'/count':x.numel()}
    if not x.numel():return result
    result.update({prefix+'/mean':x.mean().item(),prefix+'/std':x.std(unbiased=False).item(),prefix+'/min':x.min().item(),prefix+'/max':x.max().item()})
    for q,v in zip((10,50,90,99),torch.quantile(x,torch.tensor([.1,.5,.9,.99],dtype=torch.float64)).tolist()):result[prefix+f'/p{q}']=v
    return result


def weight_metrics(raw,lengths):
    x=torch.tensor(raw,dtype=torch.float64);w=x/x.mean();n=len(w)
    result=distribution(raw,'etpo/sft_raw_weight')
    result.update(distribution(w,'etpo/sft_normalized_weight'))
    result.update(distribution(lengths,'etpo/sft_trajectory_tokens'))
    ess=(w.sum().square()/w.square().sum()).item()
    result.update({'etpo/sft_weight_ess':ess,'etpo/sft_weight_ess_fraction':ess/n,'etpo/sft_top10pct_weight_share':(w.topk(max(1,math.ceil(n*.1))).values.sum()/w.sum()).item()})
    y=torch.tensor(lengths,dtype=torch.float64)
    if n>1 and w.std()>0 and y.std()>0:result['etpo/sft_weight_length_correlation']=torch.corrcoef(torch.stack([w,y]))[0,1].item()
    return result


def rollout_metrics(batch,indices):
    n=batch.non_tensor_batch;units={}; groups=defaultdict(list)
    eligible={(str(n['traj_uid'][i]),int(n['traj_idx'][i])) for i in indices}
    for i in range(len(batch)):
        key=(str(n['traj_uid'][i]),int(n['traj_idx'][i]));turn=int(n['turn_idx'][i])
        units.setdefault(key,{})[turn]=i
    records=[]
    for key,turns in units.items():
        rows=list(turns.values());group=(str(n['uid'][rows[0]]),key[1]);won=any(bool(n['etpo_env_won'][i]) for i in rows)
        record=dict(trajectory=key[0],task_position=key[1],group=group[0],won=won,eligible=key in eligible,valid_actions=all(bool(n['is_action_valid'][i]) for i in rows),reward=sum(float(n['rewards'][i]) for i in rows),turns=len(rows),tokens=sum(int(batch.batch['response_mask'][i].sum()) for i in rows))
        records.append(record);groups[group].append(record)
    for rows in groups.values():
        success=sum(r['eligible'] for r in rows); rate=success/len(rows)
        for r in rows:r.update(group_size=len(rows),eligible_success_rate=rate,raw_sft_weight=(1/rate if r['eligible'] else 0.))
    count=len(records);won=sum(r['won'] for r in records);valid_success=sum(r['eligible'] for r in records)
    m={'etpo/rollout_trajectories':count,'etpo/env_success_fraction':won/max(1,count),'etpo/eligible_success_fraction':valid_success/max(1,count),'etpo/success_rejected_count':won-valid_success,'etpo/invalid_trajectory_fraction':sum(not r['valid_actions'] for r in records)/max(1,count),'etpo/zero_success_group_fraction':sum(not any(r['eligible'] for r in rows) for rows in groups.values())/max(1,len(groups))}
    m.update(distribution([rows[0]['eligible_success_rate'] for rows in groups.values()],'etpo/group_eligible_success_rate'))
    for label in ('success','failure'):
        subset=[r for r in records if r['won']==(label=='success')]
        m.update(distribution([r['tokens'] for r in subset],f'etpo/{label}_trajectory_tokens'))
    return m,records


def credit_outcome_metrics(batch,before,after,teacher_logp):
    mask=batch.batch['response_mask'].bool(); gap=teacher_logp-batch.batch['old_log_probs']
    n=batch.non_tensor_batch;wins=defaultdict(bool)
    for i in range(len(batch)):wins[(str(n['traj_uid'][i]),int(n['traj_idx'][i]))]|=bool(n['etpo_env_won'][i])
    outcome=torch.tensor([wins[(str(n['traj_uid'][i]),int(n['traj_idx'][i]))] for i in range(len(batch))],device=mask.device)
    result={};denom=after[mask].abs().sum().item()
    for label,row_mask in [('success',outcome),('failure',~outcome)]:
        valid=mask & row_mask[:,None];old=before[valid].float();new=after[valid].float()
        result[f'etpo/{label}_credit_tokens']=int(valid.sum())
        result[f'etpo/{label}_adv_abs_mass_share']=new.abs().sum().item()/max(denom,1e-30)
        if old.numel():
            result[f'etpo/{label}_adv_abs_ratio']=new.abs().sum().item()/max(old.abs().sum().item(),1e-30)
            result[f'etpo/{label}_teacher_gap_mean']=gap[valid].float().mean().item()
    active=mask & (before!=0)
    if active.any():
        scale=(after[active]/before[active]).float()
        result['etpo/credit_near_lower_bound_fraction']=(scale<.55).float().mean().item()
        result['etpo/credit_near_upper_bound_fraction']=(scale>1.45).float().mean().item()
    return result


def write_audit(name,record):
    root=os.environ.get('ETPO_DIAGNOSTICS_DIR')
    if root:
        d=Path(root);d.mkdir(parents=True,exist_ok=True)
        with (d/(name+'.jsonl')).open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
