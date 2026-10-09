"""CPU detached observations of actual credit and teacher objective coefficients.
No sampling, training tensor mutation or additional teacher forward passes.
"""
import math
import os
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict
from etpo.diagnostics import distribution, write_audit


def ratio(a,b):
    return float(a)/float(b) if float(b)>0 else 0.


def summary(a,g,s,prefix):
    # Inputs are flat float64 CPU tensors; s is actual after/before for nonzero A.
    b=a*s;delta=b-a;n=len(a);active=a!=0
    out={prefix+'/tokens':n,prefix+'/active_tokens':int(active.sum())}
    for name,x in [('gap',g),('abs_gap',g.abs()),('adv_before',a),('abs_adv_before',a.abs()),('adv_after',b),('abs_adv_after',b.abs()),('abs_credit_delta',delta.abs()),('scale_active',s[active])]:
        out.update(distribution(x,prefix+'/'+name))
    old=a.abs().sum();new=b.abs().sum()
    out.update({prefix+'/abs_mass_before':float(old),prefix+'/abs_mass_after':float(new),prefix+'/abs_mass_ratio':ratio(new,old),prefix+'/abs_delta_over_old_mass':ratio(delta.abs().sum(),old),prefix+'/signed_delta_sum':float(delta.sum()),prefix+'/sign_flip_fraction':ratio((a.sign()!=b.sign()).sum(),n)})
    for name,mask in [('adv_positive',a>0),('adv_negative',a<0),('adv_zero',a==0),('gap_positive',g>0),('gap_negative',g<0),('gap_zero',g==0),('aligned',a*g>0),('conflict',a*g<0)]:
        out[prefix+'/'+name+'/token_fraction']=ratio(mask.sum(),n)
        out[prefix+'/'+name+'/before_abs_mass_share']=ratio(a[mask].abs().sum(),old)
        out[prefix+'/'+name+'/after_abs_mass_share']=ratio(b[mask].abs().sum(),new)
        out[prefix+'/'+name+'/before_abs_mean']=ratio(a[mask].abs().sum(),mask.sum())
        out[prefix+'/'+name+'/after_abs_mean']=ratio(b[mask].abs().sum(),mask.sum())
        out[prefix+'/'+name+'/abs_mass_ratio']=ratio(b[mask].abs().sum(),a[mask].abs().sum())
        out[prefix+'/'+name+'/gap_abs_mass_share']=ratio(g[mask].abs().sum(),g.abs().sum())
        out[prefix+'/'+name+'/gap_abs_sum']=float(g[mask].abs().sum())
        out[prefix+'/'+name+'/gap_abs_mean']=ratio(g[mask].abs().sum(),mask.sum())
    out[prefix+'/gap_abs_mass_total']=float(g.abs().sum())
    out[prefix+'/gap_positive_negative_abs_mass_ratio']=ratio(g[g>0].sum(),-g[g<0].sum())
    out[prefix+'/positive_negative_abs_mass_ratio']=ratio(b[b>0].sum(),-b[b<0].sum())
    for sign,sel in [('positive',a>0),('negative',a<0)]:
        for direction,gm in [('positive',g>0),('negative',g<0),('zero',g==0)]:
            m=sel&gm;key=prefix+f'/joint_adv_{sign}_gap_{direction}'
            out[key+'/token_fraction']=ratio(m.sum(),n)
            out[key+'/abs_mass_share']=ratio(b[m].abs().sum(),new)
    # Lower inclusive, upper exclusive except final bin. Zero-A tokens excluded.
    for lo,hi in zip([.5,.6,.7,.8,.9,1.,1.1,1.2,1.3,1.4],[.6,.7,.8,.9,1.,1.1,1.2,1.3,1.4,1.5]):
        m=active&((s>=lo) if lo>.5 else torch.ones_like(active))&((s<hi) if hi<1.5 else torch.ones_like(active));k=prefix+f'/scale_bin_{lo:.1f}_{hi:.1f}'
        out[k+'/active_token_fraction']=ratio(m.sum(),active.sum())
        out[k+'/before_abs_mass_share']=ratio(a[m].abs().sum(),old)
        out[k+'/after_abs_mass_share']=ratio(b[m].abs().sum(),new)
    out[prefix+'/scale_out_of_range_fraction']=ratio((active&((s<.5-1e-6)|(s>1.5+1e-6))).sum(),active.sum())
    for name,m in [('suppressed',active&(s<1-1e-6)),('unchanged',active&((s-1).abs()<=1e-6)),('enhanced',active&(s>1+1e-6))]:
        out[prefix+'/'+name+'/active_token_fraction']=ratio(m.sum(),active.sum())
        out[prefix+'/'+name+'/abs_delta_share']=ratio(delta[m].abs().sum(),delta.abs().sum())
    return out


def credit_diagnostics(batch,before,after,teacher_logp,step,beta):
    mask=batch.batch['response_mask'].detach().bool().cpu()
    a=before.detach().double().cpu();b=after.detach().double().cpu()
    old=batch.batch['old_log_probs'].detach().double().cpu();teacher=teacher_logp.detach().double().cpu();g=teacher-old
    s=torch.ones_like(a);active=mask&(a!=0);s[active]=b[active]/a[active]
    av,gv,sv=a[mask],g[mask],s[mask]
    metrics=summary(av,gv,sv,'etpo_detail/all')
    for name,x in [('student_logp',old[mask]),('teacher_logp',teacher[mask]),('probability_ratio_clamped',gv.clamp(-30,30).exp())]:metrics.update(distribution(x,'etpo_detail/'+name))
    metrics['etpo_detail/probability_ratio_clipped_fraction']=ratio((gv.abs()>30).sum(),len(gv))
    # Observational alternatives, never applied to advantages.
    for alternative_beta in (1.,2.,5.,10.):
        for amplitude in (.1,.25,.5):
            alt=1+2*amplitude*(torch.sigmoid(alternative_beta*av.sign()*gv)-.5)
            key=f'etpo_counterfactual/beta{alternative_beta:g}_amplitude{amplitude:g}'
            metrics[key+'/abs_mass_ratio']=ratio((av*alt).abs().sum(),av.abs().sum())
            metrics[key+'/abs_delta_ratio']=ratio((av*(alt-1)).abs().sum(),av.abs().sum())
            metrics[key+'/strong_change_fraction']=ratio(((alt-1).abs()>=.25).logical_and(av!=0).sum(),(av!=0).sum())
    n=batch.non_tensor_batch;units=defaultdict(list)
    for i in range(len(batch)):units[(str(n['traj_uid'][i]),int(n['traj_idx'][i]))].append(i)
    records=[];outcome=torch.zeros(len(batch),dtype=torch.bool)
    # Training aggregates include padding rows, trajectory audits deduplicate turns.
    for (uid,pos),rows in units.items():
        turns={int(n['turn_idx'][i]):i for i in rows};chosen=list(turns.values());won=any(bool(n['etpo_env_won'][i]) for i in rows);outcome[rows]=won
        m=mask[chosen];ta=a[chosen][m];tg=g[chosen][m];ts=s[chosen][m]
        rec={'trajectory':uid,'task_position':pos,'group':str(n['uid'][rows[0]]),'won':won,'turns':len(chosen),'metrics':summary(ta,tg,ts,'credit')}
        records.append(rec)
    for label,selection in [('success',outcome),('failure',~outcome)]:
        m=mask&selection[:,None];metrics.update(summary(a[m],g[m],s[m],'etpo_detail/'+label))
    # Token-rank position within response, independent of prompt padding.
    rank=mask.cumsum(1);length=mask.sum(1).clamp_min(1)[:,None];position=(rank-.5)/length
    for i in range(4):
        m=mask&(position>=i/4)&(position<(i+1)/4)
        metrics.update(summary(a[m],g[m],s[m],f'etpo_position/quartile{i+1}'))
    root=os.environ.get('ETPO_DIAGNOSTICS_DIR')
    if root:
        dest=Path(root)/'tokens';dest.mkdir(parents=True,exist_ok=True)
        row,col=mask.nonzero(as_tuple=True)
        # Complete valid-token observations: float32 logps/advantages, no text.
        np.savez_compressed(dest/f'step_{step:04d}.npz',row=row.numpy(),column=col.numpy(),
            response_token=batch.batch['responses'].detach().cpu()[mask].numpy(),
            student_logp=old[mask].float().numpy(),teacher_logp=teacher[mask].float().numpy(),
            advantage_before=av.float().numpy(),advantage_after=b[mask].float().numpy(),
            trajectory=np.array([str(x) for x in n['traj_uid']]),task_position=np.asarray(n['traj_idx'],dtype=np.int32),
            turn=np.asarray(n['turn_idx'],dtype=np.int32),group=np.array([str(x) for x in n['uid']]),
            won=outcome.numpy())
    write_audit('credit_trajectories' ,{'step':step,'rows_include_padding':False,'trajectories':records})
    write_audit('credit_steps',{'step':step,'actual_beta':beta,'metrics':metrics})
    return metrics


def teacher_objective_diagnostics(raw,lengths,metadata,weighted,step,update):
    r=torch.tensor(raw,dtype=torch.float64);length=torch.tensor(lengths,dtype=torch.float64);n=len(r)
    assert n and (r>0).all() and (length>0).all()
    uniform=torch.ones_like(r)/n;token_mass=length/length.sum();inverse=r/r.sum()
    actual=inverse if weighted else token_mass
    result={};schemes={'actual':actual,'unweighted_token_mean':token_mass,'uniform_trajectory_mean':uniform,'inverse_success_trajectory_mean':inverse}
    for name,mass in schemes.items():
        p='etpo_sft_detail/'+name
        result.update(distribution(mass,p+'/trajectory_mass'))
        result.update(distribution(mass/length,p+'/per_token_coefficient'))
        result[p+'/ess']=float(1/mass.square().sum());result[p+'/ess_fraction']=result[p+'/ess']/n
        for pct in (.01,.1,.2):result[p+f'/top{int(100*pct)}pct_mass_share']=float(mass.topk(max(1,math.ceil(n*pct))).values.sum())
        result[p+'/entropy']=float(-(mass*mass.log()).sum())
        for rate in (1/8,2/8,3/8,4/8,5/8,6/8,7/8,1.):
            m=torch.isclose(1/r,torch.tensor(rate,dtype=r.dtype));key=p+f'/success_rate_{rate:g}'
            result[key+'/trajectory_fraction']=ratio(m.sum(),n);result[key+'/mass_share']=float(mass[m].sum())
        result.update(distribution(mass/token_mass,p+'/mass_ratio_to_token_mean'))
    result['etpo_sft_detail/weighting_enabled']=int(weighted)
    result['etpo_sft_detail/length_normalization_total_variation']=float((uniform-token_mass).abs().sum()/2)
    result['etpo_sft_detail/inverse_success_total_variation']=float((inverse-uniform).abs().sum()/2)
    result['etpo_sft_detail/actual_vs_token_mean_total_variation']=float((actual-token_mass).abs().sum()/2)
    ages=[step-m['collected_step'] for m in metadata];result.update(distribution(ages,'etpo_sft_detail/buffer_age_steps'))
    records=[]
    for i in range(n):
        records.append(dict(metadata[i],tokens=lengths[i],raw_inverse_success=raw[i],success_rate=1/raw[i],normalized_weight=float(r[i]/r.mean()),actual_mass=float(actual[i]),actual_per_token_coefficient=float(actual[i]/length[i]),token_mean_mass=float(token_mass[i]),uniform_trajectory_mass=1/n,inverse_success_mass=float(inverse[i])))
    return result,{'step':step,'update':update,'weighting_enabled':weighted,'trajectories':records,'metrics':result}
