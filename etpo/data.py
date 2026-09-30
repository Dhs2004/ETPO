"""Build the teacher context while preserving the exact student response tokens."""
import copy
import re
import torch
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

EMPTY_SKILL = '(The skill document is currently empty. No skills have been distilled yet.)'


def student_observation(text):
    """Remove only SkillRise's empty-skill preamble; never strip task history."""
    pattern = r'## Current Skill Document\n.*?' + re.escape(EMPTY_SKILL) + r'\n*'
    return re.sub(pattern, '', text, count=1, flags=re.DOTALL)


def build_teacher_batch(batch, tokenizer, skill_bank, config):
    if 'multi_modal_inputs' in batch.non_tensor_batch:
        raise ValueError('ETPO currently supports text environments only')
    if 'raw_prompt' not in batch.non_tensor_batch or 'etpo_task_type' not in batch.non_tensor_batch:
        raise ValueError('ETPO requires raw student messages and task types')
    responses = batch.batch['responses'].clone()
    response_mask = batch.batch['response_mask'].clone().long()
    prompt_length = batch.batch['input_ids'].shape[1] - responses.shape[1]
    prompts = []
    thinking = config.actor_rollout_ref.model.get('enable_thinking', False)
    for i in range(len(batch)):
        messages = batch.non_tensor_batch['raw_prompt'][i]
        if hasattr(messages, 'tolist'):
            messages = messages.tolist()
        messages = copy.deepcopy(list(messages))
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
        expected = tokenizer.encode(rendered, add_special_tokens=False)
        actual = batch.batch['input_ids'][i, :prompt_length][batch.batch['attention_mask'][i, :prompt_length].bool()].tolist()
        if expected != actual:
            raise ValueError('Student raw chat differs from rollout tokens; truncation or template mismatch is unsafe')
        skill = skill_bank.get(batch.non_tensor_batch['etpo_task_type'][i])
        if len(tokenizer.encode(skill, add_special_tokens=False)) > config.etpo.max_skill_tokens:
            raise ValueError('Initial skill exceeds max_skill_tokens; edit the skill or increase the explicit budget')
        context = 'Use the following initial skills when applicable. The current task and observed history remain authoritative.\n\n' + skill
        if messages and messages[0]['role'] == 'system':
            messages[0]['content'] += '\n\n' + context
        else:
            messages.insert(0, {'role': 'system', 'content': context})
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > config.etpo.teacher_max_prompt_length:
            raise ValueError(f'Teacher prompt ({len(tokens)}) exceeds teacher_max_prompt_length; no silent truncation')
        prompts.append(tokens)
    width = max(map(len, prompts))
    device = responses.device
    teacher_prompts = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    prompt_mask = torch.zeros_like(teacher_prompts)
    for i, tokens in enumerate(prompts):
        teacher_prompts[i, -len(tokens):] = torch.tensor(tokens, dtype=torch.long, device=device)
        prompt_mask[i, -len(tokens):] = 1
    attention = torch.cat([prompt_mask, response_mask], dim=1)
    return DataProto.from_dict(tensors={
        'prompts': teacher_prompts, 'responses': responses,
        'input_ids': torch.cat([teacher_prompts, responses], dim=1),
        'attention_mask': attention, 'position_ids': compute_position_id_with_mask(attention),
        'response_mask': response_mask,
    }, meta_info={'temperature': 1.0})


def supervised_subset(teacher, indices, world_size):
    """Pad each distributed SFT call equally; padding has exactly zero loss weight."""
    selected = teacher.select_idxs(indices)
    selected.batch['sft_mask'] = selected.batch['response_mask'].clone().float()
    total_tokens = int(selected.batch['sft_mask'].sum().item())
    if not total_tokens:
        raise ValueError('Successful trajectories contain no supervised response tokens')
    pad = (-len(selected)) % world_size
    if pad:
        extra = selected.select_idxs([0] * pad)
        extra.batch['sft_mask'] = torch.zeros_like(extra.batch['sft_mask'])
        selected = DataProto.concat([selected, extra])
    selected.meta_info.update(sft_global_tokens=total_tokens, sft_world_size=world_size, temperature=1.0)
    return selected
