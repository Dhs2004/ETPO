"""FIFO of complete successful tasks, compact CPU tokens, checkpointable."""
import torch
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


class SuccessBuffer:
    def __init__(self):
        self.tasks = []
        self.raw_weights = []
        self.audit_metadata = []
        self.teacher_updates = 0
        self.consumed_tasks = 0

    def __len__(self):
        return len(self.tasks)

    def add(self, teacher, source, indices, raw_weights=None, collected_step=0):
        groups = {}
        for i in indices:
            key = (str(source.non_tensor_batch['traj_uid'][i]), int(source.non_tensor_batch['traj_idx'][i]))
            groups.setdefault(key, []).append(i)
        for key, rows in groups.items():
            task = []
            for i in rows:
                b = teacher.batch
                width = b['prompts'].shape[1]
                prompt = b['prompts'][i][b['attention_mask'][i, :width].bool()].detach().cpu().clone()
                response = b['responses'][i][b['response_mask'][i].bool()].detach().cpu().clone()
                task.append({'prompt': prompt, 'response': response})
            self.tasks.append(task)
            self.audit_metadata.append({'trajectory':key[0], 'task_position':key[1], 'group':str(source.non_tensor_batch['uid'][rows[0]]), 'collected_step':collected_step})
            self.raw_weights.append(None if raw_weights is None else float(raw_weights[key]))

    def batch(self, count, pad_token_id, success_rate_weighting=False):
        if not 0 < count <= len(self):
            raise ValueError('Insufficient complete successes for teacher update')
        rows = [row for task in self.tasks[:count] for row in task]
        pw = max(len(row['prompt']) for row in rows)
        rw = max(len(row['response']) for row in rows)
        prompts = torch.full((len(rows), pw), pad_token_id, dtype=torch.long)
        responses = torch.full((len(rows), rw), pad_token_id, dtype=torch.long)
        pm, rm = torch.zeros_like(prompts), torch.zeros_like(responses)
        for i, row in enumerate(rows):
            p, y = row['prompt'], row['response']
            prompts[i, -len(p):], pm[i, -len(p):] = p, 1
            responses[i, :len(y)], rm[i, :len(y)] = y, 1
        mask = torch.cat([pm, rm], 1)
        result = DataProto.from_dict(tensors={'prompts': prompts, 'responses': responses,
            'input_ids': torch.cat([prompts, responses], 1), 'attention_mask': mask,
            'position_ids': compute_position_id_with_mask(mask), 'response_mask': rm},
            meta_info={'temperature': 1.0})
        if success_rate_weighting:
            raw = self.raw_weights[:count]
            if any(w is None for w in raw):
                raise ValueError('Buffered trajectories lack success-rate weights; cannot enable weighting on a legacy buffer')
            weights = torch.tensor(raw, dtype=torch.float64)
            if not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError('Trajectory weights must be finite and positive')
            weights = weights / weights.mean()  # Normalize globally before any rank splitting.
            coefficients = torch.zeros_like(rm, dtype=torch.float32)
            offset = 0
            for task, weight in zip(self.tasks[:count], weights):
                tokens = sum(row['response'].numel() for row in task)
                if tokens == 0:
                    raise ValueError('Successful trajectory has no response tokens')
                coefficients[offset:offset+len(task)] = rm[offset:offset+len(task)] * float(weight / tokens)
                offset += len(task)
            result.batch['sft_token_weights'] = coefficients
            result.meta_info.update(sft_global_trajectories=count,
                                    sft_trajectory_weight_sum=float(weights.sum()),
                                    sft_trajectory_weight_min=float(weights.min()),
                                    sft_trajectory_weight_max=float(weights.max()))
        return result

    def consume(self, count):
        del self.tasks[:count]
        del self.raw_weights[:count]
        del self.audit_metadata[:count]
        self.teacher_updates += 1
        self.consumed_tasks += count

    def save(self, path):
        torch.save({'version': 3, 'audit_metadata':self.audit_metadata, 'tasks': self.tasks, 'raw_weights': self.raw_weights, 'teacher_updates': self.teacher_updates, 'consumed_tasks': self.consumed_tasks}, path)

    def load(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['version'] not in (1, 2, 3):
            raise ValueError('Unsupported ETPO success buffer version')
        self.tasks = state['tasks']
        self.audit_metadata = state.get('audit_metadata', [{'collected_step':0} for _ in self.tasks])
        self.raw_weights = state.get('raw_weights', [None] * len(self.tasks))
        if len(self.raw_weights) != len(self.tasks):
            raise ValueError('Success buffer weight count mismatch')
        self.teacher_updates = state.get('teacher_updates', 0)
        self.consumed_tasks = state.get('consumed_tasks', 0)
