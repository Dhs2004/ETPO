"""One-time initialization of an independent teacher (including FSDP1 shards)."""
import torch


def initialize_teacher_from_student(student, teacher):
    if student is teacher:
        raise ValueError('Teacher must be a distinct model')
    with torch.no_grad():
        for source_items, target_items in ((student.named_parameters(), teacher.named_parameters()),
                                           (student.named_buffers(), teacher.named_buffers())):
            source, target = dict(source_items), dict(target_items)
            if source.keys() != target.keys():
                raise ValueError('Teacher/student structures differ')
            for name, value in source.items():
                destination = target[name]
                if value is destination or (value.numel() and value.device == destination.device and value.data_ptr() == destination.data_ptr()):
                    raise ValueError('Teacher and student must not share tensor storage')
                if value.shape != destination.shape:
                    raise ValueError(f'Teacher/student shard shapes differ: {name}')
                destination.copy_(value)
