# Method

For a sampled response token, let `g = log p_teacher(token | history, skill) - log p_old_student(token | history)`. ETPO replaces the original GRPO advantage `A` with

`A_adjusted = A * (0.5 + sigmoid(beta * sign(A) * g))`.

This preserves the sign of the advantage. The teacher scores and scaling factors are detached from autograd. The student minimizes the usual clipped policy-gradient loss using adjusted token advantages. Approximate PPO KL may be logged as a diagnostic; it is not an added teacher-to-student KL objective.

The teacher is independently optimized on accepted successful trajectories. A task's within-rollout success rate determines its inverse-success weight. Weights are normalized to mean one within the teacher update batch. Each trajectory's response-token negative log likelihood is normalized by its response length, weighted, and averaged over trajectories. Only full teacher update batches are consumed; the remainder stays in the checkpointed buffer.

The unweighted implementation instead averages response-token losses. Disabling weighting therefore also changes trajectory-length normalization; it is not a uniform-trajectory-weight control.

Per training iteration: collect student rollouts; compute rewards, GRPO advantages and old-policy log probabilities; accumulate successful trajectories and perform available teacher updates; score skill-augmented response contexts; scale advantages; apply the student policy update. Teacher updates do not backpropagate through the student.

Implementation entry points:

- `etpo/core.py`: credit scaling, successful trajectory selection and SFT reduction.
- `etpo/buffer.py`: successful trajectory accumulation and batch normalization.
- `etpo/data.py`: token-aligned teacher prompts and supervised batches.
- `etpo/training.py`: update ordering.
- `verl/workers/actor/dp_actor.py`: separate policy and teacher optimization.
- `verl/workers/fsdp_workers.py`: independent teacher model and checkpoints.

Teacher SFT loss measures fit to accepted trajectories; it does not by itself establish training stability or teacher quality. No performance claims are included in this release.
