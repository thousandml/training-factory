"""Numerical safeguards for frozen-CQ LoRA fine-tuning."""
import torch


def prepare_trainable_lora(model):
    """Keep adapter weights and their optimizer state in FP32."""
    count = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if not parameter.is_floating_point():
                raise TypeError(f'Trainable parameter {name} is not floating point')
            parameter.data = parameter.data.float()
            count += 1
    if not count:
        raise ValueError('No trainable adapter parameters')
    return count


def install_nonfinite_gradient_guards(model):
    """Raise before any optimizer step that would apply a nonfinite adapter gradient.

    The previous per-parameter backward hook called ``.item()`` for every LoRA
    tensor on every micro-batch (~14 per decoder layer), forcing hundreds of
    CPU/GPU synchronizations per backward and stalling kernel queueing. A
    nonfinite value survives gradient accumulation and clipping, so checking
    the accumulated grads once, right before ``optimizer.step()``, gives the
    same guarantee (nothing nonfinite is ever applied) with one sync per step.
    """
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    owned = {id(p) for _, p in params}

    def check(optimizer, args, kwargs):
        grads = [(name, p.grad) for name, p in params if p.grad is not None]
        if not grads or not any(id(p) in owned for group in optimizer.param_groups for p in group['params']):
            return
        bad = torch.stack([~torch.isfinite(g).all() for _, g in grads])
        if bad.any().item():
            names = [name for (name, _), b in zip(grads, bad.tolist()) if b]
            raise FloatingPointError(
                f'Nonfinite LoRA gradient in {names[:5]}{" ..." if len(names) > 5 else ""} '
                f'({len(names)} tensors); aborting before optimizer step. '
                'Inspect CQ backward and activation magnitudes; do not disable this check.')

    from torch.optim.optimizer import register_optimizer_step_pre_hook
    return [register_optimizer_step_pre_hook(check)]
