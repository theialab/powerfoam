import numpy as np


def get_cosine_scheduler(
    lr_init,
    lr_final,
    max_steps,
    warmup_steps=0,
):

    def helper(step):
        if warmup_steps and step < warmup_steps:
            return lr_init * step / warmup_steps
        elif step > max_steps:
            return lr_final
        lr_cos = lr_final + 0.5 * (lr_init - lr_final) * (
            1 + np.cos(np.pi * (step - warmup_steps) / (max_steps - warmup_steps))
        )
        return lr_cos

    return helper


def get_exp_scheduler(
    lr_init,
    lr_final,
    max_steps,
    warmup_steps=0,
):

    def helper(step):
        if warmup_steps and step < warmup_steps:
            return lr_init * step / warmup_steps
        elif step > max_steps:
            return lr_final
        t = np.clip((step - warmup_steps) / (max_steps - warmup_steps), 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return log_lerp

    return helper
