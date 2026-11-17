"""Run upstream Jevlike training with independent best-epoch snapshots.

Uses the same arguments as `python -m jevlike.train`. The upstream source is
unchanged; only this process's snapshot function is replaced. In particular,
this does not add checkpoint resume, encoder fine-tuning, or new model layers.
"""

from jevlike import train


def cloned_trainable_state(model):
    # .cpu() alone aliases live parameters when they are already on the CPU.
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


if __name__ == "__main__":
    train.trainable_state = cloned_trainable_state
    train.main()
