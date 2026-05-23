import functools
import torch


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    if weight is not None:
        loss = loss * weight
    if avg_factor is not None:
        return loss.sum() / avg_factor
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def weighted_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction="mean", avg_factor=None, **kwargs):
        loss = loss_func(pred, target, **kwargs)
        return weight_reduce_loss(loss, weight, reduction, avg_factor)

    return wrapper


__all__ = ["weight_reduce_loss", "weighted_loss"]
