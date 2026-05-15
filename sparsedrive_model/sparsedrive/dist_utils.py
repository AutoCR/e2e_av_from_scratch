import torch.distributed as dist


def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    t = tensor.clone()
    dist.all_reduce(t.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return t
