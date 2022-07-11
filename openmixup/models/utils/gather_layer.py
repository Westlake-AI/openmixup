import numpy as np
import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [
            torch.zeros_like(input) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

        *** Warning: torch.distributed.all_gather has no gradient. ***
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def batch_shuffle_ddp(x, idx_shuffle=None, no_repeat=False):
    """Batch shuffle (no grad), for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        return: x, idx_shuffle, idx_unshuffle.
        *** no repeat (09.23 update) ***
    
    Args:
        idx_shuffle: Given shuffle index if not None.
        no_repeat: The idx_shuffle does not have any repeat index as
            the original indice [i for i in range(N)]. It's used in
            mixup methods (self-supervisedion).
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    if idx_shuffle is None:
        # generate shuffle idx
        idx_shuffle = torch.randperm(batch_size_all).cuda()
        # each idx should not be the same as the original
        if bool(no_repeat) == True:
            idx_original = torch.tensor([i for i in range(batch_size_all)]).cuda()
            idx_repeat = False
            for i in range(20):  # try 20 times
                if (idx_original == idx_shuffle).any() == True:  # find repeat
                    idx_repeat = True
                    idx_shuffle = torch.randperm(batch_size_all).cuda()
                else:
                    idx_repeat = False
                    break
            # repeat hit: prob < 1.8e-4
            if idx_repeat == True:
                fail_to_shuffle = True
                idx_shuffle = idx_original.clone()
                for i in range(3):
                    # way 1: repeat prob < 1.5e-5
                    rand_ = torch.randperm(batch_size_all).cuda()
                    idx_parition = rand_ > torch.median(rand_)
                    idx_part_0 = idx_original[idx_parition == True]
                    idx_part_1 = idx_original[idx_parition != True]
                    if idx_part_0.shape[0] == idx_part_1.shape[0]:
                        idx_shuffle[idx_parition == True] = idx_part_1
                        idx_shuffle[idx_parition != True] = idx_part_0
                        if (idx_original == idx_shuffle).any() != True:  # no repeat
                            fail_to_shuffle = False
                            break
                # fail prob -> 0
                if fail_to_shuffle == True:
                    # way 2: repeat prob = 0, but too simple!
                    idx_shift = np.random.randint(1, batch_size_all-1)
                    idx_shuffle = torch.tensor(  # shift the original idx
                        [(i+idx_shift) % batch_size_all for i in range(batch_size_all)]).cuda()
    else:
        assert idx_shuffle.size(0) == batch_size_all, \
            "idx_shuffle={}, batchsize={}".format(idx_shuffle.size(0), batch_size_all)

    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this], idx_shuffle, idx_unshuffle


@torch.no_grad()
def batch_unshuffle_ddp(x, idx_unshuffle):
    """Undo batch shuffle (no grad).

        *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this]


def grad_batch_shuffle_ddp(x, idx_shuffle=None, no_repeat=False):
    """Batch shuffle (with grad). (SimCLR GatherLayer version)
        *** Only support DistributedDataParallel (DDP) model. ***
        return: x, idx_shuffle, idx_unshuffle.
        *** no repeat (09.23 update) ***
    
    Args:
        idx_shuffle: Given shuffle index if not None.
        no_repeat: The idx_shuffle does not have any repeat index as
            the original indice [i for i in range(N)]. It's used in
            mixup methods (self-supervisedion).
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = torch.cat(GatherLayer.apply(x), dim=0)  # with grad
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    if idx_shuffle is None:
        # generate shuffle idx
        idx_shuffle = torch.randperm(batch_size_all).cuda()
        # each idx should not be the same as the original
        if bool(no_repeat) == True:
            idx_original = torch.tensor([i for i in range(batch_size_all)]).cuda()
            idx_repeat = False
            for i in range(20):  # try 20 times
                if (idx_original == idx_shuffle).any() == True:  # find repeat
                    idx_repeat = True
                    idx_shuffle = torch.randperm(batch_size_all).cuda()
                else:
                    idx_repeat = False
                    break
            # repeat hit: prob < 1.8e-4
            if idx_repeat == True:
                fail_to_shuffle = True
                idx_shuffle = idx_original.clone()
                for i in range(3):
                    # way 1: repeat prob < 1.5e-5
                    rand_ = torch.randperm(batch_size_all).cuda()
                    idx_parition = rand_ > torch.median(rand_)
                    idx_part_0 = idx_original[idx_parition == True]
                    idx_part_1 = idx_original[idx_parition != True]
                    if idx_part_0.shape[0] == idx_part_1.shape[0]:
                        idx_shuffle[idx_parition == True] = idx_part_1
                        idx_shuffle[idx_parition != True] = idx_part_0
                        if (idx_original == idx_shuffle).any() != True:  # no repeat
                            fail_to_shuffle = False
                            break
                # fail prob -> 0
                if fail_to_shuffle == True:
                    # way 2: repeat prob = 0, but too simple!
                    idx_shift = np.random.randint(1, batch_size_all-1)
                    idx_shuffle = torch.tensor(  # shift the original idx
                        [(i+idx_shift) % batch_size_all for i in range(batch_size_all)]).cuda()
    else:
        assert idx_shuffle.size(0) == batch_size_all, \
            "idx_shuffle={}, batchsize={}".format(idx_shuffle.size(0), batch_size_all)

    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
    return x_gather[idx_this], idx_shuffle, idx_unshuffle


def grad_batch_unshuffle_ddp(x, idx_unshuffle):
    """Undo batch shuffle. (SimCLR GatherLayer version)

        *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = torch.cat(GatherLayer.apply(x), dim=0)  # with grad
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
    return x_gather[idx_this]
