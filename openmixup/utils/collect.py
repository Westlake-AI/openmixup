import numpy as np
import os.path as osp
import pickle
import shutil
import tempfile
from typing import Optional
from torch.autograd import Variable
from einops import rearrange

import torch
import torch.distributed as dist
import torchvision.transforms
from torchvision.utils import save_image

import mmcv
from mmcv.runner import get_dist_info
from .gather import gather_tensors_batch


def nondist_forward_collect(func, data_loader, length):
    """Forward and collect network outputs.

    This function performs forward propagation and collects outputs.
    It can be used to collect results, features, losses, etc.

    Args:
        func (function): The function to process data. The output must be
            a dictionary of CPU tensors.
        length (int): Expected length of output arrays.

    Returns:
        results_all (dict(np.ndarray)): The concatenated outputs.
    """
    results = []
    prog_bar = mmcv.ProgressBar(len(data_loader))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = func(**data)
        results.append(result)
        prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        results_all[k] = np.concatenate(
            [batch[k].numpy() for batch in results], axis=0)
        assert results_all[k].shape[0] == length
    return results_all


def dist_forward_collect(func, data_loader, rank, length, ret_rank=-1):
    """Forward and collect network outputs in a distributed manner.

    This function performs forward propagation and collects outputs.
    It can be used to collect results, features, losses, etc.

    Args:
        func (function): The function to process data. The output must be
            a dictionary of CPU tensors.
        rank (int): This process id.
        length (int): Expected length of output arrays.
        ret_rank (int): The process that returns.
            Other processes will return None.

    Returns:
        results_all (dict(np.ndarray)): The concatenated outputs.
    """
    results = []
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(data_loader))
    for idx, data in enumerate(data_loader):
        with torch.no_grad():
            result = func(**data)  # dict{key: tensor}
        results.append(result)

        if rank == 0:
            prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        results_cat = np.concatenate([batch[k].numpy() for batch in results],
                                     axis=0)
        if ret_rank == -1:
            results_gathered = gather_tensors_batch(results_cat, part_size=20)
            results_strip = np.concatenate(results_gathered, axis=0)[:length]
        else:
            results_gathered = gather_tensors_batch(
                results_cat, part_size=20, ret_rank=ret_rank)
            if rank == ret_rank:
                results_strip = np.concatenate(
                    results_gathered, axis=0)[:length]
            else:
                results_strip = None
        results_all[k] = results_strip
    return results_all


def collect_results_cpu(result_part: list,
                        size: int,
                        tmpdir: Optional[str] = None) -> Optional[list]:
    """Collect results under cpu mode.

    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it.

    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    part_file = osp.join(tmpdir, f'part_{rank}.pkl')  # type: ignore
    mmcv.dump(result_part, part_file)
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')  # type: ignore
            part_result = mmcv.load(part_file)
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)  # type: ignore
        return ordered_results


def collect_results_gpu(result_part: list, size: int) -> Optional[list]:
    """Collect results under gpu mode.

    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.

    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
    else:
        return None


def occlusion_forward_collect(func, data_loader, length, random_drop, drop_size):

    # create a mask for occlusion test
    patch = 224 // drop_size
    patch_num = patch * patch
    mask_num = round(patch_num * random_drop) # need mask number

    print(f"patch size is {patch} with the total tokens {patch_num}")
    print(f"occlusion ratio is {random_drop * 100}% and masked tokens are {mask_num}")

    results = []
    prog_bar = mmcv.ProgressBar(len(data_loader))
    for i, data in enumerate(data_loader):
        img = rearrange(data['img'], 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=drop_size, p2=drop_size)
        row = np.random.choice(range(patch_num), size=mask_num, replace=False)
        img[:, row, :] = 0.0
        img = rearrange(img, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=patch, w=patch, p1=drop_size, p2=drop_size)

        data['img'] = img
        with torch.no_grad():
            result = func(**data)
        results.append(result)
        prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        results_all[k] = np.concatenate(
            [batch[k].numpy() for batch in results], axis=0)
        assert results_all[k].shape[0] == length
    return results_all


def fgsm_nondist_forward_collect(func, data_loader, length, head, dataset='cifar'):

    eps = 8
    if dataset == 'cifar':
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.201]
    else:
        mean, std =[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    criterion = torch.nn.CrossEntropyLoss()
    results = []
    prog_bar = mmcv.ProgressBar(len(data_loader))
    for i, data in enumerate(data_loader):
        img = data['img']
        input = Variable(img, requires_grad=True)
        optimizer_input = torch.optim.SGD([input], lr=0.1)
        data['img'] = input
        output = func(**data)
        loss = criterion(output[head], data['gt_label'])
        optimizer_input.zero_grad()
        loss.backward()

        sign_data_grad = input.grad.sign()

        input = input + eps / 255. * sign_data_grad
        eta = torch.clamp(input - img, min=-eps / 255., max=eps / 255.)
        input = torch.clamp(img + eta, min=0, max=1).detach()
        data['img'] = input

        with torch.no_grad():
            result = func(**data)
        results.append(result)
        prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        results_all[k] = np.concatenate(
            [batch[k].numpy() for batch in results], axis=0)
        assert results_all[k].shape[0] == length
    return results_all
