import numpy as np
import torch
import torch.nn as nn
import math
import random
import torch.nn.functional as F

@torch.no_grad()
def smmix(img,
          gt_label,
          attn=None,
          return_mask=False,
          dist_mode=False,
          lam=None,
          side=14,
          min_side_ratio=0.25, 
          max_side_ratio=0.75,
          **kwargs):
    r""" SMMix augmentation.

    "SMMix: SMMix: Self-Motivated Image Mixing for Vision Transformers"
    Basic Image Classification (https://arxiv.org/abs/2212.12977)". In ICCV, 2023.
        https://github.com/ChenMnZ/SMMix

    Args:
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        gt_label (Tensor): Ground-truth labels (one-hot).
        alpha (float): To sample Beta distribution.
        attn (Tensor): Attention Maps.
        lam (float): The given mixing ratio. If lam is None, sample a lam
            from Beta distribution.
        min_side_ratio (int): lower bound on uniform sampling
        max_side_ratio (int): upper bound on uniform sampling
        side: side length of attention map in image shape
        dist_mode (bool): Whether to do cross gpus index shuffling and
            return the mixup shuffle index, which support supervised
            and self-supervised methods.
        return_mask (bool): Whether to return the cutting-based mask of
            shape (N, 1, H, W). Defaults to False.
    """

    def batch_index_generate(x, idx):
        if len(x.size()) == 3:
            B, N, C = x.size()
            offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
            idx = idx + offset
            return idx.reshape(-1)
        elif len(x.size()) == 2:
            B, N = x.size()
            offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
            idx = idx + offset
            return idx
        else:
            raise NotImplementedError

    def index_translate(rectangle_index, rectangle_size=(3, 3), token_size=(7, 7)):
        total_index = torch.arange(token_size[0] * token_size[1]).reshape(1, 1, token_size[0], token_size[1]).cuda()
        total_index_list = torch.nn.functional.unfold(total_index.float(), rectangle_size, stride=1).transpose(1,2).long()
        sequence_index=total_index_list.index_select(dim=1, index=rectangle_index).squeeze(0)

        return sequence_index

    min_side = int(side * min_side_ratio)
    max_side = int(side * max_side_ratio)
    # Masks list
    rectangle_size_list = []
    for i in range(min_side, max_side + 1):
        rectangle_size_list.append((i,i))

    # normal mixup process
    if not dist_mode:
        b, _, h, w = img.size()
        y_a = gt_label
        y_b = gt_label.flip(0)

        rectangle_size = random.choice(rectangle_size_list)
        if lam is None:
            lam = (side**2 - rectangle_size[0] * rectangle_size[1]) / side**2

        patch_size = h // side
        # inputs = batch size, 196, hidden
        inputs = torch.nn.functional.unfold(img, patch_size, stride=patch_size).transpose(1, 2)
        if attn is not None: # batch size, 1, 14, 14
            attn = torch.mean(attn[:, :, 0, 1:], dim=1).reshape(-1, side, side).unsqueeze(1)
            # aggregating the image attention score of each candidate region
            rectangle_attn = torch.nn.functional.unfold(attn, rectangle_size, stride=1)
            # E.g. rectangele size = (9, 9) and total tokens = 196 (14x14)
            # Splites the 196 tokens to 6x6 patches by the size of 9x9
            # And sum the dimentaion 1 for ranking according to the attention score
            rectangle_attn = rectangle_attn.sum(dim=1)
            # print(rectangle_attn.shape) (100, 36)

        # generating path index of mixed regions, and find the max and mim region
        min_region_center_index = torch.argmin(rectangle_attn, dim=1)
        max_region_center_index = torch.argmax(rectangle_attn, dim=1)

        # Batch-level, length of the index = batch size.
        min_region_index = index_translate(min_region_center_index, rectangle_size, token_size=(side, side))
        max_region_index = index_translate(max_region_center_index, rectangle_size, token_size=(side, side))

        # Token-level, 
        min_region_index = batch_index_generate(inputs, min_region_index)
        max_region_index = batch_index_generate(inputs, max_region_index.flip(0))
        
        # image mixing 
        B, N, C = inputs.shape
        inputs_ = inputs.flip(0)
        inputs = inputs.reshape(B * N, C)
        inputs_ = inputs_.reshape(B * N, C)
        inputs[min_region_index] = inputs_[max_region_index]
        inputs = inputs.reshape(B, N, C)
        # Resizing the tokens to images --> (batch size, tokens, hidden) -> (batch size, 3, hight, width)
        inputs = torch.nn.functional.fold(inputs.transpose(1,2), h, patch_size, stride=patch_size)

        if return_mask:
            # source_mask: indicate the source region in mixed image
            # target_mask: indicate the target region in mixed image
            source_mask = torch.zeros_like(attn)
            source_mask = source_mask.reshape(-1)
            source_mask[min_region_index] = 1
            source_mask = source_mask.reshape(B, 1, side, side) # 100, 196
            source_mask = F.interpolate(source_mask, scale_factor=patch_size, mode='nearest')
            target_mask = 1 - source_mask

            mask = (target_mask, source_mask) # target mask -- img, source_mask -- img[rand_index]
            
            return inputs, (y_a, y_b, lam), mask
        
        return inputs, (y_a, y_b, lam)
        
        