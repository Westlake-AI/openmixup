import cv2
import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


class PlotTensor:
    """Plot torch tensor as matplotlib figure.

    Args:
        apply_inv (bool): Whether to apply inverse normalization.
    """

    def __init__(self, apply_inv=True) -> None:
        trans_cifar = [
            torchvision.transforms.Normalize(
                mean=[ 0., 0., 0. ], std=[1 / 0.2023, 1 / 0.1994, 1 / 0.201]),
            torchvision.transforms.Normalize(
                mean=[-0.4914, -0.4822, -0.4465], std=[ 1., 1., 1. ])]
        trans_in = [
            torchvision.transforms.Normalize(
                mean=[ 0., 0., 0. ], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            torchvision.transforms.Normalize(
                mean=[-0.485, -0.456, -0.406], std=[ 1., 1., 1. ])]
        if apply_inv:
            self.invTrans_cifar = torchvision.transforms.Compose(trans_cifar)
            self.invTrans_in = torchvision.transforms.Compose(trans_in)

    def plot(self,
             img, nrow=4, title_name=None, save_name=None,
             make_grid=True, dpi=None, cmap='gray', apply_inv=True, overwrite=False):
        assert save_name is not None

        if make_grid:
            # make grid and save as plt images
            assert img.size(0) % nrow == 0
            ncol = img.size(0) // nrow
            if ncol > nrow:
                ncol = nrow
                nrow = img.size(0) // ncol
            img_grid = torchvision.utils.make_grid(img, nrow=nrow, pad_value=0)

            if img.size(1) == 1:
                cmap = getattr(plt.cm, cmap, plt.cm.jet)
            else:
                cmap = None
            if apply_inv:
                if img.size(2) <= 64:
                    img_grid = self.invTrans_cifar(img_grid)
                else:
                    img_grid = self.invTrans_in(img_grid)
            img_grid = torch.clip(img_grid * 255, 0, 255).int()
            img_grid = np.transpose(img_grid.detach().cpu().numpy(), (1, 2, 0))
            
            fig = plt.figure(figsize=(nrow * 2, ncol * 2))
            plt.imshow(img_grid, cmap=cmap)
            if title_name is not None:
                plt.title(title_name)
            if not os.path.exists(save_name) or overwrite:
                plt.savefig(save_name, dpi=dpi)
            plt.close()
        else:
            # save each image sperately
            save_name = save_name.split('.')  # split into a name and a suffix
            if apply_inv:
                if img.size(2) <= 64:
                    img = self.invTrans_cifar(img)
                else:
                    img = self.invTrans_in(img)
            img = torch.clip(img * 255, 0, 255).detach().cpu().numpy()
            img = np.transpose(img, (0, 2, 3, 1))
            title_name = '' if title_name is None else title_name.replace(' ', '')

            for i in range(img.shape[0]):
                img_ = img[i, ...]
                # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
                cv2.imwrite("{}_{}.{}".format(save_name[0], title_name+str(i), save_name[1]), img_)
