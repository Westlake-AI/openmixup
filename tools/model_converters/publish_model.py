"""
Extract parameters and publish the model.

Example command:
python tools/model_converters/publish_model.py [PATH/to/checkpoint] [PATH/to/output]
"""
import argparse
import subprocess

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    parser.add_argument('out_file', help='output checkpoint filename')
    parser.add_argument('--swap_ema', action='store_true', default=False,
                        help='whether to extract parameters of the EMA model')
    parser.add_argument('--decode', action='store_true', default=False,
                        help='whether to add sha256sum in the output name')
    args = parser.parse_args()
    return args


def process_checkpoint(in_file, out_file, swap_ema=False, decode=False):
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']

    # swap EMA to the model
    if swap_ema:
        new_ckpt = dict()

        for key, value in checkpoint['state_dict'].items():
            if key.startswith('ema'):
                continue
            # find the ema key
            ema_key = f"ema_{key.replace('.', '_')}"
            if checkpoint['state_dict'].get(ema_key, None) is not None:
                new_ckpt[key] = checkpoint['state_dict'][ema_key]
                print("replace key {} -> {}".format(key, ema_key))
            else:
                new_ckpt[key] = value
            # new_ckpt[key] = value
        checkpoint['state_dict'] = new_ckpt

    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    if torch.__version__ >= '1.6':
        torch.save(checkpoint, out_file, _use_new_zipfile_serialization=False)
    else:
        torch.save(checkpoint, out_file)

    if decode:
        sha = subprocess.check_output(['sha256sum', out_file]).decode()
        if out_file.endswith('.pth'):
            out_file_name = out_file[:-4]
        else:
            out_file_name = out_file
        final_file = out_file_name + f'-{sha[:8]}.pth'
        subprocess.Popen(['mv', out_file, final_file])


def main():
    args = parse_args()
    process_checkpoint(args.in_file, args.out_file, args.swap_ema, args.decode)


if __name__ == '__main__':
    main()
