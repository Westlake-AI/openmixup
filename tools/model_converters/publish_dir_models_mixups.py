import copy
import os
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('dir_path', help='checkpoint file')
    args = parser.parse_args()
    return args


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    args = parse_args()
    dir_path = args.dir_path
    assert os.path.exists(dir_path) and dir_path.find("work_dirs") != -1
    save_path = os.path.join("work_dirs/my_pretrains", dir_path.split("work_dirs/")[1])
    mkdir(save_path)

    ckpt_list = os.listdir(dir_path)

    for ckpt in ckpt_list:
        if ckpt.find(".pth") == -1:
            print("bad ckeckpoint name:", ckpt)
            continue

        ckpt_path = os.path.join(dir_path, ckpt)

        try:
            ck = torch.load(ckpt_path, map_location=torch.device('cpu'))
        except:
            print("ERROR in loading:", cfg)
            continue

        output_dict = dict(state_dict=dict(), author="openmixup")

        for key, value in ck['state_dict'].items():
            # remove 'EMA'
            if key.startswith('ema_'):
                continue
            # remove 'module'
            if key.startswith('module'):
                key = key[7:]
            new_key = copy.copy(key)

            # replace head keys
            if new_key.find('mix_block') != -1:
                new_key = new_key.replace('query.weight', 'query.conv.weight')
                new_key = new_key.replace('query.bias', 'query.conv.bias')
                new_key = new_key.replace('key.weight', 'key.conv.weight')
                new_key = new_key.replace('key.bias', 'key.conv.bias')

            # replace mixblock keys
            if new_key.find('fc_cls') != -1:
                new_key = new_key.replace('fc_cls', 'fc')

            output_dict['state_dict'][new_key] = value
            print("keep key {} -> {}".format(key, new_key))

        torch.save(output_dict, ckpt_path)
        print("save ckpt:", ckpt_path)


if __name__ == '__main__':
    main()
