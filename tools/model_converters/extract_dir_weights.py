"""
Extract only the backbone parameters under a folder.

It requires the folder built as follows:
└── [PATH/to/checkpoint/folder]
    └── xxx_ep100
        ├── epoch_100.pth
        ├── ...
    └── xxx_ep300
        ├── epoch_300.pth
        ├── ...

Example command:
python tools/model_converters/extract_dir_weights.py [PATH/to/checkpoint/folder]
"""
import os
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('dir_path', help='checkpoint file')
    parser.add_argument('--backbone_only', action='store_true', help='only keep keys started by `backbone`')
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
    
    cfg_list = os.listdir(dir_path)
    
    for cfg in cfg_list:
        if cfg.find("_ep") == -1:
            print("bad config name or dir:", cfg)
            continue
        
        epoch_num = cfg.split("_ep")[1]
        ckpt_path = os.path.join(dir_path, cfg, "epoch_"+epoch_num+".pth")
        save_name = os.path.join(save_path, cfg+".pth")

        try:
            ck = torch.load(ckpt_path, map_location=torch.device('cpu'))
        except:
            print("unfinished task:", cfg)
            continue
        
        output_dict = dict(state_dict=dict(), author="openmixup")
        has_backbone = False
        for key, value in ck['state_dict'].items():
            if key.startswith('encoder_q'):
                if args.backbone_only:
                    continue
                output_dict['state_dict'][key[10:]] = value
                has_backbone = True
                print("keep key {} -> {}".format(key, key[10:]))
            elif key.startswith('backbone'):
                output_dict['state_dict'][key[9:]] = value
                has_backbone = True
                print("keep key {} -> {}".format(key, key[9:]))
        if not has_backbone:
            raise Exception("Cannot find a backbone module in the checkpoint.")
        torch.save(output_dict, save_name)
        print("save ckpt:", ckpt_path)


if __name__ == '__main__':
    main()
