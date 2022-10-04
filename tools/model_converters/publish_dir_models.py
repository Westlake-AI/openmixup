import copy
import os
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('dir_path', help='checkpoint file')
    parser.add_argument('--convert_all', default=False, type=bool, help='whether to convert all checkpoints')
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
        
        if args.convert_all:
            epoch_num = os.listdir(os.path.join(dir_path, cfg))
        else:
            epoch_num = ["epoch_" + cfg.split("_ep")[1] + ".pth"]
        
        for ep_num in epoch_num:
            ckpt_path = os.path.join(dir_path, cfg, ep_num)
            if len(epoch_num) == 1:
                save_name = os.path.join(save_path, cfg+".pth")
            else:
                mkdir(os.path.join(save_path, cfg))
                save_name = os.path.join(save_path, cfg, ep_num.replace("epoch", "checkpoint"))

            try:
                ck = torch.load(ckpt_path, map_location=torch.device('cpu'))
            except:
                print("unfinished task:", cfg)
                continue
            
            output_dict = dict(state_dict=dict(), author="openmixup")
            
            for key, value in ck['state_dict'].items():
                # remove 'module'
                if key.startswith('module'):
                    key = key[7:]
                new_key = copy.copy(key)
                
                # remove backbone keys
                for prefix_k in ['encoder_q', 'backbone', 'timm_model',]:
                    if new_key.startswith(prefix_k):
                        new_key = new_key[len(prefix_k) + 1: ]
                # remove head keys
                for head_k in ['fc', 'fc_cls',]:
                    start_idx = key.find(head_k)
                    if start_idx != -1:
                        new_key = new_key[:start_idx] + new_key[start_idx + len(head_k)+1: ]
                
                output_dict['state_dict'][new_key] = value
                print("keep key {} -> {}".format(key, new_key))

            torch.save(output_dict, save_name)
            print("save ckpt:", ckpt_path)


if __name__ == '__main__':
    main()

# usage exam:
# python tools/publish_dir_models.py [PATH of the dir to checkpoints]
