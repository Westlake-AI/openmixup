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
    
    cfg_list = os.listdir(dir_path)
    
    for cfg in cfg_list:
        
        epoch_num = os.listdir(os.path.join(dir_path, cfg))
        
        for ep_num in epoch_num:
            if not ep_num.endswith(".pth"):
                continue
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
            
            ck = ck['state_dict'] if ck.get('state_dict', None) is not None else ck
            ck = ck['model'] if ck.get('model', None) is not None else ck

            for key, value in ck.items():
                new_key = copy.copy(key)

                # replace timm to openmixup
                if new_key.find('patch_embed.proj.') != -1:
                    new_key = new_key.replace('patch_embed.proj.', 'patch_embed.projection.')
                if new_key.find('mlp.fc1.') != -1:
                    new_key = new_key.replace('mlp.fc1.', 'ffn.layers.0.0.')
                if new_key.find('mlp.fc2.') != -1:
                    new_key = new_key.replace('mlp.fc2.', 'ffn.layers.1.')
                
                if new_key.find('blocks') != -1:
                    new_key = new_key.replace('blocks', 'layers')
                if new_key.find('.norm') != -1:
                    new_key = new_key.replace('.norm', '.ln')
                if new_key == 'norm.weight':
                    new_key = 'ln1.weight'
                if new_key == 'norm.bias':
                    new_key = 'ln1.bias'
                
                output_dict['state_dict'][new_key] = value
                print("keep key {} -> {}".format(key, new_key))

            torch.save(output_dict, save_name)
            print("save ckpt:", ckpt_path)


if __name__ == '__main__':
    main()

# usage exam:
# python tools/timm_to_openmixup_dir.py [PATH of the dir to checkpoints]
