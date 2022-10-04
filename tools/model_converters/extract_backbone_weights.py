import copy
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'output', type=str, help='destination file name')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.output.endswith(".pth")
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    output_dict = dict(state_dict=dict(), author="openmixup")
    has_backbone = False
    if ck.get('state_dict', None) is not None:
        ck = ck['state_dict']
    
    for key, value in ck.items():
        if key.find('momentum') != -1:
            continue
        # remove 'module'
        if key.startswith('module'):
            key = key[7:]
        new_key = copy.copy(key)
        # remove backbone keys
        for prefix_k in ['backbone', 'encoder', 'encoder_q', 'base_encoder', 'timm_model',]:
            if new_key.startswith(prefix_k):
                has_backbone = True
                new_key = new_key[len(prefix_k) + 1: ]
        if new_key == key:
            print("remove key:", key)
            continue
        
        output_dict['state_dict'][new_key] = value
        print("keep key {} -> {}".format(key, new_key))
    if not has_backbone:
        raise Exception("Cannot find a backbone module in the checkpoint.")
    torch.save(output_dict, args.output)


if __name__ == '__main__':
    main()

# usage exam:
# python tools/extract_backbone_weights.py [PATH of the checkpoints]
