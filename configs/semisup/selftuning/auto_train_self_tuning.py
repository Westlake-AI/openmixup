import json
import os

from datetime import datetime
from mmcv import Config
from numpy.core.fromnumeric import prod, var
from functools import reduce
from operator import getitem
from itertools import product


class ConfigGenerator:
    def __init__(self, base_path: str, num_device: int) -> None:
        self.base_path = base_path
        self.num_device = num_device

    def _path_parser(self, path: str) -> str:
        assert isinstance(path, str)
        base_dir = os.path.join(*self.base_path.split('/')[:-1])
        base_name = self.base_path.split('/')[-1] # file name
        base_prefix = base_name.split('.')[0] # prefix
        backbone = base_prefix.split('_')[0]

        return base_dir, backbone, base_prefix

    def _combinations(self, var_dict: dict) -> list:
        assert isinstance(var_dict, dict)
        ls = list(var_dict.values())
        cbs = [x for x in product(*ls)] # all combiantions

        return cbs

    def set_nested_item(self, dataDict: dict, mapList: list, val) -> dict:
        """Set item in nested dictionary"""
        reduce(getitem, mapList[:-1], dataDict)[mapList[-1]] = val

        return dataDict

    def generate(self, model_var: dict, gm_var: dict, abbs: dict) -> None:
        assert isinstance(model_var, dict)
        assert isinstance(gm_var, dict)
        cfg = dict(Config.fromfile(self.base_path))
        base_dir, backbone, base_prefix = self._path_parser(self.base_path) # analysis path
        model_cbs = self._combinations(model_var)
        gm_cbs = self._combinations(gm_var)

        # params for global .sh file
        port = 99999
        time = datetime.today().strftime('%Y%m%d_%H%M%S')
        with open('{}_{}.sh'.format(os.path.join(base_dir, base_prefix), time), 'a') as shfile:
            # model setting
            for c in model_cbs:
                cfg_n = cfg # reset
                config_dir = os.path.join(base_dir, backbone)
                for i, kv in enumerate(zip(list(model_var.keys()), c)):
                    k = kv[0].split('.')
                    v = kv[1]
                    cfg_n = self.set_nested_item(cfg_n, k, v) # assign value
                    config_dir += '/{}{}'.format(str(k[-1]), str(v))
                comment = ' '.join(config_dir.split('/')[-i-1:]) # e.g. alpha1.0 mask_layer 1
                shfile.write('# {}\n'.format(comment))

                # base setting
                for b in gm_cbs:
                    base_params = ''
                    for kv in zip(list(gm_var.keys()), b):
                        a = kv[0].split('.')
                        n = kv[1]
                        cfg_n = self.set_nested_item(cfg_n, a, n)
                        base_params += '_{}{}'.format(str(a[-1]), str(n))

                    # saving json config
                    config_dir = config_dir.replace('.', '_')
                    base_params = base_params.replace('.', '_')
                    for word, abb in abbs.items():
                        base_params = base_params.replace(word, abb)
                    if not os.path.exists(config_dir):
                        os.makedirs(config_dir)
                    file_name = os.path.join(config_dir, '{}{}.json'.format(base_prefix, base_params))
                    with open(file_name, 'w') as configfile:
                        json.dump(cfg, configfile, indent=4)

                    # write cmds for .sh
                    port += 1
                    cmd = 'CUDA_VISIBLE_DEVICES=0 PORT={} bash tools/dist_train.sh {} {} &\nsleep 0.1s \n'.format(
                        port, file_name, self.num_device)
                    shfile.write(cmd)
                shfile.write('\n')
    print('Generation completed. Please modify the bash file to run experiments!')


def main():
    # Please generate Self-Tuning configs by auto_train.py
    # For example: generate the optimal config for Aircrafts in
    #   |-- configs/semisup/selftuning/aircrafts/r18_per_15/proj_dim_1024/
    #   |   |--> r18_per_15_bs24_cos_no_randaug_weight_decay0_0001_nesterovTrue_ep120.py

    # Aircrafts
    base_path = "configs/semisup/selftuning/aircrafts/r18_per_15/r18_per_15_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/aircrafts/r18_per_30/r18_per_30_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/aircrafts/r18_per_50/r18_per_50_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/aircrafts/r50_per_15/r50_per_15_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/aircrafts/r50_per_30/r50_per_30_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/aircrafts/r50_per_50/r50_per_50_bs24_cos_no_randaug.py"
    # # Cars
    # base_path = "configs/semisup/selftuning/cars/r18_per_15/r18_per_15_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cars/r18_per_30/r18_per_30_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cars/r18_per_50/r18_per_50_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cars/r50_per_15/r50_per_15_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cars/r50_per_30/r50_per_30_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cars/r50_per_50/r50_per_50_bs24_cos_no_randaug.py"
    # # CUB
    # base_path = "configs/semisup/selftuning/cub200/r18_per_15/r18_per_15_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cub200/r18_per_30/r18_per_30_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cub200/r18_per_50/r18_per_50_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cub200/r50_per_15/r50_per_15_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cub200/r50_per_30/r50_per_30_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cub200/r50_per_50/r50_per_50_bs24_cos_no_randaug.py"

    # abbreviation of long attributes
    abbs = {
        'total_epochs': 'ep'
    }
    # create nested dirs
    model_var = {
        'model.proj_dim': [1024,],  # default: 1024
    }
    # adjust sub-attributes
    gm_var = {
        'optimizer.weight_decay': [5e-4, 1e-4],  # default: 1e-4
        'optimizer.nesterov': [True, False],  # default: True
        'total_epochs': [120,]  # 15%, Aircrafts
        # 'total_epochs': [100,]  # 15%, Cars
        # 'total_epochs': [130,]  # 15%, CUB
        # 'total_epochs': [150,]  # 30%, Aircrafts
        # 'total_epochs': [120,]  # 30%, Cars
        # 'total_epochs': [160,]  # 30%, CUB
        # 'total_epochs': [200,]  # 50%, Aircrafts
        # 'total_epochs': [160,]  # 50%, Cars
        # 'total_epochs': [210,]  # 50%, CUB
    }
    
    num_device = 1
    
    generator = ConfigGenerator(base_path=base_path, num_device=num_device)
    generator.generate(model_var, gm_var, abbs)


if __name__ == '__main__':
    main()