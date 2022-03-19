from openmixup.utils import ConfigGenerator


def main():
    # Please generate Self-Tuning configs by auto_train.py
    # For example: generate the optimal config for Aircrafts in
    #   |-- configs/semisup/selftuning/aircrafts/r18/proj_dim_1024/
    #   |   |--> r18_per_15_bs24_cos_no_randaug_weight_decay0_0001_nesterovTrue_ep120.py

    # Aircrafts
    base_path = "configs/semisup/selftuning/aircrafts/r18/r18_per_15_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/aircrafts/r18/r18_per_30_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/aircrafts/r18/r18_per_50_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/aircrafts/r50/r50_per_15_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/aircrafts/r50/r50_per_30_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/aircrafts/r50/r50_per_50_bs24_cos_no_randaug.py"
    # # Cars
    # base_path = "configs/semisup/selftuning/cars/r18/r18_per_15_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cars/r18/r18_per_30_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cars/r18/r18_per_50_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cars/r50/r50_per_15_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cars/r50/r50_per_30_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cars/r50/r50_per_50_bs24_cos_no_randaug.py"
    # # CUB
    # base_path = "configs/semisup/selftuning/cub200/r18/r18_per_15_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cub200/r18/r18_per_30_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cub200/r18/r18_per_50_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cub200/r50/r50_per_15_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cub200/r50/r50_per_30_bs24_cos_no_randaug.py"
    # base_path = "configs/semisup/selftuning/cub200/r50/r50_per_50_bs24_cos_no_randaug.py"

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