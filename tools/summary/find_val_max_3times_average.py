"""
Summarize the maximum results (3 times) of the key from the folder

It requires the folder built as follows:
└── [PATH/to/exp_dir]
    └── xxx_ep100
        ├── xxx_ep100_1_yyy.log.json
        ├── xxx_ep100_2_yyy.log.json
        ├── xxx_ep100_3_yyy.log.json
        ├── ...

Example command:
python tools/summary/find_val_max_3times_average.py [PATH/to/exp_dir] [metric_name]
"""

import argparse
import numpy as np
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='Read a classification json file to report val results.')
    parser.add_argument('path', type=str, help='input json filename.')
    parser.add_argument('key', type=str, help='head keyword in the json files.')
    args = parser.parse_args()
    return args.__dict__


def read_json_max(path, print_all=True, keyword=None, **kwargs):
    record_str = list()
    record_acc = dict()
    if keyword is None:
        keyword = ['head0']
    elif isinstance(keyword, str):
        keyword = [keyword]
    for k in keyword:
        record_acc[k] = list()
    assert path.find("json") != -1
    # read each line
    with open(path, "r") as f:
        for line in f.readlines():
            line = json.loads(line)
            if line.get("mode", None) == "val":
                res = f"{line['epoch']}e, "
                for k in keyword:
                    try:
                        res += "{}: {:.2f}, ".format(k, line[k])
                        record_acc[k].append(line[k])
                    except:
                        pass
                record_str.append(res)
    # output records
    print_str = "Max -- "
<<<<<<< HEAD
    if print_all:
        max_num = min(len(record_str), 5)
        for l in record_str[-max_num:]:
=======
    for l in record_str:
        if print_all:
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
            print(l)
    for k in keyword:
        record_acc[k] = np.array(record_acc[k])
        record_acc[k] = \
            (np.max(record_acc[k]) + np.percentile(record_acc[k], 99)) / 2
        print_str += "{}: {:.2f},".format(k, record_acc[k])
    if print_all:
        print(print_str)
    return record_acc


if __name__ == '__main__':
    """ find the median of val results in latest N epochs """
    args = parse_args()
    print(args)

<<<<<<< HEAD
    keyword = args.get("key", ["head0"])
    if isinstance(keyword, str):
        keyword = keyword.split("-")

    # read record of a dir
    if args["path"].find(".json") == -1:
=======
    # read record of a dir
    if args["path"].find(".json") == -1:
        keyword = args.get("key", ["head0"])
        if isinstance(keyword, str):
            keyword = keyword.split("-")
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
        assert os.path.exists(args["path"])
        cfg_list = os.listdir(args["path"])
        cfg_list.sort()

        for cfg in cfg_list:
            cfg_args = args.copy()
            cfg_args["keyword"] = keyword
            cfg_path = os.path.join(args["path"], cfg)
<<<<<<< HEAD
            if not os.path.isdir(cfg_path):
                continue
=======
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
            # find latest json file
            json_list = list()
            for p in os.listdir(cfg_path):
                if p.find(".json") != -1:
                    json_list.append(p)
            if len(json_list) == 0:
                print(f"find empty dir={cfg_path}")
                continue
            
            if len(json_list) > 1:
                json_list.sort()
            # find 3 times average results
            score = dict()
            for j in range(3):
                try:
                    cfg_args["path"] = os.path.join(cfg_path, json_list[-(1+j)])
                    cfg_args["print_all"] = False
                    result = read_json_max(**cfg_args)
                    for k in keyword:
                        if j == 0:
                            score[k] = list()
                        score[k].append(result[k])
                except:
                    print("empty json", j)
            print("*"*100)
            print(cfg)
            print_str = "3 times average --- "
            for k in keyword:
<<<<<<< HEAD
                _str = "{}={:.2f} ({:.2f}), ".format(k, np.average(np.array(score[k])), np.std(np.array(score[k])))
                print_str += _str
            print(print_str, '\n')
=======
                try:
                    _str = "{}={:.2f} ({:.2f}), ".format(k, np.average(np.array(score[k])), np.std(np.array(score[k])))
                    print_str += _str
                except:
                    print("ERROR")
            print(print_str)
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)

    # read a json, returm max results
    else:
        args["print_all"] = True
<<<<<<< HEAD
        args["keyword"] = keyword
=======
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
        read_json_max(**args)
