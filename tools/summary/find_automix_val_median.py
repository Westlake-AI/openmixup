"""
Summarize results of the key from json logs for AutoMix

Usage 1: summary results of a json file.
   python tools/summary/find_automix_val_median.py [PATH/to/xxx.json] [total eposh] [last n epoch for median]
Usage 2: summary results of a dir of training results (as json files).
   python tools/summary/find_automix_val_median.py [PATH/to/exp_dir] [total eposh] [last n epoch for median]

It requires the folder built as follows:
└── [PATH/to/exp_dir]
    └── xxx_ep100
        ├── [PATH/to/xxx.json] (i.e., xxx_ep100_yyy.log.json)
        ├── ...
    └── xxx_ep300
        ├── xxx_300_zzz.log.json
        ├── ...

For example:
└── work_dirs/classification/cifar100/automix/r18
    └── r18_1_400ep
        ├── xxx.json
    └── r18_2_400ep
        ├── yyy.json

Usage 1: [PATH/to/xxx.json] = 'work_dirs/classification/cifar100/automix/r18/r18_1_400ep/xxx.json'
Usage 2: [PATH/to/exp_dir] = 'work_dirs/classification/cifar100/automix/r18/r18_1_400ep'
"""

import argparse
import numpy as np
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='Read a classification json file to report val results.')
    parser.add_argument('path', type=str, help='input json filename')
    parser.add_argument('epoch_num', type=int, default=200, help='input total epoch num')
    parser.add_argument('record_num', type=int, default=20, help='valid record range')
    args = parser.parse_args()
    return args.__dict__


def read_json(path, epoch_num=1200, record_num=20, print_all=True):
    record_str = list()
    record_one_q_top1 = list()
    record_one_k_top1 = list()
    record_mix_q_top1 = list()
    record_mix_k_top1 = list()
<<<<<<< HEAD

    record_one_q_top5 = list()
    record_one_k_top5 = list()
    record_mix_q_top5 = list()
    record_mix_k_top5 = list()

=======
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
    assert path.find("json") != -1, \
        "bad json path={}".format(path)
    
    bias = 0 if record_num < 10 else 1
    # read each line
    with open(path, "r") as f:
        for line in f.readlines():
            line = json.loads(line)
            if line.get("mode", None) == "val":
                if line.get("epoch") >= epoch_num - record_num - bias:
                    # print(line)
                    if line.get("acc_one_k_top1", None) is None:
                        line["acc_one_k_top1"] = line["acc_one_q_top1"]
<<<<<<< HEAD
                        line["acc_one_k_top5"] = line["acc_one_q_top5"]
                    if line.get("acc_mix_k_top1", None) is None:
                        line["acc_mix_k_top1"] = line["acc_mix_q_top1"]
                        line["acc_mix_k_top5"] = line["acc_mix_q_top5"]
=======
                    if line.get("acc_mix_k_top1", None) is None:
                        line["acc_mix_k_top1"] = line["acc_mix_q_top1"]
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
                    record_str.append("{}e, mix_k_top1: {:.2f}, one_k_top1: {:.2f}, mix_q_top1: {:.2f}, one_q_top1: {:.2f}".format(
                        line["epoch"], line["acc_mix_k_top1"], line["acc_one_k_top1"],
                        line["acc_mix_q_top1"], line["acc_one_q_top1"]))
                    record_one_q_top1.append(line["acc_one_q_top1"])
                    record_mix_q_top1.append(line["acc_mix_q_top1"])
                    record_one_k_top1.append(line["acc_one_k_top1"])
                    record_mix_k_top1.append(line["acc_mix_k_top1"])
<<<<<<< HEAD

                    record_one_q_top5.append(line["acc_one_q_top5"])
                    record_mix_q_top5.append(line["acc_mix_q_top5"])
                    record_one_k_top5.append(line["acc_one_k_top5"])
                    record_mix_k_top5.append(line["acc_mix_k_top5"])
=======
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
    # output records
    for l in record_str:
        if print_all:
            print(l)
    bias = 1
    # find best median
    best_index = list()
<<<<<<< HEAD
    best_index_ = list()
    bias_index = [len(record_one_q_top1) - bias, len(record_one_q_top1), len(record_one_q_top1) + bias]
    bias_index_ = [len(record_one_q_top5) - bias, len(record_one_q_top5), len(record_one_q_top5) + bias]
=======
    bias_index = [len(record_one_q_top1) - bias, len(record_one_q_top1), len(record_one_q_top1) + bias]
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
    for i in bias_index:
        best_index.append(max(
            np.median(np.array(record_one_k_top1[ :i])), np.median(np.array(record_mix_k_top1[ :i])),
            np.median(np.array(record_one_q_top1[ :i])), np.median(np.array(record_mix_q_top1[ :i]))
        ))
<<<<<<< HEAD
        best_index_.append(max(
            np.median(np.array(record_one_k_top5[:i])), np.median(np.array(record_mix_k_top5[:i])),
            np.median(np.array(record_one_q_top5[:i])), np.median(np.array(record_mix_q_top5[:i]))
        ))

    index = bias_index[ np.argmax(np.array(best_index)) ]
    index_ = bias_index_[np.argmax(np.array(best_index_))]

=======
    index = bias_index[ np.argmax(np.array(best_index)) ]
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
    # print(index)
    record_one_q_top1 = record_one_q_top1[ :index]
    record_mix_q_top1 = record_mix_q_top1[ :index]
    record_one_k_top1 = record_one_k_top1[ :index]
    record_mix_k_top1 = record_mix_k_top1[ :index]
<<<<<<< HEAD

    record_one_q_top5 = record_one_q_top5[:index]
    record_mix_q_top5 = record_mix_q_top5[:index]
    record_one_k_top5 = record_one_k_top5[:index]
    record_mix_k_top5 = record_mix_k_top5[:index]
    # results
    print("Acc_Top1 k={:.2f}, q={:.2f}".format(
        max(np.median(np.array(record_one_k_top1)), np.median(np.array(record_mix_k_top1)) ),
        max(np.median(np.array(record_one_q_top1)), np.median(np.array(record_mix_q_top1)) ),
    ))
    print("Acc_Top5 k={:.2f}, q={:.2f}".format(
        max(np.median(np.array(record_one_k_top5)), np.median(np.array(record_mix_k_top5))),
        max(np.median(np.array(record_one_q_top5)), np.median(np.array(record_mix_q_top5))),
    ))
=======
    # results
    print("k={:.2f}, q={:.2f}".format(
        max(np.median(np.array(record_one_k_top1)), np.median(np.array(record_mix_k_top1)) ),
        max(np.median(np.array(record_one_q_top1)), np.median(np.array(record_mix_q_top1)) ),
    ))
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)


if __name__ == '__main__':
    """ find the median of val results in latest N epochs """
    args = parse_args()
    print(args)

    # read record of a dir
    if args["path"].find(".json") == -1:
        assert os.path.exists(args["path"])
        cfg_list = os.listdir(args["path"])
        cfg_list.sort()

        for cfg in cfg_list:
            cfg_args = args.copy()
            cfg_path = os.path.join(args["path"], cfg)
            # find latest json file
            json_list = list()
            for p in os.listdir(cfg_path):
                if p.find(".json") != -1:
                    # # remove .pth
                    # os.system("rm {}/*.pth".format(cfg_path))
                    json_list.append(p)
            assert len(json_list) != 0
            if len(json_list) > 1:
                json_list.sort()
            cfg_args["path"] = os.path.join(cfg_path, json_list[-1])
            cfg_args["print_all"] = False

            print("*"*100)
            print(cfg)
            read_json(**cfg_args)

    # read a json
    else:
        read_json(**args)
    print("\n *** finished ***")
