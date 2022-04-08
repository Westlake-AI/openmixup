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
                    if line.get("acc_mix_k_top1", None) is None:
                        line["acc_mix_k_top1"] = line["acc_mix_q_top1"]
                    record_str.append("{}e, mix_k_top1: {:.2f}, one_k_top1: {:.2f}, mix_q_top1: {:.2f}, one_q_top1: {:.2f}".format(
                        line["epoch"], line["acc_mix_k_top1"], line["acc_one_k_top1"], line["acc_mix_q_top1"], line["acc_one_q_top1"]))
                    record_one_q_top1.append(line["acc_one_q_top1"])
                    record_mix_q_top1.append(line["acc_mix_q_top1"])
                    record_one_k_top1.append(line["acc_one_k_top1"])
                    record_mix_k_top1.append(line["acc_mix_k_top1"])
    # output records
    for l in record_str:
        if print_all:
            print(l)
    bias = 1
    # find best median
    best_index = list()
    bias_index = [len(record_one_q_top1) - bias, len(record_one_q_top1), len(record_one_q_top1) + bias]
    for i in bias_index:
        best_index.append(max(
            np.median(np.array(record_one_k_top1[ :i])), np.median(np.array(record_mix_k_top1[ :i])),
            np.median(np.array(record_one_q_top1[ :i])), np.median(np.array(record_mix_q_top1[ :i]))
        ))
    index = bias_index[ np.argmax(np.array(best_index)) ]
    # print(index)
    record_one_q_top1 = record_one_q_top1[ :index]
    record_mix_q_top1 = record_mix_q_top1[ :index]
    record_one_k_top1 = record_one_k_top1[ :index]
    record_mix_k_top1 = record_mix_k_top1[ :index]
    # results
    print("k={:.2f}, q={:.2f}".format(
        max(np.median(np.array(record_one_k_top1)), np.median(np.array(record_mix_k_top1)) ),
        max(np.median(np.array(record_one_q_top1)), np.median(np.array(record_mix_q_top1)) ),
    ))


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

# Usage 1: summary results of a json file.
#    python tools/summary/find_automix_val_median.py [full_path to xxx.json] [total eposh] [last n epoch for median]
# Usage 2: summary results of a dir of training results (as json files).
#    python tools/summary/find_automix_val_median.py [full_path to the dir] [total eposh] [last n epoch for median]
#
# For example: 
# - work_dirs/classification/cifar100/automix/r18/r18_1_400ep/xxx.json
# - work_dirs/classification/cifar100/automix/r18/r18_2_400ep/xxx.json
# Usage 1: [full_path to xxx.json]=work_dirs/classification/cifar100/automix/r18/r18_1_400ep/xxx.json
# Usage 2: [full_path to the dir]=work_dirs/classification/cifar100/automix/r18/r18_1_400ep
