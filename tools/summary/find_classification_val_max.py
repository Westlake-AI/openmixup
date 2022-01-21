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


def read_json_max(path, epoch_num=1200, record_num=20, print_all=True):
    record_str = list()
    record_top1 = list()
    record_top5 = list()
    assert path.find("json") != -1
    # read each line
    with open(path, "r") as f:
        for line in f.readlines():
            line = json.loads(line)
            if line.get("mode", None) == "val":
                if line.get("epoch") >= epoch_num - record_num:
                    # print(line)
                    record_str.append("{}e, acc_mix_top1: {:.2f}, acc_mix_top5: {:.2f}".format(
                        line["epoch"], line["acc_mix_top1"], line["acc_mix_top5"]))
                    record_top1.append(line["acc_mix_top1"])
                    record_top5.append(line["acc_mix_top5"])

                    record_str.append("{}e, acc_one_top1: {:.2f}, acc_one_top5: {:.2f}".format(
                        line["epoch"], line["acc_one_top1"], line["acc_one_top5"]))
                    record_top1.append(line["acc_one_top1"])
                    record_top5.append(line["acc_one_top5"])
    # output records
    for l in record_str:
        if print_all:
            print(l)
    result = [np.max(np.array(record_top1)), np.max(np.array(record_top5))]
    if print_all:
        print("max top1, top5: {:.2f}, {:.2f}".format( result[0], result[1] ))
    return result


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
                    json_list.append(p)
            assert len(json_list) != 0
            if len(json_list) > 1:
                json_list.sort()
            # find 3 times average results
            top1, top5 = list(), list()
            for j in range(3):
                try:
                    cfg_args["path"] = os.path.join(cfg_path, json_list[-(1+j)])
                    cfg_args["print_all"] = False
                    cfg_args["record_num"] = cfg_args["epoch_num"]
                    result = read_json_max(**cfg_args)
                    top1.append(result[0])
                    top5.append(result[1])
                except:
                    print("empty json", j)
            print("*"*100)
            print(cfg)
            print("3 times average, top 1 & 5={:.2f}, {:.2f}".format(
                np.average(np.array(top1)), np.average(np.array(top5))))

    # read a json, returm max results
    else:
        args["print_all"] = True
        read_json_max(**args)

# The usage of this tools is the same as find_automix_val_median.py
