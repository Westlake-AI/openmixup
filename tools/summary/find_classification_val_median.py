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
    parser.add_argument('key', type=str, help='head keyword in the json files.')
    args = parser.parse_args()
    return args.__dict__


def read_json(path, epoch_num=1200, record_num=20, print_all=True, keyword=None, **kwargs):
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
                if line.get("epoch") >= epoch_num - record_num:
                    res = f"{line['epoch']}e, "
                    for k in keyword:
                        try:
                            res += "{}: {:.2f}, ".format(k, line[k])
                            record_acc[k].append(line[k])
                        except:
                            pass
                    record_str.append(res)
    # output records
    print_str = "Median -- "
    if print_all:
        max_num = min(len(record_str), 5)
        for l in record_str[-max_num:]:
            print(l)
    for k in keyword:
        record_acc[k] = np.median(np.array(record_acc[k]))
        print_str += "{}: {:.2f},".format(k, record_acc[k])
    if print_all:
        print(print_str)
    return record_acc


if __name__ == '__main__':
    """ find the median of val results in latest N epochs """
    args = parse_args()
    print(args)

    keyword = args.get("key", ["head0"])
    if isinstance(keyword, str):
        keyword = keyword.split("-")

    # read record of a dir
    if args["path"].find(".json") == -1:
        assert os.path.exists(args["path"])
        cfg_list = os.listdir(args["path"])
        cfg_list.sort()

        for cfg in cfg_list:
            cfg_args = args.copy()
            cfg_args["keyword"] = keyword
            cfg_path = os.path.join(args["path"], cfg)
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
            cfg_args["path"] = os.path.join(cfg_path, json_list[-1])
            cfg_args["print_all"] = False

            print("*"*100)
            score = read_json(**cfg_args)
            print_str = cfg + f" -- median of last {cfg_args['record_num']}ep in {cfg_args['epoch_num']}ep\n"
            for k in keyword:
                _str = "{}={:.2f}, ".format(k, score[k])
                print_str += _str
            print(print_str)

    # read a json
    else:
        args["print_all"] = True
        args["keyword"] = keyword
        read_json(**args)


# Usage 1: summary results of a json file.
#    python tools/summary/find_classification_val_median.py [full_path to xxx.json] [total eposh] [last n epoch for median] [keys]
# Usage 2: summary results of a dir of training results (as json files).
#    python tools/summary/find_classification_val_median.py [full_path to the dir] [total eposh] [last n epoch for median] [keys]
