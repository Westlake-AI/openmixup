import os
import argparse
import numpy as np
import json
from tqdm import tqdm


def parse_args():
    """ linear or semi-supervised mode """
    parser = argparse.ArgumentParser(
        description='Read a classification json file to report val results.')
    parser.add_argument('config_dir', type=str, help='input configs path')
    parser.add_argument('weight_name', type=str, help='input pretrain weight name')
    parser.add_argument('epoch_num', type=int, default=200, help='input total epoch num')
    parser.add_argument('record_num', type=int, default=20, help='valid record range')
    args = parser.parse_args()
    return args.__dict__


def read_json(path, epoch_num=1200, record_num=20):
    # record_str = list()
    record_top1 = list()
    record_top5 = list()
    record_last_top1 = list()
    record_last_top5 = list()
    if path.find("json") == -1:
        json_list = list()
        for p in os.listdir(path):
            if p.find("json") != -1:
                json_list.append(p)
        assert len(json_list) != 0
        if len(json_list) > 1:
            json_list.sort()
        path = os.path.join(path, json_list[-1])
    
    # read each line
    with open(path, "r") as f:
        for line in f.readlines():
            line = json.loads(line)
            if line.get("mode", None) == "val":
                # record_str.append("{}e, head0_top1: {:.2f}, head0_top5: {:.2f}".format(
                #     line["epoch"], line["head0_top1"], line["head0_top5"]))
                record_top1.append(line["head0_top1"])
                record_top5.append(line["head0_top5"])
                if line.get("epoch") >= epoch_num - record_num:
                    record_last_top1.append(line["head0_top1"])
                    record_last_top5.append(line["head0_top5"])
    # output records
    # for l in record_str:
    #     print(l)
    results = dict()
    results['median'] = ( round(np.median(np.array(record_last_top1)), 2), round(np.median(np.array(record_last_top5)), 2) )
    results['max'] = ( round(np.max(np.array(record_top1)), 2), round(np.max(np.array(record_top5)), 2) )
    results['best'] = round(np.max(np.array(record_top1)), 4)
    return results


def results_summary(args):
    # read exp configs
    results_list = list()
    configs_list = list()
    if os.path.exists(args["config_dir"]):
        work_path = "work_dirs" + args["config_dir"].split("configs")[1]
        # print(work_path)
        configs_list = os.listdir(args["config_dir"])
        for cfg in tqdm(configs_list):
            if cfg.find("base") == -1 and cfg.find(".sh") == -1:  # except: "xx_base.py", "xxx.sh"
                cfg_name = cfg.split(".py")[0]
                input_args = dict(
                    path=os.path.join(work_path, cfg_name, args["weight_name"]),
                    epoch_num=50, record_num=6,
                )
                try:
                    results = read_json(**input_args)
                    results_list.append(results)
                except:
                    print("bad configs: {}, {}".format(cfg_name, args["weight_name"]))
    
    results_list.sort(key=lambda k: (k.get("best", 0)), reverse=True)
    print(args["weight_name"], results_list[0])



if __name__ == '__main__':
    """ find the median of val results in latest N epochs """
    args = parse_args()
    print(args)

    if os.path.exists(args["weight_name"]):
        weight_list = os.listdir(args["weight_name"])
        weight_list.sort()
        print("\n  ***** Extract All Weights Results ******\n")
        for weight in weight_list:
            args["weight_name"] = weight
            results_summary(args)
        print("\n  ***** Finished All Results Summaries ******\n")
    else:
        results_summary(args)


# Usage 1: summary results of [weight name]
#    python tools/results_summary_classification.py [path to semi-supervised test configs] [weight name] 50 10
# Usage 2: summary results of a list of weights in [weight dirs]
#    python tools/results_summary_classification.py [path to semi-supervised test configs] [weight dirs] 50 10
