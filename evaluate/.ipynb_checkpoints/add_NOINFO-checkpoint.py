import json
import argparse
import pandas as pd
import sys
sys.path.append('../')
from utils import load_simple_json,load_json
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path",
                        default="../pert_large/claim/e20_b4_doc100_cluster_top5_B2/test.jsonl",
                        type=str,)
    parser.add_argument("--ori_path",default="../preprocess/public_test.jsonl",type=str,)
    parser.add_argument("--save_path",default="../submissions/test_doc100_cluster_top5_NB2.jsonl",type=str,)
    args = parser.parse_args()
    ori_data = load_json(args.ori_path)
    pred_data = load_json(args.pred_path)
    mapping_idx = {d["id"]:idx for idx,d in enumerate(pred_data)}
    re_lst = []
    for d in ori_data:
        if d["id"] in mapping_idx.keys():
            data = pred_data[mapping_idx[d["id"]]]
            re_lst.append({"id":data["id"],"predicted_label":data["predicted_label"],"predicted_evidence":data["predicted_evidence"]})
        else:
            re_lst.append({"id":d["id"],"predicted_label":"NOT ENOUGH INFO","predicted_evidence":None})
    with open(args.save_path,"w",encoding="utf8",) as f:
            for i, d in enumerate(re_lst):
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(args.save_path, "saved")
    print("process done")
