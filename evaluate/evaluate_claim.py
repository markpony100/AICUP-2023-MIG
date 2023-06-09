import json
import argparse
import pandas as pd
import sys
sys.path.append('../')
from utils import load_simple_json,load_json




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path",
                        default="../pert_large/claim/e20_b4_doc100_cluster_top5_B2/test.json",type=str,)
    parser.add_argument("--ori_path",
                        default="../pert_large/sent_data/test_doc100_cluster_rerank_thr001_sent_top5_.jsonl",
                        type=str,)
    parser.add_argument("--save_path",
                        default="../pert_large/claim/e20_b4_doc100_cluster_top5_B2/test.jsonl",
                        type=str,)#../pert_large/claim/test_doc100_cluster_thr001.jsonl
    parser.add_argument('--predict_only', default=False, type=bool,)
    args = parser.parse_args()
    ori_data = load_json(args.ori_path)
    pred_data = load_simple_json(args.pred_path)
    re_lst=[]
    label_mapping = { 0:"supports", 1:"refutes", 2:"NOT ENOUGH INFO"}
    for i in ori_data:
        qid=i["id"]
        evidence_set = []
        pred_pages = "IDK"
        label="IDK"
        claim = i["claim"]
        if not args.predict_only:
            evidence_set=i["evidence"]
            label = i["label"]
            pred_pages = i["predicted_pages"]
        if str(qid) in pred_data.keys() and len(i["predicted_evidence"])>=1 :
            re_lst.append({"id":qid,"claim":claim,
                           "predicted_label":label_mapping[pred_data[str(qid)]["predict_label"]],"label":label,
                           "predicted_evidence":i["predicted_evidence"],
                           "evidence":evidence_set,"predicted_pages":pred_pages})
        else:
            re_lst.append({"id":qid,"claim":claim,"predicted_label":label_mapping[2],"label":label,
                           "predicted_evidence":i["predicted_evidence"],
                           "evidence":evidence_set,"predicted_pages":pred_pages})
    count = [i["predicted_label"] for i in re_lst]
    for i in [0,1,2]:
        print(label_mapping[i],"count: ",count.count(label_mapping[i]))
    if args.save_path:
        with open(args.save_path,"w",encoding="utf8",) as f:
                for i, d in enumerate(re_lst):
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(args.save_path, "saved")
    print("process done")
