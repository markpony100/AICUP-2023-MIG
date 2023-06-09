import json
import argparse
from dataclasses import dataclass
import pandas as pd
import sys
sys.path.append('../')
from utils import load_simple_json,load_json

def process_prediction(pred,thresh=0.5,topk=None):
    '''
    input: raw prediction
    return: dictionary with id and page value
    '''
    re_dic={}
    for qid in pred:
        buf_lst=[]
        scores = pred[qid]["score"]
        if topk:
            sorted_lst = sorted(scores,reverse=True)[:topk]
            [buf_lst.append(pred[qid]["page_ids"][scores.index(i)]) for i in sorted_lst]
        else:
            [buf_lst.append(pred[qid]["page_ids"][idx]) for idx,value in 
                    enumerate(scores) if value >= thresh]
        re_dic[qid]=buf_lst
    return re_dic
def join_data_with_preds(ori_data,preds):
    '''
    input: original data and processed predcition
    return: joint data with prediciton
    '''
    re_lst=[]
    for i in ori_data:
        id_buf = str(i["id"])
        if id_buf in preds.keys():
            i["predicted_pages_1"]=preds[id_buf]
            re_lst.append(i)
    return re_lst

def calculate_precision(data,predictions: pd.Series) -> None:
    precision = 0
    count = 0

    for i, d in enumerate(data):
        if d["label"] == "NOT ENOUGH INFO":
            continue

        # Extract all ground truth of titles of the wikipedia pages
        # evidence[2] refers to the title of the wikipedia page
        gt_pages = set([
            evidence[2]
            for evidence_set in d["evidence"]
            for evidence in evidence_set
        ])

        predicted_pages = predictions.iloc[i]
        hits = predicted_pages.intersection(gt_pages)
        if len(predicted_pages) != 0:
            precision += len(hits) / len(predicted_pages)

        count += 1

    # Macro precision
    print(f"Precision: {precision / count}")
def at_least_get_one(data,predictions: pd.Series) -> None:
    precision = 0
    count = 0
    hit_counts=0
    for i, d in enumerate(data):
        if d["label"] == "NOT ENOUGH INFO":
            continue
        predicted_pages = predictions.iloc[i]
        # Extract all ground truth of titles of the wikipedia pages
        # evidence[2] refers to the title of the wikipedia page
        at_least_get_one = False
        for evidence_set in d["evidence"]:
            evid_buf=[]
            for evidence in evidence_set:
                evid_buf.append(evidence[2])
            hit_count = 0
            for evid in evid_buf:
                if evid in predicted_pages:
                    hit_count+=1
            if hit_count == len(evid_buf):
                at_least_get_one=True
        hit_counts+=int(at_least_get_one)
        count+=1
    # Macro precision
    print(f"at least get one rate : {hit_counts / count}")


def calculate_recall(data,predictions: pd.Series) -> None:
    recall = 0
    count = 0

    for i, d in enumerate(data):
        if d["label"] == "NOT ENOUGH INFO":
            continue

        gt_pages = set([
            evidence[2]
            for evidence_set in d["evidence"]
            for evidence in evidence_set
        ])
        predicted_pages = predictions.iloc[i]
        hits = predicted_pages.intersection(gt_pages)
        recall += len(hits) / len(gt_pages)
        count += 1

    print(f"Recall: {recall / count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #path args
    parser.add_argument("--thresh",default=0.01,type=float,)
    parser.add_argument("--pred_path",default="../pert_large/page/doc100_e20b8_recall/val_test.json",type=str,)
    parser.add_argument("--ori_path",default="../../baseline/data/train_doc100_ensemble_no_pre.jsonl",type=str,)
    parser.add_argument("--save_path",default="",type=str,)
    parser.add_argument("--key",default="predicted_pages_1",type=str,)
    parser.add_argument('--topk', default=None, type=int,)
    parser.add_argument('--predict_only', default=False, type=bool,)
    args = parser.parse_args()
    ori_data = load_json(args.ori_path)
    if args.key == "predicted_pages_1":
        preds = load_simple_json(args.pred_path)
        processed_preds = process_prediction(preds,args.thresh,args.topk)
        paired_data = join_data_with_preds(ori_data,processed_preds)
    else:
        paired_data = ori_data
    predictions = pd.Series([set(elem[args.key]) for elem in paired_data])
    if not args.predict_only:
        calculate_precision(paired_data,predictions)
        calculate_recall(paired_data,predictions)
        at_least_get_one(paired_data,predictions)
    if args.save_path:
        with open(args.save_path,"w",encoding="utf8",) as f:
            for i, d in enumerate(paired_data):
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("preocess done")
    

