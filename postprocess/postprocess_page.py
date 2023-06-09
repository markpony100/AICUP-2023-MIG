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
def join_data_with_preds(ori_data,preds,key = "predicted_pages_1"):
    '''
    input: original data and processed predcition
    return: joint data with prediciton
    '''
    re_lst=[]
    for i in ori_data:
        id_buf = str(i["id"])
        if id_buf in preds.keys():
            i[key]=preds[id_buf]
            re_lst.append(i)
    return re_lst
def unique(input_lst):
    elem_set = []
    for elem in input_lst:
        if elem not in elem_set:
            elem_set.append(elem)
    return elem_set
def join_kfold_preds(pred_lst):
    buf_dic = pred_lst[0]
    for i in range(len(pred_lst)):
        if i ==0:
            continue
        for key in pred_lst[i]:
            if key in buf_dic.keys():
                buf_dic[key]+=pred_lst[i][key]
            else:
                buf_dic[key]=pred_lst[i][key]
    for key in buf_dic:#post process clean duplicate
        buf_dic[key]=unique(buf_dic[key])
    return buf_dic
if __name__ =="__main__":
    #parameters
    ori_data_path = "../preprocess/test_all.jsonl"
    test_data_folder = "../pert_large/page/0522_base_clu_folds/"
    save_test_path = "../pert_large/page/0522_base_clu_folds/Ttest_ens5.jsonl"
    #load data
    ori_data = load_json(ori_data_path)
    processed_preds=[]
    preds = ["all_test_f0","all_test_f1","all_test_f2","all_test_f3","all_test_f4"]
    #thr_param =[0.001, 0.001,0.004,0.004,]
    for idx,pred_path in enumerate(preds):
        pred = load_simple_json(test_data_folder+pred_path+".json")
        processed_preds.append(process_prediction(pred,0.5,None))
    paired_data = join_data_with_preds(ori_data,join_kfold_preds(processed_preds))
    with open(save_test_path,"w",encoding="utf8",) as f:
        for i, d in enumerate(paired_data):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("process done")