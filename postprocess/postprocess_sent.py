#from evaluate_sent_retrieval import evaluate_retrieval,process_prediction
import json
from dataclasses import dataclass
import pandas as pd
import sys
sys.path.append('../')
from utils import load_simple_json,load_json
def process_kf_prediction(pred,thresh=0.5,topk=None):
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
            [buf_lst.append(pred[qid]["predicted_evidence"][scores.index(i)]) for i in sorted_lst]
        else:
            sorted_lst = sorted(scores,reverse=True)[:]
            [buf_lst.append(pred[qid]["predicted_evidence"][scores.index(value)]) for idx,value in
                    enumerate(sorted_lst) if value >= thresh]
        re_dic[int(qid)]=buf_lst
    return re_dic
def sort_by_frequency(input_list):#complex edition hehe
    elem_set = []
    for elem in input_list:
        if elem not in elem_set:
            elem_set.append(elem)
    count_lst =[]
    for elem in elem_set:
        count_lst.append(input_list.count(elem))
    sorted_indices = sorted(range(len(count_lst)), key=lambda x: count_lst[x],reverse=True)
    return [elem_set[i] for i in sorted_indices]
def join_kf_dict(dic_lsts,topk=5):
    buf_dic={}
    for dic in dic_lsts:
        for key in dic:
            if key in buf_dic:
                buf_dic[key]+=dic[key]
            else:
                buf_dic[key]=dic[key]
    re_dic={}
    for key in buf_dic:
        re_dic[key]=sort_by_frequency(buf_dic[key])[:topk]
    return re_dic
def turn_to_eval_format(dic,ori_data):
    data_map =  {d["id"]:idx for idx,d in enumerate(ori_data)}
    re_lst = []
    for key in dic:
        for sent in dic[key]:
            re_lst.append({"qid":key,"claim":ori_data[data_map[key]]["claim"],"predicted_evidence":sent,"prob":1})
    return re_lst
def pair_pred(pred_dic,data):
    re_lst=[]
    data_map = {d["id"]:idx for idx,d in enumerate(data)}
    for key in pred_dic:
        if key in data_map.keys():
            buf = data[data_map[key]]
            buf["predicted_evidence"]=pred_dic[key]
            re_lst.append(buf)
    return re_lst
if __name__ == "__main__":
    #parameters
    ori_data_path = "../preprocess/test_all.jsonl"
    test_data_folder = "../pert_large/sent/0522_base_clu_folds_PN13/"
    save_test_path = "../pert_large/sent/0522_base_clu_folds_PN13/Ttest.jsonl"
    TOPK = 5
    #load data
    ori_data = load_json(ori_data_path)
    processed_preds=[]
    processed_dicts=[]
    preds = ["all_test_f0","all_test_f1","all_test_f2","all_test_f3","all_test_f4"]
    for pred_path in preds:
        pred = load_simple_json(test_data_folder+pred_path+".json")
        processed_dict = process_kf_prediction(pred,thresh=0.1,topk=10)
        processed_dicts.append(processed_dict)
    preds = join_kf_dict(processed_dicts,topk=TOPK)
    with open(save_test_path,"w",encoding="utf8",) as f:
        for i, d in enumerate(pair_pred(preds,ori_data)):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("process done")