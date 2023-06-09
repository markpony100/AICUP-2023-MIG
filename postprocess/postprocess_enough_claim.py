import json
from dataclasses import dataclass
import pandas as pd
import sys
import numpy as np
sys.path.append('../')
from utils import load_simple_json,load_json
from sklearn.metrics import recall_score,precision_score,confusion_matrix,accuracy_score
def join_pred_data(pred_dic,data):
    re_lst=[]
    data_map = {i["id"]:idx for idx,i in enumerate(data)}
    for key in pred_dic:
        buf=None
        if int(key) in data_map.keys():
            buf = data[data_map[int(key)]]
            buf["enough"]=pred_dic[key]['predict_label']
            re_lst.append(buf)
    return re_lst

def join_kf_dic(in_dics,thresh=None):#majority voting
    dics=in_dics.copy()
    re_dic={}
    for dic in dics:
        for key in dic:
            if key in re_dic:
                re_dic[key]+=[dic[key]["prob"]]
            else:
                re_dic[key]=[dic[key]["prob"]]
    for key in re_dic:
        if thresh:
            re_dic[key]=sum(np.array(re_dic[key]))/len(in_dics)
        else:
            re_dic[key] = int(np.median(re_dic[key]))
    return re_dic
if __name__ =="__main__":
    #params
    TEST_DATA_PATH ="../pert_large/sent/0522_base_clu_folds_PN13/all_test_clu_page_sent_ens5.jsonl"
    ENOUGH_PATH = "../pert_large/enough/0522_nclu_neg_no_pid_folds/"
    CLAIM_PATH = "../pert_large/claim/0522_clu_w_noise_npid_folds/"
    SAVE_PATH = "./page_sent_claim_clu_enough_nclu_prob.jsonl"
    #load TEST original data
    TEST_DATA = load_json(TEST_DATA_PATH)
    #pair with enough prediction
    processed_preds=[]
    processed_dicts=[]
    preds = ["all_test_f0","all_test_f1","all_test_f2","all_test_f3","all_test_f4"]
    pred_dics =[]
    for pred_path in preds:
        d_buf = load_simple_json(ENOUGH_PATH+pred_path+".json")
        pred_dics.append(d_buf)
    pred_dic = join_kf_dic(pred_dics,thresh=0.05)
    pred_dic = {key:{"predict_label":pred_dic[key]} for key in pred_dic}
    pair_data = join_pred_data(pred_dic,TEST_DATA)
    #pair with claim prediction
    processed_preds=[]
    processed_dicts=[]
    preds = ["all_test_f0","all_test_f1","all_test_f2","all_test_f3","all_test_f4"]
    pred_dics =[]
    for pred_path in preds:
        d_buf = load_simple_json(CLAIM_PATH+pred_path+".json")
        pred_dics.append(d_buf)
    pred_dic = join_kf_dic(pred_dics,thresh=0.05)
    pred_dic = {key:{"predict_label":pred_dic[key]} for key in pred_dic}
    pair_data = join_pred_data(pred_dic,pair_data)
    with open(SAVE_PATH,"w",encoding="utf8",) as f:
        for i, d in enumerate(pair_data):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("process done")
