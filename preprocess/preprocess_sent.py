import json
import numpy as np
import pandas as pd
from pandarallel import pandarallel
import sys
sys.path.append('../')
from utils import (
    generate_evidence_to_wiki_pages_mapping,
    jsonl_dir_to_df,
    load_json,
    load_simple_json
)
#from preprocess_sentence_pair import pair_with_wiki_sentences_custom,pair_with_wiki_sentences_eval
from pathlib import Path
import argparse
pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=10)
def pair_with_wiki_sentences_custom(mapping,df,negative_ratio:float = 0.5,pred_page:str = "predicted_pages",with_evid=True) -> pd.DataFrame:
    """Only for creating train sentences."""
    claims = []
    sentences = []
    labels = []
    re_lst =[]
    if with_evid:
        print("using only evidence pages")
    # positive
    for i in range(len(df)):
        if df["label"].iloc[i] == "NOT ENOUGH INFO" :
            continue
        claim = df["claim"].iloc[i]
        evidence_sets = df["evidence"].iloc[i]
        evid_dic ={}
        for evidence_set in evidence_sets:
            sents = []
            for evidence in evidence_set:
                # evidence[2] is the page title
                page = evidence[2].replace(" ", "_")
                sent_idx = str(evidence[3])
                if page not in evid_dic.keys():
                    evid_dic[page]=[sent_idx]
                else:
                    evid_dic[page].append(sent_idx)
        #pred_pages = df[pred_page].iloc[i]
        if with_evid:
            #pred_pages = list(set(pred_pages+list(evid_dic.keys())))
            pred_pages = list(evid_dic.keys())
        for page in pred_pages:#evid_dic:
            try:
                page_sent_id_pairs = [(page, k) for k in mapping[page]]
            except KeyError:
                print(f"{page} is not in our Wiki db.")
                continue
            for page_name, sentence_id in page_sent_id_pairs:
                text = mapping[page][sentence_id].replace(" ","")
                if page in evid_dic.keys():#correct page negitive sample
                    if sentence_id in evid_dic[page] or (text != "" and np.random.rand(1) <= negative_ratio):
                        re_lst.append({"sentence1":claim,
                            "sentence2":page+":"+sentence_id+":"+text,#page+":"+text,
                            "qid":int(df["id"].iloc[i]),
                            "predicted_page":page_name,
                            "sent_id":int(sentence_id),
                            "label":int(sentence_id in evid_dic[page])})
                else:#other page negitive sample
                    if (text != "" and np.random.rand(1) <= negative_ratio):
                        re_lst.append({"sentence1":claim,
                            "sentence2":page+":"+sentence_id+":"+text,
                            "qid":int(df["id"].iloc[i]),
                           "predicted_page":page_name,
                            "sent_id":int(sentence_id),
                            "label": 0 })

                    #claims.append(claim)
                    #sentences.append(page+":"+text)
                    #labels.append(int(sentence_id in evid_dic[page]))
    count=[elem["label"] for elem in re_lst]
    print("positve sent: ",count.count(1))
    print("negitive sent: ",count.count(0))
    return re_lst
def pair_with_wiki_sentences_eval(mapping,df: pd.DataFrame,is_testset: bool = False,pred_page: str ="predicted_pages") -> pd.DataFrame:
    """Only for creating dev and test sentences."""
    claims = []
    sentences = []
    evidence = []
    predicted_evidence = []
    re_lst=[]
    # negative
    for i in range(len(df)):
        # if df["label"].iloc[i] == "NOT ENOUGH INFO":
        #     continue
        claim = df["claim"].iloc[i]
        qid = df["id"].iloc[i]
        predicted_pages = df[pred_page][i]
        for page in predicted_pages:
            page = page.replace(" ", "_")
            try:
                page_sent_id_pairs = [(page, k) for k in mapping[page]]
            except KeyError:
                print(f"{page} is not in our Wiki db.")
                continue

            for page_name, sentence_id in page_sent_id_pairs:
                text = mapping[page][sentence_id].replace(" ","")
                buf_dic ={}
                if text != "" :
                    buf_dic["sentence1"]=claim
                    buf_dic["sentence2"]=page+":"+text
                    if not is_testset:
                        buf_dic["evidence"] = df["evidence"].iloc[i]
                    buf_dic["predicted_page"]=page_name
                    buf_dic["sent_id"]=int(sentence_id)
                    buf_dic["qid"]=int(qid)
                    re_lst.append(buf_dic)
    print("eval label counts: ",len(set([i["qid"] for i in re_lst])))
    return re_lst
if __name__ =="__main__":
    #parameters
    NEG_RATIO=0.4
    TRAIN_SAVE = "../pert_large/sent_data/Ttrain_all.json"
    TEST_SAVE = "../pert_large/sent_data/Ttest_all.json"
    train_idx_path = "./train_fold_index_clu.json"
    wiki_path = "../wiki-pages"
    train_path = "./train_all.jsonl"
    test_path = "../pert_large/page/0522_base_clu_folds/Ttest_ens5.jsonl"
    #PN for positive rate : negative rate = 1:3
    #TRAIN data
    TRAIN_DATA = load_json(train_path)
    TRAIN_MAP = {i["id"]:idx for idx,i in enumerate(TRAIN_DATA)}
    #loading pre-split index
    kfold_idx = load_simple_json(train_idx_path)
    #loading wiki mapping
    wiki_pages = jsonl_dir_to_df(wiki_path)
    mapping = generate_evidence_to_wiki_pages_mapping(wiki_pages)
    del wiki_pages
    #kfold sentnece mapping
    save_dic={}
    for i,fold_idx in enumerate(kfold_idx):
        train_data = [TRAIN_DATA[TRAIN_MAP[idx]] for idx in fold_idx[0] if idx in TRAIN_MAP.keys()]
        val_data = [TRAIN_DATA[TRAIN_MAP[idx]] for idx in fold_idx[1] if idx in TRAIN_MAP.keys()]
        train= pair_with_wiki_sentences_custom(mapping,pd.DataFrame(train_data),NEG_RATIO, pred_page="predicted_pages_1",with_evid=True)
        val= pair_with_wiki_sentences_custom(mapping,pd.DataFrame(val_data),NEG_RATIO, pred_page="predicted_pages_1",with_evid=True)
        save_dic["train_f"+str(i)]=train
        save_dic["val_f"+str(i)]=val
    with open(TRAIN_SAVE,"w") as f:
        json.dump(save_dic,f)
    #TEST data
    TEST_DATA = load_json(test_path)
    TEST_MAP = {i["id"]:idx for idx,i in enumerate(TEST_DATA)}
    test= pair_with_wiki_sentences_eval(mapping,pd.DataFrame(TEST_DATA),pred_page="predicted_pages_1",is_testset=True)
    with open(TEST_SAVE,"w") as f:
        json.dump({"test":test},f)
    print("process done")

