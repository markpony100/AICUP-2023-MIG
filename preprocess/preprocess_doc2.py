from sklearn.model_selection import train_test_split
import json
import sys
from pandarallel import pandarallel
import argparse
pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=10)

sys.path.append('../')
from sklearn.model_selection import KFold

from utils import (
    load_json,jsonl_dir_to_df,
    load_simple_json,generate_evidence_to_wiki_pages_mapping
)
#from preprocess_rerank_page import claim_with_wiki_content
def claim_with_wiki_content(index,data,mapping,mode="train",with_page_name=False,with_evids=False):
    re_lst=[]
    for idx in index:
        if mode == "test":
            evidence_set = []
        else:
            if  data[idx]["label"]=="NOT ENOUGH INFO":
                if mode =="train":#ignore NO INFO data if in train mode
                    continue
                evidence_set = []
            else:
                evidence_set = set([evidence[2] for evidences in data[idx]["evidence"]  for evidence
                    in evidences])
        predicted_pages = data[idx]["predicted_pages"]
        if with_evids and mode !="test":#only add evidence when training
            predicted_pages = list(set(predicted_pages+list(evidence_set)))
        for pred_page in predicted_pages:
            if pred_page in mapping.keys():#if page name not match
                sent2 = "".join([mapping[pred_page][k].replace(" ","") for k in mapping[pred_page] ])#skip empty
                if with_page_name:
                    sent2 = pred_page+":"+sent2
                re_lst.append({"sentence1":data[idx]["claim"],
                             "sentence2":sent2,
                              "page_id":pred_page,
                             "qid":data[idx]["id"],"label":int(pred_page in evidence_set)})#label:0
    return re_lst
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
#calculate diswtribution
def check_dist(trainIdx,testIdx,data):
    label_mapping = {"supports": 0,"refutes": 1,"NOT ENOUGH INFO": 2}
    tr_dist,te_dist = [0,0,0],[0,0,0]
    for d in data:
        if d["id"] in trainIdx:
            tr_dist[label_mapping[d["label"]]]+=1
        else:
            te_dist[label_mapping[d["label"]]]+=1
    score = 0
    score+=(sum(tr_dist[:1])/sum(tr_dist)-sum(te_dist[:1])/sum(te_dist))**2 
    score+=((tr_dist[0])/(tr_dist[1])-(te_dist[0])/(te_dist[1]))**2
    print("train label ratio: {}, test label ratio:{}".format(tr_dist,te_dist))
    print("tr pos ratio: {}, te pos ratio: {}".format(sum(tr_dist[:1])/sum(tr_dist),sum(te_dist[:1])/sum(te_dist)))
    print("tr sup/ref ratio: {}, te sup/ref ratio: {}".format((tr_dist[0])/(tr_dist[1]),(te_dist[0])/(te_dist[1])))
    print("sum: ",sum(tr_dist+te_dist))
    return tr_dist,te_dist,score

if __name__=="__main__":
    #parameter for cluster data split
    rand_seed = 1658
    train_idx_save = "train_fold_index_clu.json"
    train_path = './train_all.jsonl'
    doc1_train =  "../processed_data/pre_train_wikisearch_base.jsonl"
    train_save = "../pert_large/page_data/train_cluster_folds.json"
    test_path = './test_all.jsonl'
    test_save= "../pert_large/page_data/all_test.json"
    doc1_pub_test = "../processed_data/test_wikisearch_base.jsonl"
    doc1_pri_test = "../processed_data/private_wikisearch_base.jsonl"
    wiki_path = "../wiki-pages/"
    cluster_path = "./cluster/processing_file/clusters.json"#cc_clusters_96
    #TRAIN DATA
    clusters_lst = load_simple_json(cluster_path)["clusters"]
    clusters_lst = [clusters_lst[key] for key in clusters_lst]
    TRAIN_DATA = load_json(train_path)
    TRAIN_MAP = {d["id"]:idx for idx,d in enumerate(TRAIN_DATA)}
    wiki_emb = load_json(doc1_train)
    wiki_preds = {str(i["id"]):i["result"] for i in wiki_emb}
    paired_data = join_data_with_preds(TRAIN_DATA,wiki_preds,key="predicted_pages")
    #TEST DATA
    TEST_DATA = load_json(test_path)#"../../baseline/data/public_test.jsonl"
    test_emb = load_json(doc1_pub_test)+load_json(doc1_pri_test)
    wiki_preds = {str(i["id"]):i["result"] for i in test_emb}
    paired_test = join_data_with_preds(TEST_DATA,wiki_preds,key="predicted_pages")
    TEST_DATA = paired_test
    #mapping wiki
    wiki_pages = jsonl_dir_to_df(wiki_path)
    mapping = generate_evidence_to_wiki_pages_mapping(wiki_pages,)
    del wiki_pages
    #deal with kfold training data
    folds =[] #non cluster rand state 894, cluster rand state 1658 (tested)
    fold_index=[]
    kf = KFold(n_splits=5,random_state=rand_seed,shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(clusters_lst)):
        print("fold ",i)
        trainIdx = [clusters_lst[i] for i in train_index.tolist()]
        testIdx = [clusters_lst[i] for i in test_index.tolist()]
        trainIdx = [j for i in trainIdx for j in i]
        testIdx = [j for i in testIdx for j in i]
        fold_index.append([trainIdx,testIdx])
        train_idx = [TRAIN_MAP[idx] for idx in trainIdx if idx in TRAIN_MAP.keys()]#map id into index
        val_idx = [TRAIN_MAP[idx] for idx in testIdx if idx in TRAIN_MAP.keys()]
        train_dist,test_dist,score = check_dist(trainIdx,testIdx,TRAIN_DATA)
        folds.append([train_idx,val_idx,(train_dist,test_dist)])
    #save kfold index
    with open(train_idx_save,"w")as f:
        json.dump(fold_index,f)
    #save kfold data
    save_dic ={}
    for idx,i in enumerate(folds):
        trainIdx, testIdx = i[0],i[1]
        train_pair = claim_with_wiki_content(trainIdx,TRAIN_DATA,mapping,mode="train",with_page_name=True,with_evids=True)
        val_pair = claim_with_wiki_content(testIdx,TRAIN_DATA,mapping,mode="train",with_page_name=True,with_evids=True)
        #save_dic["info"]["fold"+str(idx)]=i[2]
        save_dic["train_f"+str(idx)]=train_pair
        save_dic["val_f"+str(idx)]=val_pair
    with open(train_save,"w")as f:
        json.dump(save_dic,f)
    #deal with TEST DATA
    test_pair = claim_with_wiki_content(list(range(len(TEST_DATA))),TEST_DATA,mapping,mode="test",with_page_name=True)
    with open(test_save,"w")as f:
        json.dump({"test":test_pair},f)
    print("process done")
    