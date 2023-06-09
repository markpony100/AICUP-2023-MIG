import argparse
import json
import sys
sys.path.append('../')
from utils import (
    load_json,jsonl_dir_to_df,
    load_simple_json
)
if __name__ == "__main__":
    #parameter
    CLUSTER_PATH = "./cluster/processing_file/clusters.json"
    TRAIN_PATH = "../raw_data/public_train.jsonl"
    TRAIN_PATH1 = "../raw_data/public_train_0522.jsonl"
    PRI_TEST_PATH = "../raw_data/private_test_data.jsonl"
    PUB_TEST_PATH = "../raw_data/public_test.jsonl"
    SAVE_PATH = "train_all.jsonl"
    TEST_SAVE_PATH = "test_all.jsonl"
    #load data
    clusters_lst = load_simple_json(CLUSTER_PATH)["clusters"]
    TRAIN_DATA = load_json(TRAIN_PATH)+load_json(TRAIN_PATH1)
    #preprocess data
    for i in TRAIN_DATA:
        i["claim"]= i["claim"].replace(" ","")
    #remove duplicated claim or duplicated claim with different label
    #hard coded changes
    TRAIN_DATA[1821]['label']="supports"
    TRAIN_DATA[7670]['label']='refutes'
    TRAIN_DATA[7950]['label']='refutes'
    TRAIN_DATA[6105]['label']='refutes'
    #find same claim
    same=[]
    found_same=[]
    for i in range(len(TRAIN_DATA)):
        same_buf=[]
        if i in found_same:
            continue
        for j in range(len(TRAIN_DATA)):
            if i!=j:
                if TRAIN_DATA[i]["claim"] == TRAIN_DATA[j]["claim"]:
                    same_buf.append((j,TRAIN_DATA[j]["claim"],TRAIN_DATA[j]["label"],TRAIN_DATA[j]["evidence"]))
                    found_same.append(j)
        if same_buf:
            same_buf.append((i,TRAIN_DATA[i]["claim"],TRAIN_DATA[i]["label"],TRAIN_DATA[i]["evidence"]))
            found_same.append(i)
            same.append(same_buf)
    #change NOT ENOGUH INFO label to supports or refutes
    for i in same:
        label  =[(j[2],j[3]) for j in i if j[2]!= 'NOT ENOUGH INFO']#do with supports or refutes data
        evids = [b[1] for b in label]
        label = [b[0] for b in label]
        if len(set(label))==1:#there's no opposite label
            all_evids = []
            for evids_ in evids:
                for evid_set in evids_:
                    evid_set_buf = []
                    for evid in evid_set:
                        evid_set_buf.append((evid[2],evid[3]))
                    all_evids.append(tuple(evid_set_buf))
            final_evid = []
            for set_evid in set(all_evids):
                evid_buf = []
                for buf in set_evid:
                    evid_buf.append([0,0,buf[0],buf[1]])
                final_evid.append(evid_buf)
            index = [j[0] for j in i]
            for idx in index :
                TRAIN_DATA[idx]["label"] =label[0]
                TRAIN_DATA[idx]["evidence"] =final_evid
    #remvoe duplicated data
    filtered_train = []
    same_lst = [j[0]for i in same for j in i]
    for idx,i in enumerate(TRAIN_DATA):
        if idx not in same_lst:
            filtered_train.append(i)
    for i in same:
        index = i[0][0]
        filtered_train.append(TRAIN_DATA[index])
    hit_str = "remove duplicated training data. original data size: {}, filtered data size: {}"
    print(hit_str.format(len(TRAIN_DATA),len(filtered_train)))
    with open(SAVE_PATH,"w",encoding="utf8",) as f:
        for i, d in enumerate(filtered_train):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    #deal with TEST DATA
    TEST_DATA = load_json(PRI_TEST_PATH)+load_json(PUB_TEST_PATH)
    print("test data size: {}".format(len(TEST_DATA)))
    with open(TEST_SAVE_PATH,"w",encoding="utf8",) as f:
        for i, d in enumerate(TEST_DATA):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("process done")