import json
sys.path.append('../')

from utils import (
    load_json,jsonl_dir_to_df,
    load_simple_json,generate_evidence_to_wiki_pages_mapping
)
def find_identical(data,test):
    re_lst=[]
    for i in test:
        for j in data:
            if i["claim"].replace(" ","") == j["claim"].replace(" ",""):
                re_lst.append({"id":i["id"],"label":j["label"]})
    return re_lst
if __name__ == "__main__":
    #load data
    train_data = load_json("../preprocess/train_all.jsonl")
    test_data = load_json("../preprocess/test_all.jsonl")
    pred_data = load_json("./page_sent_claim_clu_enough_nclu_prob.jsonl")
    enough_pred = load_json("../preprocess/cluster/ans_cc_cluster.jsonl")
    #find identical claim and copy their label
    script = find_identical(train_data,test_data)
    for i in script:
        if i["id"] in pred_map:
            if pred_data[pred_map[i["id"]]]["predicted_label"]!=i["label"]:
                #print(i)
                pred_data[pred_map[i["id"]]]["predicted_label"]=i["label"]
    #preprocess cluster logistic enough data
    enough_dic = {i["id"]:i["predicted_label"] for i in enough_pred}
    for i in enough_dic:
        if i in pred_map.keys():
            if enough_dic[i]=="NOT ENOUGH INFO":
                pred_data[pred_map[i]]['cluster_enough']=0
            else:
                pred_data[pred_map[i]]['cluster_enough']=1
        else:
            0
    #if enough not sure use cluster logistic
    for i in pred_data:
        if abs(i["enough"]-0.5)<0.4:
            i["enough"]=i['cluster_enough']
        else:
            i["enough"]=1 if i["enough"]>0.5 else 0
    #mapping labels
    c_lst=[]
    for i in pred_data:#mapping to labels
        if i["enough"]:
            if i['predict_label']>0.5:
                i["predicted_label"]="refutes"
            else:
                i["predicted_label"]="supports"
        else:
            i["predicted_label"]="NOT ENOUGH INFO"
        c_lst.append(i["predicted_label"])
    print("total: ",len(c_lst))
    print("ratio: ",c_lst.count("supports"),c_lst.count("refutes"),c_lst.count("NOT ENOUGH INFO"))
    #add not found data
    ori_data = load_json("../preprocess/all_test.jsonl")
    mapping_idx = {d["id"]:idx for idx,d in enumerate(pred_data)}
    re_lst = []
    for d in ori_data:
        if d["id"] in mapping_idx.keys():
            data = pred_data[mapping_idx[d["id"]]]
            re_lst.append({"id":data["id"],
                           "predicted_label":data["predicted_label"],
                           "predicted_evidence":data["predicted_evidence"]})
        else:
            re_lst.append({"id":d["id"],"predicted_label":"NOT ENOUGH INFO",
                           "predicted_evidence":[]})
    with open("../submissions/best_score.jsonl","w",encoding="utf8",) as f:
        for i, d in enumerate(re_lst):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
