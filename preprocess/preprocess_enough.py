import json
import pandas as pd
from pandarallel import pandarallel
import sys
sys.path.append('../')
pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=10)
from utils import (
    generate_evidence_to_wiki_pages_mapping,
    jsonl_dir_to_df,
    load_json,
    load_simple_json
)
import argparse
import itertools
def extract_evi_pair(input_list):
    """
    Extracts sublists from the input list based on the desired output format.

    Args:
    input_list (list): Input list of lists

    Returns:
    list: List of sublists in the desired output format
    """
    output_list = []
    for sub_list in input_list:
        if isinstance(sub_list, list):
            # For each sublist in the input list
            extracted_sublist = []
            for sublist in sub_list:
                # Extract the sublist based on the desired output format
                extracted_sublist.append((sublist[2:]))
            output_list.append(extracted_sublist)
        else:
            # For non-sublist elements in the input list
            output_list.append([(sub_list[2:])])
    return output_list
def join_with_evidence(df: pd.DataFrame,mapping: dict,
    mode: str = "train",topk: int = 5,with_pageID=True) -> list:
    """join_with_topk_evidence join the dataset with topk evidence.
    Args:
        df (pd.DataFrame): The dataset with evidence.
        wiki_pages (pd.DataFrame): The wiki pages dataframe
        topk (int, optional): The topk evidence. Defaults to 5.
        cache(Union[Path, str], optional): The cache file path. Defaults to None.
            If cache is None, return the result directly.
    Returns:
        pd.DataFrame: The dataset with topk evidence_list.
            The `evidence_list` column will be: List[str]
    """
    df["evidence_list"] = pd.Series([]).astype('object')
    # format evidence column to List[List[Tuple[str, str, str, str]]]
    if "evidence" in df.columns and mode == "train":
        df["evidence"] = df["evidence"].parallel_map(
            lambda x: [[x]] if not isinstance(x[0], list) else [x]
            if not isinstance(x[0][0], list) else x)

    #print(df["evidence"])
    print(f"Extracting evidence_list for the {mode} mode ...")
    if mode == "eval" or mode == "test":
        # extract evidence
         for index, row in df.iterrows():
            # Extract the "evidence" value from the current row
            pred_evids=[tuple(elem) for elem in row["predicted_evidence"]]
            # Initialize an empty list to store processed evidence
            evidence_list = []
            joined_evidence = []
            # Iterate over each element in the "evi_list" tuple
            for evi_id, evi_idx in pred_evids:
                # Get the corresponding value from the "mapping" dictionary
                # using "evi_id" and "evi_idx" as keys
                evi_value = mapping.get(evi_id, {}).get(str(evi_idx), "")
                # Join the evidence value to the "joined_evidence" string
                if with_pageID:
                    joined_evidence.append(evi_id+":"+evi_value)
                else:
                    joined_evidence.append(evi_value)
            # Add the joined evidence string to the "evidence_list"
                #vidence_list.append(evi_value)
            evidence_list=joined_evidence#+["PAD"]*(topk-len(joined_evidence))   # Remove trailing SEP
            evidence_list = "[SEP]".join(evidence_list[:topk])
            # Update the "evidence_list" column in the DataFrame with the processed evidence
            df.at[index, "evidence_list"] = evidence_list.replace(" ","")
    else:
        # Iterate over each row in the DataFrame
        re_lst = []
        for index, row in df.iterrows():
            # Extract the "evidence" value from the current row
            buf_dic={"id":row["id"],"claim":row["claim"],"label":row["label"]}
            evidence = row["evidence"]
            pred_evids=[(elem) for elem in row["predicted_evidence"]]
            # Initialize an empty list to store processed evidence
            evidence_list = []
            if row["label"] == "NOT ENOUGH INFO":#skip no info
                joined_evidence = []
                for evi_id, evi_idx in pred_evids:
                    evi_value = mapping.get(evi_id, {}).get(str(evi_idx), "")
                    if with_pageID:
                        joined_evidence.append(evi_id+":"+evi_value)
                    else:
                        joined_evidence.append(evi_value)
                evidence_list = "[SEP]".join(joined_evidence[:topk])
                buf_dic["evidence_list"] = evidence_list.replace(" ","")
                re_lst.append(buf_dic)
            else:#if sup or refu
                evidence_pair = extract_evi_pair(evidence)
                joined_evidence = []
                # Iterate over each element in the "evi_list" tuple
                for evids in evidence_pair:
                    evidss = evids
                    if topk:
                        add_pred = [pair for pair in pred_evids if pair not in evids]
                        evidss+=add_pred[:topk-len(evids)]
                    for evi_id, evi_idx in evidss:
                        evi_value = mapping.get(evi_id, {}).get(str(evi_idx), "")
                        if with_pageID:
                            joined_evidence.append(evi_id+":"+evi_value)
                        else:
                            joined_evidence.append(evi_value)
                    evidence_list = "[SEP]".join(joined_evidence[:topk])
                    buf_dic["evidence_list"] = evidence_list.replace(" ","")
                    re_lst.append(buf_dic)
        df = re_lst
    return df.to_dict(orient='records') if type(df) == pd.DataFrame else df
def add_neg(lst,with_preds=False):#augment with evidence or predicted pages
    re_lst = []
    for d in lst:
        if d["label"] =="NOT ENOUGH INFO":
            continue
        evidence_set = d["evidence"]
        predicted_evidence = [(elem) for elem in d["predicted_evidence"]]
        #evid_buf =[] # all evidence augment
        for evidences in evidence_set:
            evid_buf =[]
            for evidence in evidences:#set wise augment1
                evid_buf.append(list((evidence[2],evidence[3])))
            if with_preds:
                #evid_buf +=predicted_evidence
                evid_buf =+predicted_evidence
            if len(evid_buf)>1:#n >1 neg, >0 negv1, 
                del_evid = evid_buf
                neg_evid = [i for i in predicted_evidence if i not in del_evid]
                if len(neg_evid)>0:#<2 negv2
                    neg_evid = list(neg_evid)[:5]
                    re_lst.append({"id": d["id"], "label":"NOT ENOUGH INFO" , "claim":d["claim"] ,
                                   "evidence": d["evidence"], "predicted_evidence":neg_evid})
    return re_lst
def to_bert_format(data):
    label_mapping = {"supports": 1,"refutes": 1,"NOT ENOUGH INFO": 0}
    re_lst=[]
    count=[]
    for i in data:
        count.append(label_mapping[i["label"]])
        re_lst.append({"sentence1":i["claim"],"qid":i["id"],
                      "sentence2":i["evidence_list"],"label":label_mapping[i["label"]]})
    print("label distribution")
    for i in [0,1,2]:
        print(i,":",count.count(i),sep ="\t")
    return re_lst
if __name__ == "__main__":
    #parameter
    TRAIN_PATH = "../processed_data/all_ens5_top10.jsonl"
    KFOLD_IDX_PATH = "./train_fold_index.json"
    TRAIN_SAVE_PATH = "../pert_large/enough_data/train_folds.json"
    TEST_PATH = "../pert_large/sent/0522_base_clu_folds_PN13/all_test.jsonl"
    TEST_SAVE_PATH = "../pert_large/enough_data/all_test.json"
    #wiki mapping
    wiki_pages = jsonl_dir_to_df("../../baseline/data/wiki-pages")
    mapping = generate_evidence_to_wiki_pages_mapping(wiki_pages,)
    del wiki_pages
    #train
    TRAIN_DATA = load_json(TRAIN_PATH)
    TRAIN_MAP = {i["id"]:idx for idx,i in enumerate(TRAIN_DATA)}
    kfold_idx = load_simple_json(KFOLD_IDX_PATH)
    #Kfold
    save_dic={}
    for i,fold_idx in enumerate(kfold_idx):
        train_data = [TRAIN_DATA[TRAIN_MAP[idx]] for idx in fold_idx[0] if idx in TRAIN_MAP.keys()]
        val_data = [TRAIN_DATA[TRAIN_MAP[idx]] for idx in fold_idx[1] if idx in TRAIN_MAP.keys()]
        train_data=train_data+add_neg(train_data)
        val_data=val_data+add_neg(val_data)
        train = join_with_evidence(pd.DataFrame(train_data),mapping,mode="train",with_pageID=False)
        val = join_with_evidence(pd.DataFrame(val_data),mapping,mode="train",with_pageID=False)
        save_dic["train_f"+str(i)]=to_bert_format(train)
        save_dic["val_f"+str(i)]=to_bert_format(val)
    with open(TRAIN_SAVE_PATH,"w")as f:
        json.dump(save_dic,f)
    #TEST DATA
    TEST_DATA = load_json(TEST_PATH)
    test = join_with_evidence(pd.DataFrame(TEST_DATA),mapping,mode="test",with_pageID=True)
    for i in test:#use fake label for test data too lazy to adjust function
        i["label"]="supports"
    with open(TEST_SAVE_PATH,"w")as f:
        json.dump({"test":to_bert_format(test)},f)
    print("process done")