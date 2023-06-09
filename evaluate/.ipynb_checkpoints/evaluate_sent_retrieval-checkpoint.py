import json
import argparse
from dataclasses import dataclass
import pandas as pd
import sys
sys.path.append('../')
from utils import load_simple_json,load_json
def evidence_macro_precision(instance: dict,top_rows: pd.DataFrame,) -> tuple:
    """Calculate precision for sentence retrieval
    This function is modified from fever-scorer.
    https://github.com/sheffieldnlp/fever-scorer/blob/master/src/fever/scorer.py

    Args:
        instance (dict): a row of the dev set (dev.jsonl) of test set (test.jsonl)
        top_rows (pd.DataFrame): our predictions with the top probabilities

        IMPORTANT!!!
        instance (dict) should have the key of `evidence`.
        top_rows (pd.DataFrame) should have a column `predicted_evidence`.

    Returns:
        Tuple[float, float]:
        [1]: relevant and retrieved (numerator of precision)
        [2]: retrieved (denominator of precision)
    """
    this_precision = 0.0
    this_precision_hits = 0.0

    # Return 0, 0 if label is not enough info since not enough info does not
    # contain any evidence.
    if instance["label"].upper() != "NOT ENOUGH INFO":
        # e[2] is the page title, e[3] is the sentence index
        all_evi = [[e[2], e[3]]
                   for eg in instance["evidence"]
                   for e in eg
                   if e[3] is not None]
        claim = instance["claim"]
        predicted_evidence = top_rows[top_rows["claim"] == claim]["predicted_evidence"].tolist()

        for prediction in predicted_evidence:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

        return (this_precision /
                this_precision_hits) if this_precision_hits > 0 else 1.0, 1.0

    return 0.0, 0.0
def evidence_macro_recall(instance:dict,top_rows: pd.DataFrame,) -> tuple:
    """Calculate recall for sentence retrieval
    This function is modified from fever-scorer.
    https://github.com/sheffieldnlp/fever-scorer/blob/master/src/fever/scorer.py

    Args:
        instance (dict): a row of the dev set (dev.jsonl) of test set (test.jsonl)
        top_rows (pd.DataFrame): our predictions with the top probabilities

        IMPORTANT!!!
        instance (dict) should have the key of `evidence`.
        top_rows (pd.DataFrame) should have a column `predicted_evidence`.

    Returns:
        Tuple[float, float]:
        [1]: relevant and retrieved (numerator of recall)
        [2]: relevant (denominator of recall)
    """
    # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"].upper() != "NOT ENOUGH INFO":
        # If there's no evidence to predict, return 1
        if len(instance["evidence"]) == 0 or all(
            [len(eg) == 0 for eg in instance]):
            return 1.0, 1.0

        claim = instance["claim"]

        predicted_evidence = top_rows[top_rows["claim"] ==
                                      claim]["predicted_evidence"].tolist()

        for evidence_group in instance["evidence"]:
            evidence = [[e[2], e[3]] for e in evidence_group]
            if all([item in predicted_evidence for item in evidence]):
                # We only want to score complete groups of evidence. Incomplete
                # groups are worthless.
                return 1.0, 1.0
        return 0.0, 1.0
    return 0.0, 0.0
def evaluate_retrieval(df_evidences: pd.DataFrame,ground_truths: pd.DataFrame,
    cal_scores: bool = True,save_name: str = None,) -> dict:
    """Calculate the scores of sentence retrieval

    Args:
        probs (np.ndarray): probabilities of the candidate retrieved sentences
        df_evidences (pd.DataFrame): the candiate evidence sentences paired with claims
        ground_truths (pd.DataFrame): the loaded data of dev.jsonl or test.jsonl
        top_n (int, optional): the number of the retrieved sentences. Defaults to 2.

    Returns:
        Dict[str, float]: F1 score, precision, and recall
    """
    #df_evidences["prob"] = probs
    top_rows = df_evidences
    if cal_scores:
        macro_precision = 0
        macro_precision_hits = 0
        macro_recall = 0
        macro_recall_hits = 0

        for i, instance in enumerate(ground_truths):
            macro_prec = evidence_macro_precision(instance, top_rows)
            macro_precision += macro_prec[0]
            macro_precision_hits += macro_prec[1]

            macro_rec = evidence_macro_recall(instance, top_rows)
            macro_recall += macro_rec[0]
            macro_recall_hits += macro_rec[1]

        pr = (macro_precision /
              macro_precision_hits) if macro_precision_hits > 0 else 1.0
        rec = (macro_recall /
               macro_recall_hits) if macro_recall_hits > 0 else 0.0
        f1 = 2.0 * pr * rec / (pr + rec)

    if save_name is not None:
        # write doc7_sent5 file
        with open(f"data/{save_name}", "w") as f:
            for instance in ground_truths:
                claim = instance["claim"]
                predicted_evidence = top_rows[
                    top_rows["claim"] == claim]["predicted_evidence"].tolist()
                instance["predicted_evidence"] = predicted_evidence
                f.write(json.dumps(instance, ensure_ascii=False) + "\n")

    if cal_scores:
        return {"F1 score": f1, "Precision": pr, "Recall": rec}


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
            [buf_lst.append(pred[qid]["predicted_evidence"][scores.index(i)]) for i in sorted_lst]
        else:
            sorted_lst = sorted(scores,reverse=True)[:]
            [buf_lst.append(pred[qid]["predicted_evidence"][scores.index(value)]) for idx,value in
                    enumerate(sorted_lst) if value >= thresh]
        if len(buf_lst)>5:
            buf_lst = buf_lst[:5]
        re_dic[int(qid)]=buf_lst
    re_lst = []
    for key in re_dic:
        claim = pred[str(key)]["claim"]
        for sent in re_dic[key]:
            re_lst.append({"qid":key,"claim":claim,"predicted_evidence":sent,"prob":1})
    return re_lst,re_dic


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    #path args
    parser.add_argument("--thresh",default=0.5,type=float,)
    parser.add_argument("--pred_path",
                        default="../pert_large/sent/doc100_cluster_e20_b8_recall/test.json",
                        type=str,)
    parser.add_argument("--ori_path",
                        default="../../baseline/data/test_doc100_ensemble.jsonl",
                        type=str,)
    parser.add_argument("--save_path",
                default="../pert_large/sent_data/test_doc100_cluster_rerank_thr001_sent_top5_.jsonl",
                        type=str,help="jsonl file")
    parser.add_argument('--topk', default=None, type=int,)
    parser.add_argument('--predict_only', default=False, type=bool,)
    args = parser.parse_args()
    preds = load_simple_json(args.pred_path)
    processed_preds,processed_dict = process_prediction(preds,thresh=args.thresh,topk=args.topk)
    qids = list(processed_dict.keys())
    ori_data = [i for i in load_json(args.ori_path)if (i["id"]) in qids]
    if not args.predict_only:
        print(evaluate_retrieval(pd.DataFrame(processed_preds),ori_data))
    if args.save_path:
        with open(args.save_path,"w",encoding="utf8",) as f:
            for i, d in enumerate(ori_data):
                d["predicted_evidence"] = processed_dict[d["id"]]
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(args.save_path, "saved")
    print("process done")
