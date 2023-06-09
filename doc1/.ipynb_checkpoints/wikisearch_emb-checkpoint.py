# Function: search the evidence from wikipedia
import pickle
from pathlib import Path
import opencc
import hanlp
import pandas as pd
import numpy as np
import json
import os
import argparse
import logging
from collections import Counter
from utils.WikiSearch import WikiSearch
# %%
predictor = (hanlp.pipeline().append(
    hanlp.load("FINE_ELECTRA_SMALL_ZH"),
    output_key="tok",
).append(
    hanlp.load("CTB9_CON_ELECTRA_SMALL"),
    output_key="con",
    input_key="tok",
))

# %%
CONVERTER_T2S = opencc.OpenCC("t2s.json")
CONVERTER_S2T = opencc.OpenCC("s2t.json")


def do_st_corrections(text: str) -> str:
    simplified = CONVERTER_T2S.convert(text)

    return CONVERTER_S2T.convert(simplified)


def get_nps_hanlp(
        predictor, d):
    claim = d
    tree = predictor(claim)["con"]
    nps = [
        do_st_corrections("".join(subtree.leaves()))
        for subtree in tree.subtrees(lambda t: t.label() == "NP")
    ]
    return nps


# %%


def han_build(filename, data):
    hanlp_file = f"data/{filename}.pkl"
    if Path(hanlp_file).exists():
        with open(hanlp_file, "rb") as f:
            hanlp_results = pickle.load(f)
    else:
        hanlp_results = [get_nps_hanlp(predictor, d) for d in data.claim]
        with open(hanlp_file, "wb") as f:
            pickle.dump(hanlp_results, f)
    return hanlp_results


def write_file(filename, data):
    re_lst = [dict(data.iloc[i]) for i in range(len(data.index))]
    with open(filename, "w", encoding="utf8") as f:
        for i, d in enumerate(re_lst):
            d['id'] = int(d['id'])
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

# %%


def check_ans(ans_list, answer, acc):
    correct_answer = []
    for i in ans_list:
        correct_answer.append(i[2])
        if i[2] == None:
            acc['None'] += 1
        elif i[2] in answer:
            acc['correct'] += 1
        else:
            acc['incorrect'] += 1
    logging.info('answer:' + str(correct_answer))
    return correct_answer


def res_ans(save_filename, hanlp, trainData, train=True):
    acc = Counter()
    if Path(save_filename).exists():
        backup_df = pd.read_json(path_or_buf=save_filename, lines=True)
        answers = list(backup_df.result)
        correct_answers = list(backup_df.correct)
        q_id = list(backup_df.id)
        start = len(q_id)
    else:
        answers = []
        correct_answers = []
        q_id = []
        start = 0
    for index, item in enumerate(hanlp[start:]):
        index += start
        answer = serch.wikiSearch(item, trainData.claim[index])
        logging.info(str(index)+str(answer))
        if train == True:
            if type(trainData.evidence[index][0][0]) != list:
                correct_answer = check_ans(
                    trainData.evidence[index], answer, acc)
            else:
                correct_answer = check_ans(
                    trainData.evidence[index][0], answer, acc)
            correct_answers.append(correct_answer)
        else:
            correct_answers.append(None)
        answers.append(answer)
        q_id.append(trainData['id'][index])
        if index % 50 == 0:
            trainAnswer = pd.DataFrame(
                {'id': q_id, 'result': answers, 'correct': correct_answers})
            write_file(save_filename, trainAnswer)
    trainAnswer = pd.DataFrame(
        {'id': q_id, 'result': answers, 'correct': correct_answers})
    write_file(save_filename, trainAnswer)

# %%


def calculate_precision(correct_labels, data, predictions):
    precision = 0
    count = 0
    for i, d in enumerate(data):
        if correct_labels[i] == "NOT ENOUGH INFO":
            continue

        # Extract all ground truth of titles of the wikipedia pages
        # evidence[2] refers to the title of the wikipedia page
        gt_pages = set(list(d))
        predicted_pages = set(list(predictions[i]))
        hits = predicted_pages.intersection(gt_pages)
        if len(predicted_pages) != 0:
            precision += len(hits) / len(predicted_pages)
        count += 1

    # Macro precision
    print(f"Precision: {precision / count}")


def calculate_recall(correct_labels, data, predictions):
    recall = 0
    count = 0

    for i, d in enumerate(data):
        if correct_labels[i] == "NOT ENOUGH INFO":
            continue
        gt_pages = set(list(d))
        predicted_pages = set(list(predictions[i]))
        hits = predicted_pages.intersection(gt_pages)
        recall += len(hits) / len(gt_pages)
        count += 1
    print(f"Recall: {recall / count}")


def load_wikidata(wikidata_dir):
    if not wikidata_dir.endswith('/'):
        wikidata_dir += '/'
    if Path('dataset/wikidata.csv').exists():
        wikiData = pd.read_csv('dataset/wikidata.csv')
    else:
        pathJson = os.listdir(wikidata_dir)
        print(pathJson)
        wikiData = pd.DataFrame()
        for i in pathJson:
            jsonObj = pd.read_json(path_or_buf=wikidata_dir+i, lines=True)
            wikiData = pd.concat([jsonObj, wikiData],
                                 join='outer', ignore_index=True)
            del jsonObj
        wikiData.to_csv('dataset/wikidata.csv', index=False)
    return wikiData


# %%
# logging.basicConfig(level=logging.DEBUG)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("claim_data", type=str, default='../raw_data/public_train_0522.jsonl',
                        help="set the claim data, default is public_train_0316.jsonl")
    parser.add_argument("-w", "--wikidata_dir", default='../wiki-pages', type=str,
                        help="set the wikidata directory, default is ../wiki-pages")
    parser.add_argument("-t", "--task_mode", type=str, default='train',
                        help="set the task mode, train or test, default is train")
    parser.add_argument("-o", "--output_filename", type=str, default='_wikisearch_stage2emb.jsonl',
                        help="set the output filename, default is type_wikisearch_stage2emb.jsonl")
    parser.add_argument("-l", "--log_filename", type=str, default='log/wikisearch_stage2emb_',
                        help="set the log filename, default is wikisearch_stage2emb_type.log")
    args = parser.parse_args()
    print(args)
    FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    log_filename = args.log_filename + args.task_mode + '.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename, filemode='w', format=FORMAT)
    wikiData = load_wikidata(args.wikidata_dir)
    claim_data = pd.read_json(path_or_buf=args.claim_data, lines=True)
    # print(claim_data)
    # exit()
    serch = WikiSearch(wikiData)
    filename = args.output_filename
    han_filename = args.claim_data.split('/')[-1].split('.')[0] + '_han'
    han_result = han_build(han_filename, claim_data)
    if args.task_mode == 'train':
        res_ans(filename, han_result, claim_data, True)
        backup = pd.read_json(path_or_buf=filename, lines=True)
        calculate_precision(claim_data.label, backup.correct, backup.result)
        calculate_recall(claim_data.label, backup.correct, backup.result)
    else:
        res_ans(filename, han_result, claim_data, False)
