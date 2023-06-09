import argparse
from pathlib import Path
import transformers
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
#transformers.logging.set_verbosity_error()#this disabled training progress bar also warnings
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import numpy as np
import json
import torch
def save_log(file_name,log_info):
    '''save in text format'''
    with open(file_name,'w')as f:
        f.write(log_info)
    return
def compute_metrics(eval_pred):
    pred, labels = eval_pred
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
def compute_metrics_multi_class(eval_pred):
    pred, labels = eval_pred
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred,average="micro")
    precision = precision_score(y_true=labels, y_pred=pred,average="micro")
    f1 = 2.0*recall*precision/(recall+precision)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
def tokenize_data(args,tokenizer,train,validation,test):
    concate_keys  = args.concate_keys.split("/")
    buf_lst = [train,validation,test]
    re_lst = []
    for dataset in buf_lst:
        if dataset:
            if len(concate_keys)==2: 
                re_lst.append(dataset.map(lambda examples:
                    tokenizer(examples[concate_keys[0]],examples[concate_keys[1]],max_length=args.max_length,truncation_strategy="longest_first"), batched=True))
            else:
                re_lst.append(dataset.map(lambda examples:
                    tokenizer(examples[concate_keys[0]],max_length=args.max_length,truncation_strategy="longest_first"), batched=True))
        else: re_lst.append({})
    return re_lst
def save_predict_prob(data,predictions,save_path,mode="page"):
    buf_dic={}
    if mode == "page":
        probs = torch.softmax(torch.tensor(predictions), dim=1)[:, 1].tolist()
        for page_id,qid,claim,prob in zip(data["page_id"],data["qid"],data["sentence1"],probs):
            if qid not in buf_dic.keys():
                buf_dic[qid]={"page_ids":[page_id],"claim":claim,"score":[prob]}
            else:
                buf_dic[qid]['page_ids']+=[page_id]
                buf_dic[qid]['score']+=[prob]
    elif mode=="claim":
        preds = np.argmax(predictions, axis=1)
        probs = torch.softmax(torch.tensor(predictions), dim=1)[:, 1].tolist()
        idx = 0
        for qid,claim,pred in zip(data["qid"],data["sentence1"],preds):
            buf_dic[qid]={"claim":claim,"predict_label":int(pred),"prob":probs[idx]}
            idx+=1
    else:
        probs = torch.softmax(torch.tensor(predictions), dim=1)[:, 1].tolist()
        for pred_page,sent_id,qid,claim,prob in zip(data["predicted_page"],data["sent_id"],data["qid"],data["sentence1"],probs):
            if qid not in buf_dic.keys():
                buf_dic[qid]={"predicted_evidence":[[pred_page,sent_id]],"claim":claim,"score":[prob]}
            else:
                buf_dic[qid]['predicted_evidence'].append([pred_page,sent_id])
                buf_dic[qid]['score']+=[prob]
    with open(save_path,'w') as f:
        json.dump(buf_dic,f)
def main(args):
    #set gpu usage
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    #path check
    if not Path(args.output_path).exists():
        Path(args.output_path).mkdir(parents=True)

    #training setting
    train_dataset,val_dataset,test_dataset={},{},{}
    if not args.predict_only:
        if not args.eval_only:
            print("loading training data")
            train_dataset = load_dataset("json", data_files=args.data_path
            ,field=args.train_key)["train"]
        print("loading validation data")
        val_dataset = load_dataset("json", data_files=args.data_path ,field=args.val_key)["train"]
    else:
        print("loading test data")
        test_dataset = load_dataset("json", data_files=args.data_path
        ,field=args.predict_key)["train"]
    #using glue metric for f1 ro acc
    #define tokenizer
    print("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    #tokenize data
    print("tokenzing data")
    tokenzied_data = tokenize_data(args,tokenizer,train_dataset,val_dataset,test_dataset)
    token_train,token_val,token_test = tokenzied_data
    #load pretrained model if provided
    if len(args.model_path):
        print("loading prtrained model")
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path,num_labels=args.num_class)
    else:
        print("loading base model from huggingface")
        model = AutoModelForSequenceClassification.from_pretrained(args.base_model,num_labels=args.num_class)
    #train augment

    train_args = TrainingArguments(args.output_path+"/train/",evaluation_strategy = args.evaluation_strategy,
                save_strategy = args.save_strategy,save_steps = args.save_steps,learning_rate=args.lr,
                per_device_train_batch_size=args.batch_size,per_device_eval_batch_size=args.batch_size,
                num_train_epochs=args.epoch,weight_decay=args.weight_decay,load_best_model_at_end=True,
                metric_for_best_model=args.metric_for_best_model,save_total_limit= args.save_total_limit,
                eval_steps = args.eval_steps
                )
    if args.num_class>2:
        trainer = Trainer(model,train_args,train_dataset=token_train,
        eval_dataset=token_val,tokenizer=tokenizer,compute_metrics=compute_metrics_multi_class,
        )
    else:#binary
        trainer = Trainer(model,train_args,train_dataset=token_train,
        eval_dataset=token_val,tokenizer=tokenizer,compute_metrics=compute_metrics,
        )
        
    if not args.predict_only:
        if not args.eval_only:
            print("training ...")
            trainer.train()
            model.save_pretrained(args.output_path+"/best_model/")
            print("evaluating...")
            eval_result = trainer.evaluate()
            print(eval_result)
            save_log(args.output_path+"/evaluate.txt",str(args)+"\n"+str(eval_result))
        else:
            print("evaluating...")
            eval_result = trainer.evaluate()
            print(eval_result)
            save_log(args.output_path+"/evaluate1.txt",str(args)+"\n"+str(eval_result))
    else:
        print("predicting...")
        predictions, labels, _  = trainer.predict(token_test)
        save_predict_prob(test_dataset,predictions,args.output_path+args.prediction_name+".json",mode=args.mode)
    print("process completed")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #mode args
    parser.add_argument("--mode",default="page",type=str,help="page,sent mode")
    parser.add_argument("--predict_only",default=False,type=bool,help="predict only")
    parser.add_argument("--eval_only",default=False,type=bool,help="eval only")

    #dataset args
    parser.add_argument('--data_path', default="./dataset.json", type=str,
                        help='dataset path')
    parser.add_argument('--concate_keys', default="sentence1/sentence2", type=str,
                        help='concate if needed, need to be saved as key in json')
    parser.add_argument('--train_key', default="train",type=str,
                        help="keyword of training data in json")
    parser.add_argument('--val_key', default="val",type=str,
                        help="keyword of validation data in json")
    parser.add_argument('--predict_key', default="test",type=str,
                        help="keyword of test data in json(")
    #training parameter
    parser.add_argument('--gpus', default="0", type=str,)
    parser.add_argument('--base_model', default="bert-base-chinese", type=str,)
    parser.add_argument('--model_path', default="", type=str,)
    parser.add_argument('--max_length', default=512, type=int,)
    parser.add_argument('--num_class', default=2, type=int,)
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='learning rate')
    parser.add_argument('--epoch', default=10, type=int,
                        help='epoch')
    parser.add_argument('--batch_size', default=32, type=int,)
    parser.add_argument('--evaluation_strategy', default="steps", type=str,)
    parser.add_argument('--save_strategy', default="steps", type=str,)
    parser.add_argument('--save_steps', default=500, type=int,
                        help="steps for saving")
    parser.add_argument('--eval_steps', default=500, type=int,
                        help="steps for evaluate")
    parser.add_argument('--weight_decay', default=0.01, type=float,)
    parser.add_argument('--metric_for_best_model', default="f1", type=str,)
    parser.add_argument('--save_total_limit', default=2, type=int,)

    #output args
    parser.add_argument('--output_path', default="./output/",type=str,
                        help="save model/predict path")
    parser.add_argument('--prediction_name', default="prediction",type=str,
                        help="save predict filename")
    args = parser.parse_args()
    print(args)
    main(args)
