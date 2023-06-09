
#SUP or refutes training
# python3 bert_main.py --data_path ./pert_large/easy_claim/easy_claim.json\
#     --gpus 2\
#     --epoch 5\
#     --batch_size 8\
#     --output_path pert_large/easy_claim/easy_claim_02/\
#     --base_model "hfl/chinese-pert-large"\
#     --train_key train\
#     --val_key val\
#     --metric_for_best_model "f1"\
#     --num_class 3

#Enough training
python3 bert_main.py --data_path ./pert_large/enough_data/0522_fix_neg_no_pid_folds.json\
    --gpus 1\
    --epoch 5\
    --batch_size 4\
    --output_path pert_large/enough/deberta_test/fold4\
    --base_model "microsoft/DeBERTa-v3-large"\
    --metric_for_best_model "f1"\
    --num_class 2\
    --train_key train_f4\
    --val_key val_f4

# python3 bert_main.py --data_path ./pert_large/sent_data2/PN13_w_evid_pid_folds.json\
#     --gpus 1\
#     --epoch 5\
#     --batch_size 8\
#     --output_path pert_large/sent2/PN13_w_evid_pid_folds/fold2\
#     --base_model "hfl/chinese-pert-large"\
#     --train_key train_f2\
#     --val_key val_f2\
#     --metric_for_best_model "f1"\
#     --num_class 2 

# python3 bert_main.py --data_path ./pert_large/doc100_rerank_thr001.json\
#     --gpus 1\
#     --epoch 20\
#     --batch_size 8\
#     --save_steps 500\
#     --output_path pert_large/sent/doc100_e20_b8_recall/\
#     --base_model "hfl/chinese-pert-large"\
#     --metric_for_best_model "recall"
    



