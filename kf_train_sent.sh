pred_folder="0522_base_clu_folds_PN13"
pred_data="Ttrain_all.json"
gpu="0"
python3 bert_main.py --data_path ./pert_large/sent_data/$pred_data\
    --gpus 0\
    --epoch 5\
    --batch_size 8\
    --output_path pert_large/sent/$pred_folder/fold0\
    --base_model "hfl/chinese-pert-large"\
    --train_key train_f0\
    --val_key val_f0\
    --metric_for_best_model "recall"\
    --num_class 2 
python3 bert_main.py --data_path ./pert_large/sent_data/$pred_data\
    --gpus 0\
    --epoch 5\
    --batch_size 8\
    --output_path pert_large/sent/$pred_folder/fold1\
    --base_model "hfl/chinese-pert-large"\
    --train_key train_f1\
    --val_key val_f1\
    --metric_for_best_model "recall"\
    --num_class 2 
python3 bert_main.py --data_path ./pert_large/sent_data/$pred_data\
    --gpus 0\
    --epoch 5\
    --batch_size 8\
    --output_path pert_large/sent/$pred_folder/fold2\
    --base_model "hfl/chinese-pert-large"\
    --train_key train_f2\
    --val_key val_f2\
    --metric_for_best_model "recall"\
    --num_class 2 
python3 bert_main.py --data_path ./pert_large/sent_data/$pred_data\
    --gpus 0\
    --epoch 5\
    --batch_size 8\
    --output_path pert_large/sent/$pred_folder/fold3\
    --base_model "hfl/chinese-pert-large"\
    --train_key train_f3\
    --val_key val_f3\
    --metric_for_best_model "recall"\
    --num_class 2 
python3 bert_main.py --data_path ./pert_large/sent_data/$pred_data\
    --gpus 0\
    --epoch 5\
    --batch_size 8\
    --output_path pert_large/sent/$pred_folder/fold4\
    --base_model "hfl/chinese-pert-large"\
    --train_key train_f4\
    --val_key val_f4\
    --metric_for_best_model "recall"\
    --num_class 2 
