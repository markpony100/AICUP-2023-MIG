folder="0522_base_clu_folds"
data="train_cluster_folds.json"
python3 bert_main.py --data_path ./pert_large/page_data/$data\
    --gpus 0\
    --epoch 2\
    --batch_size 8\
    --output_path pert_large/page/$folder/fold0\
    --train_key train_f0\
    --val_key val_f0\
    --base_model "hfl/chinese-pert-large"\
    --metric_for_best_model "recall"  
python3 bert_main.py --data_path ./pert_large/page_data/$data\
    --gpus 0\
    --epoch 2\
    --batch_size 8\
    --output_path pert_large/page/$folder/fold1\
    --train_key train_f1\
    --val_key val_f1\
    --base_model "hfl/chinese-pert-large"\
    --metric_for_best_model "recall"  
python3 bert_main.py --data_path ./pert_large/page_data/$data\
    --gpus 0\
    --epoch 2\
    --batch_size 8\
    --output_path pert_large/page/$folder/fold2\
    --train_key train_f2\
    --val_key val_f2\
    --base_model "hfl/chinese-pert-large"\
    --metric_for_best_model "recall"  
python3 bert_main.py --data_path ./pert_large/page_data/$data\
    --gpus 0\
    --epoch 2\
    --batch_size 8\
    --output_path pert_large/page/$folder/fold3\
    --train_key train_f3\
    --val_key val_f3\
    --base_model "hfl/chinese-pert-large"\
    --metric_for_best_model "recall"  
python3 bert_main.py --data_path ./pert_large/page_data/$data\
    --gpus 0\
    --epoch 2\
    --batch_size 8\
    --output_path pert_large/page/$folder/fold4\
    --train_key train_f4\
    --val_key val_f4\
    --base_model "hfl/chinese-pert-large"\
    --metric_for_best_model "recall"  

