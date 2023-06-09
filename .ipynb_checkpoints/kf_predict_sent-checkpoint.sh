pred_folder="0522_base_clu_folds_PN13"
pred_data="all_test.json"
pred_key="all_test"
gpu="2"
python3 bert_main.py --data_path pert_large/sent_data/$pred_data\
    --gpus $gpu\
    --batch_size 8\
    --output_path ./pert_large/sent/$pred_folder/\
    --max_length 512\
    --base_model "hfl/chinese-pert-large"\
    --model_path ./pert_large/sent/$pred_folder/fold0/best_model\
    --prediction_name "${pred_key}_f0"\
    --predict_key test\
    --mode sent\
    --predict_only True 
python3 bert_main.py --data_path pert_large/sent_data/$pred_data\
    --gpus $gpu\
    --batch_size 8\
    --output_path ./pert_large/sent/$pred_folder/\
    --max_length 512\
    --base_model "hfl/chinese-pert-large"\
    --model_path ./pert_large/sent/$pred_folder/fold1/best_model\
    --prediction_name "${pred_key}_f1"\
    --predict_key test\
    --mode sent\
    --predict_only True 
python3 bert_main.py --data_path pert_large/sent_data/$pred_data\
    --gpus $gpu\
    --batch_size 8\
    --output_path ./pert_large/sent/$pred_folder/\
    --max_length 512\
    --base_model "hfl/chinese-pert-large"\
    --model_path ./pert_large/sent/$pred_folder/fold2/best_model\
    --prediction_name "${pred_key}_f2"\
    --predict_key test\
    --mode sent\
    --predict_only True 
python3 bert_main.py --data_path pert_large/sent_data/$pred_data\
    --gpus $gpu\
    --batch_size 8\
    --output_path ./pert_large/sent/$pred_folder/\
    --max_length 512\
    --base_model "hfl/chinese-pert-large"\
    --model_path ./pert_large/sent/$pred_folder/fold3/best_model\
    --prediction_name "${pred_key}_f3"\
    --predict_key test\
    --mode sent\
    --predict_only True 
python3 bert_main.py --data_path pert_large/sent_data/$pred_data\
    --gpus $gpu\
    --batch_size 8\
    --output_path ./pert_large/sent/$pred_folder/\
    --max_length 512\
    --base_model "hfl/chinese-pert-large"\
    --model_path ./pert_large/sent/$pred_folder/fold4/best_model\
    --prediction_name "${pred_key}_f4"\
    --predict_key test\
    --mode sent\
    --predict_only True 


