pred_folder="0522_nclu_neg_no_pid_folds"
pred_data="all_test.json"
pred_key="all_test"
gpu="3"
python3 bert_main.py --data_path ./pert_large/enough_data/$pred_data\
    --gpus $gpu\
    --batch_size 8\
    --output_path pert_large/enough/$pred_folder/\
    --max_length 512\
    --base_model "hfl/chinese-pert-large"\
    --model_path pert_large/enough/$pred_folder/fold0/best_model\
    --predict_only True\
    --prediction_name "${pred_key}_f0"\
    --predict_key test\
    --num_class 2\
    --mode claim 
python3 bert_main.py --data_path ./pert_large/enough_data/$pred_data\
    --gpus $gpu\
    --batch_size 8\
    --output_path pert_large/enough/$pred_folder/\
    --max_length 512\
    --base_model "hfl/chinese-pert-large"\
    --model_path pert_large/enough/$pred_folder/fold1/best_model\
    --predict_only True\
    --prediction_name "${pred_key}_f1"\
    --predict_key test\
    --num_class 2\
    --mode claim 
python3 bert_main.py --data_path ./pert_large/enough_data/$pred_data\
    --gpus $gpu\
    --batch_size 8\
    --output_path pert_large/enough/$pred_folder/\
    --max_length 512\
    --base_model "hfl/chinese-pert-large"\
    --model_path pert_large/enough/$pred_folder/fold2/best_model\
    --predict_only True\
    --prediction_name "${pred_key}_f2"\
    --predict_key test\
    --num_class 2\
    --mode claim 
python3 bert_main.py --data_path ./pert_large/enough_data/$pred_data\
    --gpus $gpu\
    --batch_size 8\
    --output_path pert_large/enough/$pred_folder/\
    --max_length 512\
    --base_model "hfl/chinese-pert-large"\
    --model_path pert_large/enough/$pred_folder/fold3/best_model\
    --predict_only True\
    --prediction_name "${pred_key}_f3"\
    --predict_key test\
    --num_class 2\
    --mode claim 
python3 bert_main.py --data_path ./pert_large/enough_data/$pred_data\
    --gpus $gpu\
    --batch_size 8\
    --output_path pert_large/enough/$pred_folder/\
    --max_length 512\
    --base_model "hfl/chinese-pert-large"\
    --model_path pert_large/enough/$pred_folder/fold4/best_model\
    --predict_only True\
    --prediction_name "${pred_key}_f4"\
    --predict_key test\
    --num_class 2\
    --mode claim 





