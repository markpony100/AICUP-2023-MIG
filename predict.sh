# python3 bert_main.py --data_path ./pert_large/claim_data/neg_no_pid_test_ens5.json\
#     --gpus 3\
#     --batch_size 8\
#     --output_path pert_large/claim/0522_w_noise_npid_folds/\
#     --max_length 512\
#     --base_model "hfl/chinese-pert-large"\
#     --model_path pert_large/claim/0522_w_noise_npid_folds//fold0/best_model\
#     --predict_only True\
#     --prediction_name test_f0\
#     --predict_key test\
#     --num_class 2\
#     --mode claim
# python3 bert_main.py --data_path ./pert_large/enough_data/0522_base_clu_PN13_test_ens5_no_pid.json\
#     --gpus 3\
#     --batch_size 8\
#     --output_path pert_large/enough/0522_fix_no_pid_folds/\
#     --max_length 512\
#     --base_model "hfl/chinese-pert-large"\
#     --model_path pert_large/enough/0522_fix_no_pid_folds/fold0/best_model\
#     --predict_only True\
#     --prediction_name test_f0\
#     --predict_key test\
#     --num_class 2\
#     --mode claim 

# python3 bert_main.py --data_path ./pert_large/claim_data/0522_03_fix_pub_test_v1.json\
#     --gpus 0\
#     --batch_size 8\
#     --output_path pert_large/claim/0522_03_w_noise/\
#     --max_length 512\
#     --base_model "hfl/chinese-pert-large"\
#     --model_path pert_large/claim/0522_03_w_noise/best_model\
#     --predict_only True\
#     --prediction_name pub_test_v1\
#     --predict_key test\
#     --num_class 2\
#     --mode claim 



#page prediction
# python3 bert_main.py --data_path pert_large/page_data/0522_base_test.json\
#     --gpus 1\
#     --batch_size 8\
#     --output_path ./pert_large/page/0522_base_clu_folds/\
#     --max_length 512\
#     --base_model "hfl/chinese-pert-large"\
#     --model_path ./pert_large/page/0522_base_clu_folds/fold1/best_model\
#     --prediction_name test_f1\
#     --predict_key test\
#     --mode page\
#     --predict_only True 


    
python3 bert_main.py --data_path pert_large/sent_data2/test_ens5_w_evid_pid.json\
    --gpus 3\
    --batch_size 8\
    --output_path ./pert_large/sent2/PN13_w_evid_pid_folds/\
    --max_length 512\
    --base_model "hfl/chinese-pert-large"\
    --model_path ./pert_large/sent2/PN13_w_evid_pid_folds/fold0/best_model\
    --prediction_name pub_test_f0\
    --predict_key test\
    --mode sent\
    --predict_only True &
python3 bert_main.py --data_path pert_large/sent_data2/test_ens5_w_evid_pid.json\
    --gpus 3\
    --batch_size 8\
    --output_path ./pert_large/sent2/PN13_w_evid_pid_folds/\
    --max_length 512\
    --base_model "hfl/chinese-pert-large"\
    --model_path ./pert_large/sent2/PN13_w_evid_pid_folds/fold1/best_model\
    --prediction_name pub_test_f1\
    --predict_key test\
    --mode sent\
    --predict_only True &
python3 bert_main.py --data_path pert_large/sent_data2/test_ens5_w_evid_pid.json\
    --gpus 3\
    --batch_size 8\
    --output_path ./pert_large/sent2/PN13_w_evid_pid_folds/\
    --max_length 512\
    --base_model "hfl/chinese-pert-large"\
    --model_path ./pert_large/sent2/PN13_w_evid_pid_folds/fold2/best_model\
    --prediction_name pub_test_f2\
    --predict_key test\
    --mode sent\
    --predict_only True &
python3 bert_main.py --data_path pert_large/sent_data2/test_ens5_w_evid_pid.json\
    --gpus 3\
    --batch_size 8\
    --output_path ./pert_large/sent2/PN13_w_evid_pid_folds/\
    --max_length 512\
    --base_model "hfl/chinese-pert-large"\
    --model_path ./pert_large/sent2/PN13_w_evid_pid_folds/fold3/best_model\
    --prediction_name pub_test_f3\
    --predict_key test\
    --mode sent\
    --predict_only True &
python3 bert_main.py --data_path pert_large/sent_data2/test_ens5_w_evid_pid.json\
    --gpus 3\
    --batch_size 8\
    --output_path ./pert_large/sent2/PN13_w_evid_pid_folds/\
    --max_length 512\
    --base_model "hfl/chinese-pert-large"\
    --model_path ./pert_large/sent2/PN13_w_evid_pid_folds/fold4/best_model\
    --prediction_name pub_test_f4\
    --predict_key test\
    --mode sent\
    --predict_only True 



#class prediction
# python3 bert_main.py --data_path ./easy_cls_03.json\
#     --gpus 0\
#     --batch_size 8\
#     --output_path pert_large/easy_cls/recall99/\
#     --max_length 256\
#     --base_model "hfl/chinese-pert-large"\
#     --model_path pert_large/easy_cls/recall99/best_model\
#     --predict_only True\
#     --predict_key test\
#     --prediction_name test\
#     --mode claim