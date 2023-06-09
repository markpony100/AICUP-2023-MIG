python3 evaluate_reranker.py --pred_path ../pert_large/e10_b8_lr1e-5_wd001/train_test.json\
    --thresh 0.01\
    --save_path ../pert_large/train_reranked.jsonl
#    
#python3 evaluate_reranker.py --pred_path ../pert_large/e10_b8_lr1e-5_wd001/test.json\
#    --thresh 0.01\
#    --save_path ../pert_large/test_reranked.jsonl\
#    --predict_only True\
#    --ori_path ../../baseline/data/test_doc5.jsonl
