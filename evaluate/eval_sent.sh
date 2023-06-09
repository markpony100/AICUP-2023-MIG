python3 evaluate_sent_retrieval.py \
    --top 5 --pred_path ../pert_large/sent/e10_b8_512/train_test.json\
    --ori_path ../../baseline/data/train_doc5.jsonl\
    --save_path ../pert_large/sent_data/train_test_doc5_rerank_sent_top5.jsonl
# python3 evaluate_sent_retrieval.py \
#     --top 5 --pred_path ../pert_large/sent/e10_b8_512/val_test.json\
#     --ori_path ../../baseline/data/train_doc5.jsonl\
#     --save_path ../pert_large/sent_data/val_test_doc5_rerank_sent_top5.jsonl
# python3 evaluate_sent_retrieval.py \
#     --top 5 --pred_path ../pert_large/sent/e10_b8_512/test.json\
#     --ori_path ../../baseline/data/train_doc5.jsonl\
#     --save_path ../pert_large/sent_data/test_doc5_rerank_sent_top5.jsonl

# python3 evaluate_sent_retrieval.py \
#     --save_path ../pert_large/val_test_doc5_rerank_sent_thr001.jsonl\
#     --thresh 0.01 --pred_path ../pert_large/sent/e10_b8_512/val_test.json\
#     --ori_path ../../baseline/data/train_doc5.jsonl