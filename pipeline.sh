
#preprocess data
echo preprocess data
cd preprocess
python3 preprocess_train.py

#cluster claim cost around 8~12 hr
#you can use processed one in processed data but put it in ./preprocess/cluster/processing_file
echo clustering
cd cluster
python3 full.py

#do document retrieval1
echo doc retrieval 1
cd ../../doc1
#will do wiki search and approximately cost more than 48hr
python3 wikisearch_emb.py ../preprocess/train_all.jsonl -o ../preprocessed_data/pre_train_wikisearch_base.jsonl 
python3 wikisearch_emb.py ../raw_data/private_test_data.jsonl -o ../preprocessed_data/private_wikisearch_base.jsonl -t test
python3 wikisearch_emb.py ../preprocess/public_test.jsonl -o ../preprocessed_data/test_wikisearch_base.jsonl -t test
echo doc retrieval2
#do document retrieval 2 preprocess
cd ../preprocess
python3 preprocess_doc2.py
#do document retreival 2 training
#cost around 40 hr for only using one gpu
cd ..
bash kf_train_page.sh
bash kf_predict_page.sh
#postprocess doc retrieval 2
cd postprocess
python3 postprocess_page.py

#do sent retreival preprocess
echo sent retrieval
cd ../preprocess
python3 preprocess_sent.py
#train sent retrieval
bash kf_train_sent.sh
bash kf_predict_sent.sh
#postprocess sent retrieval
cd ../postprocess
python3 postprocess_sent.py

#preprocess enough 
echo enough verification
cd ../preprocess
python3 preprocess_enough.py
#train enough
cd ..
bash kf_train_enough
bash kf_predict_enough

#preprocess claim
echo claim verification
cd preprocess
python3 preprocess_claim.py
#train claim
cd ..
bash kf_train_claim.sh
bash kf_predict_claim.sh

#do postprocess
cd postprocess
python3 postprocess_enough_claim.py
python3 postprocess_submission.py
