# AICUP-2023-MIG
## Environment setting
CUDA = 11.1
please install conda and use following code to set up environments

```bash
conda create -n AICUP2023
conda activate AICUP2023
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip3 install -r requirements.txt
pip3 install accelerate -U

#Directory tree
├── README.md
├── bert_main.py
├── doc1
├── kf_predict_claim.sh
├── kf_predict_enough.sh
├── kf_predict_page.sh
├── kf_predict_sent.sh
├── kf_train_claim.sh
├── kf_train_enough.sh
├── kf_train_page.sh
├── kf_train_sent.sh
├── pipeline.sh
├── postprocess
├── predict.sh
├── preprocess
├── requirements.txt
├── submissions
├── train.sh
└── utils.py

## please download our data_and_model.zip, unzip and put it in main directory
[data and model checkpoint](https://CSMIG.quickconnect.to/d/s/tuUh6pvfGgeQ596AutCmef6wWfOK2FM5/4ov2FFGow2wOV3K_wTAPq1JpsVgSfgdI-PL9gAzD2gAousp=drive_link)


## run pipeline
bash pipeline.sh
