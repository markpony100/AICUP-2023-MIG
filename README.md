# AICUP-2023-MIG
## Environment setting
CUDA = 11.1
please install conda and use following code to set up environments

conda create -n AICUP2023
conda activate AICUP2023
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip3 install -r requirements.txt
pip3 install accelerate -U

## please download our data and model zip data, unzip and put it in main directory
data and model checkpoint: https://docs.google.com/document/d/1h1ZzWUkmqbb994-Exv9vkVZihn6VP8ro3hDMqAOaFzA/edit?usp=drive_link

## run pipeline
bash pipeline.sh
