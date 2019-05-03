printf "\nHere is your current config...\n\n"
source activate base
python -V
nvidia-smi
nvcc -V
pip -V
printf "\nCreating the ml environment (ml)\n\n"
conda create -n ml python=3.7
conda activate ml
printf "\nInstalling PyTorch with Cuda Toolkit 10.0"
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
printf "\nInstalling NVIDIA Apex for Mixed Precision Training"
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
cd ..
printf "\nGetting pytorch_pretrained_bert\n\n"
pip install pytorch_pretrained_bert
printf "\nInstalling jupyter to this environment"
conda install jupyter notebook 
printf "\nInstalling Seaborn\n\n"
conda install seaborn
printf "\n\nInstalling sklearn"
pip install sklearn