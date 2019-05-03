printf "\nSetting up an unbiased toxicity workspace...\n"
sudo apt-get install tree
mkdir UnbiasedToxicity
mkdir UnbiasedToxicity/Code
cd UnbiasedToxicity/Code
printf "\nGetting repo...\n"
git init
git pull https://github.com/amanrai/deep-learning.git
git remote add deep-learning https://github.com/amanrai/deep-learning.git
cd ../..
tree -d
printf "\n\n\n...and now for the tricky bit!\n"