if [ "$1" == "" ]; then
    echo "You must enter a commit message."
else
    git add .
    git commit --allow-empty-message -m $1
    git push
fi