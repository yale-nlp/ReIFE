git sparse-checkout init --no-cone
echo "/*" > .git/info/sparse-checkout
echo '!/results/' >> .git/info/sparse-checkout
git read-tree -mu HEAD