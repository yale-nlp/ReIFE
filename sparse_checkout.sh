echo "/*" > .git/info/sparse-checkout
echo '!/results/' >> .git/info/sparse-checkout
git read-tree -mu HEAD