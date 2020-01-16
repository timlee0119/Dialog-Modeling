cur=$(dirname $(realpath $0))
if [ $2 ]; then
  echo "Using model: $2"
  cd $cur/src
  python final_test.py ../struct_data/new_test_df_300.csv $cur/$1 $cur/$2
elif [ $1 ]; then
  echo "Downloading the best model..."
  cd $cur/model
  wget 'https://www.dropbox.com/s/3btmslig1c0wpfo/bert_best_1e-5_f3_A300?raw=1'
  mv 'bert_best_1e-5_f3_A300?raw=1' bert_best_1e-5_f3_A300
  cd ../src
  python final_test.py ../struct_data/new_test_df_300.csv $cur/$1 ../model/bert_best_1e-5_f3_A300
else
  echo "Usage: ./test.sh <output path> [model path]"
fi
