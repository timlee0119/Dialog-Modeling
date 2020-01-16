if [ $2 ]; then
  echo "Using model: $2"
  python final_test.py ../struct_data/new_test_df_300.csv $1 $2
elif [ $1 ]; then
  echo "Downloading the best model..."
  cd $(dirname $(realpath $0))/model
  wget 'https://www.dropbox.com/s/3btmslig1c0wpfo/bert_best_1e-5_f3_A300?raw=1'
  mv 'bert_best_1e-5_f3_A300?raw=1' bert_best_1e-5_f3_A300
  cd ../src
  python final_test.py ../struct_data/new_test_df_300.csv $1 ../model/bert_best_1e-5_f3_A300
else
  echo "Usage: ./test.sh <output path> [model path]"
fi
