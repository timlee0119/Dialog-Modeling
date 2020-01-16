if [ $2 ]; then
  echo "two args"
elif [ $1 ]; then
  echo "Downloading the best model..."
  cd $(dirname $(realpath $0))/model
  wget 'https://www.dropbox.com/s/3btmslig1c0wpfo/bert_best_1e-5_f3_A300?raw=1'
  unzip 'bert_best_1e-5_f3_A300?raw=1'
  rm 'bert_best_1e-5_f3_A300?raw=1'
  cd ../src 
else
  echo "Usage: ./test.sh <output path> [model path]"
fi
