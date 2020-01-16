if [ $3 ]; then
  cd $(dirname $(realpath $0))/src
  python3 final_train.py ../struct_data/new_train_df_f3_300.csv ../struct_data/new_valid_df_300.csv ../struct_data/new_valid_train_df_f3_300.csv $1 $2 $3
else
  echo "Usage: ./train.sh <epoch> <batch size> <learning rate>"
fi
