import sys
import pickle
import pandas as pd

from final_common import DialogueDataset, bert_model

pretrained_model_name = 'bert-base-uncased'
num_epochs = int(sys.argv[4])
batch_size = int(sys.argv[5])
lr = float(sys.argv[6])

def main(argv, arc):
    train_path = argv[1]
    val_path = argv[2]
    val_train_path = argv[3]

    train_df = pd.read_csv(train_path, dtype = {'A': 'str', 'B':'str'})
    if 'Unnamed: 0' in train_df.columns:
        train_df = train_df.drop(['Unnamed: 0'], axis =1)
    
    val_df = pd.read_csv(val_path, dtype = {'A': 'str', 'B':'str'})
    if 'Unnamed: 0' in val_df.columns:
        val_df = val_df.drop(['Unnamed: 0'], axis =1)

    val_train_df = pd.read_csv(val_train_path, dtype = {'A': 'str', 'B':'str'})
    if 'Unnamed: 0' in val_train_df.columns:
        val_train_df = val_train_df.drop(['Unnamed: 0'], axis =1)

    model = bert_model(pretrained_model_name, epoch = num_epochs, batch_size = batch_size, lr = lr)
    model.fit_and_train(train_df, val_df, val_train_df, require_grad = True)

if __name__ == '__main__':
    main(sys.argv, len(sys.argv))