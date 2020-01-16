import sys
import pickle
import pandas as pd

from final_common import DialogueDataset, bert_model

pretrained_model_name = 'bert-base-uncased'
# pre_trained_model_name = 'bert-large-uncased'
#num_epochs = 4
num_epochs = int(sys.argv[4])
#batch_size = 8
batch_size = int(sys.argv[5])
#lr = 1e-5
lr = float(sys.argv[6])

#false_num = 3
#length_sentence_A = 300

#model_type = f'SC_adamw_f{false_num}_valepo{val_fine_tuned_epo}_A{length_sentence_A}'

#model_name = f'bert_model_{lr}_{num_epochs}_lower_0103_{model_type}'
       
       

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
#     with open(f'./model/{model_name}_last_epo', 'wb') as output:
#         pickle.dump(model, output)

if __name__ == '__main__':
    main(sys.argv, len(sys.argv))