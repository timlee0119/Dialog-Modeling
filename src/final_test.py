import sys
import os
import pandas as pd

import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

from final_common import DialogueDataset, bert_model

pre_trained_model_name = 'bert-base-uncased'

def main(argv, arc):
    test_path = argv[1]
    output_path = argv[2]

    test_df = pd.read_csv(test_path, dtype = {'A': 'str', 'B':'str'})
    if 'Unnamed: 0' in test_df.columns:
        test_df = test_df.drop(['Unnamed: 0'], axis =1)

    print(f'Testing data length: {len(test_df)}')
    tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name)
    testset = DialogueDataset(test_df, 'test', tokenizer=tokenizer)

    NUM_LABELS = 2
    
    model = bert_model(pre_trained_model_name)
    model.model = BertForSequenceClassification.from_pretrained(pre_trained_model_name, num_labels=NUM_LABELS)
    model.model.load_state_dict(torch.load(argv[3]))

    print('Start testing...')
    preds = model.predict(testset)
    test_df['prob'] = preds
    groups = test_df.groupby('question')
    ans = []
    for index, data in groups:
        if 'candidate_id' in test_df.columns:
            ans.append(data.loc[data['prob'].idxmax(),'candidate_id'])
        else:
            ans.append(data.loc[data['prob'].idxmax(),'B'])

    pred_df = pd.DataFrame()
    pred_df['id'] = [f'{80000 + i}' for i in range(0, len(ans))]
    pred_df['candidate-id'] = ans
    
    # create folder if folder not exists
    print(f'Writing results to {output_path}')
    dirname = os.path.dirname(output_path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    pred_df.to_csv(output_path, index = False)

if __name__ == '__main__':
    main(sys.argv, len(sys.argv))