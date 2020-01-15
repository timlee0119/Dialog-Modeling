import sys
import numpy as np
import pickle
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from torch.utils.data import Dataset
 
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import torchvision
from torchvision import datasets, models, transforms


from transformers import BertForSequenceClassification, BertForNextSentencePrediction
from transformers import BertModel, BertTokenizer

pre_trained_model_name = 'bert-base-uncased'
# pre_trained_model_name = 'bert-large-uncased'

num_epochs = 3
batch_size = 1
lr = 1e-4

model_name = 'bert_model_1e-05_10_F3_lower_1216_SC_adamw_1e-05_torch_dict'
device = 0

TEST_BATCH = 5


class DialogueDataset(Dataset):
    def __init__(self, df, mode , tokenizer):
        assert mode in ["train", "test"]  # 一般訓練你會需要 dev set
        self.mode = mode
        self.df = df
        self.len = len(self.df)
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
    
    def __getitem__(self, idx):
        if self.mode == "test":
            text_a, text_b = self.df.iloc[idx, :2].values
            label_tensor = None
        else:
            text_a, text_b, label = self.df.iloc[idx, :].values
            label_tensor = torch.tensor(label)
            
        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces += tokens_a + ["[SEP]"]
        len_a = len(word_pieces)
        
        # 第二個句子的 BERT tokens
        tokens_b = self.tokenizer.tokenize(text_b)
        word_pieces += tokens_b + ["[SEP]"]
        len_b = len(word_pieces) - len_a
        
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, 
                                        dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len

class bert_model():
    def __init__(self, epoch = 100, batch_size = 32, lr = 1e-4, valset = None):
        self.epoch = epoch
        self.batch_size = batch_size
        self.loss_list = []
        self.val_loss_list = []
        self.val_accu_list = []
        self.lr = lr
        self.model = None
        self.gpu = torch.cuda.is_available()

    def create_mini_batch(self, samples):
        tokens_tensors = [s[0] for s in samples]
        segments_tensors = [s[1] for s in samples]

        # 測試集有 labels
        if samples[0][2] is not None:
            label_ids = torch.stack([s[2] for s in samples])
        else:
            label_ids = None

        # zero pad 到同一序列長度
        tokens_tensors = pad_sequence(tokens_tensors, 
                                      batch_first=True)
        segments_tensors = pad_sequence(segments_tensors, 
                                        batch_first=True)

        # attention masks，將 tokens_tensors 裡頭不為 zero padding
        # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
        masks_tensors = torch.zeros(tokens_tensors.shape, 
                                    dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(
            tokens_tensors != 0, 1)

        return tokens_tensors, segments_tensors, masks_tensors, label_ids
        
    
    def fit_and_train(self, train_df, val_df, require_grad):

        NUM_LABELS = 2
        max_value = 0
        tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name)
       
        trainset = DialogueDataset(train_df, "train", tokenizer=tokenizer)
        trainloader = DataLoader(trainset, batch_size=self.batch_size, collate_fn=self.create_mini_batch)

        val_batch_size = 20
        valset = DialogueDataset(val_df, 'test', tokenizer = tokenizer)
        valloader = DataLoader(valset, batch_size=val_batch_size, collate_fn=self.create_mini_batch)
        
        model = BertForSequenceClassification.from_pretrained(pre_trained_model_name, num_labels=NUM_LABELS)
        # model = BertForNextSentencePrediction.from_pretrained(PRETRAINED_MODEL_NAME)
        # if require_grad:
        #   for param in model.parameters():
        #      param.requires_grad = True
        model.train()

        if self.gpu:
            model = model.cuda(device)
        for epo in range(self.epoch):
            total = 0
            total_loss = 0
            for data in trainloader:
                if self.gpu:
                    tokens_tensors, segments_tensors, \
                    masks_tensors, labels = [x.type(torch.LongTensor).cuda(device) for x in data]
                else:
                    tokens_tensors, segments_tensors, \
                    masks_tensors, labels = [x for x in data]
                outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors, labels=labels)
   # (tensor(0.6968, grad_fn=<NllLossBackward>), tensor([[-0.0359, -0.0432]], grad_fn=<AddmmBackward>))
#   count += self.accu(preds_label, y)
                loss = outputs [0]
                loss.backward() # calculate gradientopt = torch.optim.SGD(model.parameters(), lr=self.lr,  momentum=0.9)
                opt = torch.optim.Adam(model.parameters(), lr = self.lr)
                opt.step() #update parameter
                opt.zero_grad()
                total += len(tokens_tensors)
                total_loss += loss.item() * len(tokens_tensors)
                print(f'Epoch : {epo+1}/{self.epoch} , Training Loss : {loss}', end = '\r')
            self.loss_list.append(total_loss / total)
            print(f'Epoch : {epo+1}/{self.epoch} , Training Loss : {self.loss_list[epo]}', end = ',')

            

            model.eval()
            numebr = 0

            ans = []
            with torch.no_grad():
                for data in valloader:
                    if self.gpu:
                        tokens_tensors, segments_tensors, masks_tensors, _ = [x.type(torch.LongTensor).cuda(device) if x is not None else None for x in data]
                    else:
                        tokens_tensors, segments_tensors, masks_tensors, _ = [x for x in data]
                    outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors,)
        #        (tensor([[-0.0359, -0.0432]], grad_fn=<AddmmBackward>))   
                    values = outputs[0].data[:,1].tolist()
                    ans += values
                    print(f'count : {numebr}', end = '\r')
                    numebr += val_batch_size

                count = 0
                val_len = 0
                val_df['prob'] = ans
                groups = val_df.groupby('question')
                for index, data in groups:
                    val_len += 1
                    if 'candidate_id' in val_df.columns:
                        pred_id = data.loc[data['prob'].idxmax(),'candidate_id']
                        if data.loc[data['prob'].idxmax(),'ans'] == pred_id:
                            count += 1


                val_accu = count / val_len
                if val_accu >= max_value:
                    self.model = model
                    torch.save(model.state_dict(), f'./model/{model_name}_torch_dict')
                self.val_accu_list.append(val_accu)

                print(f'Epoch : {epo+1}/{self.epoch}, Validation Accuracy : {self.val_accu_list[epo]}',end = ',')
    
    def accu(self, pred, y):
        ret = 0
        for i in range(len(pred)):
            if pred[i] == y[i]:
                ret += 1
        return ret

    def softmax(self, vec):
        out = Func.softmax(vec, dim=1) # along rows
        values, indexs = out.max(-1)
        return indexs.view(len(indexs),1)
    
    def accu(self, pred, y):
        ret = 0
        for i in range(len(pred)):
            if pred[i] == y[i]:
                ret += 1
        return ret
    
    def forward(self, x):
        out = Func.softmax(x, dim=1) # along rows
        return out[0][1].tolist()

    
    def predict(self, test_data):
        test_batch_size = TEST_BATCH
        ans = []
        if self.gpu:
            self.model = self.model.cuda(device)
        self.model.eval()
        testloader = DataLoader(test_data, batch_size=test_batch_size, collate_fn=self.create_mini_batch)
        count = 0 
        for x in testloader:
            if self.gpu:
                tokens_tensors, segments_tensors, masks_tensors, _ = [i.cuda(device) if i is not None else i for i in x ]
            else:
                tokens_tensors, segments_tensors, masks_tensors, _= [i for i in x]
            outputs = self.model(input_ids=tokens_tensors, 
                    token_type_ids=segments_tensors, 
                    attention_mask=masks_tensors,)
#        (tensor(0.6968, grad_fn=<NllLossBackward>), tensor([[-0.0359, -0.0432]], grad_fn=<AddmmBackward>))
            # first method -> batch >1
            #        (tensor([[-0.0359, -0.0432]], grad_fn=<AddmmBackward>))   
            values = outputs[0].data[:,1].tolist()
            ans += values
            # second method
            # ans.append(self.forward(outputs[0]))
            # ans.append(outputs[0].tolist()[0][1])
            count+= test_batch_size
            print(f'count : {count}', end = '\r')
        return ans

def main(argv, arc):
    test_path = argv[1]
    output_path = argv[2]

    test_df = pd.read_csv(test_path, dtype = {'A': 'str', 'B':'str'})
    if 'Unnamed: 0' in test_df.columns:
        test_df = test_df.drop(['Unnamed: 0'], axis =1)

    print(len(test_df),end = '\n')
    tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name)
    testset = DialogueDataset(test_df, 'test', tokenizer=tokenizer)


    # first way
    # with open(f'./model/{model_name}', 'rb') as input_model:
    #     model = pickle.load(input_model)

    # second way
    NUM_LABELS = 2
    
    model = bert_model()
    model.model = BertForSequenceClassification.from_pretrained(pre_trained_model_name, num_labels=NUM_LABELS)
    # model.model = BertForNextSentencePrediction.from_pretrained(pre_trained_model_name)
    model.model.load_state_dict(torch.load('./model/bert_model_1e-05_4_lower_0102_SC_adamw_f3_valepo1_A300_torch_dict_tuned_val', map_location= f'cuda:{device}'))
    print(model.val_accu_list)

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
    # pred_df['id'] = [f'{i}' for i in range(80000,82000)]
    pred_df['id'] = [f'{80000 + i}' for i in range(0, len(ans))]
    # pred_df['id'] = [82000]
    pred_df['candidate-id'] = ans
    pred_df.to_csv(output_path, index = False)

if __name__ == '__main__':
    main(sys.argv, len(sys.argv))