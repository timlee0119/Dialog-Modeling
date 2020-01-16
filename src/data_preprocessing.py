import sys
import pandas as pd
import re
import numpy as np
import random
import pickle
import math

False_num = 3
length_sentence_A = 300
# length_sentence_A = 'adjusted'
special_token = "['-\.\!\/_,$%^*()+\"\<>?:-=]+|[+——！，。？?、~@#￥%……&*（）]+"

def check_length(A, B, right_answer = False):
    if right_answer:
        length = len(B)
        # A_list = A[-(512 - length):]
        A_list = A[-length_sentence_A:]
    else:
        A_list = []
        for i in range(len(B)):
            length = len(B[i])
            # A_list.append(A[-(512 - length):])
            A_list.append(A[-length_sentence_A:])
    return A_list, B

def preprocessing(category = 'train'):
    if category == 'valid_train':
        df = pd.read_json(f'../data/valid.json')
    else:
        df = pd.read_json(f'../data/{category}.json')
    all_df = pd.DataFrame(columns = ['A', 'B', 'label'])
    
    for j in range(len(df)):
        input_df = pd.DataFrame(columns = ['A', 'B', 'label'])
        exp = df.iloc[j]
        utterance_mapping = dict()
        ## previous dialogue
        length = len(exp['messages-so-far'])
        data = list(map(lambda x : re.sub(special_token, " ",x['utterance']), list(exp['messages-so-far'])))
        A = ''
        for i in range(len(data)):
            A += data[i].lower()
        ## answer part and answers mapping
        utterance_mapping[exp['options-for-correct-answers'][0]['utterance']] = exp['options-for-correct-answers'][0]['candidate-id']
        answer_data = re.sub(special_token, " ", exp['options-for-correct-answers'][0]['utterance'])

        A_true, answer_data = check_length(A, answer_data, right_answer = True)

        input_df['A'] = [A_true]
        input_df['B'] = [answer_data.lower()]
        input_df['label'] = [1]
        # print('finished previous dialogue ', end = ',')

        correct_index = list(map(lambda x:  1 if x['candidate-id'] == exp['options-for-correct-answers'][0]['candidate-id'] else 0, exp['options-for-next']))
        correct_index = correct_index.index(1) 
        # False Label
        false_example_num = False_num
        size = len(exp['options-for-next'])
        # print(size)
        index = [k for k in range(size)]
        random.shuffle(index)
        if correct_index in index:
            index = list(set(index) - set([correct_index]))
        index = index[:false_example_num]
        false_sentence = list()
        for i in index:
            false_sentence.append(exp['options-for-next'][i]['utterance'])
        
        false_sentence = list(map(lambda x : (re.sub(special_token, " ",x)).lower(), false_sentence))

        A_false, false_sentence = check_length(A, false_sentence)

        wrong_answer = pd.DataFrame()
        wrong_answer['A'] = A_false 
        wrong_answer['B'] = false_sentence
        wrong_answer['label'] = [0 for i in range(false_example_num)]
        input_df = input_df.append(wrong_answer).reset_index(drop= True)
        input_df = input_df.replace({'': np.nan}).reset_index(drop = True)
        input_df = input_df.dropna().reset_index(drop= True)
        all_df = all_df.append(input_df).reset_index(drop= True)
        print(f'finished {j}  / {len(df)} loop', '\r')
    all_df.to_csv(f'../struct_data/new_{category}_df_f{false_example_num}_{length_sentence_A}.csv')


def preprocessing_test(category = 'test'):
    test = pd.read_json(f'./data/{category}.json')
    test_df = pd.DataFrame()
    id_list = []
    for j in range(len(test)):
        input_df = pd.DataFrame(columns = ['A', 'B', 'question', 'candidate_id'])
        exp = test.iloc[j]
        utterance_mapping = dict()
        ## previous dialogue
        length = len(exp['messages-so-far'])
        data = list(map(lambda x : re.sub(special_token, " ",x['utterance']), list(exp['messages-so-far'])))
        A = ''
        for i in range(len(data)):
            A += data[i].lower()
        

        ## candidate Label
        candidate_id = list(map(lambda x : x['candidate-id'], list(exp['options-for-next'])))
        input_df['candidate_id'] = candidate_id
        
        B = list(map(lambda x : (re.sub(special_token, " ",x['utterance'])).lower(), list(exp['options-for-next'])))

        A, B = check_length(A, B)


        input_df['B'] = B
        input_df['A'] = A

        ## question order
        input_df['question'] = [j for i in range(100)]
        if category == 'valid':
            input_df['ans'] = [exp['options-for-correct-answers'][0]['candidate-id'] for i in range(len(input_df))]

        input_df = input_df.replace({'': np.nan}).reset_index(drop = True)
        input_df = input_df.dropna().reset_index(drop= True)
        test_df = test_df.append(input_df).reset_index(drop= True)
        print(f'finished {j} / {len(test)} loop ', end = '\r')
    test_df.to_csv(f'../struct_data/new_{category}_df_{length_sentence_A}.csv', index = False)

def main(argv, arc):
    preprocessing('train')
    preprocessing('valid_train')
    preprocessing_test('valid')
    preprocessing_test('test')

if __name__ == '__main__':
    main(sys.argv, len(sys.argv))
