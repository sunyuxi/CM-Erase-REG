# -*- coding:utf8 -*-

import json
import numpy as np

def extract_glove_vector():
    glove_path='data/rsvg/glove/glove.840B.300d.txt'
    prepro_path = 'cache/prepro/rsvg/data.json'
    output_word2vec_filepath = 'rsvg_glove.840B.300d.npy'
    with open(prepro_path, 'r') as f:
        json_data = json.load(f)
        dict_word2idx = json_data['word_to_ix']
        print('total word:')
        print(len(dict_word2idx))
        min_idx = min(dict_word2idx.values())
        max_idx = max(dict_word2idx.values())
    assert min_idx == 0
    assert max_idx == len(dict_word2idx)-1

    output_word2vec = np.zeros( (len(dict_word2idx), 300), dtype=np.float32 )

    dict_word2vec = {}
    unk_wor2vect = None
    for line in open(glove_path):
        line = line.rstrip('\n')
        arr_tmp = line.split(' ')
        if len(arr_tmp)<=300:
            print(len(arr_tmp))
            print(line)
            print(arr_tmp)
        assert len(arr_tmp)>300
        one_word2vect = [float(one) for one in arr_tmp[-300:]]
        word = arr_tmp[0]
        if len(arr_tmp)>301:
            word = ' '.join(arr_tmp[0:-300])
        dict_word2vec[word] = one_word2vect
        assert len(one_word2vect) == 300
        
        if arr_tmp[0] == '<unk>':
            unk_wor2vect = one_word2vect
    assert unk_wor2vect != None

    valid_cnt = 0
    for word, idx in dict_word2idx.items():
        if word not in dict_word2vec:
            output_word2vec[idx] = unk_wor2vect
        else:
            output_word2vec[idx] = dict_word2vec[word]
            valid_cnt = valid_cnt + 1
    
    print('valid count:')
    print(valid_cnt)
    np.save(output_word2vec_filepath, output_word2vec)
    print('success!')

if __name__ == '__main__':
    extract_glove_vector()