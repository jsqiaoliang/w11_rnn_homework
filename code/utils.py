#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import random
import json
import numpy as np


def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data


def index_data(sentences, dictionary):
    shape = sentences.shape
    sentences = sentences.reshape([-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index.reshape(shape)




def get_train_data(vocabulary, batch_size, num_steps):
    ##################
    # My Code here
    ##################
    with open('dictionary.json', encoding='utf-8') as inf:
        dictionary = json.load(inf, encoding='utf-8')
##将输入的宋词转化为数字表示
    X=[]
    Y=[]
    for word in vocabulary:
        if word in dictionary.keys():
            X.append(dictionary[word])
        else:
            X.append(dictionary['UNK'])


    for word in vocabulary[1:]:
        if word in dictionary.keys():
            Y.append(dictionary[word])
        else:
            Y.append(dictionary['UNK'])

##获取数据长度
    data_length = len(X)
##补全label的数据长度
    Y.append(0)
##计算将数据分为batchsize个分区后，每个分区有多少数据
    batch_partition_length = data_length // batch_size
##初始化数据X,Y
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    #将数据X,Y按batch_size分割开
    for i in range(batch_size):
        data_x[i] = X[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = Y[batch_partition_length * i:batch_partition_length * (i + 1)]
##计算每个batch 在num_steps 训练完，每个epoch size为多少
    epoch_size = batch_partition_length // num_steps
##返回每个step的X和Y
    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)



def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
