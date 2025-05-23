# This py is used to load data using MyDataset class
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    # store and initialize some variables
    def __init__(self, all_data, w1, word_2_index):
        self.w1= w1
        self.word_2_index = word_2_index
        self.all_data = all_data

    # get a piece of data and preprocess the data
    def __getitem__(self, item):
        a_poetry_words = self.all_data[item]
        # print("a_poetry_words in MyDataset: ", a_poetry_words)  # 仓储十万发关中，伟绩今时富郑公。有米成珠资缓急，此心如秤慎初终。
        a_poetry_index = [self.word_2_index[word] for word in a_poetry_words]
        # print("a_poetry_index in MyDataset: ", a_poetry_index)  # [922, 2096, 76, 39, 270, 249, 18, 1, 2510, 1489, 79, 26, 968, 1379, 205, 2, 11, 1036, 51, 339, 1149, 1702, 768, 1, 37, 31, 19, 3531, 2271, 256, 338, 2]
        xs_index = a_poetry_index[:-1]
        ys_index = a_poetry_index[1:]

        xs_embedding = self.w1[xs_index]

        return xs_embedding, np.array(ys_index).astype(np.int64)

    # get the total length of data
    def __len__(self):
        return len(self.all_data)

