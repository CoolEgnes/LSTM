# This py is used to split the text, and embed the words using word2vec
import numpy as np
from gensim.models.word2vec import Word2Vec
import pickle
import os
import split_data


def word_embedding(split_file, org_file):
    """
    Args:
         split_file: the split file
         org_file: original file
    Return:
        org_data: original content

    """
    vec_params_file = "vec_params.pkl"
    if not os.path.exists(split_file):
        split_data.split_poetry(org_file)

    split_all_data = open(split_file, "r", encoding="utf-8").read().split("\n")
    org_data = open(org_file, "r", encoding="utf-8").read().split("\n")

    if os.path.exists(vec_params_file):
        return org_data, pickle.load(open(vec_params_file, "rb"))

    word_embedding_model = Word2Vec(split_all_data, vector_size=100, min_count=1)
    pickle.dump((word_embedding_model.syn1neg, word_embedding_model.wv.key_to_index, word_embedding_model.wv.index_to_key),open(vec_params_file,"wb"))

    return org_data, (word_embedding_model.syn1neg, word_embedding_model.wv.key_to_index, word_embedding_model.wv.index_to_key)