# -*- coding: utf-8 -*-

import gensim

if __name__ == "__main__":
    model = gensim.models.Word2Vec.load('mymodel')
    while True:
        try:
            word = input("Input word: ")
            similar_word = model.most_similar(word, topn = 1)
            print(similar_word)
        except:
            print("Not in vocabulary")