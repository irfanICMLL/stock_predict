# -*- coding: utf-8 -*-

import gensim

def export_embedding():
    model = gensim.models.Word2Vec.load('mymodel1')
    with open('data/embeddings2.txt', 'w+') as f:
        for word in model.index2word:
            f.write(word)
            f.write('\t')
            for num in model[word]:
                f.write(str(num))
                f.write(' ')
            f.write('\n')
    print("Word Embeddings Printed to \"data/embeddings.txt\"")

if __name__ == "__main__":
    export_embedding()