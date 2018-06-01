import gensim

class Word2Vec():
    def __init__(self):
        self.model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
        
    def get_closest_word(self, word):
        return self.model.most_similar([word], topn=1)[0][0]
    
    def get_closest_words(self, word, n):
        return self.model.most_similar([word], topn=n)