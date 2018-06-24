import gensim


class Word2Vec():
    def __init__(self):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            './data/GoogleNews-vectors-negative300.bin', binary=True)
    
    def contains_word(self, word):
        return word in self.model.vocab
        
    def most_similar(self, word):
        return self.model.most_similar([word], topn=1)[0][0]
    
    def most_similar_words(self, word, n):
        return self.model.most_similar([word], topn=n)