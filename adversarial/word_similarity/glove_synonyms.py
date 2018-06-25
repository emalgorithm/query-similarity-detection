import numpy as np


class GloveSynonyms:
    def __init__(self):
        self.loadGloveModel('../data/counter-fitted-vectors.txt')
        
    def get_vector(self, word):
        return self.word_vectors[self.word_to_idx[word]]
        
    def most_similar(self, word, n=1):
        word_vector = self.get_vector(word)
        distances = (np.dot(self.word_vectors, word_vector)
                       / np.linalg.norm(self.word_vectors, axis=1)
                       / np.linalg.norm(word_vector))
        
        idxs = np.argsort(-distances)[1:n+1]

        result = [self.idx_to_word[idx] for idx in idxs]
        
        return result if n > 1 else result[0]
    
    def contains_word(self, word):
        return word in self.word_to_idx
    
    def loadGloveModel(self, gloveFile):
        print("Loading Glove Model")
        f = open(gloveFile,'r')
        self.word_vectors = []
        self.word_to_idx = {}
        self.idx_to_word = {}
        idx = 0
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            self.word_vectors.append(embedding)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
            idx += 1
        self.word_vectors = np.array(self.word_vectors)
        print("Done.",len(self.word_to_idx)," words loaded!")
