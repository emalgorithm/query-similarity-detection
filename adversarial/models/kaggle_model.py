from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import pickle
import numpy as np
import json


class KaggleModel():
    def __init__(self):
        path_prefix = './data/quora-top-performer/'
        self.tokenizer = pickle.load(open(path_prefix + 'tokenizer.dump', 'rb'))
        self.word_embedding_matrix = np.load(open(path_prefix + 'word_embedding_matrix.npy', 'rb'))
        with open(path_prefix + 'nb_words.json', 'r') as f:
            self.nb_words = json.load(f)['nb_words']
        
        self.model = load_model(path_prefix + 'kaggle.h5')
        self.MAX_SEQUENCE_LENGTH = 25
            
    def predict(self, X_test):
        question1 = X_test[:, 0]
        question2 = X_test[:, 1]

        questions = list(question1) + list(question2)
        
        question1_word_sequences = self.tokenizer.texts_to_sequences(question1)
        question2_word_sequences = self.tokenizer.texts_to_sequences(question2)
        word_index = self.tokenizer.word_index
        
        q1_data = pad_sequences(question1_word_sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        q2_data = pad_sequences(question2_word_sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        
        return self.model.predict([q1_data, q2_data])
            
    def predict_single(self, q1, q2):
        return self.predict(np.array([[q1, q2]]))[0, 0]