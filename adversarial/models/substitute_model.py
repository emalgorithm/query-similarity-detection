import json
import pickle

import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Lambda, concatenate, Dropout, \
    BatchNormalization
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from util.util import load_glove, compute_embedding_matrix

PREFIX = '../data/quora-top-performer/'
WORD_EMBEDDING_MATRIX_FILE = PREFIX + 'word_embedding_matrix.npy'
NB_WORDS_DATA_FILE = PREFIX + 'nb_words.json'
TOKENIZER = PREFIX + 'tokenizer.dump'
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 25
EMBEDDING_DIM = 100
MODEL_WEIGHTS_FILE = 'question_pairs_weights.h5'
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
RNG_SEED = 13371447
NB_EPOCHS = 25
DROPOUT = 0.1
BATCH_SIZE = 32
OPTIMIZER = 'adam'


class SubstituteModel:
    def __init__(self):
        # self.word_embedding_matrix = np.load(open(WORD_EMBEDDING_MATRIX_FILE, 'rb'))
        with open(NB_WORDS_DATA_FILE, 'r') as f:
            self.nb_words = json.load(f)['nb_words']
        self.tokenizer = pickle.load(open(TOKENIZER, 'rb'))

        self.embeddings_index = load_glove()
        self.word_embedding_matrix = compute_embedding_matrix(self.tokenizer.word_index,
                                                              self.embeddings_index,
                                                              EMBEDDING_DIM)
        self.model = self.initialize_model()

    def predict(self, X_test):
        q1 = X_test[:, 0]
        q2 = X_test[:, 1]

        question1_word_sequences = self.tokenizer.texts_to_sequences(q1)
        question2_word_sequences = self.tokenizer.texts_to_sequences(q2)

        q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
        q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)

        return self.model.predict([q1_data, q2_data])

    def predict_single(self, q1, q2):
        return self.predict(np.array([[q1, q2]]))[0][0]

    def train(self, X_train, y_train):
        """Train the substitute model using the current training data"""
        question1 = X_train[:, 0]
        question2 = X_train[:, 1]

        question1_word_sequences = self.tokenizer.texts_to_sequences(question1)
        question2_word_sequences = self.tokenizer.texts_to_sequences(question2)

        q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
        q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)

        history = self.model.fit([q1_data, q2_data],
                                            y_train,
                                            epochs=NB_EPOCHS,
                                            validation_split=VALIDATION_SPLIT,
                                            verbose=2,
                                            batch_size=BATCH_SIZE)

        max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_acc']))
        print('Maximum validation accuracy = {0:.4f} (epoch {1:d})'.format(max_val_acc, idx + 1))
        return max_val_acc

    def initialize_model(self):
        question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
        question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

        q1 = Embedding(len(self.tokenizer.word_index) + 1,
                       EMBEDDING_DIM,
                       weights=[self.word_embedding_matrix],
                       input_length=MAX_SEQUENCE_LENGTH,
                       trainable=False)(question1)
        q1 = Lambda(lambda x: K.mean(x, axis=1), output_shape=(EMBEDDING_DIM,))(q1)

        q2 = Embedding(self.nb_words + 1,
                       EMBEDDING_DIM,
                       weights=[self.word_embedding_matrix],
                       input_length=MAX_SEQUENCE_LENGTH,
                       trainable=False)(question2)
        q2 = Lambda(lambda x: K.mean(x, axis=1), output_shape=(EMBEDDING_DIM,))(q2)

        merged = concatenate([q1, q2])
        merged = Dense(200, activation='relu')(merged)
        merged = Dropout(DROPOUT)(merged)
        merged = BatchNormalization()(merged)
        merged = Dense(200, activation='relu')(merged)
        merged = Dropout(DROPOUT)(merged)
        merged = BatchNormalization()(merged)

        is_duplicate = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[question1, question2], outputs=is_duplicate)
        model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

        return model