import pickle


class WordsInCommonModel():
    def __init__(self):
        path_prefix = 'models/words_in_common/'
        self.model = pickle.load(open(path_prefix + 'model.dump', 'rb'))

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_single(self, q1, q2):
        return self.model.predict_single(q1, q2)[0]