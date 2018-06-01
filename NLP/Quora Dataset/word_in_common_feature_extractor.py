from text_processor import TextProcessor
import numpy as np

class WordsInCommonFeatureExtractor():
    def __init__(self):
        self.tp = TextProcessor()
        
    def get_features(self, X):
        Xnum = np.apply_along_axis(lambda x: self.get_features_for_sample(str(x[0]), str(x[1])), 1, X)
        
    def get_features_for_sample(self, q1, q2):
        """
        There is only a single feature which is is the ratio of words that appear in both questions over words which 
        appear in either.
        Stop words are excluded.
        """
        q1_words = set(self.tp.remove_stopwords(self.tp.tokenize(q1.lower())))
        q2_words = set(self.tp.remove_stopwords(self.tp.tokenize(q2.lower())))

        words_in_common = len(q1_words & q2_words)
        total_words = len(q1_words | q2_words) 

        frac_words_in_common = words_in_common / total_words

        return [frac_words_in_common]