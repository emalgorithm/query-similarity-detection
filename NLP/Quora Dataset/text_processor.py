from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter

class TextProcessor():
    def __init__(self):
        self.tknzr = TweetTokenizer()
        self.stops = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
    
    def tokenize(self, text):
        return self.tknzr.tokenize(text)
    
    def remove_stopwords(self, text):
        return [word for word in text if word not in self.stops]
    
    def stem(self, text):
        return [self.stemmer.stem(word) for word in text]
    
    def remove_uncommon(self, text, threshold=3):
        c = Counter(text)

        return [word for word in text if c[word] >= threshold]
    
    def clean(self, text):
        text = self.tokenize(text)
        text = self.remove_stopwords(text)
        text = self.stem(text)
        text = self.remove_uncommon(text)
        
        return text
    
    