import numpy as np

class Model():
    def __init__(self, feature_extractor, classifier):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        
    def fit(self, X_train, y_train):
        X_train_feat = self.feature_extractor.get_features(X_train)
        print(X_train_feat.shape)
        self.classifier.fit(X_train_feat, y_train)
        
        return self.classifier
    
    def fit_features(self, X_train_feat, y_train):
        self.classifier.fit(X_train_feat, y_train)
        
        return self.classifier
    
    def predict(self, X_test):
        X_test_num = self.feature_extractor.get_features(X_test)
        
        return self.classifier.predict(X_test_num)
    
    def predict_feat(self, X_test_feat):
        return self.classifier.predict(X_test_feat)
    
    def predict_single(self, q1, q2):
        x_num = np.array(self.feature_extractor.get_features_single(q1, q2)).reshape(-1, 1)

        return self.classifier.predict(x_num)
        