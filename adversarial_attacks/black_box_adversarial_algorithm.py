import numpy as np
from nltk.corpus import stopwords
import time


from text_processor import TextProcessor
from util import get_balanced_data, parallel_apply_along_axis
from glove_synonyms import GloveSynonyms
from substitute_model import SubstituteModel
from adversarial_algos import adversarial_white_box_change


class BlackBoxAdversarialAlgorithm:
    """
    Class which represents an API for performing a black box adversarial attack against a given oracle.
    """
    def __init__(self, oracle, word_similarity=GloveSynonyms(), n_initial_train=1000,
                 n_test=1000, n_st_epochs=5,
                 similarity_threshold=0.5):
        # Oracle we are trying to attack. It takes a pair of questions as input and returns
        # whether they are similar (1) or not similar (0)
        self.oracle = oracle
        # Substitute model we use to emulate the oracle.
        self.substitute_model = SubstituteModel()
        # A util to perform text related operations
        self.tp = TextProcessor()
        # A utility which contains a vector embedding for words
        self.word_similarity = word_similarity
        self.similarity_threshold = similarity_threshold
        self.n_st_epochs = n_st_epochs
        self.n_test = n_test

        # Start with 100 training examples
        X_train, X_test, _, _ = get_balanced_data()
        self.X_train = X_train[:n_initial_train]

        # Keep only testing data points which are already classified as similar by the oracle
        similar_rows = (self.oracle.predict(X_test) > self.similarity_threshold)[:, 0]
        self.X_test = X_test[similar_rows]
        self.X_test = self.X_test[:self.n_test]
        assert((self.oracle.predict(self.X_test) > self.similarity_threshold).all())

    def evaluate(self, X_test):
        """
        :param X_test: A numpy array containing the data points we want to use to evaluate our
        black box method
        :return: The transferability score of the black box method. The transferability score is
        calculated as the fraction of adversarial examples which work both on the substitute
        model and the oracle, over the adversarial examples which work on the substitute model
        """
        results = []
        for i in range(self.n_test):
            q1 = X_test[-i, 0]
            q2 = X_test[-i, 1]

            if self.oracle.predict_single(q1, q2) > self.similarity_threshold:
                results.append(self.attack(q1, q2))

            if i % 10 == 0:
                print("Evaluation: {} out of {} done".format(i, self.n_test))

        results = [result for result in results if result is not None]
        transferability = sum(results) / len(results) if len(results) > 0 else 0

        print("{} attacks were successful out of {} tried".format(sum(results), len(results)))
        print("Current transferability of black box attack model is {0:.2f}%".format(
            transferability * 100))
        return transferability

    def substitute_training(self):
        for i in range(self.n_st_epochs):
            # Label
            y_train = self.label(self.X_train)
            # Train
            self.substitute_model.train(self.X_train, y_train)
            # Evaluate
            self.evaluate(self.X_test)
            # Augment
            if i < self.n_st_epochs - 1:
                self.X_train = self.augment(self.X_train)

    def attack(self, q1, q2):
        # Craft adversarial example using white-box algorithm on substitute model
        success, adv_q1, adv_q2 = self.craft_adversarial_example(q1, q2)

        if success:
            # Use previously crafted adversarial example to try to fool the oracle
            return self.oracle.predict_single(adv_q1, adv_q2) < 0.5
        else:
            return None

    def craft_adversarial_example(self, q1, q2):
        """Modify q1 and q2 so that they fool the substitute model. Hopefully, they will be able
        to fool the oracle as well"""
        return adversarial_white_box_change(q1, q2, self.substitute_model, self.tp,
                                            self. word_similarity)

    def label(self, X_train):
        """Label the current training examples by querying the oracle"""
        return self.oracle.predict(X_train) > 0.5

    def augment(self, X_train):
        """Augment current training data with examples which are closer to boundary (score of
        0.5)"""
        print("Augmenting training data")
        start_time = time.time()
        X_train_aug = np.apply_along_axis(self.replace_word_with_greatest_change, 1, X_train)
        end_time = time.time()
        print("Augmenting done. It took {0:.4} seconds".format(end_time - start_time))

        return np.concatenate((X_train, X_train_aug), 0)

    def replace_word_with_greatest_change(self, row):
        stop = set(stopwords.words('english'))
        q1_tokenized = self.tp.tokenize(row[0])
        q2_tokenized = self.tp.tokenize(row[1])
        min_dist_from_bound = 1.0

        new_q2 = q2_tokenized

        for i, word in enumerate(q2_tokenized):
            if self.word_similarity.contains_word(word) and word not in stop:
                closest_word = self.word_similarity.most_similar(word)
                q2_modified = list(q2_tokenized)
                q2_modified[i] = closest_word
                score = self.substitute_model.predict_single(self.tp.detokenize(q1_tokenized),
                                                             self.tp.detokenize(q2_modified))
                dist_from_bound = np.abs(score - 0.5)
                if dist_from_bound < min_dist_from_bound:
                    min_dist_from_bound = dist_from_bound
                    new_q2 = q2_modified

        # print("Augmentation: replaced {} with {}".format(row[1], self.tp.detokenize(new_q2)))
        row[1] = self.tp.detokenize(new_q2)
        return row
