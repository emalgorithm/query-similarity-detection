import numpy as np

from text_processor import TextProcessor
from util import get_balanced_data
from glove_synonyms import GloveSynonyms
from substitute_model import SubstituteModel
from adversarial_algos import adversarial_white_box_change

class BlackBoxAdversarialAlgorithm:
    def __init__(self, oracle):
        self.oracle = oracle
        self.substitute_model = SubstituteModel()
        self.tp = TextProcessor()
        self.word_similarity = GloveSynonyms()

        # Start with 100 training examples
        X_train, _, _, _ = get_balanced_data()
        self.X_train = X_train[:100]
        self.n_st_epochs = 1

    def substitute_training(self):
        for i in range(self.n_st_epochs):
            # Label
            y_train = self.label(self.X_train)
            # Train
            self.substitute_model.train(self.X_train, y_train)
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
        X_train_aug = np.apply_along_axis(self.replace_word_with_greatest_change, 1, X_train)

        return np.concatenate((X_train, X_train_aug), 0)

    def replace_word_with_greatest_change(self, row):
        q1_tokenized = self.tp.tokenize(row[0])
        q2_tokenized = self.tp.tokenize(row[1])
        min_dist_from_bound = 1.0

        new_q2 = q2_tokenized

        for i, word in enumerate(q2_tokenized):
            if self.word_similarity.contains_word(word):
                closest_word = self.word_similarity.most_similar(word)
                q2_modified = list(q2_tokenized)
                q2_modified[i] = closest_word
                score = self.substitute_model.predict_single(self.tp.detokenize(q1_tokenized),
                                                             self.tp.detokenize(q2_modified))
                dist_from_bound = np.abs(score - 0.5)
                if dist_from_bound < min_dist_from_bound:
                    min_dist_from_bound = dist_from_bound
                    new_q2 = q2_modified

        print("Augmentation: replaced {} with {}".format(row[1], self.tp.detokenize(new_q2)))
        row[1] = self.tp.detokenize(new_q2)
        return row
