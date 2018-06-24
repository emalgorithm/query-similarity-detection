import time

import numpy as np
from algorithms.white_box_adversarial_algorithm import WhiteBoxAdversarialAlgorithm
from word_similarity.glove_synonyms import GloveSynonyms
from models.substitute_model import SubstituteModel
from nltk.corpus import stopwords
from util.text_processor import TextProcessor

from util.util import get_balanced_data


class BlackBoxAdversarialAlgorithm:
    """
    Class which represents an API for our black-box adversarial attack.
    """
    def __init__(self, oracle, word_similarity=GloveSynonyms(), n_initial_train=1000,
                 n_test=1000, n_st_epochs=5, similarity_threshold=0.5, modify_q1=False,
                 modify_q2=True):
        """
        :param oracle: query similarity model we are trying to fool
        :param word_similarity: word embedding object
        :param n_initial_train: size of the initial training set
        :param n_test: number of data points to use when evaluating the method
        :param n_st_epochs: number of substitute epochs to carry out
        :param similarity_threshold: threshold above which two queries are defined as similar
        :param modify_q1: whether the adversarial algorithm can change the first question of the
        pair
        :param modify_q2: whether the adversarial algorithm can change the second question of the
        pair
        """
        # Oracle we are trying to attack. It takes a pair of questions as input and returns
        # whether they are similar (1) or not similar (0)
        self.oracle = oracle
        # Substitute model we use to emulate the oracle.
        self.substitute_model = SubstituteModel()
        # A util to perform text related operations
        self.tp = TextProcessor()
        # A utility which contains a vector embedding for words
        self.word_similarity = word_similarity

        # White-box algorithm that we will use on the substitute model
        self.white_box_algorithm = WhiteBoxAdversarialAlgorithm(self.substitute_model, self.tp,
                                                                self.word_similarity)

        self.similarity_threshold = similarity_threshold
        self.n_st_epochs = n_st_epochs
        self.n_test = n_test
        self.current_st_epoch = 1
        self.modify_q1 = modify_q1
        self.modify_q2 = modify_q2

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
        start_time = time.time()
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

        end_time = time.time()
        print("Evaluation done. It took {0:.4} seconds".format(end_time - start_time))
        print("{} attacks were successful out of {} tried".format(sum(results), len(results)))
        print("Transferability of black box attack model after epoch {0} is {1:.2f}%".format(
            self.current_st_epoch,
            transferability * 100))
        return transferability

    def substitute_training(self):
        """
        Carry out self.n_st_epochs epochs of substitute training
        """
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
            self.current_st_epoch += 1

    def attack(self, q1, q2):
        """
        Try attacking the oracle on a given pair of questions which the oracle classifies as similar
        :param q1: first question of the pair
        :param q2: second question of the pair
        :return: None if we are not able to fool the substitute model. Otherwise, if we are able
        to fool the substitute model, then we return True if the adversarial example for the
        substitute model fools the oracle as well, or false otherwise.
        """
        # Craft adversarial example using white-box algorithm on substitute model
        success, adv_q1, adv_q2 = self.craft_adversarial_example(q1, q2)

        if success:
            # Use previously crafted adversarial example to try to fool the oracle
            return self.oracle.predict_single(adv_q1, adv_q2) < 0.5
        else:
            return None

    def craft_adversarial_example(self, q1, q2):
        """
        Modify q1 and q2 so that they fool the substitute model.
        :param q1: first question of the pair
        :param q2: second question of the pair
        :return: crafted adversarial example
        """
        return self.white_box_algorithm.attack(q1, q2)

    def label(self, X_train):
        """Label the current training examples by querying the oracle"""
        return self.oracle.predict(X_train) > 0.5

    def augment(self, X_train):
        """
        Augment current training data with examples which are closer to boundary (score of
        0.5) following our "gradient based dataset augmentation algorithm"
        """
        print("Augmenting training data")
        start_time = time.time()
        X_train_aug = np.apply_along_axis(self.create_new_question_pair, 1, X_train)
        end_time = time.time()
        print("Augmenting done. It took {0:.4} seconds".format(end_time - start_time))

        return np.concatenate((X_train, X_train_aug), 0)

    def get_question_closer_to_boundary(self, questions_pair, n_top_words=3):
        """
        :param q_1: First question of the pair. Used to get the similarity score with q_2
        :param q_2: Second question of the pair. Used as a starting point to generate new
        questions
        :param n_new_questions: How many new questions to generate
        :param n_top_words: How many words to try when replacing a word with its closest words
        :return: n_new_questions questions which are the ones closer to the boundary of the classifier (
        score=0.5) out of all the sentences generated from q_2 by replacing one word with a synonym.
        """
        q1_tokenized = self.tp.tokenize(questions_pair[0])
        q2_tokenized = self.tp.tokenize(questions_pair[1])

        new_q2 = q2_tokenized
        min_dist_from_bound = 1.0

        for i, word in enumerate(q2_tokenized):
            if self.can_be_replaced(word):
                closest_words = self.word_similarity.most_similar(word, n=n_top_words)
                for close_word in closest_words:
                    q2_modified = list(q2_tokenized)
                    q2_modified[i] = close_word
                    score = self.substitute_model.predict_single(self.tp.detokenize(q1_tokenized),
                                                                 self.tp.detokenize(q2_modified))
                    dist_from_bound = np.abs(score - 0.5)
                    if dist_from_bound < min_dist_from_bound:
                        min_dist_from_bound = dist_from_bound
                        new_q2 = q2_modified

        questions_pair[1] = self.tp.detokenize(new_q2)
        return questions_pair

    def create_new_question_pair(self, row):
        """
        :param q_1: First question of the pair. Used to get the similarity score with q_2
        :param q_2: Second question of the pair. Used as a starting point to generate new
        questions
        :param n_new_questions: How many new questions to generate
        :param n_top_words: How many words to try when replacing a word with its closest words
        :return: n_new_questions questions which are the ones closer to the boundary of the classifier (
        score=0.5) out of all the sentences generated from q_2 by replacing one word with a synonym.
        """
        q1_tokenized = self.tp.tokenize(row[0])
        q2_tokenized = self.tp.tokenize(row[1])
        min_dist_from_bound = 1.0

        if self.modify_q2:
            new_q2 = q2_tokenized

            for i, word in enumerate(q2_tokenized):
                if self.can_be_replaced(word):
                    closest_word = self.word_similarity.most_similar(word)
                    q2_modified = list(q2_tokenized)
                    q2_modified[i] = closest_word
                    score = self.substitute_model.predict_single(self.tp.detokenize(q1_tokenized),
                                                                 self.tp.detokenize(q2_modified))
                    dist_from_bound = np.abs(score - 0.5)
                    if dist_from_bound < min_dist_from_bound:
                        min_dist_from_bound = dist_from_bound
                        new_q2 = q2_modified
            row[1] = self.tp.detokenize(new_q2)

        if self.modify_q1:
            new_q1 = q1_tokenized

            for i, word in enumerate(q1_tokenized):
                if self.can_be_replaced(word):
                    closest_word = self.word_similarity.most_similar(word)
                    q1_modified = list(q1_tokenized)
                    q1_modified[i] = closest_word
                    score = self.substitute_model.predict_single(
                        self.tp.detokenize(q1_modified),
                        self.tp.detokenize(new_q2))
                    dist_from_bound = np.abs(score - 0.5)
                    if dist_from_bound < min_dist_from_bound:
                        min_dist_from_bound = dist_from_bound
                        new_q1 = q1_modified

            row[0] = self.tp.detokenize(new_q1)
        return row

    def can_be_replaced(self, word):
        stop = set(stopwords.words('english'))
        return self.word_similarity.contains_word(word) and word not in stop
