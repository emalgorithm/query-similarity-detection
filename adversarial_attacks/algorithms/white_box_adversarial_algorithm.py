from nltk.corpus import stopwords


class WhiteBoxAdversarialAlgorithm:
    def __init__(self, model, tp, word_similarity, threshold=0.5,
                                 max_replaced_words=5, replace_stop_words=False, verbose=0):
        self.model = model
        self.tp = tp
        self.word_similarity = word_similarity
        self.threshold = threshold
        self.max_replaced_words = max_replaced_words
        self.replace_stop_words = replace_stop_words
        self.verbose = verbose

    def attack(self, q1, q2):
        """
        'q1' and 'q2' are the two questions which are detected as similar by the classifier: score >= 0.5.
        'model' is the classifier used by the oracle, which in this case returns a similarity score
          between 0 and 1 since it is an open box algorithm.
        The function changes 'q2' so that it retains the same meaning and is now detected as not similar to q2
        """
        stop = set(stopwords.words('english'))
        q1_tokenized = self.tp.tokenize(q1)
        q2_tokenized = self.tp.tokenize(q2)
        successful = False
        replaced_words = []
        replacing_words = []

        if self.verbose == 1:
            print("initial q1: {}".format(q1))
            print("initial q2: {}".format(q2))
            print("Initial similarity is {0:.3f}".format(self.model.predict_single(self.tp.detokenize(
                q1_tokenized), self.tp.detokenize(q2_tokenized))))

        while not successful or len(replaced_words) >= self.max_replaced_words:
            # Try changing each word in q2. At the end, select the change that gives the best improvement
            min_score = self.model.predict_single(self.tp.detokenize(q1_tokenized), self.tp.detokenize(q2_tokenized))
            new_q2 = q2_tokenized
            candidate_replaced_word = None

            for i, word in enumerate(q2_tokenized):
                can_replace = word not in stop or self.replace_stop_words
                if self.word_similarity.contains_word(word) and can_replace:
                    closest_word = self.word_similarity.most_similar(word)
                    q2_modified = list(q2_tokenized)
                    q2_modified[i] = closest_word
                    score = self.model.predict_single(self.tp.detokenize(q1_tokenized), self.tp.detokenize(q2_modified))

                    if score < min_score:
                        min_score = score
                        new_q2 = q2_modified
                        candidate_replaced_word = word
                        candidate_replacing_word = closest_word

            if new_q2 == q2_tokenized:
                break

            if min_score < self.model.predict_single(self.tp.detokenize(q1_tokenized), self.tp.detokenize(q2_tokenized)):
                q2_tokenized = new_q2
                replaced_words.append(candidate_replaced_word)
                replacing_words.append(candidate_replacing_word)

            if self.verbose == 1:
                print()
                print("Initial q1: '{}'".format(self.tp.detokenize(q1_tokenized)))
                print("Updated q2: '{}'".format(self.tp.detokenize(new_q2)))
                print("New similarity: {0:.3f}".format(min_score))
                for i, replaced_word in enumerate(replaced_words):
                    print("Replaced word '{}' with '{}'".format(replaced_word, replacing_words[i]))

            if min_score < self.threshold:
                successful = True
                break

        if self.verbose == 1:
            result = "SUCCESSFUL" if successful else "UNSUCCESSFUL"
            print(result)

        return successful, q1, self.tp.detokenize(new_q2)
