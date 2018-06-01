def adversarial_white_box_change(q1, q2, model, tp, w2v, threshold=0.4):
    """
    'q1' and 'q2' are the two questions which are detected as similar by the classifier: score >= 0.4.
    'model' is the classifier used by the oracle, which in this case returns a similarity score
      between 0 and 1 since it is an open box algorithm.
    The function changes 'q2' so that it retains the same meaning and is now detected as not similar to q2
    """    
    q1_tokenized = tp.tokenize(q1)
    q2_tokenized = tp.tokenize(q2)
    successful = False
    replaced_words = 0
    print("initial q1: {}".format(q1))
    print("initial q2: {}".format(q2))
    print("Initial similarity is {}".format(model(tp.detokenize(q1_tokenized), tp.detokenize(q2_tokenized))))
    
    while not successful or replaced_words >= 5:
        # Try changing each word in q2. At the end, select the change that gives the best improvement
        min_score = model(tp.detokenize(q1_tokenized), tp.detokenize(q2_tokenized))
        new_q2 = q2_tokenized
        
        for i, word in enumerate(q2_tokenized):
            if word in w2v.model.vocab:
                closest_word = w2v.get_closest_word(word)
                q2_modified = list(q2_tokenized)
                q2_modified[i] = closest_word
                score = model(tp.detokenize(q1_tokenized), tp.detokenize(q2_modified))

                if score < min_score:
                    min_score = score
                    new_q2 = q2_modified
                    
        if new_q2 == q2_tokenized:
            break
        
        if min_score < model(tp.detokenize(q1_tokenized), tp.detokenize(q2_tokenized)):
            q2_tokenized = new_q2
            replaced_words += 1
            
        
        print()
        print("q1: '{}'".format(tp.detokenize(q1_tokenized)))
        print("q2: '{}'".format(tp.detokenize(new_q2)))
        print("Replacing {} words we have a similarity of {}".format(replaced_words, min_score))
        
        if min_score < threshold:
            successful = True
            
    if not successful:
        print("UNSUCCESSFULL")