import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import multiprocessing


def get_balanced_data():
    np.random.seed(42)
    df_train = pd.read_csv('../data/train.csv').dropna()
    
    target = 'is_duplicate'
    features = ['question1', 'question2']

    X = df_train[features].as_matrix()
    y = df_train[target].as_matrix()

    neg_idx = np.where(y == 0)[0]
    pos_idx = np.where(y == 1)[0]

    neg_idx_downsampled = neg_idx[:len(pos_idx)]
    assert(len(neg_idx_downsampled) == 149263)

    idx = np.concatenate((pos_idx, neg_idx_downsampled), axis=0)
    np.random.shuffle(idx)
    assert(len(idx) == len(pos_idx) + len(neg_idx_downsampled))

    X = X[idx]
    y = y[idx]

    assert(len(y == 0) == len(y == 1))
    assert(len(idx) == 149263 * 2)
    
    print("We have {} positive samples".format(len(y == 1)))
    print("We have {} negative samples".format(len(y == 0)))
    
    return train_test_split(X, y, test_size=0.2, random_state=42)


def detokenize(q):
    return ' '.join(q)


def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

    pool = multiprocessing.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)


def unpacking_apply_along_axis(a):
    """
    Like numpy.apply_along_axis(), but and with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    func1d, axis, arr, args, kwargs = a
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


def load_glove():
    embeddings_index = {}
    f = open('../data/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index


def compute_embedding_matrix(word_index, embeddings_index, embedding_dim):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
