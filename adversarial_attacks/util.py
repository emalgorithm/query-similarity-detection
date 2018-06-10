import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

def get_balanced_data():
    np.random.seed(42)
    df_train = pd.read_csv('data/train.csv').dropna()
    
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