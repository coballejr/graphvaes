import numpy as np

def train_test_split(data, subset_size, train_prop, seed = 42):
    
    np.random.seed(seed)
    assert subset_size <= len(data)
    random_idx = np.random.choice(len(data), subset_size, replace = False)
    
    assert train_prop <= 1
    train_size = int(train_prop*subset_size)
    train_idx, test_idx = random_idx[0:train_size], random_idx[train_size:]
    
    train, test = [data[i] for i in train_idx], [data[i] for i in test_idx]
    
    return train, test