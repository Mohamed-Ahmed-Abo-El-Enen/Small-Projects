import numpy as np
np.random.seed(1)

def sample_Z(m, n):
    return np.random.uniform(-1, 1, size=[m, n])