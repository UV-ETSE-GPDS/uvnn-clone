##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):
    e = sqrt(6) / sqrt(n + m)
    A0 = random.uniform(-e, e, (m, n))
    assert(A0.shape == (m,n))
    return A0
