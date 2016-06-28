##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):
    e = sqrt(6) / sqrt(n + m + 1)
    A0 = random.uniform(-e, e, (m, n))
    assert(A0.shape == (m,n))
    return A0



# Different batch strategies

def fullbatch(sample_count, nepoch):
    ''' full batch nepoch times '''
    for i in xrange(nepoch):
        yield range(sample_count)

def minibatch(sample_count, batch_size, epoch):
    ''' return random batch of batch_size for
    epoch * sample_count / batch_size times '''
    for _ in xrange (epoch * sample_count / batch_size):
        yield random.choice(range(sample_count), batch_size, replace=False)

def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result
