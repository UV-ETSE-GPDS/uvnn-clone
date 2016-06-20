
class ModelGenerator(object):
    ''' Abastract class which given  the data input, produces performance 
    metrics, and outputs weight and topology of the model. 
    It is asembled of the following components:
        - Reader
        - Preprocessor
        - Classifier
    '''

    def __init__(self, json_args):
        ''' takes a json arg, extracts and builds parameters '''
        if json_args['classifier'] == 'softmax_regression':
            pass
