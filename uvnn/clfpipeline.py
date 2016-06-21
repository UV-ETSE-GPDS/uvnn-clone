from utils.preprocessors import BasicPreprocessor
from utils.readers import CsvReader
import numpy as np
import matplotlib.pyplot as plt

class Clfpipeline(object):
    ''' Abastract class which given  the data input, produces performance 
    metrics, and outputs weight and topology of the model. 
    It is asembled of the following components:
        - Reader
        - Preprocessor
        - Classifier
    '''

    def __init__(self, reader, classifier=None):
        ''' takes a json arg, extracts and builds parameters '''
        # TODO right now read from provide classifier later, build it 
        # up from configuration
        self.classifier = classifier
        self.reader = reader

    def set_classifier(self, classifier):
        self.classifier = classifier

    def prepare_data(self):
        self.X, self.y = self.reader.load_data()

        def hook(X, y): 
            #needed to substract 1 from labels
            return (X, y - 1)

        self.preprocessor = BasicPreprocessor(self.X, self.y, hook)
        self.preprocessor.preprocess_data()
        (self.X_train, self.y_train, self.X_dev, self.y_dev, self.X_test, 
                self.y_test) = self.preprocessor.get_splits(0.8, 0.1, 0.1)

        print self.X_test.shape, self.y_test.shape
        
    
    def train(self, **params):
        self.curve = self.classifier.train_sgd(self.X_train, self.y_train,  
                devX = self.X_dev, devy = self.y_dev, **params)
        #counts, costs, costdevs  = zip(*curve)
        y_hat_train = self.classifier.predict(self.X_train)
        y_hat_dev = self.classifier.predict(self.X_dev)
        y_hat_test = self.classifier.predict(self.X_test)
        print 'Accuracy on train', self.calc_accuracy(self.y_train, y_hat_train)
        print 'Accuracy on dev', self.calc_accuracy(self.y_dev, y_hat_dev)
        print 'Accuracy on test', self.calc_accuracy(self.y_test, y_hat_test)

    def calc_accuracy(self, y, y_hat):
        return np.count_nonzero(y == y_hat) / float(len(y_hat))

    def plot(self):
        # plot last training run
        # TODO can be moved outside of this module
        counts, costs, costdevs  = zip(*self.curve)

        plt.figure(figsize=(6,4))
        plt.plot(5*np.array(counts), costs, color='b', marker='o', linestyle='-', label=r"train_loss")
        plt.plot(5*np.array(counts), costdevs, color='g', marker='o', linestyle='-', label=r"validation_loss")

        #plt.title(r"Learning Curve ($\lambda=0.001$, costevery=5000)")
        plt.xlabel("SGD Iterations"); plt.ylabel(r"Average $J(\theta)$"); 
        plt.ylim(ymin=0, ymax=max(1.1*max(costs),3*min(costs)));
        plt.legend()

    
    def save_weights(self, filename):
        pass
