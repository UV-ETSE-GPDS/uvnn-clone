import numpy as np
import copy
import matplotlib.pyplot as plt
from Queue import PriorityQueue
from common import data_to_spike, prepare_dataset
from common import Param, Spike, str2bool, data_to_spike, prepare_dataset
from uvnn.utils.images import show_images

class SRBM_TS(object):
    ''' Spiking RBM, Time-stepped implementation '''
    
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

    def set_data(self, X, y):
        self.X = X
        self.y = y


    def run_network(self, X, y, args, evaluate=False, W_init=None, accuracies = []):
        ''' if only_evaluate is True it will only evaluate accuracy on the provided set '''
        logger = self.logger
        batch_size = args.batch_size
        

        num_train = X.shape[0]
        num_batches = (num_train - 1) / batch_size + 1

        vis_size = args.visible_size
        hid_size = args.hidden_size
        if args.train_supervised:
            vis_size += args.num_classes



        layer_sizes = [vis_size, hid_size, vis_size, hid_size]
        
        if evaluate:
            # evaluate accuracy, using uncupling pretrained network see page 29 
            logger.info('Running evaluation')
            layer_sizes.append(args.num_classes)
        
        if W_init is None:
            # use random weight
            W = np.random.random((vis_size, hid_size)) # network weights

        if evaluate is True:
            if W_init is None:
                logger.error('Weights should be provided to evaluate ' )

            # uncuple W and W_label
            W = W_init[:vis_size,:]
            W_label = W_init[vis_size:,:]
        
        for epoch in range(args.num_epoch):
            #logger.info('running Epoch #%d' %(epoch, )) 
            # initialize params
            membranes = []
            refrac_end = []
            last_spiked = []
            firings = []
            noises = []

            for lsz in layer_sizes:
                membranes.append(np.zeros((batch_size, lsz)))
                last_spiked.append(np.zeros((batch_size, lsz)))
                last_spiked[-1].fill(-10000) # not to cause first spikes
                refrac_end.append(np.zeros((batch_size, lsz)))
                refrac_end[-1].fill(-10000) # not to cause first spikes
                firings.append(np.zeros((batch_size, lsz)))
                noises.append(np.zeros((batch_size, lsz)))
           
            cur_time = 0
            classified_correctly = 0 # used when evaluating
            heatmap = np.zeros((args.num_classes, args.num_classes))
            for batch in range(num_batches):
                #if not evaluate:
                #    logger.info('Processing batch from #%d' %(batch * batch_size, ))
                label_firing_count = np.zeros((batch_size, args.num_classes)) # for classification only not for training
                for t in np.arange(cur_time, cur_time + args.timespan, args.dt): 
                    # get batch spikes
                    l = batch * batch_size
                    r = l + batch_size
                    in_current = X[l:r,:]
                    if y is not None:
                        lab_current = y[l:r]
                 
                    if args.train_supervised:
                        in_current_labels = np.zeros((batch_size, args.num_classes))
                        in_current_labels[range(batch_size), lab_current] = 0.5
                        in_current = np.hstack((in_current, in_current_labels))

                        
                        
                
                    rand_nums = np.random.random((batch_size, vis_size))
                    firings[0] =  (rand_nums < in_current)
                   # import ipdb; ipdb.set_trace()
                    last_spiked[0][firings[0]] = t
                    
                    for pop in range(1, len(layer_sizes)):
                        # decay membrane
                        membranes[pop] = membranes[pop] * np.exp(-args.dt / args.tau)
                         
                        # add impulses
                        if pop == 1 or pop == 3:
                            membranes[pop] += (t > refrac_end[pop]) *  np.dot(firings[pop - 1], W)
                        elif pop == 2:
                            # model visible layer
                            membranes[pop] += (t > refrac_end[pop]) * np.dot(firings[pop - 1], W.T)
                        elif pop == 4:
                            # this is a label layer directly connected to
                            # the hidden layer
                            membranes[pop] += (t > refrac_end[pop]) * np.dot(firings[1], W_label.T)
                        

                        # get firings
                        firings[pop] = membranes[pop] > args.thr
                        membranes[pop][firings[pop]] = 0
                        refrac_end[pop][firings[pop]] = t + args.t_refrac
                        last_spiked[pop][firings[pop]] = t

                        
                        if pop == 4:
                            # add label firing count 
                            label_firing_count += firings[pop]
                        
                    # now learn if not evaluating
                    
                    # stdp
                    if not evaluate and args.enable_update:
                        dWp = dWn = 0
                        if np.any(firings[1]):
                            dWp = args.eta * (np.dot((last_spiked[0] > t - args.stdp_lag).T, firings[1]))
                        if np.any(firings[3]):
                            dWn  =  args.eta * (np.dot((last_spiked[2] > t - args.stdp_lag).T, firings[3]))
                        
                        dW = (dWp - dWn) / batch_size
                        #if dW.any():
                        #    import ipdb; ipdb.set_trace()
                        W += dW

                cur_time = cur_time + args.timespan + args.t_gap
                for pop in range(1, len(layer_sizes)):
                    membranes[pop] *= np.exp(-args.t_gap / args.tau)
                
                if args.train_supervised:
                    examples_seen = (batch + 1) * batch_size
                    if examples_seen / args.test_every > (examples_seen - batch_size) / args.test_every:

                        logger.info('Processed #%d examples' %(examples_seen, ))
                        # time to evaluate network
                        newargs = copy.copy(args)
                        newargs.batch_size = 1
                        newargs.train_supervised = False
                        self.run_network(self.X_test, self.y_test, newargs, evaluate=True, 
                                W_init=W, accuracies=accuracies)
                
                if evaluate:
                    #print label_firing_count[0]
                    #print 'correct is ', lab_current[0]
                    y_hats = np.argmax(label_firing_count, axis=1)
                    classified_correctly += np.count_nonzero(y_hats == lab_current)
                    heatmap[y_hats, lab_current] += 1

        if evaluate:
            logger.info('Correctly classified %d out ouf %d' %(classified_correctly, len(y)))
            acc = classified_correctly / float (len(y))
            print acc
            accuracies.append(acc)
            plt.close()
            #plt.figure(figsize=(5, 5))
            print 'minmax', np.min(W), np.max(W), np.average(W)
            print heatmap
            show_images(W.T, 28, 28)
            plt.show(block=False)
        return {}, W


    def train_network(self):
        
        self.X_train, self.y_train, self.X_test, self.y_test =  prepare_dataset(self.X, self.y, self.args)
        self.logger.info('Dataset prepared, Ttrain, Test splits ready to use')
        accuracies = []
        self.run_network(self.X_train, self.y_train, self.args, evaluate=False, 
                accuracies=accuracies)


