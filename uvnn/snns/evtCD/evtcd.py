import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import copy
import sys
import argparse
import numpy as np
import myex
from uvnn.utils.readers import CsvReader
from collections import namedtuple
from Queue import PriorityQueue
from uvnn.utils.images import show_images
from collections import defaultdict
from common import Param, Spike, str2bool
from uvnn.utils.images import show_images


#unknown things
# stdp_lag ??
# inp_scale ??
# timespan ??
# min_thr ??
# axon_delay ?? 
parser = argparse.ArgumentParser(description='evtcd algorithm with simulation')

# Algorithm params
parser.add_argument("--eta", type=float, default=0.001, help="Learning rate")
parser.add_argument("--thresh_eta", type=float, default=0, 
        help="Threshold learning rate")
parser.add_argument("--numspikes", type=int, default=2000, 
        help="Number of spikes for one input sample during [timespan] seconds")
parser.add_argument("--numspikes_label", type=int, default=1000, 
        help="Number of spikes for label during [timespan] seconds (if trained in supervized way)")
parser.add_argument("--timespan", type=float, default=0.1, 
        help="durition in which [numspikes] of one sample digit happen")
parser.add_argument("--tau", type=float, default=0.005, 
        help="Membrane time constant (decay constant) ")
parser.add_argument("--thr", type=float, default=1, help="Threshold value")
parser.add_argument("--inp_scale", type=float, default=0.1, 
        help="value to use for the first layer membrane potential addition (Doesn't matter much)")
parser.add_argument("--t_refrac", type=float, default=0.005, 
        help="Refractory period")
parser.add_argument("--stdp_lag", type=float, default=0.004, 
        help="STDP window length")
parser.add_argument("--min_thr", type=float, default=-1, 
        help="minimum threshold")
parser.add_argument("--axon_delay", type=float, default=0.0001, 
        help="axon delay, how long the spike takes to travel to the next layer")
parser.add_argument("--t_gap", type=float, default=10, 
        help="time gap between different training samples")

# Network param s
parser.add_argument("--visible_size", type=int, default=784, 
        help="number of neurons in the visible layer")
parser.add_argument("--hidden_size", type=int, default=16, 
        help="number of neurons in the hidden layer, for visualisation better to be a square number")

# training params
parser.add_argument("--input_file", default=None) 
# by default kaggle set is used

parser.add_argument("--num_train", type=int, default=8, 
        help="number of samples to use for training")

parser.add_argument("--num_test", type=int, default=2, 
        help="number of samples to use for testing")

parser.add_argument("--simulate", type=str2bool, default=True, 
        help="simulation of training with visualisation (takes longer time)")

parser.add_argument("--shuffle", type=str2bool, default=True, 
        help="Whether or not should shuffle data set before training")

parser.add_argument("--batch_size", type=int, default=None, 
        help="Batch size of training samples")

parser.add_argument("--test_every", type=int, default=None, 
        help="In every how many sample, benchmarking on test set should occur")

parser.add_argument("--enable_update", type=str2bool, default=True, 
        help="determines whether weights are updated, useful when you already have learned weights")

parser.add_argument("--save_weights", default='temp.npy', 
        help="Path to save weights")

parser.add_argument("--load_weights", default=None, 
        help="Path to load weights")

parser.add_argument("--train_supervised", type=str2bool,  default=True,
        help="Train in a suprevized way, labels and num_classes must be provided")

parser.add_argument("--num_classes", type=int, default=10, 
        help="number of classes if the supervized training is enabled, labels start from 0")

# Logs, plots

parser.add_argument("--log_reconstr", type=str2bool, default=False, 
        help="logs reconstruction distributions for each sample")

parser.add_argument("--plot_curve", type=str2bool, default=False, 
        help="Plot training errors")

parser.add_argument("--show_weight_deltas", type=str2bool, default=False, 
        help="should only be used with simulate =True, visualizes weight updates")

arguments = parser.parse_args()


class SRBM(object):
    def __init__(self, args):
        self.args = args
        #self.W_dashboard = myex.DashBoard([], args.visible_size, args.hidden_size)
        #self.W_dashboard.run_vis() # just to draw windows
        #import ipdb; ipdb.set_trace()
        pass
    
    def load_data(self):
        ''' Load data from file specified by args, or load default dataset '''

        # y might be None as the algorithm is unsupervised 
        if args.input_file is None: 
            # by default we take kaggle 28x28 dataset
            if args.num_train > 10: #TODO remove later not needed
                csv_reader = CsvReader(fn='../../input/kaggle_mnist/train.csv',
                        has_header=True, label_pos=0)
            else:
                csv_reader = CsvReader(fn='../../input/kaggle_mnist/dev/train.csv',
                        has_header=True, label_pos=0)
            self.X, self.y = csv_reader.load_data()
        else:
            csv_reader = CsvReader(fn=args.input_file, has_header=False)
            self.X, self.y = csv_reader.load_data()
        print 'Loaded', self.X.shape

    def set_data(self, X, y=None):
        ''' Alternative way to set input data ''' 
        self.X = X
        self.y = y


    def data_to_spike(self, x, numspikes, timespan):
        ''' return pairs of spike_address, time during timespan '''
        probs = x / float(sum(x))
        spikes = np.random.choice(range(len(x)), numspikes, True, probs)
        times = np.sort(np.random.rand(numspikes) * timespan)
        return zip(spikes, times)

    def prepare_dataset(self):
        # Load mnist and convert them to spikes
        
        # trunc mnist for testing 
        #csv_reader = CsvReader(fn='../../input/trunc_mnist/trunc_mnist20x20_inputs.csv', 
        #        has_header=False, fn_labels='../../input/trunc_mnist/trunc_mnist20x20_targets.csv') #mnist 28

        #X, y = csv_reader.load_data()
        # normalize data
        #import ipdb; ipdb.set_trace()
        self.X = (self.X - np.min(self.X)) / float(np.max(self.X) - np.min(self.X))
        
        if self.args.shuffle:
            order = np.array(range(self.X.shape[0]))
            np.random.shuffle(order)
            self.y = self.y[order]
            self.X = self.X[order]

        num_test = self.args.num_test
        num_train = self.args.num_train
        # put away some part for testing purposes if desired
        self.X_test = self.X[:num_test]
        self.X_train = self.X[num_test: num_test + num_train]
        if self.y is not None: 
            self.y_test = self.y[:num_test]
            self.y_train = self.y[num_test: num_test + num_train]
    
    def evaluate_accuracy(self, W_full):
        new_args = copy.copy(self.args)
        new_args.log_reconstr = True

        cf = self.args 
        W = W_full[:cf.visible_size, :]
        W_label = W_full[cf.visible_size:, :]
        acc = self.run_network_classify(cf, W, W_label, self.X_test, self.y_test)
        
        plt.close()
        #plt.figure(figsize=(5, 5))
        show_images(W.T, 28, 28)
        plt.show(block=False)
        
        # visualize weights
        #self.W_dashboard.update_all_weights_plot(W)
        return acc


    def evaluate_on_test(self, W):
        ''' evaluate on a test data for weights W,
            For now it will be the average L2 norm of distribution difference 
            between true spike and network spike distributions
        '''
        new_args = copy.copy(self.args)
        new_args.log_reconstr = True

        new_args.num_train = len(self.X_test)   # our test is the train for new network
        new_args.num_test = 0
        new_args.enable_update = False
        new_args.simulate = False
        new_args.plot_curve = False
        new_args.test_every = None
        new_args.train_supervised = False 
        srbm = SRBM(new_args) # create new network
        #import ipdb; ipdb.set_trace()
        srbm.set_data(self.X_test)
        history, _ = srbm.run_network(init_weights = np.copy(W[:784]))

        # extract all reconstr distributions
        
        recon_distributions  = []
        real_distributions = []
        num_test = len(self.X_test)
        for time, events in sorted(history.items()):
            for event in events:
                if event[0] == 'RECON_DISTR':
                    distr = event[1]
                    recon_distributions.append(distr)
                elif event[0] == 'NEW_SPIKE_TRAIN':
                    distr = event[1]
                    real_distributions.append(distr)

        assert len(recon_distributions) == len(real_distributions) == num_test
        
        errors = []
        for real, recon in zip(real_distributions, recon_distributions):
            # calculate difference
            diff = np.sum((real - recon) ** 2) / len(real)
            errors.append(diff)
        
        # return average error on all tests
        avg_err = sum(errors) / float(num_test)
        print 'Testing.. average error is', avg_err
        return avg_err
    
    def run_network_classify(self, cf, W, W_label, X_test, 
            y_test=None):
        # run three layer architectur visible, hidden, label to classify 
        # samples in a supervised training
        
        pq = PriorityQueue()
        
        membranes = []
        last_spiked = []
        refrac_end = []
        last_update = []
        
        vis_size = cf.visible_size
        hid_size = cf.hidden_size
        label_size = cf.num_classes
        sizes = (vis_size, hid_size, label_size)
        
        
        # init network 
        for layer_size in sizes:
            membranes.append(np.zeros(layer_size))
            last_spiked.append(np.zeros(layer_size))
            last_spiked[-1].fill(-100)  # not to cause first spikes accidentally
            last_update.append(np.zeros(layer_size))
            refrac_end.append(np.zeros(layer_size))
    

        t_passed = 0
        correct = 0
        for (sample_num, x) in enumerate(X_test):
            spike_train_sample = self.data_to_spike(x, cf.numspikes, cf.timespan)

            for addr, time in spike_train_sample:
                # spike fired from '-1th' layer
                pq.put(Spike(time=time + t_passed, layer=0, address=addr))               

            t_passed = t_passed + cf.timespan + cf.t_gap

            spike_count_label = np.zeros(label_size) # count spikes from label layer

            while not pq.empty():
                spike_triplet = pq.get()
                
                sp_time = spike_triplet.time
                sp_layer = spike_triplet.layer
                sp_address = spike_triplet.address

                layer = sp_layer + 1

                if layer == 1:
                    last_spiked[0][sp_address] = sp_time - cf.axon_delay
                
                membranes[layer] *= np.exp(-(sp_time - last_update[layer]) / cf.tau)

                # add impules
                if layer == 1:
                    # hidden
                    membranes[layer] += W[sp_address, :] * (refrac_end[layer] < sp_time)
                elif layer == 2:
                    # hidden
                    membranes[layer] += W_label[:, sp_address] * (refrac_end[layer] < sp_time)
                
                last_update[layer] = sp_time

                newspikes = np.nonzero(membranes[layer] > cf.thr)[0]

                refrac_end[layer][newspikes] = sp_time + cf.t_refrac
                membranes[layer][newspikes] = 0
                 
                if layer == 2:
                    spike_count_label[newspikes] += 1
                     
                for newspike in newspikes:
                    rand_delay = np.random.random() * cf.axon_delay * 2
                    new_time = sp_time + cf.axon_delay + rand_delay
                    new_triplet = Spike(time= new_time, layer=layer, address=newspike)
                    if (layer != 2):
                        pq.put(new_triplet)
            
            #print spike_count_label
            y_hat = np.argmax(spike_count_label)
            y = y_test[sample_num]
            
            if y_hat == y:
                correct += 1
            #else:
                #print 'mistake mixed', y_hat, y
        
        acc = float(correct) / len(y_test)
        print 'Accuracy - ', acc

        # visualize weights

        return  acc

    def run_network(self, init_weights=None):
        pq = PriorityQueue()
        
        ''' keep_recon_distr : keeps a list of reconstructed distributions for
        each training sample '''

        self.prepare_dataset()
        cf = self.args # configurations 
        
        membranes = []
        last_spiked = []
        refrac_end = []
        last_update = []
        spike_count = [0, 0, 0, 0]
        calc_recons = [False, False, False, False]
        thr = []

        vis_size = cf.visible_size
        hid_size = cf.hidden_size
        
        # increase number of vis_size if training via supervized way
        # see page 29 in thesis, for the picture
        if cf.train_supervised:
            # last num_classes number of input vector will be preserved for
            # labels
            vis_size += cf.num_classes

        
        if init_weights is None:
            # check whether it should be loaded
            if cf.load_weights is not None:
                W = np.load(cf.load_weights)
            else:
                # fall back to random init
                W = np.random.rand(vis_size, hid_size)
        else:
            W = init_weights
        
        
        batch_W_delta = np.zeros((vis_size, hid_size)) # used for batch updates

        # history keeps track of spikes/potential_change/ .. etc, it is dict
        # with key - timestep, (rounded to cf.prec digits), and the value is the
        # list of events which occured on this timestep (the list will
        # have size 1 unless several spikes occured simultiniously. Each item in 
        # list is a tuple, first item in tuple is a string - event type, others 
        # will be event description for example in case of 'spike' other items in 
        # the tuple will be layer and address. Tuple descriptions are below.
        # -----------------------
        # ('SPIKE', layer, address)
        # ('THR', layer, address, newvalue)
        # ('MEMBRANE', layer, layer_values)
        history = defaultdict(list)

        self.log(history, 0, ('INIT_WEIGHTS', W))

        # Data and model layers
        for layer in range(4):
            layer_size = vis_size if layer % 2 == 0 else hid_size
            
            membranes.append(np.zeros(layer_size))
            last_spiked.append(np.zeros(layer_size))
            last_spiked[-1].fill(-100)  # not to cause first spikes accidentally
            last_update.append(np.zeros(layer_size))
            refrac_end.append(np.zeros(layer_size))
            th_single = np.zeros(layer_size)
            th_single.fill(cf.thr)
            thr.append(th_single)
        
        t_passed = 0
        self.errors = []   # errors on test set
        for (sample_num, x) in enumerate(self.X_train):
            
            if cf.batch_size is not None and sample_num != 0 and sample_num % cf.batch_size == 0:
                # perform batch update
                W += batch_W_delta / cf.batch_size
                batch_W_delta.fill(0)

            if cf.test_every is not None and (sample_num + 1) % cf.test_every == 0:
                # run current network on test set for training curve
                #import ipdb; ipdb.set_trace()
                print 'Trained on %d, W - max: %.3f, min %.3f, avg %.3f' % (sample_num + 1, 
                        np.max(W), np.min(W), np.average(W))
                self.errors.append(self.evaluate_accuracy(W))


            spike_train_sample = self.data_to_spike(x, cf.numspikes, cf.timespan)
            # spike train is a one digit encoded to pairs (spike_address, time)
            # example digit 8 can be represented ((2, 12), (2, 13), (4, 14) ...)
        
            spike_image = np.zeros(vis_size)
            for addr, time in spike_train_sample:
                # spike fired from '-1th' layer
                pq.put(Spike(time=time + t_passed, layer=0, address=addr))               
                spike_image[addr] += 1

            if cf.train_supervised:
                # add label spikes
                step = cf.timespan / float(cf.numspikes_label)
                # spikes for the labels are in the end
                addr = vis_size - 10 + self.y_train[sample_num]

                for time in np.arange(0, cf.timespan, step):
                    pq.put(Spike(time=time + t_passed, layer=0, address=addr))
            
            # add spike image to history, but normalize first
            if cf.simulate or cf.log_reconstr:
                history[t_passed].append(('NEW_SPIKE_TRAIN', 
                    spike_image / float(np.max(spike_image))))

            t_passed = t_passed + cf.timespan + cf.t_gap

            last_time = -1
            
            #reconstraction spike distribution for each sample, 
            # needed for benchmarking
            recon_distr = np.zeros(vis_size)      

            while not pq.empty():
                spike_triplet = pq.get()
                #import ipdb; ipdb.set_trace()
                self.process_spike(spike_triplet, cf, pq, thr, last_spiked, membranes,
                        last_update, refrac_end, spike_count, calc_recons, W, history, 
                        recon_distr, batch_W_delta)
            
            if cf.log_reconstr:
                history[t_passed].append(('RECON_DISTR', 
                    recon_distr / float(np.max(recon_distr))))

            
            
            if cf.simulate:
                # log weight update
                for hidd in range(hid_size):
                    weights_to_log = np.copy(W[:, hidd])
                    self.log(history, t_passed, ('UPDATE_WEIGHTS', hidd, 
                        weights_to_log, False))
        

        # plot errors 
        if cf.plot_curve:
            fig = plt.figure()
            fig.suptitle('\n'.join(str(self.args).split(',')), fontsize=8)
            ax2 = fig.add_subplot(111)
            plt.plot(self.errors)
            plt.show()
        
        return history, W

    def add_noise(self, membrane, layer_num):
        pass

    def log(self, history, time, data):
        history[round(time, 6)].append(data)

    def process_spike(self, spike_triplet, cf, pq, thr, last_spiked, membranes, 
            last_update, refrac_end, spike_count, calc_recons, weights, history,
            recon_distr, batch_weight_delta):
        ''' Params:
                spike triplet - has a form of (time, layer, address) 
                cf - configs
                pq - priority queue of spike trains
        '''
        
        #import ipdb; ipdb.set_trace() # BREAKPOINT
        sp_time = spike_triplet.time
        sp_layer = spike_triplet.layer
        sp_address = spike_triplet.address
        if cf.simulate:
            self.log(history, sp_time, ('SPIKE', sp_layer, sp_address))
        layer = sp_layer + 1

        # Reconstruct the imaginary first-layer action that resulted in this spike
        if layer == 1:
            last_spiked[0][sp_address] = sp_time - cf.axon_delay
            if thr[0][sp_address] < cf.min_thr:
                thr[0][sp_address] = cf.min_thr
            else:
                thr[0][sp_address] -= cf.eta * cf.thresh_eta
            spike_count[0] += 1
            if calc_recons[0]:
                recon[0] *= np.exp(- (spike_triplet.time - last_recon[0]) / cf.recon_tau)
                recon[0][sp_address] += recon_imp
                last_recon[0] = sp_time - cf.axon_delay

        # update neurons
        
        #decay membrane

        membranes[layer] *= np.exp(-(sp_time - last_update[layer]) / cf.tau)


        #add impulse
        if layer == 0:
            membranes[layer][sp_address] += cf.inp_scale
        elif layer == 2:
            # visible layer
            membranes[layer] += weights[:, sp_address] * (refrac_end[layer] < sp_time)
        elif layer == 1 or layer == 3:
            # hidden layer
            membranes[layer] += weights[sp_address, :] * (refrac_end[layer] < sp_time)


        # add noise
        self.add_noise(membranes[layer], layer) 
       
        if cf.simulate:
            self.log(history, sp_time, ('MEMBRANE', layer, membranes[layer] / thr[layer%2]))

        # update last_update

        last_update[layer] = sp_time

        # Add firings to queue
        newspikes = np.nonzero(membranes[layer] > thr[layer % 2])[0]
        
        if layer == 2:
            # model_visible layer
            recon_distr[newspikes] += 1
        
        for newspike in newspikes:
            #import ipdb; ipdb.set_trace() # BREAKPOINT
            # update counts

            spike_count[layer] += 1

            # update refrac end
            refrac_end[layer][newspike] = sp_time + cf.t_refrac

            # reset firings
            membranes[layer][newspike] = 0

            # reset time for stdp
            last_spiked[layer][newspike] = sp_time

            # stdp threshold adjustment

            thr_direction = -1 if layer < 2 else 1
            wt_direction = 1 if layer < 2 else -1
            thr[layer % 2][newspike] += thr_direction * cf.eta * cf.thresh_eta

            thr[layer % 2][thr[layer%2] < cf.min_thr] = cf.min_thr


            # STDP weight adjustment
            if (layer % 2 == 1):
                #import ipdb; ipdb.set_trace()
                weight_adj = (last_spiked[layer - 1] > (sp_time - cf.stdp_lag)) * (wt_direction * cf.eta)

                if cf.enable_update:
                    # check if batch training is on
                    if cf.batch_size is None:
                        # if its turned off update immediately
                        weights[:, newspike] += weight_adj
                    else:
                        # else save update for later
                        batch_weight_delta[:, newspike] += weight_adj

                if cf.simulate and cf.show_weight_deltas and np.any(weight_adj):
                    # log weight update
                    weights_to_log = np.copy(weights[:, newspike])
                    self.log(history, sp_time, ('UPDATE_WEIGHTS', newspike, weights_to_log))
                #print np.max(weight_adj) ,np.min(weight_adj)

            # reconstruct the layer if desired
            if calc_recons[layer]:
                recon[layer] *= np.exp(- (sp_time - last_recon[layer]) / recon_tau)
                recon[layer][newspike] += recon_imp
                last_recon[layer] = sp_time
            
            # add spikes to the queue if not in the end layer
            # also add  random delay 
            rand_delay = np.random.random() * cf.axon_delay * 2
            new_time = sp_time + cf.axon_delay + rand_delay
            new_triplet = Spike(time= new_time, layer=layer, address=newspike)
            if cf.simulate:
                self.log(history, new_time, ('SPIKE', layer, newspike))

            if (layer != 3):
                pq.put(new_triplet)


if __name__ == '__main__':
    np.random.seed(2)           # to rerun same experiments
    args = parser.parse_args()  # get algorithm arguments from cmd
    srbm = SRBM(args) # create and initialize spiking rbm network
    srbm.load_data()
    #import ipdb; ipdb.set_trace()
    history, weights = srbm.run_network()
    np.save(args.save_weights, weights)
    
    dashboard = myex.DashBoard(sorted(history.items()), args.visible_size, args.hidden_size)
    #dashboard.plot_reconstr_accuracy()
    if args.simulate:
        dashboard.run_vis()
