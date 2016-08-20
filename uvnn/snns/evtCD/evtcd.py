import time
import sys
import argparse
import numpy as np
import myex
from uvnn.utils.readers import CsvReader
from collections import namedtuple
from Queue import PriorityQueue
import matplotlib.pyplot as plt
from uvnn.utils.images import show_images
from collections import defaultdict
from common import Param, Spike, str2bool

#unknown things
# stdp_lag ??
# inp_scale ??
# timespan ??
# min_thr ??
# axon_delay ?? 
parser = argparse.ArgumentParser(description='evtcd algorithm with simulation')

# Algorithm params
parser.add_argument("--eta", type=float, default=0.01, help="Learning rate")
parser.add_argument("--thresh_eta", type=float, default=0, 
        help="Threshold learning rate")
parser.add_argument("--numspikes", type=int, default=500, 
        help="Number of spikes for one input sample during [timespan] seconds")
parser.add_argument("--timespan", type=float, default=0.1, 
        help="durition in which [numspikes] of one sample digit happen")
parser.add_argument("--tau", type=float, default=0.005, 
        help="Membrane time constant (decay constant) ")
parser.add_argument("--thr", type=float, default=1, help="Threshold value")
parser.add_argument("--inp_scale", type=float, default=0.1, 
        help="value to use for the first layer membrane potential addition (Doesn't matter much)")
parser.add_argument("--t_refrac", type=float, default=0.001, 
        help="Refractory period")
parser.add_argument("--stdp_lag", type=float, default=0.006, 
        help="STDP window length")
parser.add_argument("--min_thr", type=float, default=-1, 
        help="minimum threshold")
parser.add_argument("--axon_delay", type=float, default=0.0001, 
        help="axon delay, how long the spike takes to travel to the next layer")
parser.add_argument("--t_gap", type=float, default=4, 
        help="time gap between different training samples")

# Network params
parser.add_argument("--visible_size", type=int, default=784, 
        help="number of neurons in the visible layer")
parser.add_argument("--hidden_size", type=int, default=16, 
        help="number of neurons in the hidden layer")

# training params
parser.add_argument("--input_file", default=None) 
# by default kaggle set is used

parser.add_argument("--num_samples", type=int, default=10, 
        help="number of samples to use for the training")

parser.add_argument("--simulate", type=str2bool, default=True, 
        help="simulation of training with visualisation (teakes longer time)")




cf = Param(eta=0.01, thresh_eta=0, numspikes=500, timespan=0.1, tau=0.005, 
        thr=1, inp_scale=0.1, t_refrac=0.001, stdp_lag=0.006, min_thr=-1,
        plot_things=False, axon_delay=0.0001, t_gap = 4)

arguments = parser.parse_args()


class SRBM(object):
    def __init__(self, args):
        self.args = args

        pass
    
    def load_data(self):
        ''' Load data from file specified by args, or load default dataset '''

        # y might be None as the algorithm is unsupervised 
        if args.input_file is None: 
            # by default we take kaggle 28x28 dataset
            csv_reader = CsvReader(fn='../../input/kaggle_mnist/dev/train.csv',
                    has_header=True, label_pos=0)
            self.X, self.y = csv_reader.load_data()
        else:
            csv_reader = CsvReader(fn=args.input_file, has_header=False)
            self.X, self.y = csv_reader.load_data()


    def data_to_spike(self, x, numspikes, timespan):
        ''' return pairs of spike_address, time during timespan '''
        probs = x / float(sum(x))
        spikes = np.random.choice(range(len(x)), numspikes, True, probs)
        times = np.sort(np.random.rand(numspikes) * timespan)
        return zip(spikes, times)

    def prepare_spike_trains(self, numspikes=100, timespan = 10):
        # Load mnist and convert them to spikes
        
        # trunc mnist for testing 
        #csv_reader = CsvReader(fn='../../input/trunc_mnist/trunc_mnist20x20_inputs.csv', 
        #        has_header=False, fn_labels='../../input/trunc_mnist/trunc_mnist20x20_targets.csv') #mnist 28

        #X, y = csv_reader.load_data()
        X, y = self.X, self.y
        # normalize data
        X = (X - np.min(X)) / float(np.max(X) - np.min(X))
        # remove low valued pixels (truncated mnist problem)
        #X[X < 0.12] = 0
        
        reshuffle = True # weather or not reshuffle array
        if reshuffle:
            order = np.array(range(X.shape[0]))
            np.random.shuffle(order)
            X = X[order]
        #import ipdb; ipdb.set_trace()
        #X -= np.min(X)
        #import ipdb; ipdb.set_trace()
        #spike_trains = np.apply_along_axis(img_to_spike, 1, X, numspikes, timespan)
        # use vectorized form later
        spike_trains = [self.data_to_spike(x, numspikes, timespan) for x in X]

        return spike_trains, y


    def train_network(self, cf, vis_size, hid_size, num_samples):
        pq = PriorityQueue()
        spike_trains, labels = self.prepare_spike_trains(cf.numspikes, cf.timespan)
        
        membranes = []
        last_spiked = []
        refrac_end = []
        last_update = []
        spike_count = [0, 0, 0, 0]
        calc_recons = [False, False, False, False]
        thr = []
        W = np.random.rand(vis_size, hid_size)
        
        
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
        for spike_train_sample in spike_trains[:num_samples]:

            # spike train is a one digit encoded to pairs (spike_address, time)
            # example digit 8 can be represented ((2, 12), (2, 13), (4, 14) ...)
        
            spike_image = np.zeros(vis_size)
            for spike, time in spike_train_sample:
                # spike fired from '-1th' layer
                pq.put(Spike(time=time + t_passed, layer=0, address=spike))               
                spike_image[spike] += 1
            
            # add spike image to history, but normalize first
            history[t_passed].append(('NEW_SPIKE_TRAIN', 
                spike_image / np.max(spike_image)))

            t_passed = t_passed + cf.timespan + cf.t_gap

            last_time = -1
            while not pq.empty():
                spike_triplet = pq.get()
                #import ipdb; ipdb.set_trace()
                self.process_spike(spike_triplet, cf, pq, thr, last_spiked, membranes,
                        last_update, refrac_end, spike_count, calc_recons, W, history)

        # visualize spikes
        return history

    def add_noise(self, membrane, layer_num):
        pass

    def log(self, history, time, data):
        history[round(time, 6)].append(data)

    def process_spike(self, spike_triplet, cf, pq, thr, last_spiked, membranes, 
            last_update, refrac_end, spike_count, calc_recons, weights, history):
        ''' Params:
                spike triplet - has a form of (time, layer, address) 
                cf - configs
                pq - priority queue of spike trains
        '''
        
        #import ipdb; ipdb.set_trace() # BREAKPOINT
        sp_time = spike_triplet.time
        sp_layer = spike_triplet.layer
        sp_address = spike_triplet.address
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
        if np.any(membranes[layer] > 500):
            import ipdb; ipdb.set_trace() # BREAKPOINT


        #add impulse
        if layer == 0:
            # inp scale param not shown in thesis
            membranes[layer][sp_address] += cf.inp_scale
        elif layer % 2 == 0:
            membranes[layer] += weights[:, sp_address] * (refrac_end[layer] < sp_time)
        else:
            membranes[layer] += weights[sp_address, :] * (refrac_end[layer] < sp_time)


        # add noise
        self.add_noise(membranes[layer], layer) 
        
        self.log(history, sp_time, ('MEMBRANE', layer, membranes[layer] / thr[layer%2]))

        # update last_update

        last_update[layer] = sp_time

        # Add firings to queue
        newspikes = np.nonzero(membranes[layer] > thr[layer % 2])[0]

        
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
                #import ipdb; ipdb.set_trace() # BREAKPOINT
                weights[:, newspike] += weight_adj
                # log weight update

                #log(history, sp_time, ('UPDATE_WEIGHTS', newspike, np.random.random(weights.shape[0])))
                if np.any(weight_adj):
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
            self.log(history, new_time, ('SPIKE', layer, newspike))
            if (layer != 3):
                # TODO amb rng.nextfloat()
                pq.put(new_triplet)


if __name__ == '__main__':
    np.random.seed(0)           # to rerun same experiments
    args = parser.parse_args()  # get algorithm arguments from cmd
    srbm = SRBM(args) # create and initialize spiking rbm network
    srbm.load_data()

    history = srbm.train_network(args, args.visible_size, args.hidden_size, args.num_samples)
    
    dashboard = myex.DashBoard(sorted(history.items()), args.visible_size, args.hidden_size)
    dashboard.plot_reconstr_accuracy()
    if args.simulate:
        dashboard.plot_thigns()
