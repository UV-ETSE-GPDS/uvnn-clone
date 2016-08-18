from uvnn.utils.readers import CsvReader
import numpy as np
from collections import namedtuple
from Queue import PriorityQueue

from IPython import display #plotting purposes
import matplotlib.pyplot as plt
from uvnn.utils.images import show_images
import time
import myex
from collections import defaultdict
from common import Param, Spike
#unknown things
# stdp_lag ??
# inp_scale ??
# timespan ??
# min_thr ??
# axon_delay ?? 



cf = Param(eta=1e-1, thresh_eta=0, numspikes=100, timespan=2, tau=0.05, 
        thr=2, inp_scale=0.1, t_refrac=0.001, stdp_lag=0.002, min_thr=-1,
        plot_things=False, axon_delay=0.0001)

def img_to_spike(x, numspikes, timespan):
    ''' return pairs of spike_address, time '''
    probs = x / float(sum(x))
    spike = np.random.choice(range(len(x)), numspikes, True, probs)
    times = np.sort(np.random.rand(numspikes) * timespan)
    return zip(spike, times)

def prepare_spike_trains(numspikes=100, timespan = 10):
    # Load mnist and convert them to spikes
    csv_reader = CsvReader(fn='../../input/kaggle_mnist/train.csv', has_header=True, 
            label_pos=0) #mnist 28
    
    # trunc mnist for testing 
    csv_reader = CsvReader(fn='../../input/trunc_mnist/trunc_mnist20x20_inputs.csv', 
            has_header=False, fn_labels='../../input/trunc_mnist/trunc_mnist20x20_targets.csv') #mnist 28

    X, y = csv_reader.load_data()
    # normalize data
    X = (X - np.min(X)) / float(np.max(X) - np.min(X))
    # remove low valued pixels (truncated mnist problem)
    X[X < 0.12] = 0
    #X -= np.min(X)
    #import ipdb; ipdb.set_trace()
    #spike_trains = np.apply_along_axis(img_to_spike, 1, X, numspikes, timespan)
    # use vectorized form later
    spike_trains = [img_to_spike(x, numspikes, timespan) for x in X]
    #import ipdb; ipdb.set_trace()
   

    if False: 
        # TODO move to dashboard this things later
        print 'Real images'
        ris = np.random.choice(range(X.shape[0]), 10, False)
        show_images(X[ris], 20, 20)
        
        print 'Converted spike trains'
        #plt.subplot(1, 10)
        plt_m, plt_n = 10, 1
        plt.figure(figsize=(2 * plt_m, 2 * plt_n))
        for i, ri in enumerate(ris):
            spikes, _ = zip(*spike_trains[ri]) 
            
            xx, yy = zip(*map(lambda x: (x % 20, x / 20), spikes))
            plt.subplot(1, 10, i + 1)
            plt.xticks(np.arange(0, 20))
            plt.yticks(np.arange(0, 20, 1))
            plt.plot(yy, xx, 'o')
        plt.show()

    return spike_trains, y


def train_network(cf, vis_size, hid_size):
    pq = PriorityQueue()
    spike_trains, labels = prepare_spike_trains(cf.numspikes, cf.timespan)
    
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

    log(history, 0, ('INIT_WEIGHTS', W))

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
    for spike_train_sample in spike_trains[:200]:  # first 20 spikes

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

        t_passed += cf.timespan

        last_time = -1
        while not pq.empty():
            spike_triplet = pq.get()
            #import ipdb; ipdb.set_trace()
            process_spike(spike_triplet, cf, pq, thr, last_spiked, membranes,
                    last_update, refrac_end, spike_count, calc_recons, W, history)

    # visualize spikes
    return history

def add_noise(membrane, layer_num):
    pass

def log(history, time, data):
    history[round(time, 6)].append(data)

def process_spike(spike_triplet, cf, pq, thr, last_spiked, membranes, 
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
    log(history, sp_time, ('SPIKE', sp_layer, sp_address))
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
    add_noise(membranes[layer], layer) 
    
    log(history, sp_time, ('MEMBRANE', layer, membranes[layer] / thr[layer%2]))

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
                log(history, sp_time, ('UPDATE_WEIGHTS', newspike, weights_to_log))
            #print np.max(weight_adj) ,np.min(weight_adj)

        # reconstruct the layer if desired
        if calc_recons[layer]:
            recon[layer] *= np.exp(- (sp_time - last_recon[layer]) / recon_tau)
            recon[layer][newspike] += recon_imp
            last_recon[layer] = sp_time
        
        # add spikes to the queue if not in the end layer
        # also add  random delay 
        rand_delay = np.random.random() / 10000
        new_time = sp_time +  2 * cf.axon_delay + rand_delay
        new_triplet = Spike(time= new_time, layer=layer, address=newspike)
        log(history, new_time, ('SPIKE', layer, newspike))
        if (layer != 3):
            # TODO amb rng.nextfloat()
            pq.put(new_triplet)

    #print np.sum(weights)

    # Plot things
    if not cf.plot_things:
        return
    display.clear_output(wait=True) 
    show_images(weights.T[:5, :], 20, 20, 12)
    display.display(plt.gcf())
    time.sleep(1)

visible_size = 400
hidden_size = 4
history = train_network(cf, visible_size, hidden_size)
dashboard = myex.DashBoard(sorted(history.items()), visible_size, hidden_size)
dashboard.plot_thigns()
