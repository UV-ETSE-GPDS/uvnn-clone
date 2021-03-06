import argparse
import copy
import numpy as np
import visual
import logging
import pickle
logging.basicConfig(format='%(asctime)s %(message)s')
from event_based import SRBM_EB
from time_stepped import SRBM_TS
from common import str2bool
from common import load_data
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt

#unknown things
# stdp_lag ??
# inp_scale ??
# timespan ??
# min_thr ??
# axon_delay ?? 
parser = argparse.ArgumentParser(description='evtcd algorithm with simulation')

# Algorithm params
parser.add_argument("--implementation", choices=["TIME_STEPPED", "EVENT_BASED", "EXPERIMENT"], 
        default="EVENT_BASED", help="Log level.")

parser.add_argument("--eta", type=float, default=0.001, help="Learning rate")
parser.add_argument("--thresh_eta", type=float, default=0, 
        help="Threshold learning rate")
parser.add_argument("--dt", type=float, default=0.001, 
        help="Dleta time in case of time stepped implementation")
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
parser.add_argument("--linear_decay", type=float, default=None, 
        help="linear approximation constant for decaying")

parser.add_argument("--linear_decay_only_in_eval", type=bool, default=False, 
        help="If this is true linear decay will only be used in evaluation phase, not in training")

parser.add_argument("--stdp_lag", type=float, default=0.003, 
        help="STDP window length")
parser.add_argument("--min_thr", type=float, default=-1, 
        help="minimum threshold")
parser.add_argument("--axon_delay", type=float, default=0.0001, 
        help="axon delay, how long the spike takes to travel to the next layer")
parser.add_argument("--t_gap", type=float, default=10, 
        help="time gap between different training samples")
parser.add_argument("--noise_uniform", type=float, default=0.1, 
        help="Uniform noise to add to spike distributioin")
parser.add_argument("--noise_decay", type=float, default=1, 
        help="Decay rate of noise( decayed in every batch operation)")
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

parser.add_argument("--batch_size", type=int, default=8, 
        help="Batch size in case of time stepped implementation")

parser.add_argument("--num_epoch", type=int, default=1, 
        help="number of epochs")

parser.add_argument("--num_test", type=int, default=2, 
        help="number of samples to use for testing")

parser.add_argument("--simulate", type=str2bool, default=True, 
        help="simulation of training with visualisation (takes longer time)")

parser.add_argument("--shuffle", type=str2bool, default=True, 
        help="Whether or not should shuffle data set before training")

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

parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
        default="INFO", help="Log level.")

parser.add_argument("--seed", type=int, default=3, help="random seed to rerun experiments")

args = parser.parse_args()  # get algorithm arguments from cmd
logger = logging.getLogger()
logger.setLevel(args.log_level)

np.random.seed(args.seed)           # to rerun the same experiments

X, y = load_data(args, logger)

if args.implementation == 'EVENT_BASED':
    args.batch_size = None # batch training in event based implementation not working properly
    srbm = SRBM_EB(args, logger) # create and initialize spiking rbm network
    logger.info('Loaded Event based implementation of evtCD')
    srbm.set_data(X, y)
    srbm.train_network()
elif args.implementation == 'TIME_STEPPED':
    srbm = SRBM_TS(args, logger)
    logger.info('Loaded time-stepped implementation of evtCD')
    
    srbm.set_data(X, y)
    history, weights, accs = srbm.train_network()
    np.save(args.save_weights, weights)
    dashboard = visual.DashBoard(sorted(history.items()), args.visible_size, args.hidden_size)
elif args.implementation == 'EXPERIMENT':
    # Experimetns
    
    exps = []     # list of experiment labels and args

    # expargs[0] is reserved for baseline solution
    def makeargs(param, val):
        assert param in args.__dict__, 'unknown paramter'
        label = param + ' = ' + str(val)
        newarg = copy.copy(args)
        newarg.__dict__[param] = val
        return (label, newarg)
    
    exps.append(('baseline', args))  # baseline solution provided from cmd don't touch
    
    # EDIT EXPERIMENTS BELOW
    
    
    # number of layers
    exps.append(makeargs('hidden_size', 50))
    exps.append(makeargs('hidden_size', 150))

    #exps.append(('baseline', args))  # baseline solution provided from cmd don't touch

    # exps.append(makeargs('linear_decay_only_in_eval', True))
    
    ''' decays 
    
    exps.append(makeargs('linear_decay', 0.1))
    exps.append(makeargs('linear_decay', 0.05))
    exps.append(makeargs('linear_decay', 0.2))
    exps.append(makeargs('linear_decay', 0.03))

    '''

    # learning rates
    ''' 
    exps.append(makeargs('eta', 0.0005))
    exps.append(makeargs('eta', 0.002))
    exps.append(makeargs('eta', 0.0001))

    # dt time interval
    exps.append(makeargs('dt', 0.002))
    exps.append(makeargs('dt', 0.008))

    #batch size
    exps.append(makeargs('batch_size', 5))
    exps.append(makeargs('batch_size', 2))

    # noise
    exps.append(makeargs('noise_uniform', 0.07))
    exps.append(makeargs('noise_uniform', 0.01))
    
    # tau

    exps.append(makeargs('tau', 0.2))
    exps.append(makeargs('tau', 0.1))

    # window

    exps.append(makeargs('stdp_lag', 0.04))
    exps.append(makeargs('stdp_lag', 0.01))

    # refrac

    exps.append(makeargs('t_refrac', 0.03))
    exps.append(makeargs('t_refrac', 0.01))
    '''


    exp_results = [] # tuples of args, acccuracies 
    #import ipdb; ipdb.set_trace()
    #ax = fig.add_subplot(111)
    for ind, (label, args) in enumerate(exps):
        print 'running experiment number# ', ind
        #if ind < 10000:
        #    accs = pickle.load(open("exp_results_ld.p", "rb"))
        #    exp_results.append(accs[ind])
        #    continue
        
        srbm = SRBM_TS(args, logger) # create and initialize spiking rbm network
        np.random.seed(args.seed)  
        srbm.set_data(X, y)
        hist, w, accuracies = srbm.train_network()
        np.save(args.save_weights, w)
        exp_results.append((str(args), label, accuracies))
        #pickle.dump(exp_results, open("exp_results_to.p", "wb" )) # save just in case
        #print accuracies

    pickle.dump(exp_results, open("exp_results.p", "wb" )) # save just in case
    for ind, (arg, label, accuracies) in enumerate(exp_results):
        examples_seen, accs = zip(*accuracies)
        plt.plot(examples_seen, accs, label= label)

    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #        ncol=n_exp, mode="expand", borderaxespad=0.)
    plt.legend(loc='upper left')
    plt.show()
    


if args.simulate:
    dashboard.run_vis()
