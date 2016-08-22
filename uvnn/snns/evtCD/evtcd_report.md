# implementation of evtCD for Spiking RBM

## Introduction
This implementation is based on the Master thesis by [Online Learning in Event-based Restricted Boltzmann Machines](http://dannyneil.com/attach/dneil_thesis.pdf) by _Daniel Neil_.

## Installation

* _Requirements:_ Python 2.7, numpy, pandas, matplotlib, argparse, [pyqtgraph](http://www.pyqtgraph.org) (for visualisation).
* _Clone github repository_: `https://github.com/scientist1642/uvnn`
* Make sure to add the topmost __uvnn__  directory in your PYTHONPATH. (one - which is along readme.md file). 

 linux: you put this in your .bashrc file: `export PYTHONPATH=$PYTHONPATH:/Users/zisakadze/Projects/UVwork/uvnn`
windows : `http://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-7`
If done correctly you should be able to execute `import uvnn` from the python shell.

## Getting help
Configurations to the code are passed through command line, you can type python evtcd.py --help to get the full help.
<pre>
 --eta ETA             Learning rate
 --thresh_eta THRESH_ETA
                        Threshold learning rate
 --numspikes NUMSPIKES
                        Number of spikes for one input sample during
                        [timespan] seconds
 --timespan TIMESPAN   durition in which [numspikes] of one sample digit
                        happen
  --tau TAU             Membrane time constant (decay constant)
  --thr THR             Threshold value
  --inp_scale INP_SCALE
                        value to use for the first layer membrane potential
                        addition (Doesn't matter much)
  --t_refrac T_REFRAC   Refractory period
  --stdp_lag STDP_LAG   STDP window length
  --min_thr MIN_THR     minimum threshold
  --axon_delay AXON_DELAY
                        axon delay, how long the spike takes to travel to the
                        next layer
  --t_gap T_GAP         time gap between different training samples
  --visible_size VISIBLE_SIZE
                        number of neurons in the visible layer
  --hidden_size HIDDEN_SIZE
                        number of neurons in the hidden layer
  --input_file INPUT_FILE
  --num_train NUM_TRAIN
                        number of samples to use for training
  --num_test NUM_TEST   number of samples to use for testing
  --simulate SIMULATE   simulation of training with visualisation (takes
                        longer time)
  --shuffle SHUFFLE     Whether or not should shuffle data set before training
  --batch_size BATCH_SIZE
                        Whether or not should shuffle data set before training
  --test_every TEST_EVERY
                        In every how many sample, benchmarking on test set
                        should occur
  --enable_update ENABLE_UPDATE
                        determines whether weights are updated, useful when
                        you already have learned weights
  --save_weights SAVE_WEIGHTS
                        Path to save weights
  --load_weights LOAD_WEIGHTS
                        Path to save weights
  --log_reconstr LOG_RECONSTR
                        logs reconstruction distributions for each sample
  --plot_curve PLOT_CURVE
                        Plot training errors
  --show_weight_deltas SHOW_WEIGHT_DELTAS
                        should only be used with simulate =True, visualizes
                        weight updates
</pre>

## Providing input
One data sample is supposed to be one dimensional float or integer vector. Spike trains are then created by converting this vectors to probabilities which determine firing rates of each neuron on visible layer. For example in case of 28x28 MNIST database, where we have 60 000 training samples, input should be - 60000 X 784 comma separated values. you can provide input file by command line: `--input_file='/Users/John/file1.csv'` Or by default kaggle mnist set **28x28 mnist set** is used and you need to have two files- `train.csv` and `labels.csv` in folder `uvnn/input/kaggle_set`. You can download them from `https://www.kaggle.com/c/digit-recognizer/data`.

P.S. _If you need to provide other kind of input files where separator is other character, or labels are in the same file you can take a look at [CSVreader function.](https://github.com/scientist1642/uvnn/blob/master/uvnn/utils/readers.py)_

## General Overview

Code is mainly organized in two files evtcd.py where thre is a main functionality and myex.py for visualisation purposes. Let's take a look at evtcd.py first.

The main class is __SRBM__

* `load_data()` loads data from the specified file
* `set_data(X, y)` alternatively sets the data from the code
* `data_to_spike(x, numspikes, timespan)` converts data sample to spike train
* `prepare_dataset()` does some preprocessings and divides data by test and train set if testing is desired
* `error_on_test(W)` To tune the parameters some kind of error function was needed, we could use the accuracy as it is in thesis if we had trained the network in a supervized way, but accuracy might not give the good dynamic insight how well the input distribution is learnt. I decided to have some samples without labels in the test set, and after in every `test_in_every` episodes, run learnt weights on the unknown set of inputs. For each sample we will recieve some kind of distribution of spikes on reconstruction layer. We can meassure the distances between true and reconstructed distributions and take the mean over test samples to evaluate the error on weight W. Right now I simply use MSE error as a distance.
* `run_network(init_weights)` takes init weights if desired, initializes all arrays and data structures of the network needed to run the algorithm. Most importantly there is one priority queue which keeps track of all spikes in the network in triplets (time, layer, address). one by one triplet is taken from this queue and we call `process spike`
* `process_spike` goes and processes the spike updates membranes, causes another spikes, or logs current snapshot to history

### History
To separate the algorithm functionality from the visualisation, history is introduced which is a dictionary of (key=time, data=events) type. since in theory (very unlikely) several events can occur on one timestep (especially because it's rounded), events is a list of several different event

event is a tuple, where first element is always a string- event type. for example right now event tuple examples are:

* `(INIT_WEIGHTS, weights)` 
* `(SPIKE, layer, address)`
* `(NEW_SPIKE_TRAIN`, spike_train) 
*  and so on...

From this history myex.py visualissation part is able to go through the training process.

### Example 
Let's run the basic example: All unused parameters will be used as default. 

`$ python evtcd.py --num_train=1000 --num_test=50 
--test_every=100 --plot_curve=True --simulate=False --hidden_size=81 --save_weights='myweights.npy' `

Now trained weights are saved in myweights.npy, if we want to visualise running network we can do so by setting simulate=True and num_train to lower number and disable weight update

`python evtcd.py --num_train=10 --num_test=0  --simulate=True --enable_update=False --hidden_size=100 --load_weights 'myweights.npy'`

 



