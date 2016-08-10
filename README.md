## Instructions

**Note**: In order to run the notebooks and most of the code you need to have "... UVwork/uvnn" directory into python path, For example my project path is `/Users/johndoe/Projects/UVwork/uvnn.`

Then on
linux: you put this in your .bashrc file: `export PYTHONPATH=$PYTHONPATH:/Users/zisakadze/Projects/UVwork/uvnn`
windows : `http://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-7`
If done correctly you should be able to execute import uvnn from this terminal.


## EvtCD
* code is located in uvnn/snns/evtCD
* requirenments - python2.7, numpy, pyqtgraph (for visualisation)
* kaggle truncated mnist should be in the folder `uvnn/input/trunc_mnist`

Installation of pyqtgraph can be found on http://www.pyqtgraph.org.

algorithm is in evtcd.py while myex.py is for visualisation.
run normally 
`python evtcd.py` after several seconds you should be able to see the visualisation window. 



## Other things
### installation on windows

## Classifiers

## Autoencoders

#### __Sparse Autoencoder__
The loss function is calculated as:
 
![loss](http://bit.ly/298GPiY). 

Implemented according to [this](http://deeplearning.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity). The activation function for middle layer is sigmoid. Parameters:

1. ro - sparsity parameter, default 0.05
2. beta - weight of sparsity penalty term
3. alpha - learning rate
4. reg - weight decay 

The code is tested on truncmnist database 




## configuration (In progress)
In order to build up a model you need to load the configuration file which is in **yaml** format. Simple
configuration files are available in `configs/` directory. Example:

```yaml
Info:
	dataset_name: truncmnist
Reader:
	name: CsvReader
	fn: input/smth_input.csv
	fn_labels: input/smth_target.csv
	has_header: False # provide True if it has headers
	sep: , # separator in csv file
	label_pos:
Preprocessor:
	name: BasicPreprocessor
Classifier:
	name: MLP
	splits:
	arch:
		h1: 30
		output: 10
	learning_rate:
	regularization:
	
```
--

<img src="https://lh4.googleusercontent.com/-SfmUd8KFxLc/AAAAAAAAAAI/AAAAAAAAABE/eCr7S9qwlpc/s0-c-k-no-ns/photo.jpg" alt="alt text" width="110" height="whatever">

