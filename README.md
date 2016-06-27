<img src="https://lh4.googleusercontent.com/-SfmUd8KFxLc/AAAAAAAAAAI/AAAAAAAAABE/eCr7S9qwlpc/s0-c-k-no-ns/photo.jpg" alt="alt text" width="110" height="whatever">
## Instalation
Todo - provide instructions for windows
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

