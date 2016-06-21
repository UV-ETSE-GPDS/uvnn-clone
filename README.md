<img src="https://lh4.googleusercontent.com/-SfmUd8KFxLc/AAAAAAAAAAI/AAAAAAAAABE/eCr7S9qwlpc/s0-c-k-no-ns/photo.jpg" alt="alt text" width="110" height="whatever">
## Instalation
Todo - provide instructions for windows

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

