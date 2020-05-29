# Installation
- create a conda environment to setup the experiments
```
conda create --name comapre-env --file requirements.txt
```
- activate the environment
```
conda activate compare-env
```

# Setup
We need the wav files and the corresponding labels to set up the experiments. We also need a hardcoded directory structure utilized by the training scripts. You can set up these requirements by running:
```
./recipes/common/setup.sh 
```

# Feature generation
The file `recipes/common/feature_generator.py` contains the feature generation module. You can run the feature generation process to generate training, developent and test set features by the following command.
```
./recipes/common/run_feature_generator.sh
```
For now the above scripts produces log-mel features which are preemphaised and apply a butterworth filter to the input. You will need to modify the generator script to create other type of features.

# Training a classifier
The file `src/seqseq2d.py` trains a simple neural network model over the generated features. The python script hardcodes the location of input features and labels. You will need to update to input your own features. A simple execution of this script is given in:
```
./recipes/common/run_seqseq2d.py
```
It also includes examples of training with other features.
