# UPB FinCausal 2020 Task 1

This repository contains the ensemble used by the UPB team to obtain the 2nd place at the [FinCausal 2020](http://wp.lancs.ac.uk/cfie/fincausal2020/) Task 1. The ensemble is composed of five Transformer-based models: Bert-Large, RoBERTa-Large, ALBERT-Large, FinBERT-Base and SciBERT-base.

The ensemble can be downloaded from [here](http://swarm.cs.pub.ro/~ccercel/UPB-Fincausal2020-best-ensemble.zip).

## Installation

Make sure you have Python3 and PyTorch installed. Install the dependencies via pip:

```
pip install -r requirements.txt
```

## Prediction

To make a prediction run the `predict.py` script and give it an input file with a sentence on each line and the path to the ensemble of models. The script will output a file with 1 or 0 on each line, corresponding to the sentence being causal or not being causal, respectively.

```
python3 predict.py [input_path] [models_path] [--output_path]
```
