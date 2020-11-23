# UPB FinCausal 2020 Task 1

This repository contains the ensemble used by the UPB team to obtain the 2nd place at the [FinCausal 2020](http://wp.lancs.ac.uk/cfie/fincausal2020/) Task 1. The ensemble is composed of five Transformer-based models: Bert-Large, RoBERTa-Large, ALBERT-Base, FinBERT-Base and SciBERT-base.

<p align="center">
  <img src="https://raw.githubusercontent.com/avramandrei/UPB-FinCausal-2020-Task-1/main/resources/Ensemble-Figure.png">
</p>

The ensemble can be downloaded from [here](https://swarm.cs.pub.ro/~ccercel/UPB-Fincausal2020-best-ensemble.zip).

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

## Ensemble Performance

We depict the performance of each individual model and of the ensemble on both validation dataset (split explained in paper) and on the evaluation dataset.

| Model | Valid-Prec | Valid-Rec | Valid-F1c | Test-Prec | Test-Rec | Test-F1 |
--------| :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
BERT-Large | 95.81 | 96.15 | 95.77 | 97.10 | 97.02 | 97.05 |
RoBERTa-Large | 95.69 | 95.29 | 95.46 | 97.35 | 97.30 | 97.32 |
ALBERT-Base | 95.71 | 95.88 | 95.78 | 96.75 | 96.76 | 96.75 | 
FinBERT-Base | 93.88 | 94.71 | 93.92 | 94.08 | 94.30 | 94.18 |
SciBERT-Base | 95.67 | 95.99 | 95.75 | 96.77 | 96.83 | 96.89 | 
Ensemble | - | - | - | **97.53** | **97.59** | **97.55** |



