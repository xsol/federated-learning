# Federated Learning for Continually Learning Prototypes

This repository contains code for a federated learning algorithm based on CLP as introduced in https://arxiv.org/abs/2404.00418.
The repository contains (not computationally optimized) implementations of said algorithm, CLP and evaluation code.
The federated learning algorithm is described in a document, which will be uploaded here soon. It also contains evaluation of the Algorithm on MNIST dataset.

## Structure
The codebase is found in the `common`, `feature_extractor`, `federated_learning` and `prototypes` directories. 

Scripts for evaluation (or usage examples) are found in the root directory

## Usage

Evaluation generates plots in tensorboard. Tensorboard-related information is stored in `runs` directory. 

Do `tensorboard --logdir=runs` to see the results after using an evaluation script. Use the Timeseries and Projector tabs.

Trained or federated clients are stored in `data`, each accompanied by a file containing metadata.

Available scripts (relevant parameters can be set in main function):
- train_prototypes.py: train a client using CLP. clients are stored in `data/manual`

- evaluate_prototypes.py: evaluate (create tensorboard plots) a client (trained or federated)

- federate_prototypes.py: federate a client from specified parents stored in `data/manual`

- evaluate_federated_learning.py: large-scale evaluation for specified hyperparameters. Detailed plots are generated for each client. Clients and eval results are stored in `data/auto` and `runs/auto`, respectively.

- evaluate_federated_learning_hyperparameters.py: many large-scale evaluations to see the impact of different hyperparameters. Only overview-plots are generated. data and results are stored in directories named accordig to the hyperparameters


    

