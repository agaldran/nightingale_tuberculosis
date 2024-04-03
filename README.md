# Overview:
This repository holds the code for the second place in the Nightingale competition on [active tuberculosis bacilli detection](https://app.nightingalescience.org/contests/m3rl61qq21wo).
The data needed to train all models lives in the Nightingale platform. The structure of this project is relatively straightforward, though.
A jupyter notebook shows how I split the data into five train/validation sets.
As a result, 10 csv files are generated, e.g. tr_f1.csv, vl_f1.csv.
These two would be useful for carrying out a training run, and we do five of them per model, one is a CNN (resnext50) and the other a transformer (swin).

For training there is a script called train.py, which will take several hyperparemeters as input. Most used values are the defaults specified in the argument parser, with the exception of the architecture, learning rate, training fold and save path. The run_experiments.sh and run_experiments_2.sh bash scripts can be used to reproduce the training of the ten models.
For testing, there is the test_ensemble.py script, which loads ten models (5xresnext50 and 5xswin) from the `experiments` folder, and uses them to build predictions on the test set. Finally, the submit.py script will read a test csv and send it for evaluation.
Additionally, there are the utils folder which contains mostly auxiliary logic for data loading, model definitions etc.
Any doubt please open an issue.
