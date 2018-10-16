# Home Credit Default Risk Competition


This repository has a complete pipeline for predicting default risk on [Home Credit Default Risk Competition](https://www.kaggle.com/c/home-credit-default-risk) at Kaggle.
This model is part of the [7th place solution](https://www.kaggle.com/c/home-credit-default-risk/discussion/64580), which consists of an ensemble of 13 different models using linear regression.

In this Kaggle competition, participants must predict if a customer will default on a given credit application (binary classification). 
The effectiveness of a solution was evaluated with the AUC ROC metric, which is very common in this kind of problem.
This single model is able to achieve the 32nd position on the [leaderboard](https://www.kaggle.com/c/home-credit-default-risk/leaderboard),
with a 0.8044 private score.

The LightGBM library is used to implement a gradient boosting decision tree algorithm with Gradient-based One-Side Sampling (GOSS).
This novel technique was introduced by the Microsoft Research team and is described in [this paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf).

## Installation

1. Install Python 3 with pip
2. Clone repository and install requirements

```pip3 install -r requirements.txt```

3. Change the path variables at the config.py file

4. Run the main file

```python main.py```

### Dependencies

- Python 3
- Numpy
- Sklearn
- Lightgbm
- Hyperopt

The code was tested with Python 3.6 on a Windows 7 machine and Python 3.5 on Ubuntu 16 LTS.
