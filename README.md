# Machine Learning Model for Algorithmic Trading

## Overview:

Creating an algorithmic trading system using Python programming and machine learning, that learns and adapts to new data and improving the performance by adjusting the model’s input features to find the parameters that result in the best trading outcomes.

Following steps have been followed:

1. Implement an algorithmic trading strategy that uses machine learning to automate the trade decisions establishing a Baseline Performance

2. Tuning the Baseline Trading Algorithm by adjusting the input parameters to optimize the trading algorithm.

3. Training a new machine learning model and comparing its performance to that of a baseline model.


## Dataset:

The data used is in form of a CSV file contains OHLCV data for an MSCI–based emerging markets ETF that iShares issued  between 2015-01-21 to 2021-01-22.

## Machine learning models and performances:

### Baseline model:

`SMA windows:`
short_window (SMA_Fast) = `4`days
long_window (SMA_Slow) = `100`days

`Training data period`: 3 months (2015-04-02 15:00:00 `to` 2015-07-02 15:00:00)
`Testing data`: (2015-07-06 10:00:00 `to` 2021-01-22 15:45:00)

`Classifier model`: SVC from SKLearn's support vector machine (SVM)learning method

![](images/crypto_data.png)





