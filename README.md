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

`Classification report:`

![](Images/SVM3mplot.png)

![](Images/SVM3m.png)
The baseline model has an `accuracy score` of `0.55` with recall of `0.96` for classification category `1` and `0.04` for classification category`-1`.

The cumulative return plot shows that strategy returns reached `1.5` whereas `actual returns` reached `1.4` in the later period.

### Tuned models and performances:

#### Adjusting the size of the training dataset:

##### `Training data period`: 6 months

![](Images/SVM6mplot.png)

![](Images/SVM6m.png)

##### `Training data period`: 9 months


![](Images/SVM9mplot.png)


![](Images/SVM9m.png)


##### `Training data period`: 11 months


![](Images/SVM11m.png)


![](Images/SVM11.png)


##### What impact resulted from increasing or decreasing the training window?

* By tuning the baseline model by adjusting the size of the training dataset to `6 months`, `9 months` and `11 months` from 3 months, we observe that accuracy score improved by just 1 percent at 6 months but further started dropping on increasing the duartion of training dataset. Also, recall scores did not improve much for classification category `-1` except at 9 months where accuracy score dropped with drop in recall score for classification category `1`.`

* Looking at the plots, it is clear that the cumulative strategy returns also did not surpass actual returns majority of the times.

Thus the 3 months duration of baseline model for training data looks best of all the durations tried above.


#### Adjusting the SMA input features:

* (Keeping best period for training dataset- 3 months)


#### short_window(SMA_Fast)= 10 days

![](Images/sma_10.png)

![](Images/sma_10_plot.png)


#### short_window(SMA_Fast)= 30 days


![](Images/sma_30.png)

![](Images/sma_30_plot.png)

#### long_window(SMA_Slow)= 80 days

![](Images/sma_80.png)

![](Images/sma_80_plot.png)


#### long_window(SMA_Slow)= 120 days

![](Images/SMA_120.png)

![](Images/SMA_slow_120.png)

#### short_window (SMA_Fast) = 10 days, long_window (SMA_Slow) = 120 days


![](Images/SMA10_120.png)


![](Images/SMA10-120.png)


