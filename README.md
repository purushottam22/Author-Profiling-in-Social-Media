###### Author Profiling on Arabic Tweets using Deep learning techniques

In this era, where everyone wants to hide there personality, the author profiling is very useful in determining the profile of author. Here, we are using arabic tweets to identify the gender, age and language of the authors.

We are using python 3, Keras 2.1.1.


## Dataset
We are using Test and Train Dataset provided by Forum for Information Retrieval Evaluation (FIRE) 2019.

## Model
We devlop two models : 
- Long Short-Term Memory
- Long Short-Term Memory + Features

In the first model, we uses LSTM model, the main file for this model is 'Train_test.py'.
In the second model, we uses some features which are given in a folder named 'Features'. This folder contains emoji counter, sentence length and many other python programs which are beneficial in determining the features of the tweets.
The main program of second model is 'Train_test_feature.py'. 

We are using Softmax layer in both model.

## Accuracy
On Test Data

On Model - 1

- Gender 57.64 
- Age 27.50
- Lang 55.14

On Model - 2

- Gender 66.24 
- Age 22.22
- Lang 80.28
