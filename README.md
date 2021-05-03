# MachineLearning
This is a project I have written for the purpose of revising and learning concepts in Machine Learning.

I have started with Logistic Regression and Naive Bayes classifiers for NLP Sentiment Analysis.

## Using the sample scripts

The Python scripts 'imdb_sentiment_analysis.py' in the main directory is an example of how to use the ml package to train and test the Logistic Regression and Naive Bayes classifiers. 

To use this script, the directory paths can be modified to point to your own training and testing data, or if you wish to use it as is, you can follow these steps to get the Large Movie Review Dataset v1.0:
1. Go to https://ai.stanford.edu/~amaas/data/sentiment/
2. Click on 'Large Movie Review Dataset v1.0' to download the file aclImdb_v1(1).tar.gz
3. Move this file to the same directory as the ml package and the imdb_sentiment_analysis.py script
4. Extract the .tar file, and then extract the aclImdb directory from the .tar file.
5. Run the script 'imdb_sentiment_analysis.py'

## Contents of the ml package

This package contains four modules:

**classifiers**: Contains classes representing machine learning classifiers

**io**: Contains classes related to the input and output of data

**nlp**: Contains classes specific to the processing and feature extraction of natural language text

**test**: Contains classes used for the testing and evaluation of a classifier

## TODO:
1. K-fold cross validation in the test module
2. Import/export a trained model in the io module
3. Import/export preprocessed data in the io module
4. Add more things to this TODO list
