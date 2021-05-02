"""
A package of classifiers

LogisticRegression:
    Instance attributes:
        theta = [1,1,1]
        threshold = 0.5
    Methods:
        __init__
        train_logistic_regression
        sigmoid
        classify

NaiveBayes
    Instance attributes:
        log_likelihoods = {}
        log_prior = 0
    Methods:
        __init__
        update_log_likelihood
        log_prior
        classify
        

"""
import math as m
import copy

class LogisticRegression:
    """Class representing a logistic regression model"""

    def __init__(self, theta = [1,1,1], threshold = 0.5):
        """Constructs logistic regression model with a default theta vector [1,1,1] and a default threshold 0.5"""
        self.theta = theta
        self.threshold = threshold
        return
    
    def train(self, feature_set, attributes, learning_rate):
        """Given a feature set, list of attributes, initial theta and learning rate (alpha), this method
        uses gradient descent to maximise the log likelihood of a correct classification
        """
        min_attribute_size = min(len(feature_set[k]) for k in feature_set)
        
        for k in range(min_attribute_size):
            for (feature,y) in [ (feature_set[attributes[0]][k], 1), (feature_set[attributes[1]][k], 0) ]:
                
                h = self.sigmoid(feature)
                new_theta = []
                
                for i,t in enumerate(self.theta):
                    new_theta.append(t - (learning_rate/(len(feature_set)))*feature[i]*(h - y))
                    
                self.theta = copy.copy(new_theta)
        return self.theta
    
    def sigmoid(self, feature):
        """Logistic sigmoid function"""
        
        product = sum(f*self.theta[i] for i, f in enumerate(feature))
        
        try:    # attempt to calculate sigmoid
            sigmoid = 1/(1 + m.exp(-product))
            
        except: # in the event of an overflow, assign 0 or 1 depending on sign of product
            if product > 0:
                sigmoid = 1
            elif product < 0:
                sigmoid = 0
            
        return sigmoid

    def classify(self, feature):
        """Classifies a datapoint based on the feature vector and the theta vector.

        Threshold for positive classification is set at 0.5 by default. Returns a boolean: True for positive
        classification and False for negative"""
        sigmoid = self.sigmoid(feature)
        
        if sigmoid < self.threshold:
            return False
        elif sigmoid >= self.threshold:
            return True

class NaiveBayes:
    """Class representing a Naive Bayes Classifier"""
    def __init__(self):
        """Constructs a naive bayes classifier with an empty log likelihoods dictionary and a log prior of 0"""
        self.log_likelihoods = {}
        self.log_prior = 0
        return

    def update_log_likelihood(self, conditional_probabilities, labelled_training_dataset, attribute_pair):
        """Updates the log likelihood dictionary. Note that if a feature was already in the dictionary, it is overwritten"""
        for feature in conditional_probabilities:
            self.log_likelihoods.update({feature: m.log( conditional_probabilities[feature][attribute_pair[0]]/conditional_probabilities[feature][attribute_pair[1]])})
        return self.log_likelihoods

    def log_prior(self, labelled_training_dataset, attribute_pair):
        """Calculates the log prior of the given dataset and overwrites self.log_prior with it"""
        self.log_prior = m.log(len(labelled_training_dataset[attribute_pair[0]])/len(labelled_training_dataset[attribute_pair[1]]))
        return log_prior

    def classify(self, datapoint):
        """Classifies a datapoint based on the trained log prior and log likelihood dict"""
        score = self.log_prior
        for feature in datapoint:
            try:
                score += self.log_likelihoods[feature]
            except KeyError: # ignore words that weren't in the training set
                pass 

        if score >= 0:
            return True
        elif score < 0:
            return False


            
