"""
A package of classifiers

LogisticRegression:
    __init__
    train_logistic_regression
    sigmoid
    classify

"""
import math as m
import copy

class LogisticRegression:
    """Class representing a logistic regression model"""

    def __init__(self):
        return
    
    def train_logistic_regression(self, feature_set, attributes, theta, alpha):
        """Given a feature set, list of attributes, initial theta and learning rate (alpha), this method
        uses gradient descent to maximise the log likelihood of a correct classification
        """
        min_attribute_size = min(len(feature_set[k]) for k in feature_set)
        
        for k in range(min_attribute_size):
            for (feature,y) in [ (feature_set[attributes[0]][k], 1), (feature_set[attributes[1]][k], 0) ]:
                
                h = self.sigmoid(feature, theta)
                new_theta = []
                
                for i,t in enumerate(theta):
                    new_theta.append(t - (alpha/(len(feature_set)))*feature[i]*(h - y))
                    
                theta = copy.copy(new_theta)
        return theta
    
    def sigmoid(self, feature, theta):
        """Logistic sigmoid function"""
        
        product = sum(f*theta[i] for i, f in enumerate(feature))
        
        try:    # attempt to calculate sigmoid
            sigmoid = 1/(1 + m.exp(-product))
            
        except: # in the event of an overflow, assign 0 or 1 depending on sign of product
            if product > 0:
                sigmoid = 1
            elif product < 0:
                sigmoid = 0
            
        return sigmoid

    def classify(self, feature, theta, threshold = 0.5):
        """Classifies a datapoint based on the feature vector and the theta vector.

        Threshold for positive classification is set at 0.5 by default. Returns a boolean: True for positive
        classification and False for negative"""
        
        sigmoid = self.sigmoid(feature, theta)
        
        if sigmoid < threshold:
            return False
        elif sigmoid >= threshold:
            return True
