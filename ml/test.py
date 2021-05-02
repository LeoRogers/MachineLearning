"""
Package for testing and evaluating a model.

Test:
    __init__
    test
    trp
    tnr
    fpr
    fnr
    precision
    accuracy
    plr
    fnr
    fdr
    csi
    fb
    

"""

class Test:
    def __init__(self):
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0
        return

    def test(self, classifier, attributes, test_feature_set):
        """Tests the labelled test feature set against the given classifier for the given attribute pair.

        The classifier must return a boolean, and returning 'True' should correspond to the first attribute in the list.
        The True Positive, True Negative, False Positive and False Negative values are updated as it goes.
        """
        for attribute in attributes:
            for feature in test_feature_set[attribute]:
                classification = classifier(feature)
                if classification and attribute == attributes[0]: #Predicted: positive, Actually: positive
                    self.true_pos +=1
                elif classification and attribute == attributes[1]: #Predicted: positive, Actually: negative
                    self.false_pos += 1
                elif not classification and attribute == attributes[0]: #Predicted: negative, Actually: positive
                    self.false_neg += 1
                if not classification and attribute == attributes[1]: #Predicted: negative, Actually: negative
                    self.true_neg += 1
        
        return self.true_pos, self.false_pos, self.true_neg, self.false_neg
    
    def tpr(self):
        ''' True Positive Rate, Sensitivity '''
        return self.true_pos/(self.true_pos + self.false_neg)

    def tnr(self):
        ''' True Negative Rate, Specificity '''
        return self.true_neg/(self.true_neg + self.false_pos)

    def fpr(self):
        '''False Positive Rate'''
        return self.false_pos/(self.false_pos + self.true_neg)

    def fnr(self):
        '''False Negative Rate'''
        return self.false_neg/(self.false_neg + self.true_pos)
        
    def precision(self):
        '''Precision''' 
        return self.true_pos/(self.true_pos + self.false_pos)

    def accuracy(self):
        '''Accuracy'''
        return (self.true_pos + self.true_neg)/(self.false_pos + self.false_neg + self.true_pos + self.true_neg)

    def plr(self):
        ''' Positive likelihood ratio '''
        return self.tpr()/self.fpr()

    def fnr(self):
        '''Negative likelihood ratio'''
        return self.fnr()/self.tnr

    def fdr(self):
        '''False Discovery Rate'''
        return self.false_pos/(self.false_pos + self.true_pos)

    def csi(self):
        '''Critical Success Index '''
        return self.true_pos /(self.true_pos + self.false_neg + self.false_pos)

    def fb(self, b = 1):
        """Generalised f-score. Defaults to f1."""
        c = 1 + b**2
        return c*self.true_pos/(c*self.true_pos + self.false_neg*b**2 + self.false_pos)
