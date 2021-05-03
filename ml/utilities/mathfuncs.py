"""
Useful mathematical functions that might need to be accessed by a variety of classes

sigmoid
"""
import math as m

def sigmoid(feature, theta):
    """Logistic sigmoid function"""
    
    product = sum(f*theta[i] for i, f in enumerate(feature))
    try:    # attempt to calculate sigmoid
        sigmoid = 1/(1 + m.exp(-product))
        
    except OverflowError: # in the event of an overflow, assign 0 or 1 depending on sign of product
        if product > 0:
            sigmoid = 1
        elif product < 0:
            sigmoid = 0
        
    return sigmoid


