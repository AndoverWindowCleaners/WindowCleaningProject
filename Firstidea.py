"""
First (Ratio) Idea
We'll probably have to do noise reduction
"""
import numpy as np
#testvals- feel free to change
first=np.ones((10,5))
second=np.ones((10,5))
second[4:8,1]=1.5
def noiseReduce(X,neighbors):
    """Edit here if you want to do noise reduction 2d"""
    pass
def isWindow(First,Second,epsilon):
    """
    First and Second are numpy arrays of images
    Epsilon is a float (>=1) to determine whether a pixel has a window or not
    Function uses 1/epsilon also
    
    Returns True for window pixels, false otherwise
    """
    ratios=First/Second
    return np.logical_or(ratios<=(1/epsilon),ratios>=epsilon)