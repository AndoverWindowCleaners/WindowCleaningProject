"""
First (Ratio) Idea to Detect Windows
Functions in this function should take in a numpy array of pixel values and
end up returning distance the drone should move in each direction
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
#rint(isWindow(first,second,1.5))
def spray(processedimg,sizes):
    """
    processedimg is numpy array with True for window and False for others
    sizes is tuple of the percent of each dimension the image should check
    
    Returns boolean with whether or not drone should spray
    """
    x,y=processedimg.shape
    for size in sizes:
        #expression takes the middle part of processedimg that is size of total size
        if np.mean(processedimg[int((1-size)*x//2):int((1+size)*x//2+1),int((1-size)*y//2):int((1+size)*y//2+1)])<0.5:
            return False
    return True
print(spray(isWindow(first,second,1.5),(0.5,)))
        