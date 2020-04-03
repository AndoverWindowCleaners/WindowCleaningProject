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
    #Remember that this function also needs to determine how many windows there are and their centers
print(isWindow(first,second,1.5))
def move(coords):
    """coords should be a list of tuples
    Each tuple should contain the coordinates(in the array/picture) for the center of the window
    Depending on how sensors work, we might take a few more arguments (distance to wall and angle)
    
    Returns:
        Tuple: How far in each direction the drone should move
    """
    pass