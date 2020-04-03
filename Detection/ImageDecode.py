"""
Functions in this file should take an image file from pi camera and
return a numpy array of pixel values in b/w
Link to camera:
https://www.amazon.com/Raspberry-Camera-Module-Megapixels-Sensor/dp/B07L82XBNM/ref=asc_df_B07L82XBNM/?tag=hyprod-20&linkCode=df0&hvadid=343234125040&hvpos=1o1&hvnetw=g&hvrand=9943191476561846608&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1018095&hvtargid=pla-717544328579&psc=1&tag=&ref=&adgrpid=68968886317&hvpone=&hvptwo=&hvadid=343234125040&hvpos=1o1&hvnetw=g&hvrand=9943191476561846608&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1018095&hvtargid=pla-717544328579
We'll probably want to use pca, but maybe not if pi has enough power
"""
import numpy as np