"""
Uses method described in pdf in root director
"""

import math
alpha=math.pi/12 #random value determined at build
sin=math.sin(2*alpha)
cos=math.cos(2*alpha)
def findAD(dist1,dist2):
    """
    dist1 and dist2 are two distances from the distance sensors
    returns (angle, distance)
    WARNING: This is not done and does not include the case where ABC is an obtuse triangle
    """
    tarea=dist1*dist2*sin #double the area
    a=math.sqrt(dist1**2+dist2**2-2*dist1*dist2*cos)
    h=tarea/a
    if dist1<dist2:
        return (alpha-math.acos(h/dist1),h)
    else:
        return(alpha-math.acos(h/dist2),h)
