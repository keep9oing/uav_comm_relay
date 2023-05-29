import numpy as np

def angle_le_0(x):
    # find out the equivalent angle for any angle less than 0
    if x < 0:
        theta = abs(x)
        Q = theta/360

        if Q>=0 and Q <=1:
            op = x + 360
        elif Q>1 and Q<=2:
            op = x + 720
        elif Q>2 and Q<=3:
            op = x + 1080
        elif Q>3 and Q<=4:
            op = x + 1440
    else:
        op = x

    return op

def angle_ge_360(x):
    # find out the equivalent angle for any angle greater than 360

    theta = abs(x)
    Q = theta/360
    
    if Q>=0 and Q <=1:
        op = theta
    elif Q>1 and Q<=2:
        op = theta - 360
    elif Q>2 and Q<=3:
        op = theta - 720
    elif Q>3 and Q<=4:
        op = theta - 1080
    return op