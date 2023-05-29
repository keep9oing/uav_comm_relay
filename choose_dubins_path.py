from utils import *
import numpy as np

def choose_dubins_path(P1: np.ndarray, P2: np.ndarray, seg: list):
    # The function choose_dubins_path chooses the shortest path from the set of four Dubins paths
    # P1 and P2 are the initial and final poses of P=[x_position y_position theta Rmin] is a column vector
    # seg is a real number represetns discretization.
    # IDX = 1 represents RR path, 2 represents LLpath, 3 represents RL path, 4 represents LR path

    """sumary_line
    Inputs:
        P1: initial pose
        P2: final pose
        seg: discretization, numbers of samplingpoint on start arc, line and final arc
    Outputs:
        op: dictionary of 6 keys
        op[0]: [x y psi]
        op[1]: length of the start arc
        op[2]: length of the line
        op[3]: length of the final arc
        op[4]: total length of the path
        op[5]: path type having minimum length
    """

    # Initialization
    ps = P1
    pf = P2
    config = np.vstack((ps, pf))

    # dubins path computation by four cases
    op_RR = dubins_rr(config, seg)
    op_LL = dubins_ll(config, seg)
    op_RL = dubins_rl(config, seg)
    op_LR = dubins_lr(config, seg)
    
    RR_length = op_RR[6]
    LL_length = op_LL[6]
    RL_length = op_RL[6]
    LR_length = op_LR[6]

    # find the dubins path having a minimum path length
    min_length = min(RR_length, LL_length, RL_length, LR_length)
    Idx = np.argmin([RR_length, LL_length, RL_length, LR_length])

    op = dict()
    if Idx==0:
        op[0] = np.vstack((op_RR[0], op_RR[1], op_RR[2]))
        op[1] = op_RR[3]
        op[2] = op_RR[4]
        op[3] = op_RR[5]
        op[4] = op_RR[6]
        op[5] = 0
    elif Idx==1:
        op[0] = np.vstack((op_LL[0], op_LL[1], op_LL[2]))
        op[1] = op_LL[3]
        op[2] = op_LL[4]
        op[3] = op_LL[5]
        op[4] = op_LL[6]
        op[5] = 1
    elif Idx==2:
        op[0] = np.vstack((op_RL[0], op_RL[1], op_RL[2]))
        op[1] = op_RL[3]
        op[2] = op_RL[4]
        op[3] = op_RL[5]
        op[4] = op_RL[6]
        op[5] = 2
    elif Idx==3:
        op[0] = np.vstack((op_LR[0], op_LR[1], op_LR[2]))
        op[1] = op_LR[3]
        op[2] = op_LR[4]
        op[3] = op_LR[5]
        op[4] = op_LR[6]
        op[5] = 3
    return op

def dubins_rr(pose: np.ndarray, n: list):
    # Produces Dubins path with RSR configuration
    """
    Inputs:
        pose: initial and final pose
        seg: discretization, numbers of samplingpoint on start arc, line and final arc
    Outputs:
        op: dictionary of 7 keys
        op[0]: pose_arc1
        op[1]: pose_line
        op[2]: pose_arc2
        op[3]: arc1_length
        op[4]: line_length
        op[5]: arc2_length
        op[6]: total_length
    """
    
    M, N = pose.shape

    if M%2 !=0 and N%4 != 0:
        raise ValueError('There at least a pair of poses')
    
    Ps = pose[0,:]
    Pf = pose[1,:]

    Rs = Ps[3]
    Rf = Pf[3]

    x1 = Ps[0]
    y1 = Ps[1]
    th1 = Ps[2]
    thc1 = th1 + 90

    x2 = Pf[0]
    y2 = Pf[1]
    th2 = Pf[2]
    thc2 = th2 + 90

    thc1 = angle_le_0(thc1)
    thc1 = angle_ge_360(thc1)
    thc2 = angle_le_0(thc2)
    thc2 = angle_ge_360(thc2)

    # Center of right turning circles
    xcs = x1 - Rs*np.cos(np.deg2rad(thc1))
    ycs = y1 - Rs*np.sin(np.deg2rad(thc1))

    xcf = x2 - Rf*np.cos(np.deg2rad(thc2))
    ycf = y2 - Rf*np.sin(np.deg2rad(thc2))

    # Condition for existence of Dubins path
    d = np.sqrt((xcf-xcs)**2 + (ycf-ycs)**2)

    cnd1 = abs(Rs-d)
    cnd2 = abs(Rf-d)

    if cnd1<Rf and cnd2<Rs:
        raise ValueError('There is no RR Dubins path')
    
    Rdiff = Rf-Rs
    q = np.arctan2(Rdiff, np.sqrt(d**2-Rdiff**2)*180/np.pi)

    phi1 = q + 90
    phi2 = q + 90
    phi1 = angle_le_0(phi1)
    phi1 = angle_ge_360(phi1)
    phi2 = angle_le_0(phi2)
    phi2 = angle_ge_360(phi2)

    mtgt = np.arctan2(ycf-ycs, xcf-xcs)*180/np.pi
    mtgt = angle_le_0(mtgt)
    mtgt = angle_ge_360(mtgt)

    if xcf>=xcs * ycf<ycs:
        shy1 = phi1 + 360 + mtgt
        shy2 = phi2 + 360 + mtgt
    else:
        shy1 = phi1 + mtgt
        shy2 = phi2 + mtgt

    shy1 = angle_le_0(shy1)
    shy1 = angle_ge_360(shy1)
    shy2 = angle_le_0(shy2)
    shy2 = angle_ge_360(shy2)

    Tsp = shy1
    Tfp = shy2
    thc1p = thc1
    thc2p = thc2
    if thc1p < Tsp:
        thc1p = thc1p + 360
    if Tfp < thc2p:
        thc2p = thc2p - 360

    ths = [thc1p, Tsp, Tfp, thc2p]

    xTx = xcs + Rs*np.cos(np.deg2rad(shy1))
    yTx = ycs + Rs*np.sin(np.deg2rad(shy1))
    xTn = xcf + Rf*np.cos(np.deg2rad(shy2))
    yTn = ycf + Rf*np.sin(np.deg2rad(shy2))

    x_arc1 = xcs + Rs*np.cos(np.deg2rad(np.linspace(thc1p, Tsp, n[0])))
    y_arc1 = ycs + Rs*np.sin(np.deg2rad(np.linspace(thc1p, Tsp, n[0])))
    x_line = np.linspace(xTx, xTn, n[1])
    y_line = np.linspace(yTx, yTn, n[1])
    x_arc2 = xcf + Rf*np.cos(np.deg2rad(np.linspace(Tfp, thc2p, n[2])))
    y_arc2 = ycf + Rf*np.sin(np.deg2rad(np.linspace(Tfp, thc2p, n[2])))

    # Output generation
    pose_arc1 = np.vstack((x_arc1, y_arc1, np.linspace(thc1p, Tsp, n[0]))).T
    pose_line = np.vstack((x_line, y_line, Tsp*np.ones(n[1]))).T
    pose_arc2 = np.vstack((x_arc2, y_arc2, np.linspace(Tfp, thc2p, n[2]))).T

    pose_line = np.vstack((x_line[1:-1], y_line[1:-1], Tsp*np.ones(n[1]-2))).T

    # Length of the path
    arc1_length = Rs*abs(thc1p-Tsp)*np.pi/180
    line_length = np.sqrt((xTx-xTn)**2 + (yTx-yTn)**2)
    arc2_length = Rf*abs(Tfp-thc2p)*np.pi/180
    total_length = arc1_length + line_length + arc2_length

    op = {0: pose_arc1, 1: pose_line, 2: pose_arc2, 3: arc1_length, 4: line_length, 5: arc2_length, 6: total_length}

    return op

def dubins_ll(pose: np.ndarray, n: list):
    # Produces Dubins path with LSL configuration
    """
    Inputs:
        pose: initial and final pose
        seg: discretization, numbers of samplingpoint on start arc, line and final arc
    Outputs:
        op: dictionary of 7 keys
        op[0]: pose_arc1
        op[1]: pose_line
        op[2]: pose_arc2
        op[3]: arc1_length
        op[4]: line_length
        op[5]: arc2_length
        op[6]: total_length
    """
    
    M, N = pose.shape

    if M%2 !=0 and N%4 != 0:
        raise ValueError('There at least a pair of poses')
    
    Ps = pose[0,:]
    Pf = pose[1,:]

    Rs = Ps[3]
    Rf = Pf[3]

    x1 = Ps[0]
    y1 = Ps[1]
    th1 = Ps[2]
    thc1 = th1 - 90

    x2 = Pf[0]
    y2 = Pf[1]
    th2 = Pf[2]
    thc2 = th2 - 90

    thc1 = angle_le_0(thc1)
    thc1 = angle_ge_360(thc1)
    thc2 = angle_le_0(thc2)
    thc2 = angle_ge_360(thc2)

    # Center of right turning circles
    xcs = x1 - Rs*np.cos(np.deg2rad(thc1))
    ycs = y1 - Rs*np.sin(np.deg2rad(thc1))

    xcf = x2 - Rf*np.cos(np.deg2rad(thc2))
    ycf = y2 - Rf*np.sin(np.deg2rad(thc2))

    # Condition for existence of Dubins path
    d = np.sqrt((xcf-xcs)**2 + (ycf-ycs)**2)

    cnd1 = abs(Rs-d)
    cnd2 = abs(Rf-d)

    if cnd1<Rf and cnd2<Rs:
        raise ValueError('There is no LL Dubins path')
    
    Rdiff = Rf-Rs
    q = np.arctan2(Rdiff, np.sqrt(d**2-Rdiff**2)*180/np.pi)

    phi1 = 270 - q
    phi2 = 270 - q
    phi1 = angle_le_0(phi1)
    phi1 = angle_ge_360(phi1)
    phi2 = angle_le_0(phi2)
    phi2 = angle_ge_360(phi2)

    mtgt = np.arctan2(ycf-ycs, xcf-xcs)*180/np.pi
    mtgt = angle_le_0(mtgt)
    mtgt = angle_ge_360(mtgt)

    if xcf>=xcs and ycf<ycs:
        shy1 = phi1 + 360 + mtgt
        shy2 = phi2 + 360 + mtgt
    else:
        shy1 = phi1 + mtgt
        shy2 = phi2 + mtgt

    shy1 = angle_le_0(shy1)
    shy1 = angle_ge_360(shy1)
    shy2 = angle_le_0(shy2)
    shy2 = angle_ge_360(shy2)

    Tsp = shy1
    Tfp = shy2
    thc1p = thc1
    thc2p = thc2
    if thc1p > Tsp:
        thc1p = thc1p - 360
    if Tfp > thc2p:
        thc2p = thc2p + 360

    ths = [thc1p, Tsp, Tfp, thc2p]

    xTx = xcs + Rs*np.cos(np.deg2rad(shy1))
    yTx = ycs + Rs*np.sin(np.deg2rad(shy1))
    xTn = xcf + Rf*np.cos(np.deg2rad(shy2))
    yTn = ycf + Rf*np.sin(np.deg2rad(shy2))

    x_arc1 = xcs + Rs*np.cos(np.deg2rad(np.linspace(thc1p, Tsp, n[0])))
    y_arc1 = ycs + Rs*np.sin(np.deg2rad(np.linspace(thc1p, Tsp, n[0])))
    x_line = np.linspace(xTx, xTn, n[1])
    y_line = np.linspace(yTx, yTn, n[1])
    x_arc2 = xcf + Rf*np.cos(np.deg2rad(np.linspace(Tfp, thc2p, n[2])))
    y_arc2 = ycf + Rf*np.sin(np.deg2rad(np.linspace(Tfp, thc2p, n[2])))

    # Output generation
    pose_arc1 = np.vstack((x_arc1, y_arc1, np.linspace(thc1p, Tsp, n[0]))).T
    pose_line = np.vstack((x_line, y_line, Tsp*np.ones(n[1]))).T
    pose_arc2 = np.vstack((x_arc2, y_arc2, np.linspace(Tfp, thc2p, n[2]))).T

    pose_line = np.vstack((x_line[1:-1], y_line[1:-1], Tsp*np.ones(n[1]-2))).T

    # Length of the path
    arc1_length = Rs*abs(thc1p-Tsp)*np.pi/180
    line_length = np.sqrt((xTx-xTn)**2 + (yTx-yTn)**2)
    arc2_length = Rf*abs(Tfp-thc2p)*np.pi/180
    total_length = arc1_length + line_length + arc2_length

    op = {0: pose_arc1, 1: pose_line, 2: pose_arc2, 3: arc1_length, 4: line_length, 5: arc2_length, 6: total_length}

    return op

def dubins_rl(pose: np.ndarray, n: list):
    # Produces Dubins path with RSL configuration
    """
    Inputs:
        pose: initial and final pose
        seg: discretization, numbers of samplingpoint on start arc, line and final arc
    Outputs:
        op: dictionary of 7 keys
        op[0]: pose_arc1
        op[1]: pose_line
        op[2]: pose_arc2
        op[3]: arc1_length
        op[4]: line_length
        op[5]: arc2_length
        op[6]: total_length
    """
    
    M, N = pose.shape

    if M%2 !=0 and N%4 != 0:
        raise ValueError('There at least a pair of poses')
    
    Ps = pose[0,:]
    Pf = pose[1,:]

    Rs = Ps[3]
    Rf = Pf[3]

    x1 = Ps[0]
    y1 = Ps[1]
    th1 = Ps[2]
    thc1 = th1 + 90

    x2 = Pf[0]
    y2 = Pf[1]
    th2 = Pf[2]
    thc2 = th2 - 90

    thc1 = angle_le_0(thc1)
    thc1 = angle_ge_360(thc1)
    thc2 = angle_le_0(thc2)
    thc2 = angle_ge_360(thc2)

    # Center of right turning circles
    xcs = x1 - Rs*np.cos(np.deg2rad(thc1))
    ycs = y1 - Rs*np.sin(np.deg2rad(thc1))

    xcf = x2 - Rf*np.cos(np.deg2rad(thc2))
    ycf = y2 - Rf*np.sin(np.deg2rad(thc2))

    # Condition for existence of Dubins path
    d = np.sqrt((xcf-xcs)**2 + (ycf-ycs)**2)

    cnd = abs(Rs+Rf)

    if cnd > d:
        raise ValueError('There is no RL Dubins path')
    
    Rsum = Rf + Rs
    q = np.arctan2(np.sqrt(d**2-Rsum**2), Rsum)*180/np.pi

    phi1 = q
    phi2 = 180 + q
    phi1 = angle_le_0(phi1)
    phi1 = angle_ge_360(phi1)
    phi2 = angle_le_0(phi2)
    phi2 = angle_ge_360(phi2)

    mtgt = np.arctan2(ycf-ycs, xcf-xcs)*180/np.pi
    mtgt = angle_le_0(mtgt)
    mtgt = angle_ge_360(mtgt)

    if xcf>=xcs and ycf<ycs:
        shy1 = phi1 + 360 + mtgt
        shy2 = phi2 + 360 + mtgt
    else:
        shy1 = phi1 + mtgt
        shy2 = phi2 + mtgt

    shy1 = angle_le_0(shy1)
    shy1 = angle_ge_360(shy1)
    shy2 = angle_le_0(shy2)
    shy2 = angle_ge_360(shy2)

    Tsp = shy1
    Tfp = shy2
    thc1p = thc1
    thc2p = thc2
    if thc1p < Tsp:
        thc1p = thc1p + 360
    if Tfp > thc2p:
        thc2p = thc2p + 360

    ths = [thc1p, Tsp, Tfp, thc2p]

    xTx = xcs + Rs*np.cos(np.deg2rad(shy1))
    yTx = ycs + Rs*np.sin(np.deg2rad(shy1))
    xTn = xcf + Rf*np.cos(np.deg2rad(shy2))
    yTn = ycf + Rf*np.sin(np.deg2rad(shy2))

    x_arc1 = xcs + Rs*np.cos(np.deg2rad(np.linspace(thc1p, Tsp, n[0])))
    y_arc1 = ycs + Rs*np.sin(np.deg2rad(np.linspace(thc1p, Tsp, n[0])))
    x_line = np.linspace(xTx, xTn, n[1])
    y_line = np.linspace(yTx, yTn, n[1])
    x_arc2 = xcf + Rf*np.cos(np.deg2rad(np.linspace(Tfp, thc2p, n[2])))
    y_arc2 = ycf + Rf*np.sin(np.deg2rad(np.linspace(Tfp, thc2p, n[2])))

    # Output generation
    pose_arc1 = np.vstack((x_arc1, y_arc1, np.linspace(thc1p, Tsp, n[0]))).T
    pose_line = np.vstack((x_line, y_line, Tsp*np.ones(n[1]))).T
    pose_arc2 = np.vstack((x_arc2, y_arc2, np.linspace(Tfp, thc2p, n[2]))).T

    pose_line = np.vstack((x_line[1:-1], y_line[1:-1], Tsp*np.ones(n[1]-2))).T

    # Length of the path
    arc1_length = Rs*abs(thc1p-Tsp)*np.pi/180
    line_length = np.sqrt((xTx-xTn)**2 + (yTx-yTn)**2)
    arc2_length = Rf*abs(Tfp-thc2p)*np.pi/180
    total_length = arc1_length + line_length + arc2_length

    op = {0: pose_arc1, 1: pose_line, 2: pose_arc2, 3: arc1_length, 4: line_length, 5: arc2_length, 6: total_length}

    return op

def dubins_lr(pose: np.ndarray, n: list):
    # Produces Dubins path with LSR configuration
    """
    Inputs:
        pose: initial and final pose
        seg: discretization, numbers of samplingpoint on start arc, line and final arc
    Outputs:
        op: dictionary of 7 keys
        op[0]: pose_arc1
        op[1]: pose_line
        op[2]: pose_arc2
        op[3]: arc1_length
        op[4]: line_length
        op[5]: arc2_length
        op[6]: total_length
    """
    
    M, N = pose.shape

    if M%2 !=0 and N%4 != 0:
        raise ValueError('There at least a pair of poses')
    
    Ps = pose[0,:]
    Pf = pose[1,:]

    Rs = Ps[3]
    Rf = Pf[3]

    x1 = Ps[0]
    y1 = Ps[1]
    th1 = Ps[2]
    thc1 = th1 - 90

    x2 = Pf[0]
    y2 = Pf[1]
    th2 = Pf[2]
    thc2 = th2 + 90

    thc1 = angle_le_0(thc1)
    thc1 = angle_ge_360(thc1)
    thc2 = angle_le_0(thc2)
    thc2 = angle_ge_360(thc2)

    # Center of right turning circles
    xcs = x1 - Rs*np.cos(np.deg2rad(thc1))
    ycs = y1 - Rs*np.sin(np.deg2rad(thc1))

    xcf = x2 - Rf*np.cos(np.deg2rad(thc2))
    ycf = y2 - Rf*np.sin(np.deg2rad(thc2))

    # Condition for existence of Dubins path
    d = np.sqrt((xcf-xcs)**2 + (ycf-ycs)**2)

    cnd = abs(Rs+Rf)

    if cnd > d:
        raise ValueError('There is no LR Dubins path')
    
    Rsum = Rf + Rs
    q = np.arctan2(np.sqrt(d**2-Rsum**2), Rsum)*180/np.pi

    phi1 = 360 - q
    phi2 = 180 - q
    phi1 = angle_le_0(phi1)
    phi1 = angle_ge_360(phi1)
    phi2 = angle_le_0(phi2)
    phi2 = angle_ge_360(phi2)

    mtgt = np.arctan2(ycf-ycs, xcf-xcs)*180/np.pi
    mtgt = angle_le_0(mtgt)
    mtgt = angle_ge_360(mtgt)

    if xcf>=xcs and ycf<ycs:
        shy1 = phi1 + 360 + mtgt
        shy2 = phi2 + 360 + mtgt
    else:
        shy1 = phi1 + mtgt
        shy2 = phi2 + mtgt

    shy1 = angle_le_0(shy1)
    shy1 = angle_ge_360(shy1)
    shy2 = angle_le_0(shy2)
    shy2 = angle_ge_360(shy2)

    Tsp = shy1
    Tfp = shy2
    thc1p = thc1
    thc2p = thc2
    if thc1p > Tsp:
        thc1p = thc1p + 360
    if Tfp < thc2p:
        thc2p = thc2p + 360

    ths = [thc1p, Tsp, Tfp, thc2p]

    xTx = xcs + Rs*np.cos(np.deg2rad(shy1))
    yTx = ycs + Rs*np.sin(np.deg2rad(shy1))
    xTn = xcf + Rf*np.cos(np.deg2rad(shy2))
    yTn = ycf + Rf*np.sin(np.deg2rad(shy2))

    x_arc1 = xcs + Rs*np.cos(np.deg2rad(np.linspace(thc1p, Tsp, n[0])))
    y_arc1 = ycs + Rs*np.sin(np.deg2rad(np.linspace(thc1p, Tsp, n[0])))
    x_line = np.linspace(xTx, xTn, n[1])
    y_line = np.linspace(yTx, yTn, n[1])
    x_arc2 = xcf + Rf*np.cos(np.deg2rad(np.linspace(Tfp, thc2p, n[2])))
    y_arc2 = ycf + Rf*np.sin(np.deg2rad(np.linspace(Tfp, thc2p, n[2])))

    # Output generation
    pose_arc1 = np.vstack((x_arc1, y_arc1, np.linspace(thc1p, Tsp, n[0]))).T
    pose_line = np.vstack((x_line, y_line, Tsp*np.ones(n[1]))).T
    pose_arc2 = np.vstack((x_arc2, y_arc2, np.linspace(Tfp, thc2p, n[2]))).T

    pose_line = np.vstack((x_line[1:-1], y_line[1:-1], Tsp*np.ones(n[1]-2))).T

    # Length of the path
    arc1_length = Rs*abs(thc1p-Tsp)*np.pi/180
    line_length = np.sqrt((xTx-xTn)**2 + (yTx-yTn)**2)
    arc2_length = Rf*abs(Tfp-thc2p)*np.pi/180
    total_length = arc1_length + line_length + arc2_length

    op = {0: pose_arc1, 1: pose_line, 2: pose_arc2, 3: arc1_length, 4: line_length, 5: arc2_length, 6: total_length}

    return op