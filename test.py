import numpy as np
import copy
import choose_dubins_path


R_c = 20 # communication range km
P = np.array([30, 30]) # position of the base station km
Pn = np.array([[25,60],[35,40],[42,54]]) # position of NFZ km
rho_s = 3 # radius of NFZ km
Wf = np.array([[5,20],[8,55],[25,87],[42,95],[55,59],[42,39],[32,37],[14,33],[7,20]]) # waypoint sequence of FF km
Vf = 0.05 # speed of FF km/s
width_f = 3 # width of FF km
length_f = 0.7 # length of FF km
ds = 2.5 # separation distance Km
Puc_init = np.array([18,25]) # position of UAV km
V_max = 0.05 # maximum speed of UAV km/s
n_max = 0.001 # Maximum lateral acceleration of UAV km/s^2
k0 = 0.9 # overlap coefficient
dt = 1 # time step
yes_speed = False
yes_NFZ = True

# turning radius
turn_R = V_max**2/(n_max * 9.81)
Wf_org = copy.deepcopy(Wf)

N_u = 3 # number of UAVs
N_nfz = len(Pn) # number of NFZs

# waypoint history of UAV center according to gcs position
## angle between gcs and waypoints
theta = np.arctan2(Wf[:,1]-P[1], Wf[:,0]-P[0])
# heading angle at between waypoints
theta_w = np.arctan2(Wf[1:,1]-Wf[:-1,1], Wf[1:,0]-Wf[:-1,0])

# formation shape of uavs' of relative to the UAV center
uav_form = np.zeros((N_u,2))
for i in range(1,N_u+1):
    if i%2 == 1: # uavs having an odd number
        uav_form[i-1,0] = (-(np.ceil(N_u/2)-1) + 2*np.floor(i/2))*ds
        uav_form[i-1,1] = -ds
    else: # uavs having an even number
        uav_form[i-1,0] = (-(np.ceil(N_u/2)-1) + 2*(np.floor(i/2)-1))*ds
        uav_form[i-1,1] = ds

Pu = np.zeros((N_u,2))
A = np.array([[np.cos(theta_w[0]), -np.sin(theta_w[0])],[np.sin(theta_w[0]), np.cos(theta_w[0])]])
for i in range(N_u):
    Pu[i,:] = Puc_init + np.matmul(uav_form[i,:], A.T)

## position of UAV center
Wuc = P + R_c * k0 * np.array([np.cos(theta), np.sin(theta)]).T
## asign first element of Wuc as the center of UAVs at the beginning
Wuc[0,0] = Puc_init[0]
Wuc[0,1] = Puc_init[1]

# check Wuc is in the communication range
index_GCS = np.zeros(len(Wuc))
for i in range(len(Wuc)):
    dist_to_GCS = np.linalg.norm(Wf[i,:]-P)
    if dist_to_GCS >= R_c+0.001: # 1m margin by numerical error
        index_GCS[i] = 1
    else:
        index_GCS[i] = 0

# generate path of FF
t = [0]
ff = [] # path of FF
ff_traj = [] # trajectory of FF
t_arv_org = [] # arrival time of FF
t_arv = [] # arrival time of FF
## vertices of FF
x_ff = Wf[0,0]
y_ff = Wf[0,1]
ff_traj.append(np.array([x_ff, y_ff]))
bd_ff = np.array([[-width_f/2, length_f/2],[width_f/2, length_f/2],[width_f/2, -length_f/2],[-width_f/2, -length_f/2]])

## rotation matrix
A = np.array([[np.cos(theta_w[0]), -np.sin(theta_w[0])],[np.sin(theta_w[0]), np.cos(theta_w[0])]])
## rotated boundary of FF
temp = np.matmul(bd_ff, A.T)
## add FF
ff.append(temp + np.ones((len(bd_ff),1))*np.array([[x_ff, y_ff]]))

i = 1
j = 0
k = 0
while j < len(theta_w):
    t.append(t[i-1]+dt)
    x_ff = x_ff + Vf*np.cos(theta_w[j])*dt
    y_ff = y_ff + Vf*np.sin(theta_w[j])*dt
    ff_traj.append(np.array([x_ff, y_ff]))

    A = np.array([[np.cos(theta_w[j]), -np.sin(theta_w[j])],[np.sin(theta_w[j]), np.cos(theta_w[j])]])
    temp = np.matmul(bd_ff, A.T)
    ff.append(temp + np.ones((len(bd_ff),1))*np.array([[x_ff, y_ff]]))

    # Arrival time estimation to the next waypoint
    if np.linalg.norm(np.array([x_ff,y_ff])-Wf_org[k+1,:]) < Vf*dt*2:
        t_arv_org.append(t[i])
        k = k + 1

    if np.linalg.norm(np.array([x_ff,y_ff])-Wf[j+1,:]) < Vf*dt*2:
        t_arv.append(t[i])
        j = j + 1

    i = i + 1

# waypoint history of individual UAVs
UAVs = dict()
uav_form2 = np.zeros((N_u,2))
for i in range(N_u):
    UAVs[i] = np.zeros((len(Wf),2))
    for j in range(len(Wf)):
        if j==0:
            A = np.array([[np.cos(theta_w[j]), -np.sin(theta_w[j])],[np.sin(theta_w[j]), np.cos(theta_w[j])]])
            temp = np.matmul(uav_form[i,:], A.T)
            UAVs[i][j,:] = temp + Wuc[j,:]
        else:
            if index_GCS[j] == 1:
                # a ratio of the distance from GCS to FF over R_c
                kf = np.linalg.norm(Wf[j,:]-P)/R_c
                Nm = np.floor(kf/k0 - 1)

                # rotation matrix: align normal to the line between GCS and FF
                h_temp = np.arctan2(Wf[j,1]-P[1], Wf[j,0]-P[0]) - np.pi/2

                if kf > 1+k0:
                    if i+1 > N_u - Nm:
                        h_temp2 = h_temp + np.pi/2
                        A = np.array([[np.cos(h_temp2), -np.sin(h_temp2)],[np.sin(h_temp2), np.cos(h_temp2)]])
                        ESD = -(N_u - i)*k0*R_c
                        uav_form2[i,0] = ESD
                        uav_form2[i,1] = 0
                        temp = np.matmul(uav_form2[i,:], A.T)
                        UAVs[i][j,:] = temp + Wf[j,:]
                    else:
                        A = np.array([[np.cos(h_temp), -np.sin(h_temp)],[np.sin(h_temp), np.cos(h_temp)]])
                        uav_form2[i,0] = ((-(N_u-Nm-1) + 2*(i)))*ds
                        uav_form2[i,1] = 0
                        temp = np.matmul(uav_form2[i,:], A.T)
                        UAVs[i][j,:] = temp + Wuc[j,:]
                else:
                    A = np.array([[np.cos(h_temp), -np.sin(h_temp)],[np.sin(h_temp), np.cos(h_temp)]])
                    uav_form2[i,0] = (-(N_u-1) + 2*(i))*ds
                    uav_form2[i,1] = 0
                    temp = np.matmul(uav_form2[i,:], A.T)
                    UAVs[i][j,:] = temp + Wuc[j,:]
            else:
                h_temp = np.arctan2(Wf[j,1]-P[1], Wf[j,0]-P[0]) - np.pi/2
                A = np.array([[np.cos(h_temp), -np.sin(h_temp)],[np.sin(h_temp), np.cos(h_temp)]])
                uav_form2[i,0] = (-(N_u-1) + 2*(i))*ds
                uav_form2[i,1] = 0
                temp = np.matmul(uav_form2[i,:], A.T)
                UAVs[i][j,:] = temp + Wuc[j,:]

# Time elapsed waypoints of FF
t_arv = np.insert(t_arv, 0, 0)
t_arv_diff = np.diff(t_arv)

# save original waypoint of UAVS
UAVs_org = copy.deepcopy(UAVs)

# # Speed condtraint
if yes_speed:
    for i in range(N_u):
        for j in range(len(UAVs[i])-1):
            # Radius able to arrive on time with V_max
            R_max = t_arv_diff[j]*V_max - 2*np.pi*turn_R
            if R_max < np.linalg.norm(UAVs[i][j+1,:]-UAVs[i][j,:]):
                print(i,j)
                # next waypoint
                x2 = UAVs[i][j+1,0]
                y2 = UAVs[i][j+1,1]
                # current waypoint
                x1 = UAVs[i][j,0]
                y1 = UAVs[i][j,1]

                # compute closest point to the next waypoint
                mm = (y2-y1)/(x2-x1)
                bb = y1 - mm*x1
                A = mm**2 + 1
                B = -(x1 - bb*mm + y1*mm)
                C = x1**2 + bb**2 - 2*y1*bb + y1**2 - R_max**2
                x_temp1 = (-B + np.sqrt(B**2 - A*C))/A
                x_temp2 = (-B - np.sqrt(B**2 - A*C))/A
                if abs(x2-x_temp1) <= abs(x2-x_temp2):
                    x_init = x_temp1
                else:
                    x_init = x_temp2
                y_init = mm*x_init + bb

                UAVs[i][j+1,:] = np.array([x_init, y_init])

# NFZ constraint
t_arv_NFZ = dict()
W_uav_NFZ = dict()

for i in range(N_u):
    t_arv_NFZ[i] = t_arv
    W_uav_NFZ[i] = copy.deepcopy(UAVs[i])
    NFZ_count = 0
    for j in range(len(UAVs[i])-1):
        temp = np.zeros(N_nfz)
        for k in range(N_nfz):
            temp[k] = np.linalg.norm(UAVs[i][j,:]-Pn[k,:])
        # get sorted order index of temp
        NFZ_order = np.argsort(temp)

        # look for the crossing point of the line
        for kk in range(N_nfz):
            k = NFZ_order[kk]

            # current UAV position
            a0 = UAVs[i][j,0]
            b0 = UAVs[i][j,1]
            # NFZ
            a1 = Pn[k,0]
            b1 = Pn[k,1]
            # next UAV position
            a2 = UAVs[i][j+1,0]
            b2 = UAVs[i][j+1,1]
            # gradient of line between current and next UAV position
            m = (b2-b0)/(a2-a0)
            b = -m*a0 + b0

            # yes or no of crossing the NFZ
            # distance from the line to the NFZ
            d = abs(m*(a1-a2)+b2-b1)/np.sqrt(m**2+1)
            # distance from the NFZ to the current UAV position
            d1 = np.linalg.norm(np.array([a1,b1])-np.array([a0,b0]))
            # distance from the NFZ to the next UAV position
            d2 = np.linalg.norm(np.array([a1,b1])-np.array([a2,b2]))

            # NFZ margin
            safe = 1.2

            if d < safe*rho_s:
                criterion1 = -1/m*(a0-a1)+b1-b0
                criterion2 = -1/m*(a2-a1)+b1-b2
                # perfectly cut
                if d1 >= safe*rho_s and d2 > safe*rho_s and criterion1*criterion2 < 0:
                    A = (a1-a0)**2 - (rho_s*safe)**2
                    B = (a1-a0)*(b0-b1)
                    C = (b0-b1)**2 - (rho_s*safe)**2
                    x_temp1 = (-B + np.sqrt(B**2 - A*C))/A
                    x_temp2 = (-B - np.sqrt(B**2 - A*C))/A

                    if np.imag(x_temp1) != 0:
                        print("imaginary",i,j)

                    if abs(m-x_temp1) <= abs(m-x_temp2):
                        m1 = x_temp1 # tangential line's gradient
                    else:
                        m1 = x_temp2 # tangential line's gradient
                    
                    m2 = -1/m # gradient normal to line between current and next UAV position
                    if m2 != np.inf:
                        x_init = (m1*a0 - m2*a1 + b1 - b0)/(m1-m2)
                        y_init = m1*(x_init-a0) + b0
                    else:
                        x_init = a1
                        y_init = m1*(x_init-a0) + b0

                    temp_d1 = np.linalg.norm(np.array([x_init,y_init])-UAVs[i][j,:])
                    temp_d2 = np.linalg.norm(np.array([x_init,y_init])-UAVs[i][j+1,:])
                    t_new = t_arv[j] + temp_d1/(temp_d1+temp_d2)*t_arv_diff[j]

                    if yes_NFZ:                    
                        W_uav_NFZ[i] = np.insert(W_uav_NFZ[i], j+1+NFZ_count, np.array([x_init, y_init]), axis=0)
                        t_arv_NFZ[i] = np.insert(t_arv_NFZ[i], j+1+NFZ_count, t_new)
                        NFZ_count = NFZ_count + 1

            elif d1 >= safe*rho_s and d2 <= safe*rho_s:                
                A = (a1-a0)**2 - (rho_s*safe)**2
                B = (a1-a0)*(b0-b1)
                C = (b0-b1)**2 - (rho_s*safe)**2
                x_temp1 = (-B + np.sqrt(B**2 - A*C))/A
                x_temp2 = (-B - np.sqrt(B**2 - A*C))/A

                if np.imag(x_temp1) != 0:
                    print("imaginary",i,j)

                if abs(m-x_temp1) <= abs(m-x_temp2):
                    m1 = x_temp1
                else:
                    m1 = x_temp2

                m2 = -1/m1
                if m2 != np.inf:
                    x_init = (m1*a0 - b0 - m2*a2 + b2)/(m1-m2)
                    y_init = m1*(x_init-a0) + b0
                else:
                    print("infinite gradient",i,j)
                    x_init = a2
                    y_init = m1*(x_init-a0) + b0

                # speed constraint
                if yes_speed:
                    R_max = t_arv_diff[j]*V_max - 2*np.pi*turn_R
                    if R_max < np.linalg.norm(np.array([x_init,y_init])-UAVs[i][j,:]):
                        a2 = x_init
                        b2 = y_init
                        x1 = UAVs[i][j,0]
                        y1 = UAVs[i][j,1]

                        mm = (y1-b2)/(x1-a2)
                        bb = mm*(-x1)+y1
                        A = 1 + mm**2
                        B = -(x1-bb*mm+y1*mm)
                        C = x1**2 + y1**2 + bb**2 - 2*y1*bb - R_max**2
                        x_temp1 = (-B + np.sqrt(B**2 - A*C))/A
                        x_temp2 = (-B - np.sqrt(B**2 - A*C))/A
                        if abs(a2-x_temp1) <= abs(a2-x_temp2):
                            x_init = x_temp1
                        else:
                            x_init = x_temp2
                        y_init = mm*x_init + bb

                if yes_NFZ:
                    W_uav_NFZ[i][j+N_nfz+1,:] = np.array([x_init, y_init])
                    UAVs[i][j+1,:] = np.array([x_init, y_init])


# Dubins path input of UAV
h_uav = dict()
waypoints = dict()
for k in range(N_u):
    h_uav[k] = np.zeros(len(W_uav_NFZ[k])-1)
    waypoints[k] = np.zeros((len(W_uav_NFZ[k]),4))
    # Heading hsitory of UAV between the waypoints
    h_uav[k] = np.arctan2(np.diff(W_uav_NFZ[k][:,1]), np.diff(W_uav_NFZ[k][:,0]))
    h_uav_temp = np.append(h_uav[k], h_uav[k][-1])
    waypoints[k][:,:2] = W_uav_NFZ[k]
    waypoints[k][:,2] = h_uav_temp*180/np.pi

    # Turning radius for each waypoint
    waypoints[k][:,3] = turn_R
    for i in range(len(W_uav_NFZ[k])-1):
        if h_uav_temp[i] == h_uav_temp[i+1] and (i+2) != len(W_uav_NFZ[k]):
            waypoints[k][i,3] = 0
            waypoints[k][i+1,3] = 0
        
        # conditions for existence of Dubins path
        d = np.linalg.norm(W_uav_NFZ[k][i,:]-W_uav_NFZ[k][i+1,:])
        Rs = waypoints[k][i,3]
        Rf = waypoints[k][i+1,3]
        cnd1 = abs(Rs-d)
        cnd2 = abs(Rf-d)
        # preventing Dubins error
        if ((cnd1 < Rf+2) or (cnd2 < Rs+2)):
            waypoints[k][i,3] = 0
            waypoints[k][i+1,3] = 0

# Initial Dubins path
# in order to compute the lengths of sub-segments
seg = [1000, 10, 1000]
N1 = np.zeros(N_u)
Length = dict()
Length_seg = dict()
for k in range(N_u):
    N, M = waypoints[k].shape
    N1[k] = N-1 # no. of path between waypoints
    Length[k] = np.zeros(int(N1[k]))
    Length_seg[k] = np.zeros((int(N1[k]),3))
    for i in range(int(N1[k])):
        # 'choose_dubins_path': compute the dubins path between input waypoints
        nposes = choose_dubins_path.choose_dubins_path(waypoints[k][i,:], waypoints[k][i+1,:], seg)
        Length[k][i] = nposes[4] # total length of Dubins path
        Length_seg[k][i,:] = np.array([nposes[1],nposes[2],nposes[3]]) # length of sub-segments

# Dubins path considering speed between segments
poses = dict()
for k in range(N_u):
    t_arv_NFZ_diff = np.diff(t_arv_NFZ[k])
    poses[k] = []
    for i in range(int(N1[k])):
        # Number of sub-segment to get the time-based position history
        seg = np.round(Length_seg[k][i,:]/Length[k][i] * t_arv_NFZ_diff[i]/dt)
        seg = seg.astype(int).tolist()
        poses[k].append(choose_dubins_path.choose_dubins_path(waypoints[k][i,:], waypoints[k][i+1,:], seg))

min_len = np.inf
uav_traj = dict()
for k in range(N_u):
    uav_traj[k] = []
    for i in range(int(N1[k])):
        if i < int(N1[k])-1:
            uav_traj[k].append(poses[k][i][0][:-1,:])
        else:
            uav_traj[k].append(poses[k][i][0][:,:])
    uav_traj[k]=np.vstack(uav_traj[k])
    min_len = np.min([len(t), len(uav_traj[k]), min_len])

t = t[:int(min_len)]
ff_traj = ff_traj[:int(min_len)]

for k in range(N_u):
    uav_traj[k] = uav_traj[k][:int(min_len),:]
