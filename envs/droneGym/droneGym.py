import gym
import pandas as pd
from gym import spaces
import numpy as np
import json
import os
import time
import csv
from collections import deque
import PIDcontrol
from simulator import *

# Physical Constants
m = 1.07         #kg
Ixx = 0.0093   #kg-m^2
Iyy = 0.0093   #kg-m^2
# Izz = 0.9*(Ixx + Iyy) #kg-m^2 (Assume nearly flat object, z=0)
Izz = .0151
Ir = .0066 #rotor moment of inertia
dx = 0.214     #m
# dy = 0.0825     #m
dy = dx
dragCoef = 4.406*10**-7 #kg*m^2*s^-1
g = 9.81  #m/s/s
DTR = 1/57.3; RTD = 57.3
thrustCoef = 1.5108 * 10**-5 #kg*m


class droneGym(gym.Env):
    """Custom Environment that follows gym interface"""
    dt: float
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(droneGym, self).__init__()
        self.dt = .01
        self.t = 0
        self.startTime = time.time()

        self.diagPath = os.path.join(os.getcwd(), 'details.csv')
        os.remove(self.diagPath) if os.path.exists(self.diagPath) else None
        with open(self.diagPath,'w',newline = '') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['Time','Simulated Time','Failed on','Reward','Altitude (m)', 'Roll','Pitch','Yaw', 'X_distance','Y_distance'])

        # Define action and observation space
        # self.action_space = spaces.Box(low = np.array((0,0,0,0)), high = np.array((1,1,1,1)))
        self.action_space = spaces.Box(low = np.array((0,0,0)), high = np.array((1,1,1)))
        # self.action_space = spaces.Box(low = np.array((0,0,0,0)), high = np.array((100,100,100,100)))
        # self.action_space = spaces.Tuple((spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2)))

        self.action_space.n = 3

        self.ds = droneSim()

        self.x = self.stateMatrixInit()

        # self.observation_space = spaces.Box(low=np.array((-100,-100,-100,0,0,0,0,0,0,-100,-100,-100,-100,-100,-100)), high=np.array((100,100,100,7,7,7,7,7,7,100,100,100,100,100,100)))
        # self.observation_space = spaces.Box(low=np.array((-np.inf,-np.inf,-np.inf,-np.inf)), high=np.array((np.inf,np.inf,np.inf,np.inf)))
        self.observation_space = spaces.Box(low=np.array((-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)), high=np.array((np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf)))

        self.rateLimitUp = 4
        self.rateLimitDown = 4#8

        self.reward_range = np.array((-np.inf,1))

        self.times = [self.t]
        self.xdot_b = []#latitudinal velocity body frame
        self.ydot_b = []#latitudinal velocity body frame
        self.zdot_b = []#latitudinal velocity body frame
        self.p = []#rotational velocity body frame
        self.q = []#rotational velocity body frame
        self.r = []#rotational velocity body frame
        self.phi = []#euler rotation global frame
        self.theta = []#euler rotation global frame
        self.psi = []#euler rotation global frame
        self.xpos = []#global x position
        self.y = []#global y position
        self.z = []#global z position

        self.u1 = []
        self.u2 = []
        self.u3 = []
        self.u4 = []

        self.prevU = np.zeros(4)

        self.rewardList = []

        self.altController =  PIDcontrol.PIDControl('Alt', Kp =50, Ki = 6, Kd = 28, timeStep = self.dt, open = False)
        self.rollController = PIDcontrol.PIDControl('Roll', Kp=2, Ki=.1, Kd=0, timeStep=self.dt, open=False)
        self.pitchController = PIDcontrol.PIDControl('Pitch', Kp=2, Ki=.1, Kd=0, timeStep=self.dt, open=False)
        self.yawController = PIDcontrol.PIDControl('Yaw', Kp=600, Ki=0, Kd=0, timeStep=self.dt, open=False)
        self.zSet = 10
        self.temp = []

    def checkActionStepSize(self, action):
        #limit step-to-step action size (imitating motor inertia)
        limitedActions = np.zeros(4)
        for i,n in enumerate(action):
            diff = n - self.prevU[i]
            if diff > self.rateLimitUp:
                limitedActions[i] = self.prevU[i] + self.rateLimitUp
            elif diff < -self.rateLimitDown:
                limitedActions[i] = self.prevU[i] - self.rateLimitDown
            else:
                limitedActions[i] = n

        self.prevU = limitedActions
        return limitedActions
        # return action


    def createUs(self, state, action):
        errs = np.zeros(4)
        errs[0] = action[0] - state[11]
        errs[1] = action[1] - state[6]
        errs[2] = action[2] - state[7]
        errs[3] = action[3] - state[8]

        u = np.zeros(4)
        bsZerr = self.zSet - state[11]
        u[0] = self.altController.updateControl(bsZerr)
        u[1] = self.rollController.updateControl(errs[1])
        u[2] = self.pitchController.updateControl(errs[2])
        u[3] = self.yawController.updateControl(errs[3])

        return u

    def step(self, action):
        # Execute one time step within the environment
        #This is where we're accepting the action requirements from the RL agent
        #Right now since we're producing sigmoid outputs, we're modifying the inputs accordingly
        #Action[0] = z target (in meters?  Currently unused)
        #Action[1] = Phi Reference Angle (x, in drone reference frame)
        #Action[2] = Theta Reference Angle (y, in drone reference frame)

        maxAngleAllowed = .6457718 #around 37 degrees
        # if len(action) < 3:
        #     newAct = np.zeros(3)
        #     for i,n in enumerate(action):
        #         newAct[i] = 1/(1+ np.exp(-n))
        #     action = newAct

        # action[0] = (action[0]/200 + .5) * 100
        # action[0] = (action[0] -.5)*10 #Z velocity estimate?
        # action[0] = (action[0]) * 50 #Z position Target?
        # action[1:] = [(i-.5)*maxAngleAllowed for i in action[1:]]
        # action = np.append(action,0)
        temp = action
        self.actionnn = action
        action = action * 100
        action[1:] = action[1:] - 50

        # action[0] = action[0]*1.5
        # if action[0]>100: action[0] = 100

        temp = action

        action = self.checkActionStepSize(action)
        self.actionActual = action[0:3]

        if np.isnan(action[0]):
            print('tt')

        # self.uActions = self.createUs(self.x, action)
        # x_next = self.numericalIntegration(self.x,self.uActions,self.dt)

        # x_next = self.numericalIntegration(self.x,action,self.dt)
        self.ds.x = self.x[0:12]
        tempX, currU, xdot = self.ds.numericalIntegration(self.x[0:12], action, self.dt, errsIsControl=True)
        x_next = np.zeros(16)
        x_next[0:12] = tempX
        self.t += self.dt

        self.globalAngularVel = xdot[[6,7,8]]

        if x_next[11] < .05:
            x_next[2] = np.min([self.x[2],x_next[2],0])
            x_next[11] = np.max([self.x[11],x_next[11],0])

            # x_next[3] = 0
            # x_next[4] = 0
            # x_next[5] = 0

        reward, done = self.calcReward(self.x)

        # x_next[14] = x_next[11] - self.zSet
        # x_next[13] = x_next[10] - self.ySet
        # x_next[12] = x_next[9] - self.xSet
        x_next[14] = x_next[5] - self.angular_rate_sp[2]
        x_next[13] = x_next[4] - self.angular_rate_sp[1]
        x_next[12] = x_next[3] - self.angular_rate_sp[0]

        x_next[15] = x_next[11] - 6

        self.x = x_next
        self.memory(self.x, temp)

        return self.add_noise(self.x[[12,13,14,15,6,7,8]]), reward, done, {}

    def add_noise(self, x):

        for i in range(0,len(x)):
            x[i] = np.random.uniform(-.05,.05) * x[i] + x[i]

        return x

    def reset(self):
        # Reset the state of the environment to an initial state
        self.t = 0
        self.ds = droneSim()
        self.x = self.stateMatrixInit()

        self.times = [self.t]
        self.xdot_b = []#latitudinal velocity body frame
        self.ydot_b = []#latitudinal velocity body frame
        self.zdot_b = []#latitudinal velocity body frame
        self.p = []#rotational velocity body frame
        self.q = []#rotational velocity body frame
        self.r = []#rotational velocity body frame
        self.phi = []#euler rotation global frame
        self.theta = []#euler rotation global frame
        self.psi = []#euler rotation global frame
        self.xpos = []#global x position
        self.y = []#global y position
        self.z = []#global z position

        self.u1 = []
        self.u2 = []
        self.u3 = []
        self.u4 = []

        self.prevU = np.zeros(4)

        self.rewardList = []

        self.altController =  PIDcontrol.PIDControl('Alt', Kp =50, Ki = 6, Kd = 28, timeStep = self.dt, open = False)
        self.rollController = PIDcontrol.PIDControl('Roll', Kp=2, Ki=.1, Kd=0, timeStep=self.dt, open=False)
        self.pitchController = PIDcontrol.PIDControl('Pitch', Kp=2, Ki=.1, Kd=0, timeStep=self.dt, open=False)
        self.yawController = PIDcontrol.PIDControl('Yaw', Kp=600, Ki=0, Kd=.1, timeStep=self.dt, open=False)
        self.temp = []

        posNegMult =np.sign(np.random.randint(0,2) -.1)
        self.xSetRandom = np.random.uniform(.5,1)*10*posNegMult
        self.ySetRandom = np.random.uniform(.5,1)*10*posNegMult

        self.prevDist = np.sqrt(self.xSetRandom**2 + self.ySetRandom**2)

        self.prev_shaping = None
        self.prev_action = np.zeros(self.action_space.n)#self.action_space.sample()
        # self.angular_rate_sp = [np.random.random()*.6457718, np.random.random()*.6457718, np.random.random()*.6457718]
        self.angular_rate_sp = [0,0,0]#

        return self.x[[12,13,14,11]]

    def render(self, mode='human', close=False, epNum = 0):
        # Render the environment to the screen
        newfileName = "Flight_run_ep_" + str(epNum)

        df = pd.DataFrame(list(zip(self.times, self.xdot_b, self.ydot_b, self.zdot_b, self.p, self.q, self.r, self.phi, self.theta, self.psi, self.xpos, self.y, self.z)),
                          columns=['t', 'xdot_b', 'ydot_b', 'zdot_b', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'x', 'y',
                                   'z'])
        # self.u1 = np.zeros(len(self.u1)) + 1
        self.u4 = np.zeros(len(self.u4)) + 1


        dfAction = pd.DataFrame(list(zip(self.u1, self.u2, self.u3, self.u4)), columns = ['U1','U2','U3','U4'])
        with open(os.path.join(r'C:\Users\Stephen\PycharmProjects\QuadcopterSim\visualizer',newfileName + '.js'), 'w') as outfile:
            outfile.truncate(0)
            outfile.write("var sim_data = [ \n")
            json.dump([i for i in self.times[0:-1]], outfile, indent=4)
            outfile.write(",\n")
            parsed1 = json.loads(
                dfAction[['U1','U2','U3','U4']].T.to_json(orient='values'))
            json.dump(parsed1, outfile, indent=4)
            outfile.write(",\n")
            parsed = json.loads(
                df[['xdot_b', 'ydot_b', 'zdot_b', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'x', 'y', 'z']].T.to_json(
                    orient='values'))
            json.dump(parsed, outfile, indent=4)
            outfile.write("]")



    def calcReward(self, state):
        #calculate reward based off of distance from setpoints
        done = False

        # if self.t < 5:
        xSet = self.xSetRandom
        ySet = self.ySetRandom
        zSet = 10
        # else:# self.t >= 5 and self.t<10:
        #     xSet = -self.ySetRandom
        #     ySet = -self.xSetRandom
        #     zSet = 10
        # elif self.t >= 10 and self.t < 15:
        #     xSet = -5
        #     ySet = 5
        #     zSet = 12
        # else:
        # xSet = 0
        # ySet = 0
        # zSet = 10

        self.xSet = xSet
        self.ySet = ySet
        self.zSet = zSet

        zErr, xErr, yErr = self.calculateError(state, [zSet, xSet, ySet])
        # desiredZvel = -self.altControl.updateControl(zErr)
        # desiredXvel = self.xControl.updateControl(xErr)
        # desiredYvel = self.yControl.updateControl(yErr)
        phiRef, thetaRef = self.globalNeededThrust(state, xErr, yErr)
        # totRefs = [0, 0, 0, phiRef, thetaRef, 0]
        # totRefs = [0, 0, 0, 0, 0, 0]
        totRefs = [phiRef, thetaRef, 0]


        ###
        #This is the calcualtion of the reward generated at each time step
        #The currently uncommented reward calculaction has a penalty of -1, and a reduction for when the drone moves towards the target
        #Other rewards listed here focus on minimizing the absolute distance, or rewarding small angles and short lasting flights
        ###


        # reward_unlog = 2.23606797749979 - np.sqrt(self.distin3d(state[9], state[10], state[11], xSet, ySet, zSet)) #sqrt of 0,0,5 position
        # reward = 10 * (np.exp(reward_unlog)/(np.exp(reward_unlog) + 1) - .5)
        # dist = self.distin3d(state[9], state[10], state[11], xSet, ySet, zSet)  # sqrt of 0,0,5 position
        # dist2d = self.distin3d(state[9], state[10], zSet, xSet, ySet, zSet)
        # # reward = round(reward)
        # # reward = -np.clip(np.sum(np.abs([(state[n]-totRefs[i]) for i,n in enumerate([6,7,8])])),0,1)
        # reward = -1 + 10*(self.prevDist-dist2d)#/(np.sqrt(self.xSet**2 + self.ySet**2))
        #
        # self.prevDist = dist2d
        maxAngleAllowed = 0.5745329
        # pitch_bad = not(-maxAngleAllowed < state[6] < maxAngleAllowed) and self.t > .4
        # roll_bad = not(-maxAngleAllowed < state[7] < maxAngleAllowed) and self.t > .4
        roll_bad = ((2*np.pi)-maxAngleAllowed) > np.abs(state[6]) > maxAngleAllowed
        pitch_bad = ((2*np.pi)-maxAngleAllowed) > np.abs(state[7]) > maxAngleAllowed
        alt_bad = not(.1 < state[11] < 100) and self.t > 1
        #
        # self.rewardList.append(reward)
        # goodDist = 3 #in m

        self.angular_rate_sp = np.zeros(3)
        self.true_error = self.angular_rate_sp - np.array([state[3], state[4], state[5]])
        self.true_error += self.angular_rate_sp - self.globalAngularVel
        shaping = -np.sum(self.true_error**2)

        e_penalty = 0
        if self.prev_shaping is not None:
            e_penalty = shaping - self.prev_shaping
        self.prev_shaping = shaping

        min_y_reward = 0

        threshold = np.maximum(np.abs(self.angular_rate_sp) * 0.1, np.array([2]*3))
        inband = (np.abs(self.true_error) <= threshold).all()
        percent_idle = 0.12
        max_min_y_reward = 1000
        if np.average(self.actionnn) < percent_idle:
            min_y_reward = max_min_y_reward * (1 - percent_idle) * inband
        else:
            min_y_reward = max_min_y_reward * (1 - np.average(self.actionnn)) * inband

        if roll_bad or pitch_bad:
            angleThreshPunishment = -1000000000
        else:
            angleThreshPunishment = 0

        rewards = [
            -100 * np.max(np.abs(self.actionActual - self.prev_action)),
            min_y_reward,
            angleThreshPunishment,
            10000000*e_penalty,
            -1e6 * np.sum(self.oversaturation_high()),
            self.doing_nothing_penalty(),
            self.on_the_ground_penalty(state),
            self.repeatedActionsPenalty(self.actionActual, self.prev_action)
        ]

        reward = np.sum(rewards)
        self.rewardList.append(reward)


        if self.t > 6: #or pitch_bad or roll_bad or alt_bad:# or dist<goodDist:
            # if self.t > 9.8:
            #     reward = 400
            #     # reward = 1
            #     self.rewardList.append(reward)
            # else:
            #     reward = 10000*self.t - 4000
            #     # reward = -10
            #     self.rewardList.append(reward)
            done = True

            failer = ''
            if pitch_bad:
                failer += "Pitch"
            if roll_bad:
                failer += 'Roll'
            if alt_bad:
                failer += "Alt"
            # if dist<goodDist:
            #     failer += "Dist"
            #     reward = 80


            with open(self.diagPath, 'a', newline = '') as csvFile:
                writer = csv.writer(csvFile)
                totReward = np.sum(self.rewardList)
                writer.writerow([round(time.time()-self.startTime,1), round(self.t,2), failer, round(totReward/self.t,3),
                                 round(state[11],3),round(state[6],3),round(state[7],3),round(state[8],3), np.ceil(state[12]), np.ceil(state[13])])

        self.prev_action = self.actionActual

        return reward/100000000, done

    def repeatedActionsPenalty(self,action, prevAction, penalty=1e4):
        numInfring = 0
        for i, n in enumerate(action):
            if n == prevAction[i]:
                numInfring += 1

        return numInfring*penalty

    def on_the_ground_penalty(self, state, penalty = 1e7):
        total_penalty = 0

        if state[11] < .06 and self.t > .15:
            total_penalty -= penalty

        return total_penalty

    def doing_nothing_penalty(self, penalty=1e7):
        total_penalty = 0

        if np.sum(self.actionnn == 0) > 1:# and not (self.angular_rate_sp == np.zeros(3)).all():
            total_penalty -= penalty

        if (self.actionnn ==1).all():
            total_penalty -= penalty

        return total_penalty

    def oversaturation_high(self):

        ac = np.maximum(self.actionnn, np.zeros(self.action_space.n))
        # return np.maximum(ac - np.ones(4), np.zeros(4))
        if len(np.where(ac == 1)[0]) > 0:
            return 1
        else:
            return 0

    def distin3d(self,x1,y1,z1,x2,y2,z2):
        #calculate distance in 3d space
        return(np.sqrt(np.power(x2-x1,2) + np.power(y2-y1,2) + np.power(z2-z1,2)))

    def memory(self,x, action):
        self.times.append(self.times[-1] + self.dt)
        self.xdot_b.append(x[0])
        self.ydot_b.append(x[1])
        self.zdot_b.append(x[2])
        self.p.append(x[3])
        self.q.append(x[4])
        self.r.append(x[5])
        self.phi.append(x[6])
        self.theta.append(x[7])
        self.psi.append(x[8])
        self.xpos.append(x[9])
        self.y.append(x[10])
        self.z.append(x[11])

        self.u1.append(action[0])
        self.u2.append(action[1])
        self.u3.append(action[2])
        # self.u4.append(action[3])
        self.u4.append(0)


    def stateMatrixInit(self):
        x = np.zeros(16)
        # x[2] = -.049
        x[11] = 8#.049
        x[12] = 0#9.951
        x[3] = (np.random.random()-.5)*1.8915436
        x[4] = (np.random.random()-.5)*1.8915436
        # x[5] = np.random.random()*.3457718
        # x0 = xdot_b = latitudinal velocity body frame
        # x1 = ydot_b = latitudinal velocity body frame
        # x2 = zdot_b = latitudinal velocity body frame
        # x3 = p = rotational velocity body frame
        # x4 = q = rotational velocity body frame
        # x5 = r = rotational velocity body frame
        # x6 = phi = euler rotation global frame
        # x7 = theta = euler rotation global frame
        # x8 = psi = euler rotation global frame
        # x9 = x = global x position
        # x10 = y = global y position
        # x11 = z = global z position
        return x

    # def processControlInputs(self, u):
    #
    #     #linearized motor response
    #     w_o = np.zeros(4)
    #     thrustForce = np.zeros(4)
    #
    #     modifyDef = 100000   #initial Val = 10000000
    #     lesDef = .013385701848569465 * modifyDef
    #
    #     for i,n in enumerate(u):
    #         # thrustForce[i] = .447675* n / 10
    #         try:
    #             w_o[i] = modifyDef *(-2/(1+np.e**((n/10)-5)) + 2) - lesDef #rough log equation mapping control signal (voltage) to rps
    #         except FloatingPointError as e:
    #             w_o[i] = modifyDef
    #         thrustForce[i] = thrustCoef * w_o[i]
    #
    #     F1 = thrustForce[0] + thrustForce[2] + thrustForce[3]/2
    #     F2 = thrustForce[0] - thrustForce[1] - thrustForce[3]/2
    #     F3 = thrustForce[0] - thrustForce[2] + thrustForce[3]/2
    #     F4 = thrustForce[0] + thrustForce[1] - thrustForce[3]/2
    #
    #     return F1, F2, F3, F4
    #
    # def stateTransition(self, x, u):
    #     xdot = np.zeros(16)
    #
    #     # Store values in a readable format
    #     ub = x[0]
    #     vb = x[1]
    #     wb = x[2]
    #     p = x[3]
    #     q = x[4]
    #     r = x[5]
    #     phi = x[6]
    #     theta = x[7]
    #     psi = x[8]
    #     xE = x[9]
    #     yE = x[10]
    #     hE = x[11]
    #
    #     F1, F2, F3, F4 = self.processControlInputs(u)
    #     # Calculate forces from propeller inputs
    #     # F1 = u#Fthrust(x, u[0], dx, dy)
    #     # F2 = u#Fthrust(x, u[1], -dx, -dy)
    #     # F3 = u#Fthrust(x, u[2], dx, -dy)
    #     # F4 = u#Fthrust(x, u[3], -dx, dy)
    #     Fz = F1 + F2 + F3 + F4
    #
    #     L = dy * (F4 - F2)
    #     M = dx * (F1 - F3)
    #     N = .01 * (F1 - F2 + F3 - F4)  # .01 = drag coef?  random scaling for yaw
    #
    #     # Pre-calculate trig values
    #     cphi = np.cos(phi)
    #     sphi = np.sin(phi)
    #     cthe = np.cos(theta)
    #     sthe = np.sin(theta)
    #     cpsi = np.cos(psi)
    #     spsi = np.sin(psi)
    #
    #     # Calculate the derivative of the state matrix using EOM
    #     xdot[0] = (1/m) * (g*sthe)
    #     xdot[1] = g * sphi / m
    #     xdot[2] = (1 / m) * (-Fz) + (g * cphi * cthe)
    #     xdot[3] = 1 / Ixx * (L + (Iyy - Izz) * q * r)  # = pdot
    #     xdot[4] = 1 / Iyy * (M + (Izz - Ixx) * p * r)  # = qdot
    #     xdot[5] = 1 / Izz * (N + (Ixx - Iyy) * p * q)  # = rdot
    #     xdot[6] = p + (q * sphi + r * cphi) * sthe / cthe  # = phidot
    #     xdot[7] = q * cphi - r * sphi  # = thetadot
    #     xdot[8] = (q * sphi + r * cphi) / cthe  # = psidot
    #     xdot[9] = cthe * cpsi * ub + (-cthe * spsi + sphi * sthe * cpsi) * vb + \
    #               (sphi * spsi + cphi * sthe * cpsi) * wb  # = xEdot
    #     xdot[10] = cthe * spsi * ub + (cphi * cpsi + sphi * sthe * spsi) * vb + \
    #                (-sphi * cpsi + cphi * sthe * spsi) * wb  # = yEdot
    #     xdot[11] = -1 * (-sthe * ub + sphi * cthe * vb + cphi * cthe * wb)  # = zEdot
    #
    #     #keep the target setpoints the same for now
    #     xdot[12] = x[12]
    #     xdot[13] = x[13]
    #     xdot[14] = x[14]
    #     xdot[15] = x[15]
    #
    #
    #     return xdot
    #
    # def numericalIntegration(self, x, action, dt):
    #     # for now accept whatever we get from the derivative, maybe in future use Runge
    #     x_next = x + self.stateTransition(x, action) * dt
    #
    #     for i,n in enumerate(x_next):
    #         if i in [0,1,2,9,10,11,12,13,14]:
    #             continue
    #         else:
    #             if np.abs(n)>2*np.pi:
    #                 x_next[i] = n % (2*np.pi)
    #
    #     if np.sum(np.isnan(x_next)):
    #         print('tt')
    #
    #     return x_next
    #
    def globalNeededThrust(self,x, u_x, u_y):
        #from https://liu.diva-portal.org/smash/get/diva2:1129641/FULLTEXT01.pdf, page 48

        phi = x[6]
        theta = x[7]
        psi = x[8]

        # Pre-calculate trig values
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cthe = np.cos(theta)
        sthe = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        maxAngleAllowed = 0.2745329 #10 degrees, actual max angle is ~36.86
        Kmax = np.sin(maxAngleAllowed)**2 / (1 - np.sin(maxAngleAllowed)**2)

        k = min(Kmax, u_x ** 2 + u_y ** 2)

        if u_x == 0 and u_y == 0:
            phiRef = 0
        else:
            phiRef = np.arcsin(np.sqrt(k/((1+k)*(u_x**2 + u_y**2)))*(u_x*spsi - u_y*cpsi))

        s = np.sign(u_y * spsi + u_x*cpsi)
        if np.cos(phiRef)*np.sqrt(1+k) < 1:
            thetaRef = 0
        else:
            thetaRef = s*np.arccos(1/(np.cos(phiRef)*np.sqrt(1+k)))
        # thetaRef = 0

        return phiRef, thetaRef

    def calculateError(self, x, setpoints):
        # setpoints = [alt, roll, pitch, yaw]
        altError = setpoints[0] - x[11]
        xError = setpoints[1] - x[9]
        yError = setpoints[2] - x[10]

        return altError, xError, yError