import numpy as np
from collections import deque
import pandas as pd
import sys

class PIDControl():
    def __init__(self, call, Kp = .05, Ki = .001, Kd = .002, timeStep = .01, open = False):
        self.call = call
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.open = open
        self.integratorEps = []#deque(maxlen = 100)
        self.eps = [0]
        self.us = [0]
        self.prevU = 0
        self.du = 0
        self.rateLimitUp = 2
        self.rateLimitDown = 8
        self.dt = timeStep

        self._integral = 0

        #for hover, all reference signals should be zero
        #except for altitude, which should be whatever

        #control signals u1-u4 definitions:
        #b is thrust coef, d is drag coef.  Tuneable....

        # u1 = b*(w1**2 + w2**2 + w3**2 + w4**2)
        # u2 = b*(w2**2 - w4**2)
        # u3 = b*(w3**2 - w1**2)
        # u4 = d*(w1**2 - w2**2 + w3**2 - w4**2)

    def updateControl(self, e):

        self.eps.append(e)
        self.integratorEps.append(e)
        # weights = np.arange(1,len(self.integratorEps)+1)

        P = self.Kp * self.eps[-1]
        # I = self.Ki * np.sum(self.integratorEps)
        if self.us[-1] < 100 :#and self.us[-1] > 0
            self._integral += self.Ki * self.eps[-1] * self.dt
        D = self.Kd * (self.eps[-1]-self.eps[-2])/self.dt
        # if len(self.integratorEps) == 0:
        # else:
        #     I = self.Ki * np.average(self.integratorEps, weights=weights) * len(self.integratorEps)
        I = self._integral

        u = P + I + D


        if u < 0 and self.call == "Alt":
            self.integratorEps.pop()
            u = 0
        elif u > 100 and self.call == "Alt":
            self.integratorEps.pop()
            u = 100

        if u > self.du + self.rateLimitUp:
            u = self.du + self.rateLimitUp
        elif u < self.du - self.rateLimitDown:
            u = self.du - self.rateLimitDown


        self.du = u - self.prevU

        if self.open == True:
            u = 0

        self.us.append(u)
        return u
