import simulator
import pandas as pd
import numpy as np
import shutil
import os
import matplotlib.pyplot as plt
from envs.droneGym import droneGym


pat = r'C:\Users\Stephen\PycharmProjects\QuadcopterSim\visualizer\Flight_run_ep_0.js'
times = []

def translateJStoPython(pat):
    pat2 = pat[0:-3] + 'temp.csv'
    cols = ['t', 'u1','u2','u3','u4','xdot_b', 'ydot_b', 'zdot_b', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'x', 'y','z']
    shutil.copyfile(pat,pat2)

    #find the first ], which is where the t array ends, staying the len of each section.  Then remove , and other brackets
    df2 = pd.read_table(pat2, skiprows = 1)
    df2[df2.columns[0]].str.find(']')
    qq = list(map(lambda x: x == '],', df2[df2.columns[0]]))
    lenOfRun = qq.index(True)

    df2 = df2.replace({',':''},regex = True)
    df2 = df2[pd.to_numeric(df2['['], errors = 'coerce').notnull()]


    #reshape into an array
    dfFinal = pd.DataFrame(np.reshape(df2.to_numpy(), (lenOfRun, len(cols)), order='F'), columns=cols).astype('float')

    os.remove(pat2)

    return dfFinal

def comparativePlots(x_sim, y_sim, z_sim, x_gym, y_gym, z_gym, times, title, labels):

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(times, x_sim, 'k.-', label='PID')
    ax[0].plot(times, x_gym, 'b*-', label='PPO')
    ax[0].set_ylabel(labels[0])
    ax[0].legend()
    ax[0].set_title(title)
    ax[0].grid()

    ax[1].plot(times, y_sim, 'k.-', label='PID')
    ax[1].plot(times, y_gym, 'b*-', label='PPO')
    ax[1].set_ylabel(labels[1])
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(times, z_sim, 'k.-', label='PID')
    ax[2].plot(times, z_gym, 'b*-', label='PPO')
    ax[2].set_ylabel(labels[2])
    ax[2].legend()
    ax[2].grid()


if __name__ == "__main__":
    plt.ion()
    ds = simulator.droneSim()
    ds_sim = simulator.droneSim()
    env = droneGym()
    env.prev_shaping = 0
    df = translateJStoPython(pat)
    dt = .01

    # df['u1'] = 60
    # df['u2'] = 60
    # df['u3'] = 50
    # df['u4'] = 50

    ds.stateMatrixInit()
    x_init = df.loc[0,['xdot_b', 'ydot_b', 'zdot_b', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'x', 'y','z']].to_list()
    # ds.x = list(np.zeros(len(['xdot_b', 'ydot_b', 'zdot_b', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'x', 'y', 'z'])))
    # ds.x[11] = 800
    ds.x = x_init.copy()
    x = x_init.copy()

    # rewardDf = pd.DataFrame(columns=['actionSize','angleThresh','error','oversaturation','ground','doNothing','repeatedActions','offset',
    #                                  'currError'])
    rewardDf = pd.DataFrame(columns=['actionSize','angleThresh','error','oversaturation','ground','doNothing','repeatedActions','offset',
                                     'currError'])

    for i in range(0, len(df)):
        us = np.array([df.loc[i,'u1'], df.loc[i,'u2'],df.loc[i,'u3'],0])#df.loc[i,'u4']])
        times.append(df.loc[i,'t'])
        # us[1] = us[1] - 50
        # us[2] = us[2] - 50

        usLimited = ds.checkActionStepSize(us)
        df.loc[i, 'u1lim'] = usLimited[0]
        df.loc[i, 'u2lim'] = usLimited[1]
        df.loc[i, 'u3lim'] = usLimited[2]
        df.loc[i, 'u4lim'] = usLimited[3]
        # if i > 300:
        #     df['u2'] = 40


        x_next, currU, _ = ds.numericalIntegration(x, usLimited, dt, errsIsControl=True)
        if x[11] < .05:
            x_next[2] = np.min([x[2], x_next[2], 0])
            x_next[11] = np.max([x[11], x_next[11], .05])
        ds.memory(x_next)
        env.angular_rate_sp = [0, 0, 0]
        if i > 0:
            env.prev_action = df.loc[i-1, ['u1','u2','u3']]
            env.actionnn = df.loc[i, ['u1','u2','u3']]
            env.actionActual = df.loc[i, ['u1lim','u2lim','u3lim']]
            currError = -np.sum((np.abs(np.array([x_next[3], x_next[4], x_next[5]])))**2)
            temp = env.calcReward(x_next, output_for_sim=True)
            temp.append(currError)
            rewardDf.loc[len(rewardDf)-1] = temp
            env.prev_shaping = currError
        x = x_next


    # df1 = simulator.plotStuff(times, ds.xdot_b, ds.ydot_b, ds.zdot_b, ds.p, ds.q, ds.r, ds.phi, ds.theta, ds.psi, ds.x,
    #                          ds.y, ds.z, df['u1'], df['u2'], df['u3'], df['u4'], title = ': Simulator', onlyOne = True)
    #
    df2 = simulator.plotStuff(times, df.xdot_b, df.ydot_b, df.zdot_b, df.p, df.q, df.r, df.phi, df.theta, df.psi, df.x,
                             df.y, df.z, df['u1lim'], df['u2lim'], df['u3lim'], df['u4lim'], title=": Gym")
    df3 = ds_sim.run_sim(x_init, plot=True)

    comparativePlots(df3['p'], df3['q'], df3['r'], df2['p'], df2['q'], df2['r'], times, 'Body Frame of Reference',
                     ['P (rad/sec)', 'Q (rad/sec)', 'R (rad/sec)'])
    # comparativePlots(df3['phi'], df3['theta'], df3['psi'], df2['phi'], df2['theta'], df2['psi'], times, 'Body Frame of Reference',
    #                  ['Pi (rad/sec)', 'Theta (rad/sec)', 'Psi (rad/sec)'])
    comparativePlots(df3['xdot_b'], df3['ydot_b'], df3['zdot_b'], df2['xdot_b'], df2['ydot_b'], df2['zdot_b'], times, 'Body Frame of Reference',
                     ['xdot_b (m/s)', 'ydot_b (m/s)', 'zdot_b (m/s)'])

    plt.figure()
    plt.plot(times[1:], rewardDf['actionSize'], '.', label='actionSize')
    plt.plot(times[1:], rewardDf['error'], '.', label='error')
    plt.plot(times[1:], rewardDf['offset'], '.', label='offset')
    plt.plot(times[1:], rewardDf['currError'], '.', label='currError')
    plt.plot(times[1:], rewardDf['currError'].diff(), '.', label='currError_dot')


    plt.legend()


    print('tt')