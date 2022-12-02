import simulator
import pandas as pd
import numpy as np
import shutil
import os
import matplotlib.pyplot as plt
from envs.droneGym import droneGym
import envs
import gym
from stable_baselines.common.policies import MlpPolicy, ActorCriticPolicy, register_policy, nature_cnn, mlp_extractor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C
import tensorflow as tf

env = gym.make('droneGym-v0')

pat = r'C:\Users\Stephen\PycharmProjects\QuadcopterSim\visualizer\Flight_run_ep_0.js'
times = []


def mergeDFs(runDict, key):

    lookup2 = {}
    locZeroCrossing = []

    if type(runDict) == pd.DataFrame:
        return runDict[key], None

    for i in runDict.keys():
        lookup2[i] = np.abs(runDict[i][key])

        arr = runDict[i].loc[0:100, key].to_numpy()

        if arr[0] > 0:
            speed = ((arr[:-1] * arr[1:]) < 0).argmax() + 1
        else:
            speed = ((arr[:-1] * arr[1:]) > 0).argmin() + 1

        locZeroCrossing.append(speed)


    return pd.DataFrame(lookup2), locZeroCrossing


def comparativePlots(runDict_rl, runDict_pid, times, title, labels, ylabs, repeated=False):

    fig, ax = plt.subplots(3, 1, sharex=True)
    fig.set_figheight(9)
    fig.set_figwidth(16)
    rlSpeed = {}
    pidSpeed = {}

    for i, n in enumerate(labels):
        rl, rlSpeed[n] = mergeDFs(runDict_rl, n)
        pid, pidSpeed[n] = mergeDFs(runDict_pid, n)

        rl = rl.mean(axis=1)
        pid = pid.mean(axis=1)

        if repeated == True:
            # Used when error is injected into the sim repeatedly over the run, produces 1 jump-averaged plots
            # Assumed .5 sec for now
            qq = rl.to_numpy()
            wid = int(len(qq)/50)
            qq = qq.reshape(wid, 50)
            rl = np.mean(qq, axis=0)

            qq = pid.to_numpy()
            wid = int(len(qq)/50)
            qq = qq.reshape(wid, 50)
            pid = np.mean(qq, axis=0)

            times = np.linspace(0, .5, 50)

        ax[i].plot(times, rl, 'k.-', label="AI")
        ax[i].plot(times, pid, 'b*-', label='PID')

        ax[i].legend(loc='upper right')
        ax[i].grid()
        ax[i].set_ylabel(ylabs[i], fontsize=14)
        ax[i].tick_params(axis='x', labelsize=12)
        ax[i].tick_params(axis='y', labelsize=12)

        if i == 0:
            ax[i].set_title(title, fontsize=18)

        if i == 2:
            ax[i].set_xlabel("Time in Seconds", fontsize=14)


    return rlSpeed, pidSpeed


if __name__ ==  "__main__":
    plt.ion()
    model = PPO2(MlpPolicy, env, verbose=0, nminibatches=1, tensorboard_log="./drone_tensorboard/", gamma=.999,
                 ent_coef=0.00000444, lam=.95, learning_rate=0.0120750887045793, n_steps=512, noptepochs=20)
    model = model.load('mayAsWell.zip')
    # model = model.load('nonlinearTest.zip')
    model.env = DummyVecEnv([lambda: env])

    ds = simulator.droneSim()
    runDict_rl = {}
    runDict_pid = {}

    ds_sim = simulator.droneSim()
    env = droneGym()
    env.batch = True
    env.prev_shaping = 0
    dt = .01
    t_max = 6000
    next = False

    # df['u1'] = 60
    # df['u2'] = 60
    # df['u3'] = 50
    # df['u4'] = 50

    ds.stateMatrixInit()


    for n in range(0, 1):
        x_init = env.reset()
        ds.x = x_init.copy()
        x = x_init.copy()

        df_rl = pd.DataFrame(columns=['error_p', 'error_q', 'error_r', 'z', 'xdot_b', 'ydot_b', 'zdot_b',
                                   'u1', 'u2', 'u3', 't'])

        df_pid = ds.run_sim(env.render(start=True)[0:12], plot=False, updateSetpoints=True, t_max = t_max)
        df_pid = df_pid.loc[0:t_max-1, :]
        df_pid['error_p'] = df_pid['p'] - ds.pSetpoints[1:]
        df_pid['error_q'] = df_pid['q'] - ds.qSetpoints[1:]
        df_pid['error_r'] = df_pid['r'] - 0

        runDict_rl[n] = df_rl

        for i in range(1, t_max+1):
            # if next == True:
            #     env.x[3] = (np.random.random()) * 1.8915436 * 3
            #     env.x[4] = (np.random.random()) * 1.8915436 * 3
            #     next = False
            # if i % 50 == 0:
            #     next = True

            env.updateSetpoint([ds.pSetpoints[i], ds.qSetpoints[i], 0])
            action, _states = model.predict(x)
            x, rewards, dones, info, act = env.step(action)

            # same_action = np.array([info[0]*100, (info[1]-.5)*100, (info[2]-.5)*100])
            new_obs = np.append(x, act)
            new_obs = np.append(new_obs, i)

            df_rl.loc[i] = new_obs

        #env.render()
        runDict_rl[n] = df_rl
        runDict_pid[n] = df_pid


    # comparativePlots(runDict_rl, runDict_pid, np.linspace(0, 6, t_max), "Internal Attitude Error, 100 Run Average",
    #                  ['error_p', 'error_q', 'error_r'],
    #                  ['Error in P (rad/sec)', 'Error in Q (rad/sec)', 'Error in R (rad/sec)'])
    #
    # comparativePlots(runDict_rl, runDict_pid, np.linspace(0, 6, t_max), "Control Signals, 100 Run Average",
    #                  ['u1', 'u2', 'u3'], ['Altitude Control (u1)', 'P Control (u2)', 'Q Control (u3)'])

    fig, ax = plt.subplots(2)
    p = df_rl['error_p'] + ds.pSetpoints[1:]
    q = df_rl['error_q'] + ds.qSetpoints[1:]
    ax[0].plot(df_rl['t'] / 100, ds.pSetpoints[1:], '*', markersize=10, label="P Setpoint")
    ax[0].plot(df_rl['t'] / 100, df_pid['p'], 'o', markersize=10, label="PID P Response")
    ax[0].plot(df_rl['t'] / 100, p, '.', markersize=10, label="RL P Response")
    ax[0].legend()
    ax[0].grid()
    ax[0].set_ylabel("P (rad/sec)", fontsize=18)
    ax[0].set_title('Comparison of RL and PID for Nonlinear Input, Single Run', fontsize=20)
    ax[1].plot(df_rl['t'] / 100, ds.qSetpoints[1:], '*', markersize=10, label="Q Setpoint")
    ax[1].plot(df_rl['t'] / 100, df_pid['q'], 'o', markersize=10, label="PID Q Response")
    ax[1].plot(df_rl['t'] / 100, q, '.', markersize=10, label="RL Q Response")
    ax[1].legend()
    ax[1].grid()
    ax[1].set_ylabel("Q (rad/sec)", fontsize=18)
    plt.xlabel('Time (s)', fontsize=18)

    print('tt')

#if doing nonlinear input stuff
    # fig, ax = plt.subplots(2)
    # p = df_rl['error_p'] + ds.pSetpoints[1:]
    # q = df_rl['error_q'] + ds.qSetpoints[1:]
    # ax[0].plot(df_rl['t'] / 100, ds.pSetpoints[1:], '*', markersize=10, label="P Setpoint")
    # ax[0].plot(df_rl['t'] / 100, df_pid['p'], 'o', markersize=10, label="PID P Response")
    # ax[0].plot(df_rl['t'] / 100, p, '.', markersize=10, label="RL P Response")
    # ax[0].legend()
    # ax[0].grid()
    # ax[0].set_ylabel("P (rad/sec)", fontsize=18)
    # ax[0].set_title('Comparison of RL and PID for Nonlinear Input, Single Run', fontsize=20)
    # ax[1].plot(df_rl['t'] / 100, ds.qSetpoints[1:], '*', markersize=10, label="Q Setpoint")
    # ax[1].plot(df_rl['t'] / 100, df_pid['q'], 'o', markersize=10, label="PID Q Response")
    # ax[1].plot(df_rl['t'] / 100, q, '.', markersize=10, label="RL Q Response")
    # ax[1].legend()
    # ax[1].grid()
    # ax[1].set_ylabel("Q (rad/sec)", fontsize=18)
    # plt.xlabel('Time (s)', fontsize=18)
