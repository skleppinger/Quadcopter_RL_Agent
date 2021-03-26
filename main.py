import numpy as np
import envs
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from matplotlib import pyplot as plt

env = gym.make('droneGym-v0')

def lrGenerator(t):
    tt = -1*(t-1) * 10000
    lr = .0025*np.cos(tt/2000*np.pi) + .0025

    return lr

env = gym.make('droneGym-v0')

# model = PPO2(MlpPolicy, env, verbose=0, learning_rate= .00005, n_steps = 10000, nminibatches=1)
# model = model.load('testOfVariedAngles.zip')
model = PPO2(MlpPolicy, env, verbose = 0,n_steps = 3000, nminibatches=1,tensorboard_log="./drone_tensorboard/")
model.learning_rate = lrGenerator
model.env = DummyVecEnv([lambda: env])
model.learn(total_timesteps=10000000)



# set up plotter, hope it's dynamic
plt.ion()
fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
h1, = ax1.plot([], [], 'b.', label='Reward')
h2, = ax2.plot([], [], 'y.', label='Loss')
ax1.legend()
ax2.legend()
plt.show()

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

    h1.set_xdata(np.append(h1.get_xdata(), i))
    h1.set_ydata(np.append(h1.get_ydata(), rewards))
    # h2.set_xdata(np.append(h2.get_xdata(), i))
    # h2.set_ydata(np.append(h2.get_ydata(), running_loss))
    ax1.relim()
    ax1.autoscale_view()
    # ax2.relim()
    # ax2.autoscale_view()
    # ax.plot(range(0,i+1), running_reward,'b.', label = 'Reward')
    # ax.plot(range(0,i+1), running_loss,'y.', label = 'Loss')
    # ax.legend()
    plt.draw()
    fig.canvas.flush_events()

env.render()
env.close()


print('tt')