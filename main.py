import numpy as np
import envs
import gym
from stable_baselines.common.policies import MlpPolicy, ActorCriticPolicy, register_policy, nature_cnn, mlp_extractor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C
from matplotlib import pyplot as plt
import tensorflow as tf


env = gym.make('droneGym-v0')

# Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
# with a nature_cnn feature extractor
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)
        self._is_graph_network = True

        with tf.compat.v1.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            # pi_latent2, vf_latent2 = mlp_extractor(self.processed_obs,net_arch = [128, dict(vf=[156, 156], pi=[128])], act_fun = tf.nn.relu, **kwargs)
            # actionSpace = tf.compat.v1.layers.dense(pi_latent2, ac_space.n, activation= 'sigmoid', name = 'pf')
            # value_fn = tf.compat.v1.layers.dense(vf_latent2, 1, name='vf')
            # vf_latent = vf_latent2

            shapesShared = [256]
            extracted_features = mlp_extractor(self.processed_obs, shapesShared, activ)
            # extracted_features = mlp_extractor(extracted_features, shapesShared, activ)

            pi_h = extracted_features[0]
            shapesp = [128, 64]
            for i, layer_size in enumerate(shapesp):
                if i == len(shapesp)-1:
                    pi_h = tf.nn.sigmoid(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
                else:
                    pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
            pi_latent = pi_h

            vf_h = extracted_features[1]
            shapesv = [64,64]
            for i, layer_size in enumerate(shapesv):
                vf_h = activ(tf.compat.v1.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
            value_fn = tf.compat.v1.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

def lrGenerator(t):
    tt = -1*(t-1) * 10000
    lr = .00025*np.cos(tt/2000*np.pi) + .00025

    return lr

env = gym.make('droneGym-v0')

model = PPO2(MlpPolicy, env, verbose = 0,n_steps = 3000, nminibatches=1,tensorboard_log="./drone_tensorboard/")
# model = PPO2(CustomPolicy, env, verbose = 0, n_steps = 3000, nminibatches=1,tensorboard_log="./drone_tensorboard/")
model = model.load('notbad____.zip')
model.full_tensorboard_log = True
model.tensorboard_log = "./drone_tensorboard/"
# model.learning_rate = lrGenerator
model.learning_rate = .00013
model.env = DummyVecEnv([lambda: env])
model.learn(total_timesteps=1000000000)



# set up plotter, hope it's  
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