from gym.envs.registration import register

register(
    id='droneGym-v0',
    entry_point='envs.droneGym:droneGym',
    max_episode_steps=20000
)