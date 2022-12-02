# Full Attitude Control of a Quadcopter by Reinforcement Learning Agent

The focus of this Master's Thesis work was to develop an environment and a RL agent capable of learning the mechanisms of attitude control of an under-actuated quadcopter entirely through Reinforcement learning techniques.  The paper describing the underlying techniques can be found here:
https://www.researchgate.net/publication/363736123_Comparison_of_Traditional_Optimal_Control_Methodologies_to_Reinforcement_Learning_Agents_in_Quadcopter_Attitude_Control_Applications

This project is focused on creating a computationally simplified environment describing quadcopter motion as well as a novel implementation of the PPO2 RL training method that was able to solve this motion efficiently.

A direct port to a HIL test environment would likely not work due to some simplifications in the fluid dynamics modelling done around the propellers, but that could be added for future tests.

The framework used for this project closely adheres to the [OpenAI Gym](https://github.com/openai/gym) style and is built off of [stable_baselines](https://github.com/hill-a/stable-baselines) model implementations. 
Due to this software versions are very specific - module versions are defined in the requirements.txt file and python needs to.  Running in a virtual environment is highly recommended.

##How to Run
Once this repo is cloned and a virtual environment is set up per the requirements file, you can then generate a new model by running the main.py file.
You can visualize this training by opening up a tensorboard server pointing at the file being generated in the drone_tensorboard folder
This starts off by default with a pre-trained model called aboutAsGood that's currently save in the repository - that model can be used as a suboptimal initial state.


##Visualization
In order to visualize the flight of a single episode, you must insert calls to the env.render() function somewhere in the training (currently it's called once at the end of the training run).
That generates a data file that can then be shown in browser by opening the sim_vis.html file in the visualizier folder (the datafile will have to be renamed to test.json for now if you don't want to update the code)
