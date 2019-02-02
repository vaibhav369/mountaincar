import os
import numpy as np
import gym

from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.models import Sequential

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy, GreedyQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'MountainCar-v0'
env = gym.make(ENV_NAME)
env = gym.wrappers.Monitor(env, './video/',video_callable=lambda episode_id: True,force = True)

obs_space_shape = env.observation_space.shape
nb_actions = env.action_space.n

model = Sequential()
model.add( Flatten(input_shape=(1, )+obs_space_shape) )
model.add( Dense(20) )
model.add( Activation('relu') )
model.add( Dense(20) )
model.add( Activation('relu') )
model.add( Dense(20) )
model.add( Activation('relu') )
model.add( Dense(nb_actions) )

policy = GreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)

dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=nb_actions,
               nb_steps_warmup=10, target_model_update=1e-2)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

model_dirname = 'models'
filename = 'mountaincar_greedy_policy.h5'

dqn.load_weights(os.path.join(model_dirname, filename))

dqn.test(env, nb_episodes=5, visualize=True)
