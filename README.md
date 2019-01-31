# mountaincar
The project uses Deep-Reinforcement-Learning to solve the Mountain-Car environment from open AI

There are two files in the project currently :-

mountaincar.ipynb :-

I have used this file for training a Deep Q Network. I have used google collaboratory for this purpose. 
Simply, executing cells one after the other, like in any other simple jupyter notebook should suffice.
However, for some reason, rendering an openAI environment remains a problem on google collaboratory. 
Therefore the visualizations have been turned off.
For Deep-Learning models, the project uses keras. For Deep-reinforcement learning models, the project uses
Keras-RL, which is a very convenient, and modular library for reinforcement learning algorithms.

mountaincar_tester.py :-

This is a simple python file, created for visualizing the performance of agent on the environment. This file can 
simply be run on local machine by putting the model file in models directory (simply downloading/cloning the repo
should do). Because this fie is run on local, visualizations are turned on.
