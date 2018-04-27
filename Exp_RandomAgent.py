
import numpy as np 
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

from GridWorld import GridWorld
from Agents import RandomAgent
from TOMnet import CharacterNet, TOMNet

''' initializing the gridworld and the agent'''
Height = 11
Width = 11
Num_Goals = 4

sample_world = GridWorld(Height, Width, Num_Goals)
sample_world.generateWalls()
sample_world.generateGoals()

random_agent = RandomAgent(sample_world)
sample_world.addAgent(random_agent)
# sample_world.showImage()

''' initializing ToM network '''
batch_size = 7
input_channel = 11 # 5+K -> 1 wall, 4 goals, 1 agent, 5 action
height = 11
width = 11
hidden_channel_cn = 8
hidden_channel_tn = 32
output_size = 2
input_size = (batch_size, input_channel, height, width)

charNet = CharacterNet(input_size, hidden_channel_cn, output_size)
tomNet = TOMNet(input_size, hidden_channel_tn)

''' Running'''
for i in range(100):
	agent_y, agent_x = sample_world.agent.get_position()
	# print(agent_y, age)
	state = sample_world.getState()
	state = torch.Tensor(state).unsqueeze(0)
	e_char = charNet(state)
	sample_world.step()







