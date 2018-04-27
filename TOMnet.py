#!/usr/bin/env python3

import numpy as np 
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from CLSTM import CLSTM

''' Custom implementation of the networks in "Machine Theory of Mind" by
	Rabinowitz et al.
'''
class CharacterNet(nn.Module):
	def __init__(self, input_size, h_channel, output_size):
		super(CharacterNet, self).__init__()
		self.in_channel = input_size[1] # K+5 [N, K+5, 11, 11]
		self.out_channel = h_channel # 8
		self.input_size = input_size 
		self.output_size = output_size
		self.kernel_size = 3

		self.conv = nn.Conv2d(self.in_channel, self.out_channel, 
							  self.kernel_size, padding=1, stride=1)
		# self.lstm = nn.LSTM(self.out_channel, self.out_channel, 1)
		self.lstm = CLSTM(self.input_size[0], self.out_channel, 
						  self.out_channel, self.input_size[2], 
						  self.input_size[3])
		self.avgpool = nn.AvgPool2d(self.kernel_size)
		self.linear = nn.Linear(self.out_channel*3*3, output_size)

	def forward(self, x):
		x = F.relu(self.conv(x))
		x = self.lstm(x).unsqueeze(0)
		x = self.avgpool(x)
		x = x.view(-1, self.out_channel*3*3)
		x = self.linear(x)
		return x

	# def init_hidden(self, batch_size):
	# 	return (Variable(torch.zeros(self.out_channel,self.input_size[2], self.input_size[3])),
	# 			Variable(torch.zeros(self.out_channel,self.input_size[2], self.input_size[3])))

class MentalNet(nn.Module):
	def __init__(self, input_size, h_channel, output_size):
		super(MentalNet, self).__init__()
		self.in_Channel = input_size[1]
		self.out_channel = h_channel
		self.input_size = input_size
		self.output_size = output_size
		self.kernel_size = 3

	def forward(self, x):
		return x

class TOMNet(nn.Module):
	def __init__(self, input_size, h_channel):
		super(TOMNet, self).__init__()
		self.input_size = input_size # [N, K, 11, 11]
		self.in_channel = input_size[1] # K
		self.out_channel = h_channel # 32
		self.output_size = 5 # num action space
		self.kernel_size = 3
		self.padding = self.kernel_size//2

		self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, 
							   self.kernel_size, padding=self.padding)
		self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, 
							   self.kernel_size, padding=self.padding)
		self.relu = nn.ReLU()
		self.avgpool = nn.AvgPool2d(self.kernel_size)
		self.linear = nn.Linear(self.out_channel*3*3, self.output_size)


	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(self.conv2(x))
		x = self.avgpool(x)
		print(x.size())
		x = x.view(-1, self.out_channel*3*3)
		x = F.softmax(self.linear(x), dim=1)

		return x











