
import numpy as np 
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.functional as F 
import torch.optim as optim


''' custom implementation of LSTM'''
class LSTMcell(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(LSTMcell, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size

		self.weight_x = nn.Linear(input_size, 4*input_size)
		self.weight_h = nn.Linear(hidden_size, 4*hidden_size)

	def forward(self, x, hc):
		h, c = hc # hc: recurrent unit
		gate_inputs = self.weight_x(x) + self.weight_h(h)
		z_gate, i_gate, f_gate, o_gate = gate_inputs.chunk(4,1)
		
		z = torch.tanh(z_gate)
		i = torch.sigmoid(i_gate)
		f = torch.sigmoid(f_gate)
		o = torch.sigmoid(o_gate)

		c_new = f*c + i*z
		h_new = o*torch.tanh(c_new)

		return (h_new, c_new)

	def init_hidden(self, batch_size):
		return (Variable(torch.zeros(batch_size, self.hidden_size)), 
				Variable(torch.zeros(batch_size, self.hidden_size)))

class LSTM(nn.Module):
	def __init__(self, batch_size, input_size, hidden_size, batch_first=True):
		super(LSTM, self).__init__()
		self.batch_size = batch_size
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.batch_first = batch_first
		self.cell = LSTMcell(self.input_size, self.hidden_size)


	def forward(self, seq):
		if self.batch_first:
			seq = seq.transpose(0,1)

		output = []
		hc = self.cell.init_hidden(self.batch_size)
		for i in range(seq.size()[0]):
			x = seq[i]
			hc = self.cell.forward(x, hc)

			if isinstance(hc, tuple):
				output.append(hc[0])
			else:
				output.append(hc)
		output = torch.cat(output, 0).view(seq.size(0), *output[0].size())

		if self.batch_first:
			output = output.transpose(0,1)
		return hc[0]


''' custom implementation of Convolutional LSTM'''
class CLSTMcell(nn.Module):
	def __init__(self, input_channel, input_height, input_width, hidden_channel, kernel_size):
		super(CLSTMcell, self).__init__()
		self.input_channel = input_channel
		self.input_height = input_height
		self.input_width = input_width
		self.hidden_channel = hidden_channel
		self.kernel_size = kernel_size # going to assume square kernel for convinience
		self.padding = kernel_size // 2

		self.conv = nn.Conv2d(self.input_channel+self.hidden_channel, 4*self.hidden_channel, self.kernel_size, padding=self.padding, stride=1)

	def forward(self, x, hc):
		h, c = hc # hc: recurrent unit
		xh = torch.cat([x,h], dim=0)
		gate_inputs = self.conv(xh.unsqueeze(0))
		gate_inputs = gate_inputs.squeeze()
		z_gate, i_gate, f_gate, o_gate = torch.split(gate_inputs, self.hidden_channel, dim=0)
		
		z = torch.tanh(z_gate)
		i = torch.sigmoid(i_gate)
		f = torch.sigmoid(f_gate)
		o = torch.sigmoid(o_gate)

		c_new = f*c + i*z
		h_new = o*torch.tanh(c_new)

		return (h_new, c_new)

	def init_hidden(self, batch_size):
		return (Variable(torch.zeros(self.hidden_channel, self.input_height, self.input_width)), 
				Variable(torch.zeros(self.hidden_channel, self.input_height, self.input_width)))


class CLSTM(nn.Module):
	def __init__(self, batch_size, input_channel, hidden_channel, height, width, batch_first=True):
		super(CLSTM, self).__init__()
		self.batch_size = batch_size
		self.input_channel = input_channel
		self.hidden_channel = hidden_channel
		self.height = height
		self.width = width
		self.batch_first = batch_first
		self.cell = CLSTMcell(self.input_channel, self.height, self.width, self.hidden_channel, 3)


	def forward(self, seq):
		# Assume the batch size (sequence length) comes first
		# if self.batch_first:
		# 	seq = seq.transpose(0,1)

		output = []
		hc = self.cell.init_hidden(self.batch_size)
		for i in range(seq.size()[0]):
			x = seq[i]
			hc = self.cell(x, hc)

			if isinstance(hc, tuple):
				output.append(hc[0])
			else:
				output.append(hc)
		output = torch.cat(output, 0).view(seq.size(0), *output[0].size())

		# if self.batch_first:
		# 	output = output.transpose(0,1)
		return hc[0]
