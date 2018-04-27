#!/usr/bin/env python3

import numpy as np 
import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

from TOMnet import CharacterNet, TOMNet

batch_size = 7
input_channel = 5 # 5+K
height = 11
width = 11
hidden_channel_cn = 8
hidden_channel_tn = 32
output_size = 2
input_size = (batch_size, input_channel, height, width)

charNet = CharacterNet(input_size, hidden_channel_cn, output_size)
tomNet = TOMNet(input_size, hidden_channel_tn)

test_input = Variable(torch.randn(batch_size,input_channel,height,width))
output = charNet(test_input)
output2 = tomNet(test_input)
print(output.size(), output2.size())