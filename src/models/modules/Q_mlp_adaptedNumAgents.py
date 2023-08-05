import random
import numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Q_MLPModel_adapt(nn.Module):
	"""
	Multi-layer perceptron module.
	"""

	def __init__(self, input_size, hidden_size_last, output_size, cuda_flag, dropout_rate=0.0, activation=None):
		super(Q_MLPModel_adapt, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size_last
		self.output_size = output_size
		self.dropout_rate = dropout_rate
		self.cuda_flag = cuda_flag
		self.dropout = nn.Dropout(p=self.dropout_rate)

		cur_hidden_size = int(self.input_size / 2)
		self.model = nn.Sequential(
			nn.Linear(self.input_size, cur_hidden_size),  # always half number of neurons
			nn.ReLU())

		while True:
			if cur_hidden_size > 200:
				# add layer with halving the number of neurons
				next_hidden_size = int(cur_hidden_size / 2)
				self.model = nn.Sequential(
					self.model,
					nn.Linear(cur_hidden_size, next_hidden_size),
					nn.ReLU())
				cur_hidden_size = next_hidden_size

			else:
				self.model = nn.Sequential(
					self.model,
					nn.Linear(cur_hidden_size, self.output_size))

				if activation is not None:
					self.model = nn.Sequential(
						self.model,
						activation)
				break


	def forward(self, inputs):
		return self.model(inputs)