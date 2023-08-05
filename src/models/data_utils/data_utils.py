import argparse
import collections
# import json
import os
import random
import sys
import time
import six
import numpy as np
import copy
import pickle

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from .parser import *

from collections import Iterable

_PAD = b"_PAD"

PAD_ID = 0
START_VOCAB_SIZE = 1
max_token_len = 5


def np_to_tensor(inp, output_type, cuda_flag, volatile_flag=False):
    if output_type == 'float':
        if volatile_flag:  # useful for eval (if no gradients need to be computed)
            with torch.no_grad():
                inp_tensor = Variable(torch.FloatTensor(inp))
        else:
            inp_tensor = Variable(torch.FloatTensor(inp))
    elif output_type == 'int':
        if volatile_flag:
            with torch.no_grad():
                inp_tensor = Variable(torch.LongTensor(inp))
        else:
            inp_tensor = Variable(torch.LongTensor(inp))
    else:
        print('undefined tensor type')
    if cuda_flag:
        inp_tensor = inp_tensor.cuda()
    return inp_tensor


def load_dataset(filename, args):  # here: samples is a list of vrp problems; one vrp problem= list of agent dicts
    with open(filename, 'rb') as f:
        samples = pickle.load(f)
    # print('Number of data samples in ' + filename + ': ', len(samples))
    return samples


class vrpDataProcessor(object):
    def __init__(self):
        self.parser = vrpParser()

    def get_batch(self, data, args, init_route_nodes, start_idx=None):

        # start idx is batch idx
        data_size = len(data)
        pool_node_id = data[0][0]["pool"]
        if start_idx is not None:
            batch_idxes = [i for i in range(start_idx, min(data_size, start_idx + args.batch_size))]
        else:
            batch_idxes = np.random.choice(len(data), args.batch_size)  # draw batch_size many numbers within [0, len(data)]
        batch_data = []
        for idx in batch_idxes:
            problem = data[idx]
            if init_route_nodes:   # in case of val they're already defined
                cur_init_route_nodes = copy.deepcopy(init_route_nodes[idx])
                dm = self.parser.parse(problem, args.heuristic, args.take_x_y, cur_init_route_nodes)  # one global VRPManager
            else:  # not defined = None, cannot do None[idx]
                dm = self.parser.parse(problem, args.heuristic, args.take_x_y, init_route_nodes)
            num_agents = len(problem)
            # add pool node
            dm.vehicle_states[num_agents] = [pool_node_id]
            batch_data.append(dm)
        return batch_data
