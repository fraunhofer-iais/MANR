import random
import numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

from ..data_utils import data_utils
import copy

#from line_profiler_pycharm import profile


class SeqLSTM(nn.Module):
    """
	LSTM to embed the input as a sequence.
    NEW class --> calc_embedding
	"""

    def __init__(self, args):
        super(SeqLSTM, self).__init__()
        self.batch_size = args.batch_size
        self.hidden_size = args.LSTM_hidden_size
        self.embedding_size = args.embedding_size
        self.num_layers = args.num_LSTM_layers
        self.dropout_rate = args.dropout_rate
        self.cuda_flag = args.cuda
        self.encoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                               batch_first=True, dropout=self.dropout_rate, bidirectional=True)

    #@profile
    def calc_embedding(self, dm_list, take_x_y):
        # call encoder for each agent individually; i.e. first call: routes of agent 0 for all vrps in batch, 2nd: routes of agent 1 for all vrps...
        # sequential information only considered within agent route; and no order on global state
        num_agents = dm_list[0].num_agents
        for agent_id in range(num_agents):
            encoder_input = []
            max_node_cnt = 0  # counts longest agent sequence in all vrps

            for dm in dm_list:
                encoder_input.append(dm.routes[agent_id][:])
                max_node_cnt = max(max_node_cnt, len(dm.routes[agent_id][:]))

            # needed to fill up sequence if shorter than longest in batch
            agent_depot_idx = dm_list[0].vehicle_states[agent_id][0]
            for i, dm in enumerate(dm_list):
                while len(encoder_input[i]) < max_node_cnt:
                    if take_x_y:
                        encoder_input[i].append([dm.nodes[agent_depot_idx].node_x, dm.nodes[agent_depot_idx].node_y,
                                                 dm.nodes[agent_depot_idx].node_x, dm.nodes[agent_depot_idx].node_y,
                                                 0.0])
                    else:
                        encoder_input[i].append([agent_depot_idx, agent_depot_idx, 0.0])

            encoder_input = np.array(encoder_input)
            encoder_input = data_utils.np_to_tensor(encoder_input, 'float', self.cuda_flag)  # ,                                                eval_mode)  # here: of shape [batchsize, length of longest route, 3]
            encoder_output, encoder_state = self.encoder(encoder_input)  # output of lstm; each embedding went from size 5 to 1024

            for dm_idx, dm in enumerate(dm_list):
                # encoder output might be longer if there are different-sized agent routes in batch; only copy those needed per node (depot was duplicated)
                num_nodes_in_cur_ag_route = len(dm.vehicle_states[agent_id])
                dm.encoder_outputs_with_cost[agent_id] = encoder_output[dm_idx][:num_nodes_in_cur_ag_route]
                # get one agent depot representation by averaging output of first and last depot
                cur_encoder_output = dm.encoder_outputs_with_cost[agent_id].clone()
                first_depot_eo = cur_encoder_output[0].clone()
                last_depot_eo = cur_encoder_output[-1].clone()
                mean_depot_eo = torch.mean(torch.stack([first_depot_eo, last_depot_eo]), axis=0)
                cur_encoder_output[0] = mean_depot_eo
                cur_encoder_output[-1] = mean_depot_eo
                dm.encoder_outputs_with_cost[agent_id] = cur_encoder_output

        return dm_list


    """
                #pytorch couldn't compute gradients due to in-place operations...
                #first_depot_eo = dm.global_encoder_outputs[agent_id][0].clone()
                #last_depot_eo = dm.global_encoder_outputs[agent_id][-1].clone()
                #mean_depot_eo = torch.mean(torch.stack([first_depot_eo, last_depot_eo]), axis=0).clone()
                #dm.global_encoder_outputs[agent_id][0] = mean_depot_eo.clone()
                #dm.global_encoder_outputs[agent_id][-1] = mean_depot_eo.clone()
              
    """
