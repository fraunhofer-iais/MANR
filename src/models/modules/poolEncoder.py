import random
import numpy as np
import time

import torch
import torch.nn as nn
from ..modules import mlp

from torch.autograd import Variable
import torch.nn.functional as F
import pdb

from ..data_utils import data_utils
# from line_profiler_pycharm import profile



class Attention(nn.Module):
    """
	Attention module to embed the pool.
    NEW class --> calc_embedding
	"""

    def __init__(self, args):
        super(Attention, self).__init__()
        self.batch_size = args.batch_size
        if args.take_x_y:
             input_size = 4
        else:
            input_size = 2
            #(args.LSTM_hidden_size * 2) # / self.num_heads  # 1024= 512*2 times two since bidirectional LSTMs used final output size; per head it is concatenated so output size per head shpuld be 1024/8 = 128

        self.dropout_rate = args.dropout_rate
        self.cuda_flag = args.cuda

        self.num_MLP_layers = args.num_MLP_layers
        self.MLP_hidden_size = args.MLP_hidden_size  # * 2  # = 512 (2*256)  ( from 5/3 to 512 to 16 in the end for key and query; for values from 5/3 to 2048 to 1024 in the end)
        self.LSTM_hidden_size = args.LSTM_hidden_size
        self.attention_size = args.attention_size
        self.sqrt_attention_size = int(np.sqrt(self.attention_size))

        self.gen_query = mlp.MLPModel(self.num_MLP_layers,
                                      input_size,
                                      self.MLP_hidden_size, self.attention_size, self.cuda_flag)
        self.gen_keys = mlp.MLPModel(self.num_MLP_layers, input_size, self.MLP_hidden_size,
                                     self.attention_size, self.cuda_flag)  # key and query can be lower dim; here attention size = output size of 16
        self.gen_values = mlp.MLPModel(self.num_MLP_layers, input_size, self.MLP_hidden_size * 4,  # in all exps so far: self.MLP_hidden_size * 4
                                       self.LSTM_hidden_size * 2, self.cuda_flag)  # output needs to be 1024 dim like other encoders (bidir.)

    #@profile
    def calc_pool_embedding(self, dm_list,
                            take_x_y):  # , eval_mode=False): # for own route; more information available here
        # compute representation for each node in the pool by doing self attention (with all other nodes in the pool)

        for vrp in dm_list:
        # compute attention outputs (needs to be done separately per vrp since queries only valid per vrp)
            query_key_value_inputs = []   # for all other nodes (IF I do want to make key_value input with fixed output, i.e. for EACH node in the graph, then here extra input. Currently: only consider the relevant nodes as input
            for idx, node_id_pool in enumerate(vrp.vehicle_states[-1]):    # pool node itself not duplicated so no need to exclude last node
                if idx == 0:   # then pool node itself
                    if take_x_y:
                        node_emb = [vrp.get_node(node_id_pool).node_x, vrp.get_node(node_id_pool).node_y,
                                    vrp.get_node(node_id_pool).node_x, vrp.get_node(node_id_pool).node_y]
                    else:
                        node_emb = [node_id_pool, node_id_pool]

                    pool_node_id = node_id_pool   # id of true pool node (always the first one)
                else:  # predecessor always true pool node
                    if take_x_y:
                        node_emb = [vrp.get_node(node_id_pool).node_x, vrp.get_node(node_id_pool).node_y,
                                    vrp.get_node(pool_node_id).node_x, vrp.get_node(pool_node_id).node_y]
                    else:
                        node_emb = [node_id_pool, pool_node_id]

                query_key_value_inputs.append(node_emb)

            # now for each node in the pool of cur vrp: input given (sometimes only pool node itself there)
            query_key_value_inputs = np.array(query_key_value_inputs)
            query_key_value_inputs = data_utils.np_to_tensor(query_key_value_inputs, 'float', self.cuda_flag)

            queries = self.gen_query(query_key_value_inputs)  # 16 dim per node; query for node i is a row
            keys = self.gen_keys(query_key_value_inputs)  # 16 dim per node; key for node i is a row
            values = self.gen_values(query_key_value_inputs)  # 1024 dim per node; query for node i is a row
            keys_T = torch.transpose(keys, 0, 1)  # now key for node i is a column
            compatibilities = torch.matmul(queries,
                                           keys_T) / self.sqrt_attention_size  # row i: compatibility of all keys with query i
            att_weights = nn.Softmax(dim=1)(compatibilities)
            fin_node_representations = torch.matmul(att_weights,values)  # aggregate values from other nodes (and nodes itself) based on compatibility, number nodes * 1024

            # write representations of nodes in pool to global encoder outputs
            vrp.encoder_outputs_with_cost[-1] = fin_node_representations
            # done for one vrp in batch

        # done for all vrps in batch
        return dm_list
