import numpy as np
import operator
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch import cuda
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F


class BaseModel(nn.Module):
    """
	Base neural rewriter model. The concrete architectures for different applications are derived from it.
	"""

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.processes = args.processes
        self.batch_size = args.batch_size
        self.LSTM_hidden_size = args.LSTM_hidden_size
        self.MLP_hidden_size = args.MLP_hidden_size
        self.num_MLP_layers = args.num_MLP_layers
        self.gradient_clip = args.gradient_clip
        # if training process continued, set learning rate appropriately
        if args.lr_decay_steps and args.resume:
            self.lr = args.lr * args.lr_decay_rate ** ((args.resume - 1) // args.lr_decay_steps)
        else:
            self.lr = args.lr

        if not args.eval:
            print('Current learning rate is {}.'.format(self.lr))
        self.dropout_rate = args.dropout_rate
        # number of rewriting steps T_iter
        self.max_reduce_steps = args.max_reduce_steps
        # T_w or T_u
        #self.num_sample_rewrite_pos = args.num_sample_rewrite_pos
        # T_w or T_u
        #self.num_sample_rewrite_op = args.num_sample_rewrite_op
        # p_c end value?
        self.value_loss_coef = args.value_loss_coef
        # decay factor in cumulative reward
        self.gamma = args.gamma
        # init p_c
        #self.cont_prob = args.cont_prob
        self.cuda_flag = args.cuda
        # new: factor tuning entropy penalty
        # ##self.penalty_scale = args.penalty_scale

    def init_weights(self, param_init):
        torch.manual_seed(2021)
        for param in self.parameters():
            param.data.uniform_(-param_init, param_init)

    def lr_decay(self, lr_decay_rate):
        self.lr *= lr_decay_rate
        print('Current learning rate is {}.'.format(self.lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr


    def train(self):
        if self.gradient_clip > 0:
            clip_grad_norm_(self.parameters(), self.gradient_clip)
            # maybe differentiate when looking at plots
            # clip_grad_norm_(self.value_estimator.parameters(), self.gradient_clip) with different values ...

        all_params = list(self.parameters())
        all_params = [p for p in all_params if p.grad is not None]
        device = all_params[0].grad.device
        # new norm is clipped
        new_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in all_params]),
            2.0)

        self.optimizer.step()
        return new_norm

    # not used:
    def train_Q(self):
        Q_params = list(self.input_encoder.parameters()) + list(self.input_encoder_globalState.parameters()) + list(
            self.value_estimator.parameters())
        if self.gradient_clip > 0:
            clip_grad_norm_(Q_params, self.gradient_clip)
        self.optimizer_Q.step()

    #not used:
    def train_rule(self):
        rule_params = list(self.policy_embedding.parameters()) + list(self.policy.parameters())
        if self.gradient_clip > 0:
            clip_grad_norm_(rule_params, self.gradient_clip)
        self.optimizer_rule.step()
