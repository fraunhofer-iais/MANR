import numpy as np
import operator
import random
import time
from multiprocessing.pool import ThreadPool

import torch
import torch.nn as nn
import torch.optim as optim
from torch import cuda
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .data_utils import data_utils
from .modules import mlp, state_encoder,  poolEncoder, Q_mlp_adaptedNumAgents, state_encoder_no_cost  #Q_mlp  # state_encoder_oneSample
from .rewriter import vrpRewriter
from .BaseModel import BaseModel
#from torchviz import make_dot
import hiddenlayer as hl

#from line_profiler_pycharm import profile

from collections import Counter

eps = 1e-3
log_eps = np.log(eps)


class vrpModel(BaseModel):
    """
    Model architecture for vehicle routing.
    """

    def __init__(self, args, num_agents, config=None):
        super(vrpModel, self).__init__(args)
        self.embedding_size = args.embedding_size   # 5
        self.attention_size = args.attention_size   # 16
        self.sqrt_attention_size = int(np.sqrt(self.attention_size))
        self.state_encoder = state_encoder.SeqLSTM(args)
        self.state_encoder_no_cost = state_encoder_no_cost.SeqLSTM(args)
        self.pool_encoder = poolEncoder.Attention(args)

        if args.rew_all_possible_embeddings:
            self.policy_embedding = mlp.MLPModel(self.num_MLP_layers,
                                                 self.LSTM_hidden_size * 4 + self.embedding_size * 3,
                                                 self.MLP_hidden_size, self.attention_size, self.cuda_flag)
        else:
            self.policy_embedding = mlp.MLPModel(self.num_MLP_layers,
                                                 self.LSTM_hidden_size * 4 + self.embedding_size * 2,
                                                 self.MLP_hidden_size, self.attention_size, self.cuda_flag)  # with depot * 6
        self.policy = mlp.MLPModel(self.num_MLP_layers,
                                   self.LSTM_hidden_size * 4, self.MLP_hidden_size,
                                   self.attention_size, self.cuda_flag)  # with depot: *4



        self.value_estimator = Q_mlp_adaptedNumAgents.Q_MLPModel_adapt(self.LSTM_hidden_size * 4 * num_agents,
                                                self.MLP_hidden_size,
                                                1,
                                                self.cuda_flag)
        #mlp.MLPModel(self.num_MLP_layers + 1, self.LSTM_hidden_size * 4 * num_agents,
                                            #self.MLP_hidden_size, 1,
                                            #self.dropout_rate, self.cuda_flag)
        self.rewriter = vrpRewriter()

        if args.optimizer == 'adam':
            """
            # take separate lr for context state embedder; need to define different param groups therefore
            context_state_encoder_params = []
            all_other_params = []
            for name, param in self.named_parameters():
                if "state_encoder_no_cost" in name:
                    context_state_encoder_params.append(param)
                else:
                    all_other_params.append(param)

            self.optimizer = optim.Adam([{'params':context_state_encoder_params}, {'params':all_other_params}], lr=self.lr)
            # smaller learning for context encoder [0] / all but context embedder [1]
            # self.optimizer.param_groups[1]['lr'] = self.lr * 0.5
            """
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

            # Q_params = list(self.input_encoder.parameters())+list(self.input_encoder_globalState.parameters())+list(self.value_estimator.parameters())
            ##for param in Q_params:
            ##    print(param.requires_grad)
            # rule_params = list(self.policy_embedding.parameters()) + list(self.policy.parameters())
            ##for param in rule_params:
            ##    print(param.requires_grad)
            # self.optimizer_Q = optim.Adam(Q_params, lr=self.lr)
            # self.optimizer_rule = optim.Adam(rule_params, lr=self.lr)
        elif args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        elif args.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
        else:
            raise ValueError('optimizer undefined: ', args.optimizer)

        # removed ones from orig
        # self.input_format = args.input_format
        # self.reward_thres = -0.01

    #@profile
    def compute_Q(self, dm, global_action_list):  # input = list of x global actions
        # Q(self, dm, global_action_list)
        # for each node in the vrp's vehicle state (apart from start and end depot)
        region_states_list = []
        rule_states_list = []
        for global_action in global_action_list:  # global action 5 entries per agent, one local agent action entry = two tuples: first (agentid, region node id in agent's vehicle state), second (agentid, rule node id in agent's vehicle state)
            region_states_one_global_action = []
            rule_states_one_global_action = []
            for agent_id, local_action in enumerate(global_action):
                if local_action:   # can be empty if pool empty and one agent isn't visiting any nodes, i.e. cannot choose a region node
                    region_spec = local_action[0]  # (agent id, id in vehicle state)
                    rule_spec = local_action[1]  # (agent id, id in vehicle state)
                    region_states_one_global_action.append(
                        dm.encoder_outputs_with_cost[region_spec[0]][region_spec[1]].unsqueeze(
                            0))
                    rule_states_one_global_action.append(
                        dm.encoder_outputs_with_cost[rule_spec[0]][rule_spec[1]].unsqueeze(0))

                # this is ok, since if one agent is not acting, we need something here global (below it means [depot,depot]
                else:  # if local action is empty, then agent is visiting no nodes. Take zeros embedding.
                    # take embedding of depot and multiply  with zero
                    region_states_one_global_action.append(0 * dm.encoder_outputs_with_cost[agent_id][0].unsqueeze(0))
                    rule_states_one_global_action.append(0 * dm.encoder_outputs_with_cost[agent_id][0].unsqueeze(0))

            #if region_states_one_global_action:  # if some agent is acting (o/w, i.e. if empty then forget about it); this contains a value either way=!

            region_states_one_global_action = torch.cat(region_states_one_global_action,
                                                        0)  # make one tensor out of the several 1024-dim region node EOs (matrix trensor) #maybe later flatten into one tensor!!
            region_states_one_global_action = torch.flatten(
                region_states_one_global_action)  # one vector for global action (regions part; other parts are later on concatenated)   --> dim = 3*1024=3072
            region_states_list.append(region_states_one_global_action)

            rule_states_one_global_action = torch.cat(rule_states_one_global_action, 0)
            rule_states_one_global_action = torch.flatten(rule_states_one_global_action)
            rule_states_list.append(rule_states_one_global_action)

        # list of tensors; I need tensor matrix --> take torch.stack (now they all have dim 5*1024 = 5120 (LSTM hidden = 512)
        region_states_list = torch.stack(region_states_list)
        rule_states_list = torch.stack(rule_states_list)
        pred_rewards_list = self.value_estimator(torch.cat([region_states_list, rule_states_list],
                                                           dim=1))  # input: x global actions; reward per global action. One global action incorporates region and rule EOs #depot currently out; info can be deleted above...
        # pred_rewards_list *= 10  # added again; in orig paper relevant since exp(10*Q); I only use Q in sense argmax Q = argmax 10*Q; but scale fits better with observed reward...; but should be learned so removed

        ###### make plot of architecture
        #make_dot(pred_rewards_list).render("attached", format="png")
        #import os
        #os.environ["PATH"] += os.pathsep + "/home/IAIS/npaul/miniconda3/envs/research/lib/python3.8/site-packages/graphviz/backend/"

        #im = hl.build_graph(self.value_estimator, torch.cat([region_states_list, rule_states_list], dim=1))#.build_dot()
        #im.save(path="/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/plots_InformedML_paper/architectures/Q", format="jpg")
        ######


        return pred_rewards_list

    # given: one vrp of one agent; rewrite_pos
    #@profile
    def rule_policy(self, batch_idx, dm, own_agent_id, rewrite_pos, allow_no_change_in_rule, rew_all_possible_embeddings, take_x_y, rule_temperature_scaling, rule_temperature,
                    eval_flag):  # local!
        # rewrite_pos is now a tuple (agent_id, rewrite_pos)

        # region node (can be local or a node from the pool)
        reg_node_idx = dm.vehicle_states[rewrite_pos[0]][rewrite_pos[1]]  # rewrite pos[0] either own agent id or id of pool
        reg_node = dm.get_node(reg_node_idx)

        # encoder output for region node
        cur_state = dm.encoder_outputs_with_cost[rewrite_pos[0]][rewrite_pos[1]].unsqueeze(0)  # [[val1,..,]] (1024 vals)

        cur_states_reg = []  # for region node encoder output (duplicated), length as above: always given
        cur_states_rule = []  # for rules node encoder outputs: always given
        new_embeddings_region_suc = []  # new embeddings of successor of region node: only given if region not from pool
        new_embeddings_region = []  # new embeddings of region nodes: only given if rule from own route (not if from pool or from some other agent)
        new_embeddings_rule_suc = []  # new embeddings of successor of rule node: only given if rule from own route (not if from pool or from some other agent)

        # compute probability for each node in vrp (fixed output definition of policy: nodeId 0, 1, 2, ...)
        feasible_node_idxes = dm.get_feasible_node_idxes(rewrite_pos, allow_no_change_in_rule, own_agent_id)
        feasible_rule_nodes = [dm.nodes[idx] for idx in feasible_node_idxes]

        for rule_node in feasible_rule_nodes:
            # find node in current tour to get its encoder_output  = (agent_id, index in route)
            agent_id,id_in_vehicle_state = dm.get_route_dIdx(rule_node.nodeId)

            cur_states_reg.append(cur_state.clone())  # region node encoder output

            if rule_node.nodeId == dm.num_nodes - 1:
                # rule node = pool node; get pool embedding by taking embedding of pool node
                cur_rule_state = dm.encoder_outputs_with_cost[dm.num_agents][0].unsqueeze(0)   # pool node embedding always first entry
            else:
                # current rule encoder output
                cur_rule_state = dm.encoder_outputs_with_cost[agent_id][id_in_vehicle_state].unsqueeze(0)

            cur_states_rule.append(cur_rule_state)

            # get new embeddings of region node, region suc and rule suc
            if take_x_y:
                # new embedding of region node +  rule successor (both depend on the rule node!)
                if agent_id == dm.num_agents:
                    # if rule node the true pool node = add region to pool --> then 0 cost

                    new_embedding_reg = [reg_node.node_x, reg_node.node_y, rule_node.node_x, rule_node.node_y, 0]
                    new_embedding_rule_suc = [0] * 5   # successor of pool node not meaningful

                    if rewrite_pos[0] == own_agent_id:   # if region node from own route (i.e. not from pool)
                        # predecessor of region
                        pre_reg_node_idx = dm.vehicle_states[rewrite_pos[0]][rewrite_pos[1] - 1]
                        pre_reg_node = dm.get_node(pre_reg_node_idx)
                        # successor of region
                        reg_suc_node_idx = dm.vehicle_states[rewrite_pos[0]][rewrite_pos[1] + 1]
                        reg_suc_node = dm.get_node(reg_suc_node_idx)
                        new_embedding_region_suc = [reg_suc_node.node_x, reg_suc_node.node_y, pre_reg_node.node_x,
                                                    pre_reg_node.node_y, dm.get_cost(reg_suc_node, pre_reg_node, own_agent_id)]
                    else:   # region node also from pool
                        new_embedding_region_suc = [0] * 5

                else:  # if rule node not from pool
                    # then rule node from own route: meaningful and cost information given
                    new_embedding_reg = [reg_node.node_x, reg_node.node_y, rule_node.node_x, rule_node.node_y, dm.get_cost(reg_node, rule_node, own_agent_id)]

                    rule_suc_node_idx = dm.vehicle_states[agent_id][id_in_vehicle_state + 1]  # successor of rule node
                    rule_suc_node = dm.get_node(rule_suc_node_idx)

                    new_embedding_rule_suc = [rule_suc_node.node_x, rule_suc_node.node_y, reg_node.node_x,reg_node.node_y,dm.get_cost(rule_suc_node, reg_node, own_agent_id)]

                    if rewrite_pos[0] == own_agent_id:   # if region node from own route (i.e. not from pool)
                        # predecessor of region
                        pre_reg_node_idx = dm.vehicle_states[rewrite_pos[0]][rewrite_pos[1] - 1]
                        pre_reg_node = dm.get_node(pre_reg_node_idx)
                        # successor of region
                        reg_suc_node_idx = dm.vehicle_states[rewrite_pos[0]][rewrite_pos[1] + 1]
                        reg_suc_node = dm.get_node(reg_suc_node_idx)
                        new_embedding_region_suc = [reg_suc_node.node_x, reg_suc_node.node_y, pre_reg_node.node_x,
                                                    pre_reg_node.node_y, dm.get_cost(reg_suc_node, pre_reg_node, own_agent_id)]
                    else:
                        new_embedding_region_suc = [0] * 5


            else: # CASE NOT PROPERLY CODED (for node ids and not x,y)
                raise NameError("case not properly coded")

            new_embeddings_region.append(new_embedding_reg)
            new_embeddings_rule_suc.append(new_embedding_rule_suc)
            new_embeddings_region_suc.append(new_embedding_region_suc)

        # now: given new embeddings for region, rule suc and region suc for each rule node
        cur_states_reg = torch.cat(cur_states_reg, 0)  # region node encoder output (duplicated)
        cur_states_rule = torch.cat(cur_states_rule, 0)  # rule node encoder output (per rule node)
        # always needed
        new_embeddings_region_suc = data_utils.np_to_tensor(new_embeddings_region_suc, 'float', self.cuda_flag)
        new_embeddings_rule_suc = data_utils.np_to_tensor(new_embeddings_rule_suc, 'float', self.cuda_flag)

        if rew_all_possible_embeddings:   # take all embeddings
            new_embeddings_region = data_utils.np_to_tensor(new_embeddings_region, 'float', self.cuda_flag)
            policy_inputs = torch.cat([cur_states_reg, cur_states_rule, new_embeddings_region, new_embeddings_region_suc, new_embeddings_rule_suc],1)
        else:  # only take always observable embeddings (i.e. exclude region node embedding which might be zero if region node outside of route)
            policy_inputs = torch.cat([cur_states_reg, cur_states_rule, new_embeddings_region_suc,new_embeddings_rule_suc],1)

        rule_cand_keys = self.policy_embedding(policy_inputs)   # in original code called ctx_embeddings # in: for each rule node a vector of size 3075 (3*1024 + 3), out: 16 dim (size of attention_size) per rule node

        # extend query: not only region node info, but aggregated x,y-info of all other nodes in the problem which are not present in the own local route --> context info
        context_nodes_encoder_outputs_no_cost = []
        for a_id in range(dm.num_agents):
            if a_id != own_agent_id:  # then all nodes from this agent are context nodes
                # already compute mean of embeddings per agent route
                context_nodes_encoder_outputs_no_cost.append(torch.mean(dm.encoder_outputs_no_cost[a_id][:-1], axis=0))  # exclude doubled last depot
        # final averaged context state
        context_state = torch.stack(context_nodes_encoder_outputs_no_cost).mean(axis=0)
        query_input = torch.cat([cur_state.squeeze(),context_state])  # have to make it to (,1) array again
        region_query = self.policy(query_input[None,:])  # need  the shape (1,x) instead of simply x;in original code called key
        # compatibility of region per rule node (unnormalized log-probs = logits)

        compatibility_feas = torch.matmul(region_query, torch.transpose(rule_cand_keys, 0, 1)) / self.sqrt_attention_size  # attention: how good does the chosen region node fit to a specific rule node? (one value per node)

        if rule_temperature_scaling:
            compatibility_feas = compatibility_feas * rule_temperature


        rule_logprobs_feas = nn.LogSoftmax(dim=1)(compatibility_feas)
        rule_probs_feas = nn.Softmax(dim=1)(compatibility_feas)  # softmax on logits of rule nodes

        compatibility_feas = compatibility_feas.squeeze(0)
        rule_logprobs_feas = rule_logprobs_feas.squeeze(0)
        rule_probs_feas = rule_probs_feas.squeeze(0)

        # create probabilities of size (len(dm.nodes)) and transfer probabilities of feasible nodes from above
        rule_probs = torch.zeros(len(dm.nodes)).cuda()
        rule_logprobs = rule_probs.log()

        for idx_in_probs, feasible_node_id in enumerate(feasible_node_idxes): #feasible_node_idxes  = rule node Ids
            rule_probs[feasible_node_id] = rule_probs_feas[idx_in_probs]
            rule_logprobs[feasible_node_id] = rule_logprobs_feas[idx_in_probs]

        if eval_flag:
            _, rule_cand_nodeIds = torch.sort(rule_probs, descending=True)  # sort rule nodes highest prob first, get indices
            rule_cand_nodeIds = rule_cand_nodeIds.data.cpu().numpy()
            chosen_rule_nodeId = rule_cand_nodeIds[0]  # take the one with highest prob
            rule_picking_entropy = None
        else:  # during training
            rule_cand_dist = Categorical(rule_probs)
            rule_picking_entropy = float(rule_cand_dist.entropy().data.cpu().numpy())
            # NEW: SAMPLE JUST ONCE (enforcing more exploration) (previously:sample from rule #rule nodes times and take node appearing most often)
            chosen_rule_nodeId = rule_cand_dist.sample().item()

        rule_agent_id, rule_id_in_route = dm.get_route_dIdx(chosen_rule_nodeId)
        # ac_probs in nodeId order
        return [rule_agent_id, rule_id_in_route], rule_probs, rule_logprobs, rule_picking_entropy

    #@profile
    def forward(self, batch_idx, dm_list, eval_flag, agent_id=None,
                candidate_flag=False, batch_cand_global_actions=None, epsilon_greedy=None, epsilon=None, softmax_Q=None,
                dm_ids_pool_filled=None, dm_pool_node_assignments=None, allow_no_change_in_rule=None, rew_all_possible_embeddings=None, take_x_y=None, Z=None, rule_temperature_scaling=None, rule_temperature=None):  # first line: candidate flag False relevant params; 2nd line: candidate flag True relevant parameters

        # if pool filled then region already given (in dm_pool_node_assignments), o/w sample from local route
        batch_size = len(dm_list)
        if candidate_flag == True:  # create Z candidate local actions per vrp in batch
            batch_cand_l_actions = []
            batch_cand_l_actions_rule_probs = []
            batch_cand_l_actions_rule_logprobs = []
            batch_cand_l_actions_rule_entropies = []

            for dm_idx in range(batch_size):  # for each vrp in the batch
                dm = dm_list[dm_idx]

                Z_l_actions_per_vrp = []
                Z_l_actions_per_vrp_rule_probs = []
                Z_l_actions_per_vrp_rule_logprobs = []
                Z_l_actions_per_vrp_rule_entropies = []

                # check if candidate Z region nodes from pool (already there) or if candidate Z region nodes shall be sampled
                if dm_idx in dm_ids_pool_filled:   # if pool filled, i.e. if it needs to get emptied
                    j = dm_ids_pool_filled.index(dm_idx)
                    # rewrite_pos is now a tuple: (id of agent, node id); if pool filled: id of agent = "pool agent" id  = num_agents
                    # in node assignment = nodeId (dm_pool_node_assignments[j][k][dm.agent_id])) written; here we need position of node in dummy pool route
                    rewrite_pos_z_list = []
                    for k in range(Z):   # for each candidate assigned pool node
                        assigned_pool_node = dm_pool_node_assignments[j][k][agent_id]

                        if assigned_pool_node or assigned_pool_node == 0:   # can be empty if there were more agents than nodes in pool; otherwise always node id, 0 id extra since bool there false
                            rewrite_pos_z_list.append(list(dm.get_route_dIdx(assigned_pool_node)))  # returns tuple (pool "agent" idx, id in pool); converted to list
                        else:
                            rewrite_pos_z_list.append(0)
                else:  # if pool empty, randomly draw regions from own route
                # !!!currently sampling z region nodes  TO DO: importance sampling like MCTS (counting number of times region node chosen etc.)
                    if len(dm.vehicle_states[agent_id]) > 2:  # if agent visits at least one node (2 there sine of start and end depot)
                        rewrite_pos_z_list = np.random.randint(len(dm.vehicle_states[agent_id]) - 2,
                                                               size=Z) + 1
                        # rewrite_pos shall be a tuple containing id of agent first
                        rewrite_pos_z_list = list(map(list, zip([agent_id]*len(rewrite_pos_z_list), rewrite_pos_z_list)))
                    else:  # agent visits no node; no region node can be chosen
                        rewrite_pos_z_list = [0] * Z  # dummy region nodes of zero

                for rewrite_pos_z in rewrite_pos_z_list:
                    if rewrite_pos_z == 0:   # if pool empty but agent is not visiting any node or if pool filled but more agents there than nodes in pool
                        Z_l_actions_per_vrp.append([])
                        Z_l_actions_per_vrp_rule_probs.append(np.nan)
                        Z_l_actions_per_vrp_rule_logprobs.append(np.nan)
                        Z_l_actions_per_vrp_rule_entropies.append(np.nan)
                    else:  # if region node there, get a corresponding rule node
                        chosen_rule_route_node_tuple, rule_probs, rule_logprobs, rule_picking_entropy = self.rule_policy(batch_idx, dm, agent_id, rewrite_pos_z, allow_no_change_in_rule, rew_all_possible_embeddings, take_x_y, rule_temperature_scaling, rule_temperature,
                            eval_flag)
                        Z_l_actions_per_vrp.append((rewrite_pos_z, chosen_rule_route_node_tuple))   # local action
                        Z_l_actions_per_vrp_rule_probs.append(rule_probs)
                        Z_l_actions_per_vrp_rule_logprobs.append(rule_logprobs)
                        Z_l_actions_per_vrp_rule_entropies.append(rule_picking_entropy)

                batch_cand_l_actions.append(Z_l_actions_per_vrp)
                batch_cand_l_actions_rule_probs.append(Z_l_actions_per_vrp_rule_probs)
                batch_cand_l_actions_rule_logprobs.append(Z_l_actions_per_vrp_rule_logprobs)
                batch_cand_l_actions_rule_entropies.append(Z_l_actions_per_vrp_rule_entropies)

            return batch_cand_l_actions, batch_cand_l_actions_rule_entropies, batch_cand_l_actions_rule_probs, batch_cand_l_actions_rule_logprobs

        else:
            # given: batch_cand_global_actions (Z per vrp)
            # compute Q scores and with prob 1-eps vote for global action with highest score (o/w vote for random)

            batch_global_candidates_Qs = []
            batch_random_vote = []
            batch_vote_action_idx = []
            batch_pred_reward_list = []

            for dm_idx in range(batch_size): # for each vrp in the batch
                dm = dm_list[dm_idx]
                Z_cand_global_actions = batch_cand_global_actions[dm_idx]
                pred_reward_list = self.compute_Q(dm, Z_cand_global_actions)
                batch_global_candidates_Qs.append(pred_reward_list)

                if epsilon_greedy:
                    if eval_flag:  # for validation always vote for best action
                        action_idx = torch.argmax(pred_reward_list).tolist()
                    else:  # during training
                        rn = np.random.random()
                        if rn < epsilon:  # with probability epsilon vote for random (global) action out of candidate pool!
                            action_idx = np.random.randint(len(pred_reward_list))
                            batch_random_vote.append(1)
                        else:  # with probability 1-epsilon take best action = action with highest predicted Q
                            action_idx = torch.argmax(pred_reward_list).tolist()
                            batch_random_vote.append(0)
                else:
                    #if softmax_Q and not eval_flag:  # sample region wrt to softmax defined on 1/10 * Q [take out again factor of 10 which was introduced artificially above]
                    #    action_probs = nn.Softmax(dim=0)(
                    #        pred_reward_list * 10)  # maybe dim=0 here since list compared to dim=1 in tensor above
                    #    action_probs = action_probs.squeeze()  # works like this; without 0; or writing 1
                    #    candidate_action_dist = Categorical(action_probs)
                    #    action_entropy = candidate_action_dist.entropy().data.cpu().numpy()
                    #    batch_action_entropies.append(action_entropy)
                    #    action_idx = candidate_action_dist.sample(sample_shape=[1])
                    #    action_idx = action_idx.data.cpu().numpy()[0]
                    #else:  # take argmax of Q (in eval always)
                    action_idx = torch.argmax(pred_reward_list).tolist()

                batch_vote_action_idx.append(action_idx)
                batch_pred_reward_list.append(pred_reward_list)

            return batch_vote_action_idx, batch_pred_reward_list, batch_random_vote
