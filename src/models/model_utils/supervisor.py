import numpy as np
import argparse
import sys
import os
import torch
import re
import json
import time
import torch.multiprocessing as mp
import math

from torch.nn.utils import clip_grad_norm

from ..data_utils import data_utils
from ..data_utils.parser import *

import pdb
from collections import Counter
import copy

#from line_profiler_pycharm import profile

CKPT_PATTERN = re.compile('^ckpt-(\d+)$')

eps = 1e-3
log_eps = np.log(eps)


class Supervisor(object):
    """ The base class to manage the high-level model execution processes. The concrete classes for different applications are derived from it. """

    def __init__(self, model, args):
        self.processes = args.processes
        self.model = model
        self.keep_last_n = args.keep_last_n
        self.dropout_rate = args.dropout_rate
        self.global_step = args.resume
        self.batch_size = args.batch_size
        self.model_dir = args.model_dir  # + "_central"
        self.Z = args.Z
        # self.softmax_Q = args.softmax_Q  # flag

    def load_pretrained(self, load_model):
        print("Read model parameters from %s." % load_model)
        checkpoint = torch.load(load_model)
        self.model.load_state_dict(checkpoint)

    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        global_step_padded = format(self.global_step, '08d')
        ckpt_name = 'ckpt-' + global_step_padded
        path = os.path.join(self.model_dir, ckpt_name)
        ckpt = self.model.state_dict()
        torch.save(ckpt, path)

        if self.keep_last_n is not None:
            ckpts = []
            for file_name in os.listdir(self.model_dir):
                matched_name = CKPT_PATTERN.match(file_name)
                if matched_name is None or matched_name == ckpt_name:
                    continue
                step = int(matched_name.group(1))
                ckpts.append((step, file_name))
            if len(ckpts) > self.keep_last_n:
                ckpts.sort()
                os.unlink(os.path.join(self.model_dir, ckpts[0][1]))


class vrpSupervisor(Supervisor):  # locally for each agent
    """ Management class for vehicle routing."""

    def __init__(self, model, args, plotter):
        super(vrpSupervisor, self).__init__(model, args)
        self.DataProcessor = data_utils.vrpDataProcessor()
        self.plotter = plotter

    #@profile
    def update_node_representations(self, dm_list, take_x_y):
        # for list of vrps: for train and eval batch-size many, for val: more --> call it batch-size-wise here
        # for each given global state, update encoder outputs for all nodes

        # can be called per batch: each agent route gets locally separately embedded by an LSTM using local cost information
        dm_list = self.model.state_encoder.calc_embedding(dm_list, take_x_y)

        #each agent route gets locally separately embedded by an LSTM solely based on x-y-coords (i.e. no cost information)
        dm_list = self.model.state_encoder_no_cost.calc_embedding(dm_list, take_x_y)

        # representations of nodes in pool (here with cost information); needs to be called per vrp
        dm_list = self.model.pool_encoder.calc_pool_embedding(dm_list, take_x_y)
        return dm_list

    #@profile
    def check_pool(self, dm_list, Z, pool_node_dicts):
        # input: list of dms, number of candidate actions Z, pool node dicts (per vrp: tells for each node in the pool which agents were already offered the node / gave it away
        # output: list of dm ids with filled pool, list of lists: one list = Z dicts with pool node - agent assignments
        num_agents = dm_list[0].num_agents
        agent_ids = set(np.arange(num_agents))
        dm_ids_pool_filled = []  # collect indices of vrps in which pool is filled
        for dm_id, dm in enumerate(dm_list):
            if len(dm.vehicle_states[-1]) > 1:  # if "pool route" contains a node (another than its own start and end pool node)
                dm_ids_pool_filled.append(dm_id)
        dm_pool_node_assignments = []  # list of lists; one list = one vrp with filled pool containing Z dictionaries; one dictionary has keys = agent_id, value = nodeId of assigned node from pool
        if dm_ids_pool_filled:  # if there are vrps with a filled pool
            # generate Z pool nodes-agents-assignments for each vrp with filled pool
            for dm_id in dm_ids_pool_filled:
                Z_assign = []    # Z assignments per vrp
                cur_pool_nodes = dm_list[dm_id].vehicle_states[-1][1:]  # first node is irrelevant =  pool node itself --> here nodeIds

                # worst case: as many nodes in pool as agents and all agents already had the node. Then need to ensure that an agents is not offered two nodes. Only this scenario with conflicts.
                eachAgentAllNodesOffered = True
                for cur_pool_node in cur_pool_nodes:
                    if len(pool_node_dicts[dm_id][cur_pool_node]) != num_agents:
                        eachAgentAllNodesOffered = False

                # always: len(cur_pool_nodes) <= num_agents, worst case both agents give something to pool. If pool filled once then you can only take out so no more nodes can added
                for k in range(Z):
                    # init
                    cur_agent_assign = {i: [] for i in range(num_agents)}

                    if eachAgentAllNodesOffered:  # each node can be assigned to any agent, randomly draw (such that no agent gets more than one node)
                        cur_sampled_agents = random.sample(range(num_agents), len(cur_pool_nodes))
                        for i, agent in enumerate(cur_sampled_agents):
                            cur_agent_assign[agent] = cur_pool_nodes[i]
                    else:
                        # look at nodes individually to see which agents it can be assigned to (it cannot be that one node was seen by both agents and the other only by one. So either both by one or both by two.
                        # still conflicts possible: 3 agents and a node seen by one agent each. Then individual sampling can lead to one agent receiving more than one node
                        available_agents_in_action = copy.deepcopy(agent_ids)
                        for cur_pool_node in cur_pool_nodes:
                            # get agents which can get the cur pool node
                            excl_agents = pool_node_dicts[dm_id][cur_pool_node]
                            poss_agents = (agent_ids - excl_agents) & available_agents_in_action
                            # sample one
                            if poss_agents:  # i.e. if not empty (can be the case if all 5 agents give node to a pool and distribution in such a way that in the end agent who gave a node to the pool is the only available left but gets excluded then...)
                                sampled_agent = random.sample(poss_agents, 1)[0]
                                cur_agent_assign[sampled_agent] = cur_pool_node
                                # this agent cannot receive another node in this action
                                available_agents_in_action.remove(sampled_agent)
                            # it might be that not all pool nodes get distributed

                    Z_assign.append(cur_agent_assign)
                dm_pool_node_assignments.append(Z_assign)
        return dm_ids_pool_filled, dm_pool_node_assignments

    #@profile
    def candidate_local_actions(self, agent_id, batch_idx, dm_list, dm_ids_pool_filled, dm_pool_node_assignments, allow_no_change_in_rule, rew_all_possible_embeddings, take_x_y, rule_temperature_scaling, rule_temperature, eval_flag):
        batch_cand_l_actions, batch_cand_l_actions_rule_entropies, batch_cand_l_actions_rule_probs, batch_cand_l_actions_rule_logprobs = \
            self.model(batch_idx, dm_list, eval_flag, agent_id=agent_id,
                       candidate_flag=True, batch_cand_global_actions=None, epsilon_greedy=None, epsilon=None, softmax_Q=None,
                       dm_ids_pool_filled=dm_ids_pool_filled, dm_pool_node_assignments=dm_pool_node_assignments, allow_no_change_in_rule=allow_no_change_in_rule, rew_all_possible_embeddings=rew_all_possible_embeddings, take_x_y=take_x_y, Z=self.Z, rule_temperature_scaling=rule_temperature_scaling, rule_temperature=rule_temperature)

        return batch_cand_l_actions, batch_cand_l_actions_rule_entropies, batch_cand_l_actions_rule_probs, batch_cand_l_actions_rule_logprobs

    #@profile
    def global_action_voting(self, batch_idx, batch_cand_global_actions, dm_list, epsilon_greedy, epsilon, eval_flag):
        batch_vote_action_idx, batch_pred_reward_list, batch_random_vote = self.model(batch_idx, dm_list, eval_flag,
                                                                                      candidate_flag=False, batch_cand_global_actions=batch_cand_global_actions, epsilon_greedy=epsilon_greedy, epsilon=epsilon, softmax_Q=None,
                                                                                      dm_ids_pool_filled=None, dm_pool_node_assignments=None, allow_no_change_in_rule=None, rew_all_possible_embeddings=None, take_x_y=None, Z=None)

        return batch_vote_action_idx, batch_pred_reward_list, batch_random_vote

    #@profile
    def get_majority_voting(self, batch_votes_agents):
        # input: one list per agent. In each agent's list: his voted for action idx for all vrps in batch
        agent_votes_per_vrp = list(map(list, zip(*batch_votes_agents)))
        chosen_action_idxs_per_vrp = []
        for j in range(len(agent_votes_per_vrp)):  # for each vrp
            c = Counter(agent_votes_per_vrp[j])
            chosen_action_idx, freq = c.most_common(1)[0]
            chosen_action_idxs_per_vrp.append(chosen_action_idx)
        return chosen_action_idxs_per_vrp

    #@profile
    def rewrite(self, dm_list, chosen_action_idxs_per_vrp, batch_cand_global_actions,
                batch_cand_pred_reward_list, batch_cand_rule_entropies_agents, batch_cand_rule_probs_agents, batch_cand_rule_logprobs_agents, take_x_y, eval_flag):

        rewritten_dm_list = []
        num_agents = dm_list[0].num_agents
        if eval_flag:
            # only for one agent enough in val; info like different rule probs not neeeded there
            batch_rewriting_info_agents = []
        else:
            batch_rewriting_info_agents = [[] for _ in range(num_agents)]

        for j, dm in enumerate(dm_list):  # for each vrp
            # collect rewriting info
            chosen_global_action = batch_cand_global_actions[j][chosen_action_idxs_per_vrp[j]]
            pred_reward_tensor = batch_cand_pred_reward_list[j][chosen_action_idxs_per_vrp[j]]
            # can also be None in case of Z=1; because there no calling of Q was necessary (or if agent couldn't do anything in a pool round)
            # I think this is only the case if Z = 1;
            if pred_reward_tensor == None:
                pred_reward = pred_reward_tensor
            else:
                pred_reward = pred_reward_tensor.tolist()[0]

            if eval_flag:
                # global action enough
                batch_rewriting_info_agents.append(chosen_global_action)
            else:
                # for train save all relevant rewriting info for loss computations
                for agent_id in range(num_agents):  # rewriting info per agent since rule was used for different things
                    rule_entropy = batch_cand_rule_entropies_agents[agent_id][j][chosen_action_idxs_per_vrp[j]]
                    rule_probs = batch_cand_rule_probs_agents[agent_id][j][chosen_action_idxs_per_vrp[j]]
                    rule_logprobs = batch_cand_rule_logprobs_agents[agent_id][j][chosen_action_idxs_per_vrp[j]]

                    rewriting_info = [chosen_global_action, pred_reward, pred_reward_tensor, rule_probs, rule_logprobs, rule_entropy]
                    batch_rewriting_info_agents[agent_id].append(rewriting_info)

            # agent move, update global vehicle state
            new_dm = self.model.rewriter.move(dm, chosen_global_action, take_x_y)
            rewritten_dm_list.append(new_dm)

        return rewritten_dm_list, batch_rewriting_info_agents

    def get_cost_from_last_feasible(self, obs_team_avg_cost, t):
        # for one vrp: record of team average cost per rewriting step
        # get cost of last feasible solution before time step t
        hist = obs_team_avg_cost[:t].copy()
        feas_sol_found = False
        while not feas_sol_found:
            if np.isnan(hist[-1]):  # if last solution is not feasible since pool filled
                hist = hist[:-1].copy()
            else:
                feas_sol_found = True
        return hist[-1]

    def exclude_pool_from_distribution(self, orig_policy_probs):  # computes tilde policy
        exclude_pool_node = orig_policy_probs[:-1]   # last node always pool node
        dist_no_pool = exclude_pool_node / exclude_pool_node.sum()   # divide each entry by sum of all probs to normalize and get new distribution
        return dist_no_pool

    def compute_entropy(self, model_rule_probs, teacher_rule_probs):
        # pool node not included in probs anymore

        # I think this here leads to nan gradients:
        #teacher_log_probs = teacher_rule_probs.log()
        #elementwise_product = teacher_log_probs * model_rule_probs
        #H_old = -elementwise_product.nansum()

        H = 0
        H_tensor = torch.tensor(H, dtype=torch.float).cuda()

        for i in range(len(teacher_rule_probs)):
            # ignore zero probabilities (= correct) to get same result as above maybe without nan gradients
            if teacher_rule_probs[i] != 0 and model_rule_probs[i] != 0:
                # only then valid summand
                H_tensor -= teacher_rule_probs[i].log() * model_rule_probs[i]
        return H_tensor

    #@profile
    def train(self, cur_dm_rec, obs_team_avg_cost, cur_rewrite_recs, args, supervisor_teacher):  # data of one vrp of one agent
        # cur_dm_rec = [init_vrp, rewritten_vrp_1, rewritten_vrp_2, rewritten_vrp_3]
        # obs_team_avg_cost = [team avg cost of init_vrp, team avg cost of rewritten_vrp_1, team avg cost of rewritten_vrp_2, ...] --> is nan if solution is not feasible
        # cur_rewrite_recs = [rewrite info agent 1, rewrite info agent 2], with rewrite info agent i = [rewrite info 1, rewrite info 2 ,... max rewriting steps]
        # --> reward prediction the same in rewrite recs, but needed for rule probs since each agent generated training data

        # actually team avg cost could be taken from cur_dm_rec but ok

        max_steps_pool_filled = args.max_steps_pool_filled
        penalty_val = args.penalty_val
        advantage_file_name = args.advantage_file_name


        A_track = []  # contains values of advantage function for all rewriting steps
        advantage_track_probs = []  # values for all rewriting steps for one vrp
        advantage_track_Qs = []  # values for all rewriting steps for one vrp

        cur_total_policy_loss = data_utils.np_to_tensor(np.zeros(1), 'float', cuda_flag=True)
        cur_pred_value_rec = []
        cur_value_target_rec = []

        num_agents = len(cur_rewrite_recs)

        normal_rewards = []  # track out of interest to see their range; because penalty val should be defined accordingly maybe

        if args.tl_entropy_distill and self.global_step < args.tl_entropy_distill_batch_thresh:
            cur_transfer_loss = data_utils.np_to_tensor(np.zeros(1), 'float', cuda_flag=True)
            H_track = []
            R_track = []

            changed_init_tl_loss_list = []  # collected for each rewriting step, whether local re-ord action happened
            # get node representations for vrps from the teacher
            cur_dm_rec_teacher = [vrp_manag.clone(args.take_x_y) for vrp_manag in cur_dm_rec]
            cur_dm_rec_teacher = supervisor_teacher.update_node_representations(cur_dm_rec_teacher, args.take_x_y)

            # compute teacher probs for all local re-ordering actions in episode (done once here; then only need to sum up resp values later)
            # can be used to compute entropy later as well
            teacher_probs_no_pool_per_vrp = []
            chosen_local_rule_nodeId_per_vrp = []   # the rule nodes chosen by the student model for each agent for local re-ordering actions; needed for the teacher log probs in R
            for dm_idx, rewriting_info in enumerate(cur_rewrite_recs[0]):
                cur_global_action = rewriting_info[0]
                teacher_probs_no_pool_cur_vrp_per_agent = []
                chosen_local_rule_nodeId_cur_vrp_per_agent = []
                for cur_agent_id, cur_local_action in enumerate(cur_global_action):
                    if cur_local_action:
                        # if local re-ordering action
                        if cur_local_action[0][0] == cur_agent_id and cur_local_action[1][0] == cur_agent_id:

                            # get corresponding vrp (from teacher rec since there teacher node encodings)
                            cur_dm_teacher = cur_dm_rec_teacher[dm_idx]

                            # get chose rule node Id for later (to access probs in the probs tensor)
                            chosen_local_rule_nodeId_cur_vrp_per_agent.append(cur_dm_teacher.vehicle_states[cur_local_action[1][0]][cur_local_action[1][1]])

                            # get teacher probabilities
                            # eval flag doesn't matter: I need whole probs anyway (but less computations done in rule policy in that case)
                            _, teacher_rule_probs, _, _ = supervisor_teacher.model.rule_policy(None, cur_dm_teacher, cur_agent_id,
                                                                                               cur_local_action[0],
                                                                                               args.allow_no_change_in_rule,
                                                                                               args.rew_all_possible_embeddings,
                                                                                               args.take_x_y,
                                                                                               rule_temperature_scaling=False,
                                                                                               rule_temperature=1,
                                                                                               eval_flag=True)
                            # exclude pool from rule policy distribution: I only want teacher and model to agree on local re-ordering actions
                            teacher_rule_probs_no_pool = supervisor_teacher.exclude_pool_from_distribution(
                                teacher_rule_probs)
                            teacher_probs_no_pool_cur_vrp_per_agent.append(teacher_rule_probs_no_pool)
                        else:
                            teacher_probs_no_pool_cur_vrp_per_agent.append(np.nan)
                            chosen_local_rule_nodeId_cur_vrp_per_agent.append(np.nan)
                    else:
                        teacher_probs_no_pool_cur_vrp_per_agent.append(np.nan)
                        chosen_local_rule_nodeId_cur_vrp_per_agent.append(np.nan)

                teacher_probs_no_pool_per_vrp.append(teacher_probs_no_pool_cur_vrp_per_agent)
                chosen_local_rule_nodeId_per_vrp.append(chosen_local_rule_nodeId_cur_vrp_per_agent)

        # deduce pool penalty from cur_dm_rec: check if pool was not emptied
        pool_sizes = []
        for vrp in cur_dm_rec:
            pool_sizes.append(len(vrp.vehicle_states[-1]))   # pool node always contained, i.e. pool is filled if size > 1
        pool_penalties = []  # one entry per rewriting step = action  #np.zeros(len(cur_dm_rec))
        count = 0
        for k, pool_size in enumerate(pool_sizes[1:]):   # first vrp = init, always pool empty. Also need penalty per action
            if pool_size > 1:
                count += 1
                if count > max_steps_pool_filled:
                    pool_penalties.append(penalty_val)
                else:
                    pool_penalties.append(0)
            else:
                count = 0
                pool_penalties.append(0)

        # take Q reward from agent 0 (doesn't matter from which)
        # for rule policy data consider each agent data each and aggregate afterwards

        # for idx, (chosen_global_action, pred_reward, pred_reward_tensor, rule_probs, rule_logprobs, rule_entropy) in enumerate(cur_rewrite_recs[0]):  # len(cur_rewrite_rec) = num_rewriting_steps
        for idx, rewriting_info in enumerate(cur_rewrite_recs[0]):
            # for one time step t: compute loss summands
            # idx corresponds to rewriting step; sum t = 0 --> T-1

            # Collect summands in Q loss function (observed and predicted)
            chosen_global_action = rewriting_info[0]   # same for agents
            pred_reward = rewriting_info[1]   # same for agents
            pred_reward_tensor = rewriting_info[2]   # same for agents
            cur_reward = 0
            if self.model.gamma > 0.0:
                decay_coef = 1.0
                num_rollout_steps = len(cur_dm_rec) - 1 - idx  # len(cur_dm_rec)-1 = num_rewriting_steps, since len(cur_dm_rec)= 1 (init dm) +  num_rewriting_steps

                # s_0, a_0, r_1, s_1, a_1, r_2, ...
                for i in range(idx, idx + num_rollout_steps): # for the rewriting steps
                    if not(np.isnan(obs_team_avg_cost[i + 1])):  # if not nan
                        # then solution is feasible; compute improvement from last feasible solution
                        cost_from_last_feasible = self.get_cost_from_last_feasible(obs_team_avg_cost, i+1)
                        cur_dec_reward_step_i_plus_1 = decay_coef * (cost_from_last_feasible - obs_team_avg_cost[i + 1])
                        normal_rewards.append(cost_from_last_feasible - obs_team_avg_cost[i + 1])
                    else:
                        # then solution is not feasible, i.e. pool is filled
                        if pool_penalties[i] != 0:
                            # then pool filled for a longer time than allowed --> penalize
                            cur_dec_reward_step_i_plus_1 = - decay_coef * pool_penalties[i]
                        else:
                            # then ok
                            cur_dec_reward_step_i_plus_1 = 0

                    cur_reward += cur_dec_reward_step_i_plus_1
                    decay_coef *= self.model.gamma

            cur_reward_tensor = data_utils.np_to_tensor(np.array([cur_reward], dtype=np.float32), 'float',cuda_flag=True, volatile_flag=True)
            cur_value_target_rec.append(cur_reward_tensor)  # collects summands of sum from t= 0 T-1 in loss function: observed reward on lhs
            cur_pred_value_rec.append(pred_reward_tensor)  # collects summands of sum from t= 0 T-1 in loss function: predicted Q on rhs

            # Rule policy loss per agent data and then aggregated
            cur_total_policy_loss_agent_sum = data_utils.np_to_tensor(np.zeros(1), 'float', cuda_flag=True)

            A_track_agents = []
            advantage_track_probs_agents = []  # values for all rewriting steps for one vrp
            advantage_track_Qs_agents = []


            # Transfer rule policy loss per agent data and then aggregated
            if args.tl_entropy_distill and self.global_step < args.tl_entropy_distill_batch_thresh:
                cur_transfer_loss_agent_sum = data_utils.np_to_tensor(np.zeros(1), 'float', cuda_flag=True)
                R_track_agents = []  # R: contains teacher decisions
                H_track_agents = []  # entropy between teacher and model
                # for entropy distill: tl loss only computed for local reordering actions; can be zero if only pool actions
                changed_init_tl_loss = 0


            for agent_id in range(num_agents):
                # each agent used the (same) policy for a different region node --> different rule probs
                agent_action = chosen_global_action[agent_id]
                agent_rewriting_info = cur_rewrite_recs[agent_id][idx]
                rule_probs = agent_rewriting_info[3]
                rule_logprobs = agent_rewriting_info[4]

                if agent_action:   # can be empty if agent didn't have any nodes in a normal round or if he didn't get a node from the pool in the pool round
                    # compute A = pred_reward - \sum_other rule prob * Q
                    agent_rule_spec = agent_action[1]
                    chosen_rule_nodeId = cur_dm_rec[idx].vehicle_states[agent_rule_spec[0]][agent_rule_spec[1]]
                    changed_global_action_list = []
                    rule_candidates_nodeIds = []   # probably not needed
                    # get candidate rule nodes with respective changed global actions
                    for j in range(len(rule_probs)):  # go through all candidate rule nodes; index j = nodeId due to definition of rule probs
                        if j != chosen_rule_nodeId:
                            if rule_probs[j] != 0:   # if prob is not zero (i.e. if feasible and if weight on it; unfeasible were masked in rule policy)
                                # true rule candidate found: node with ID j
                                rule_candidates_nodeIds.append(j)
                                # get tuple position of this node in state
                                rule_cand_tupleStateId = list(cur_dm_rec[idx].get_route_dIdx(j))
                                changed_global_action = copy.deepcopy(chosen_global_action)
                                changed_global_action[agent_id][1][0] = rule_cand_tupleStateId[0]
                                changed_global_action[agent_id][1][1] = rule_cand_tupleStateId[1]
                                changed_global_action_list.append(changed_global_action)

                    # compute Q for the changed global actions
                    # changed global action list might be empty since policy completely determined (one node as prob 1, the others zero); then advantage = 0
                    if changed_global_action_list:
                        pred_rewards_tensor_list = self.model.compute_Q(cur_dm_rec[idx], changed_global_action_list)
                        pred_rewards_list = pred_rewards_tensor_list.squeeze().data.cpu().numpy().tolist()
                        if type(pred_rewards_list) == float:  # if computation was done for one action, then a 0-d array is returned
                            pred_rewards_list = [pred_rewards_list]

                        # rule probs of rule candidates
                        rule_probs_np = rule_probs.data.cpu().numpy()    #  as numpy array
                        rel_rule_probs = [rule_probs_np[j] for j in rule_candidates_nodeIds]

                        # chosen rule node needs to be included in average as well (i.e. reward for global action with chosen rule node (needs to be included in average) + rule prob)
                        pred_rewards_list.append(pred_reward)
                        rel_rule_probs.append(rule_probs_np[chosen_rule_nodeId])

                        sum_inA = np.dot(rel_rule_probs, pred_rewards_list)   # sum_inA = how good avg rule
                        A = pred_reward - sum_inA
                    else:   # rule probs = one hot vector; only prob 1 for chosen node rest prob 0
                        rule_probs_np = rule_probs.data.cpu().numpy()
                        rel_rule_probs = rule_probs_np[chosen_rule_nodeId]
                        pred_rewards_list = []
                        A = 0

                    ac_mask = np.zeros(len(rule_logprobs))   # Needed for pytorch gradient propagation? Below multiplication of tensors.
                    ac_mask[chosen_rule_nodeId] = A   # advantage value
                    ac_mask = data_utils.np_to_tensor(ac_mask, 'float', cuda_flag=True)  # , eval_flag)

                    if args.weigh_local_reord_in_rule_loss: # when transfering knowledge from another agent setup: local reordering already well learned, don't update weights there much and consequently focus on pool interaction actions (which need to be newly learned)
                        # check if current agent action involves pool interaction
                        if agent_action[0][0] == num_agents or agent_action[1][0] == num_agents:  # num_agents = index of pool in state
                            # if region node from pool or if rule node from pool: then action contains pool usage
                            ####################loss_weight_local_reord_act = 1
                            loss_weight_local_reord_act = args.loss_weight_pool_act  # try emphasizing pool action instead of reducing influence of localReord
                        else:  # a local reordering action: give it less weight
                            ####################loss_weight_local_reord_act = args.loss_weight_local_reord_act
                            loss_weight_local_reord_act = 1   # try keeping infl. of localreordact and only increase infuence of pool action
                    else:  # usual case when training: loss equally weighted for local reordering and pool actions
                        loss_weight_local_reord_act = 1

                    cur_total_policy_loss_agent_sum += rule_logprobs[chosen_rule_nodeId] * ac_mask[chosen_rule_nodeId] * loss_weight_local_reord_act
                    # cur_total_policy_loss -= rule_logprobs[chosen_rule_nodeId] * ac_mask[chosen_rule_nodeId]  # one summand of L_u(phi) added

                    advantage_track_probs_agents.append(rel_rule_probs)
                    advantage_track_Qs_agents.append(pred_rewards_list)
                    A_track_agents.append(A)

                    if args.tl_entropy_distill and self.global_step < args.tl_entropy_distill_batch_thresh:
                        # compute R and entropy for the current agent action (if no pool interaction is involved)

                        # check if local re-ordering action (only then additional transfer loss summand)
                        if agent_action[0][0] == agent_id and agent_action[1][0] == agent_id:  # then region and rule node local, i.e. local re-ordering action (else: summand for agent is zero, nothing to update)

                            # compute H (entropy) and R (teacher feedback on episode)
                            # take teacher probs out of list generated above

                            teacher_rule_probs_no_pool = teacher_probs_no_pool_per_vrp[idx][agent_id]
                            rule_probs_no_pool = self.exclude_pool_from_distribution(rule_probs)

                            H = self.compute_entropy(rule_probs_no_pool, teacher_rule_probs_no_pool)

                            ####### convert H to pytorch tensor and put it on cuda##########H_tensor = torch.tensor(H, dtype=torch.float).cuda()
                            H_track_agents.append(H)

                            # compute R
                            R = 0
                            # sum over all future time steps given the current one (idx)
                            for t_step in range(idx+1, len(teacher_probs_no_pool_per_vrp)):
                                rel_rule_id = chosen_local_rule_nodeId_per_vrp[t_step][agent_id]
                                if not math.isnan(rel_rule_id):   # then local re-ordering action
                                    R += teacher_probs_no_pool_per_vrp[t_step][agent_id][rel_rule_id].log()

                            # instead of append R, append it as tensor
                            R_track_agents.append(torch.tensor(R, dtype=torch.float).cuda())
                            R_mask = np.zeros(len(rule_logprobs))  # Needed for pytorch gradient propagation? Below multiplication of tensors.
                            R_mask[chosen_rule_nodeId] = R  # advantage value
                            R_mask = data_utils.np_to_tensor(R_mask, 'float', cuda_flag=True)

                            # add agent transfer loss to sum over agents

                            #cur_transfer_loss_agent_sum += rule_logprobs[chosen_rule_nodeId] * R_mask[chosen_rule_nodeId] + H
                            ## above corrected:
                            cur_transfer_loss_agent_sum += -(rule_logprobs[chosen_rule_nodeId] * R_mask[chosen_rule_nodeId]) + H
                            changed_init_tl_loss += 1



                else:
                    # how used? just for information I guess
                    advantage_track_probs_agents.append(np.nan)
                    advantage_track_Qs_agents.append(np.nan)
                    A_track_agents.append(np.nan)

            # one (negative) summand for time step t in rule loss is the average loss value over agents
            cur_total_policy_loss -= cur_total_policy_loss_agent_sum/num_agents

            A_track.append(A_track_agents)
            advantage_track_probs.append(advantage_track_probs_agents)
            advantage_track_Qs.append(advantage_track_Qs_agents)

            if args.tl_entropy_distill and self.global_step < args.tl_entropy_distill_batch_thresh:
                # one (negative) summand for time step t in transfer rule loss is the average loss value over agents
                #cur_transfer_loss -= cur_transfer_loss_agent_sum/num_agents
                ## above corrected:
                cur_transfer_loss += cur_transfer_loss_agent_sum/num_agents

                H_track.append(H_track_agents)
                R_track.append(R_track_agents)
                changed_init_tl_loss_list.append(changed_init_tl_loss)


        # done with summing up all summands for t

        if args.tl_entropy_distill and self.global_step < args.tl_entropy_distill_batch_thresh:
            # aggregate losses: normal model loss + transfer learning loss

            # cur transfer loss is ALWAYS negative but should contribute to the loss function
            # corrected above; wrong here
            ########cur_transfer_loss *= -1
            cur_final_policy_loss = cur_total_policy_loss + args.tl_loss_weight * cur_transfer_loss

            num_local_reord_actions_in_rewriting_episode = len(changed_init_tl_loss_list)-changed_init_tl_loss_list.count(0)
            if num_local_reord_actions_in_rewriting_episode > 0:
                transfer_loss_for_stopping_criterion = cur_transfer_loss / num_local_reord_actions_in_rewriting_episode
            else:
                # then zero
                transfer_loss_for_stopping_criterion = cur_transfer_loss

        else:
            cur_final_policy_loss = np.nan
            cur_transfer_loss = np.nan
            H_track = np.nan
            R_track = np.nan
            transfer_loss_for_stopping_criterion = np.nan

        return cur_total_policy_loss, cur_pred_value_rec, cur_value_target_rec, A_track, advantage_track_probs, advantage_track_Qs, pool_penalties, normal_rewards,cur_final_policy_loss, cur_transfer_loss, H_track, R_track, transfer_loss_for_stopping_criterion

    def train_batch(self, dm_rec, team_avg_cost_rec, rewrite_rec_agents, epoch, args, batch_rule_entropies, supervisor_teacher):  # batch data of one agent
        # batch rule entropies not used, but if setting doesn't work we try to add a penalty based on the entropy; so not deleted yet --> to get that poliy doesn't get too certain too fast

        batch_size = len(dm_rec)
        pred_value_rec = []
        obs_value_rec = []
        total_policy_loss = data_utils.np_to_tensor(np.zeros(1), 'float', cuda_flag=True)
        # total_value_loss = data_utils.np_to_tensor(np.zeros(1), 'float', cuda_flag=True)

        final_policy_loss = data_utils.np_to_tensor(np.zeros(1), 'float', cuda_flag=True)  # aggregated loss: total policy and transfer
        transfer_loss = data_utils.np_to_tensor(np.zeros(1), 'float', cuda_flag=True)
        transfer_loss_for_stopping_criterion = data_utils.np_to_tensor(np.zeros(1), 'float', cuda_flag=True) # is averaged over number of local re-ord actions

        track_pred_reward = []
        track_obs_reward = []

        batch_advantage_track_probs = []  # one list per vrp in batch; in list: list of values per rewriting step in this vrp
        batch_advantage_track_Qs = []  # one list per vrp in batch; in list: list of values per rewriting step in this vrp
        batch_pool_penalties = []
        batch_normal_rewards = []

        batch_H_track = []
        batch_R_track = []

        for dm_idx, cur_dm_rec in enumerate(dm_rec):  # for each vrp in batch
            obs_team_avg_cost = team_avg_cost_rec[dm_idx]
            cur_rewrite_recs = [rewrite_rec_agent[dm_idx] for rewrite_rec_agent in rewrite_rec_agents]

            cur_total_policy_loss, cur_pred_value_rec, cur_obs_value_rec, A_track, advantage_track_probs, advantage_track_Qs, pool_penalties, normal_rewards,cur_final_policy_loss, cur_transfer_loss, H_track, R_track, cur_transfer_loss_for_stopping_criterion = self.train(cur_dm_rec,
                                                                                                           obs_team_avg_cost, cur_rewrite_recs, args, supervisor_teacher)
            # policy loss aggregated for one vrp (over its rewriting steps)
            # for Q: pred and observed collected (for all rewriting steps)
            batch_advantage_track_Qs.append(advantage_track_Qs)
            batch_advantage_track_probs.append(advantage_track_probs)
            batch_pool_penalties.append(pool_penalties)
            batch_normal_rewards.append(normal_rewards)
            if dm_idx == 0:
                A_track_1st_batch_vrp = A_track.copy()

            track_pred_reward.append(cur_pred_value_rec)
            track_obs_reward.append(cur_obs_value_rec)
            pred_value_rec.extend(cur_pred_value_rec)
            obs_value_rec.extend(cur_obs_value_rec)

            batch_H_track.append(H_track)
            batch_R_track.append(R_track)

            # above: values incorporating information of rewriting episode of ONE vrp. Now aggregate / collect for whole batch
            total_policy_loss += cur_total_policy_loss  #.item() TO AVOID CUDA MEMORY ERROR ADDED .ITEM() PROBLEM!!DETACHING HAPPENS HERE; maybe CUDA ERROR is the reason for the weird condition in orig code
            final_policy_loss += cur_final_policy_loss
            transfer_loss += cur_transfer_loss
            transfer_loss_for_stopping_criterion += cur_transfer_loss_for_stopping_criterion

        # now: policy loss summed up for all vrps in batch; Q values collected for all vrps in batch

        # if len(pred_value_rec) > 0:   # not sure this can still happen. If for all vrps in batch the agent never visited any node...
        pred_value_rec = torch.cat(pred_value_rec, 0)
        obs_value_rec = torch.cat(obs_value_rec, 0)
        pred_value_rec = pred_value_rec.unsqueeze(1)
        obs_value_rec = obs_value_rec.unsqueeze(1)

        # Q loss
        total_value_loss = torch.nn.MSELoss(reduction="sum")(pred_value_rec, obs_value_rec)   # orig: total_value_loss = F.smooth_l1_loss(pred_value_rec, obs_value_rec, size_average=False)
        total_value_loss /= batch_size
        total_value_loss /= len(rewrite_rec_agents[0][0])  # divided by 1/T to get loss per rewriting step and not per episode

        # policy losses
        total_policy_loss /= batch_size  # pure model
        final_policy_loss /= batch_size   # model + transfer
        transfer_loss /= batch_size   # transfer

        ######## for stopping criterion: avg transfer loss only over vrps where local re-reord actions appeared and avg additionally over their freq (x out of 30 rewriting steps)
        if args.tl_entropy_distill and self.global_step < args.tl_entropy_distill_batch_thresh:
            try:
                bool_localReOrd_in_vrp_batch = [1 if any(vrp) else 0 for vrp in batch_H_track]
                batch_size_localReord_contained_in_episode = sum(bool_localReOrd_in_vrp_batch)
                if batch_size_localReord_contained_in_episode > 0:
                    transfer_loss_for_stopping_criterion = transfer_loss_for_stopping_criterion / batch_size_localReord_contained_in_episode
                else:
                    transfer_loss_for_stopping_criterion = 10  # just to avoid that stopping criterion hits in
            except TypeError:
                print(batch_H_track)
                sys.exit(0)
        ##########

        if args.turn_off_alpha_temporary:
            if self.global_step <= 500:
                self.model.value_loss_coef = 0   # only update encoders and Q
                print("alpha set to zero")
            else:
                self.model.value_loss_coef = args.value_loss_coef
                print("alpha > 0 like usually")


        if args.dynamic_alpha:
            if (epoch > 0) and (epoch % 8 == 0):
                if self.model.value_loss_coef >= 1:
                    self.model.value_loss_coef *= 1.5
                else:
                    self.model.value_loss_coef *= 10
            # if epoch == 5:  # epoch 0-4 done
            #    self.model.value_loss_coef = 0.1  # init was 0.01
            # if epoch == 20:
            #    self.model.value_loss_coef = 1

        # OLD add a penalty term to the policy loss based on the entropy; idea: entropy is supposed to decrease slowly; CURRENTLY OUT
        # OLD since we're minimizing: subtract entropy term (with a factor as a new hyperparam)
        # #####total_loss = self.model.value_loss_coef * (total_policy_loss - self.model.penalty_scale * np.mean(batch_rule_entropies)) + total_value_loss

        if args.tl_entropy_distill and self.global_step < args.tl_entropy_distill_batch_thresh:
            # in final: transfer loss included
            total_loss = self.model.value_loss_coef * final_policy_loss + total_value_loss

            # check if mean (over batch) weighted tl loss is smaller than eps; then ausschalten in future
            ###if (torch.absolute(args.tl_loss_weight * transfer_loss) < args.eps_weighted_tl) and (
            ###        changed_init_tl_loss > 0):  # if only pool actions, then tl loss is zero
            print(transfer_loss_for_stopping_criterion)
            if torch.absolute(transfer_loss_for_stopping_criterion) < args.eps_weighted_tl: # NOW: averaged over number of local reord actions! Not sum
                args.tl_entropy_distill_batch_thresh = 1  # from now on: no more transfer loss influence
                # no more teacher; and now emphasize learning actions with pool involved
                args.weigh_local_reord_in_rule_loss = True



        else:
            total_loss = self.model.value_loss_coef * total_policy_loss + total_value_loss

        return total_loss, total_policy_loss, total_value_loss, track_pred_reward, track_obs_reward, A_track_1st_batch_vrp, batch_advantage_track_probs,  batch_advantage_track_Qs, batch_pool_penalties, batch_normal_rewards, final_policy_loss, transfer_loss, batch_H_track,batch_R_track




# for eval not implemented (only val within training)

    def batch_eval(self, eval_data, output_trace_flag, process_idx):
        cum_loss = 0
        cum_reward = 0
        data_size = len(eval_data)

        for batch_idx in range(0, data_size, self.batch_size):
            batch_data = self.DataProcessor.get_batch(eval_data, self.batch_size, batch_idx)
            cur_avg_loss, cur_avg_reward, dm_rec = self.model(batch_data, eval_flag=True)
            cum_loss += cur_avg_loss.item() * len(batch_data)
            cum_reward += cur_avg_reward * len(batch_data)
            if output_trace_flag == 'complete':
                for cur_dm_rec in dm_rec:
                    for i in range(len(cur_dm_rec)):
                        print('step ' + str(i))
                        dm = cur_dm_rec[i]
                        print(dm.tot_dis[-1])
                        for j in range(len(dm.vehicle_state)):
                            cur_pos, cur_capacity = dm.vehicle_state[j]
                            cur_node = dm.get_node(cur_pos)
                            print(cur_node.x, cur_node.y, cur_node.demand, cur_capacity, dm.tot_dis[j])
                        print('')
            print('process start idx: %d batch idx: %d pred reward: %.4f' \
                  % (process_idx, batch_idx, cur_avg_reward))
        return cum_loss, cum_reward

    #@profile
    def eval(self, data, output_trace_flag, max_eval_size=None):
        data_size = len(data)
        if max_eval_size is not None:
            data_size = min(data_size, max_eval_size)
        eval_data = data[:data_size]
        if self.processes == 1:
            cum_loss, cum_reward = self.batch_eval(eval_data, output_trace_flag, 0)
        else:
            cum_loss = 0
            cum_reward = 0
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                pass
            pool = mp.Pool(processes=self.processes)
            res = []
            batch_per_process = data_size // self.processes
            if data_size % batch_per_process > 0:
                batch_per_process += 1
            for st in range(0, data_size, batch_per_process):
                res += [
                    pool.apply_async(self.batch_eval, (eval_data[st: st + batch_per_process], output_trace_flag, st))]
            for i in range(len(res)):
                cur_cum_loss, cur_cum_reward = res[i].get()
                cum_loss += cur_cum_loss
                cum_reward += cur_cum_reward

        avg_loss = cum_loss / data_size
        avg_reward = cum_reward / data_size
        print('average pred reward: %.4f' % avg_reward)
        return avg_loss, avg_reward







""" irrelevant by now

    def choose_local_actions(self, batch_idx, batch_fictional_global_actions, dm_list_local, batch_fictional_actions_rule_entropies, batch_fictional_actions_ac_logprobs, batch_fictional_actions_ac_probs, batch_fictional_actions_chosen_candidate_acs, batch_fictional_actions_candidate_neighbor_idxes, allow_no_change_in_rule,rew_all_possible_embeddings, take_x_y,only_rule_can_do_nothing,epsilon_greedy, epsilon, batch_count_random, eval_flag = False):
        batch_chosen_local_actions, batch_rewriting_info, batch_rule_picking_entropies, batch_action_entropies,batch_count_random, batch_candidates_Qs, batch_step_bool_eps_case = self.model(batch_idx, dm_list_local, allow_no_change_in_rule,rew_all_possible_embeddings, take_x_y,only_rule_can_do_nothing,epsilon_greedy, epsilon,eval_flag, batch_count_random=batch_count_random,
                                                                                                    batch_fictional_global_actions=batch_fictional_global_actions,
                                                                                                    softmax_Q=self.softmax_Q, batch_fictional_actions_rule_entropies=batch_fictional_actions_rule_entropies, batch_fictional_actions_ac_logprobs= batch_fictional_actions_ac_logprobs, batch_fictional_actions_ac_probs=batch_fictional_actions_ac_probs, batch_fictional_actions_chosen_candidate_acs=batch_fictional_actions_chosen_candidate_acs, batch_fictional_actions_candidate_neighbor_idxes=batch_fictional_actions_candidate_neighbor_idxes)  # vrpModel --> calls forward function
        # pred rewards correspond to chosen region nodes in batch_chosen_local_actions
        # here tracking info : ac_logprobs_pred_rewards, ...
        if eval_flag:
            return batch_chosen_local_actions
        else:
            return batch_chosen_local_actions, batch_rewriting_info, batch_rule_picking_entropies, batch_action_entropies,batch_count_random, batch_candidates_Qs, batch_step_bool_eps_case


    def outside_suggestion(self, global_actions):
        # input: list of global actions potentially containing outside rules which need to be specified now
        num_agents = len(global_actions[0])
        agent_ids = np.arange(num_agents)
        reg_pools = []
        distr_suggestions = []
        for global_action in global_actions:
            reg_pool = []
            for local_action in global_action:
                if "outside" in local_action:
                    reg_pool.append(local_action[0])
            reg_pools.append(reg_pool)
            cur_distribution = dict.fromkeys(np.arange(len(reg_pool)))  # key = index of region node in reg_pool
            # randomly distribute region nodes to other potential agents
            for i in range(len(reg_pool)):
                # region node i can be given to any other agent than the one owning the region node
                pot_agents = list(set(agent_ids)-{reg_pool[i][0]})
                chosen_agent = random.sample(pot_agents, 1)
                cur_distribution[i] = chosen_agent
                # check whether distribution is accepted by agents

            distr_suggestions.append(cur_distribution)
        return distr_suggestions, reg_pools

    def correct_rewriting_info(self, agent_id, batch_rewriting_info, batch_solved_global_actions, dm_list,allow_no_change_in_rule,rew_all_possible_embeddings, take_x_y):
        batch_size = len(batch_solved_global_actions)
        corr_info = batch_rewriting_info.copy()
        for i in range(batch_size):
            # if region and rule node changed

            if batch_solved_global_actions[i][agent_id]:  # can be empty if no action was possible since agent is not visiting nodes; in that case rewriting info is ok (it contains a Q pred but maybe shouldn't)

                if [batch_solved_global_actions[i][agent_id][0][1],batch_solved_global_actions[i][agent_id][1]] != [batch_rewriting_info[i][0],batch_rewriting_info[i][3]] :
                    # then action was changed by conflict solver to doing nothing

                    # rewriting info structure: [chosen_rewrite_pos, pred_reward, pred_reward_tensor, neighbor_idx_tuple, ac_logprobs, chosen_candidate_acs, ac_probs, candidate_neighbor_idxes, rule_picking_entropy]
                    corr_info[i][0] = batch_solved_global_actions[i][agent_id][0][1]
                    # compute Q value (first float, then tensor)
                    pred_reward_tensor = self.model.compute_Q(dm_list[i], [batch_solved_global_actions[i]])[0]
                    corr_info[i][1] = pred_reward_tensor.tolist()[0]
                    corr_info[i][2] = pred_reward_tensor
                    # taken rule node (meaning doing nothing)
                    corr_info[i][3] = batch_solved_global_actions[i][agent_id][1]    # rule tuple
                    # need to compute rule picking info
                    _, ac_logprobs, _, ac_probs, candidate_neighbor_idxes, rule_picking_entropy = self.model.rule_policy(dm_list[i], batch_rewriting_info[i][0], allow_no_change_in_rule,rew_all_possible_embeddings, take_x_y, False)
                    # ac_logprobs
                    corr_info[i][4] = ac_logprobs
                    # chosen rule node index (candidate_acs) --> index of rule tuple in candidate_n
                    corr_info[i][5] = candidate_neighbor_idxes.index(batch_solved_global_actions[i][agent_id][1])
                    # ac_probs
                    corr_info[i][6] = ac_probs
                    # candidate_neighbor_idxes
                    corr_info[i][7] = candidate_neighbor_idxes
                    # rule_picking_entropy
                    corr_info[i][8] = rule_picking_entropy

        return corr_info




    def get_conflicts(self, global_action):
        num_agents = len(global_action)
        cur_conflicts = []  # collects conflicts in one global action

        for i in range(num_agents - 1):
            # if one agent visits all nodes, then actions of other agents are truly empty [] and there cannot be a conflict
            # check if action of agent i is empty; then there cannot be a conflict with the others
            if not global_action[i]:
                continue
            for j in range(i + 1, num_agents):  # check whether there's conflict with action of agent i
                # check if action of agent i is empty; then there cannot be a conflict with agent i
                if not global_action[j]:
                    continue

                # if agent i is doing nothing or if agent j is doing nothing: (ag_id, reg)-(ag_id, rule) == (0,1) in case of doing nothing
                if (np.array(global_action[i][0]) - np.array(global_action[i][1]) == np.array([0, 1])).all() or (
                        np.array(global_action[j][0]) - np.array(global_action[j][1]) == np.array([0, 1])).all():
                    # in this case only rule i = rule j can be a conflict
                    if global_action[i][1] == global_action[j][1]:
                        cur_conflicts.append([i, j])

                    continue

                # if agent i and j both have proper actions; more conflicts are possible
                if global_action[i][0] == global_action[j][1] or global_action[i][1] == global_action[j][0] or \
                        global_action[i][1] == global_action[j][
                    1]:  # if region i = rule j or rule i = rule j OR region j = rule i, then conflict
                    cur_conflicts.append([i, j])  # remember conflicting agent ids
        return cur_conflicts


    def solve_by_doing_nothing(self, dm_list, dm_id, agent_id, cur_solved_action, cur_conflicts):
        cur_num_regions = len(dm_list[dm_id].global_vehicle_states[agent_id]) - 2  # route of current agent
        possible_doingNothing_regions_randomOrder = np.arange(cur_num_regions) + 1
        random.shuffle(possible_doingNothing_regions_randomOrder)
        possible_doingNothing_actions_randomOrder = [([agent_id, region], [agent_id, region - 1]) for region in
                                                     possible_doingNothing_regions_randomOrder]
        successful = False
        for doNothing_action in possible_doingNothing_actions_randomOrder:
            test_cur_solved_action = cur_solved_action.copy()
            test_cur_solved_action[agent_id] = doNothing_action
            # check if a new conflict was inserted by testing whether number of conflicts was reduced or not
            if len(self.get_conflicts(test_cur_solved_action)) < len(cur_conflicts):
                # then it was successful
                successful = True
                cur_solved_action = test_cur_solved_action.copy()
                break

        return successful, cur_solved_action

    def conflict_solving(self, batch_chosen_global_actions, dm_list, fict_actions=False):  # input: per vrp in batch one global action

        num_agents = len(batch_chosen_global_actions[0])
        batch_solved_global_actions = []
        batch_num_conflicts_per_action = []
        problems_with_conflict_solving = []
        for act_id, global_action in enumerate(batch_chosen_global_actions):
            cur_conflicts = self.get_conflicts(global_action)   # list of agent ids involved in conflicts
            # now: conflicts for one global action collected
            flat_conflicts = [item for sublist in cur_conflicts for item in sublist]
            agent_dict_freq_conflicts = Counter(flat_conflicts)  # Counter({0:2, 1: 4, 2: 4, 3: 2, 4: 2})
            batch_num_conflicts_per_action.append(len(cur_conflicts))
            cur_solved_action = global_action.copy()
            while cur_conflicts:  # while conflicts not empty
                cur_agent_id, cur_freq = agent_dict_freq_conflicts.most_common(1)[
                    0]  # return list with one entry: tuple (agent id, freq)
                # don't delete respective agent's action like cur_solved_action[cur_agent_id] = [], but replace with doing nothing action
                # take random doing nothing action which doesn't lead to a conflict
                # get corresponding vrp to see which region nodes are possible
                if fict_actions:   # in fict case: Z many actions per vrp in set
                    cur_dm_id = int(act_id/self.Z)
                else:   # usual case if actions have been chosen; one action per vrp then
                    cur_dm_id = act_id

                successful, cur_solved_action = self.solve_by_doing_nothing(dm_list, cur_dm_id, cur_agent_id, cur_solved_action, cur_conflicts)
                if not successful:  # then no doNothing action lead to a reduction in conflicts  [example: routes [10,9,10], [11,3,8,6,...,11] --> global action ([0,1],[1,2]), ([1,2],[0,0]). Giving agent 0 doing nothing action does not solve the problem. Agent 1 has to do nothing
                    # can happen if an agent only has one possiblity to do nothing (since only one region node there) which collides with action of other agent
                    # replace action of other agent with doing nothing
                    other_agents_involved_in_conflict = list(set(agent_dict_freq_conflicts.keys())-{cur_agent_id})
                    for other_agent_id in other_agents_involved_in_conflict:
                        successful, cur_solved_action = self.solve_by_doing_nothing(dm_list, cur_dm_id,
                                                                                          other_agent_id,
                                                                                          cur_solved_action,
                                                                                          cur_conflicts)
                        if successful:
                            break


                if not successful:
                    problems_with_conflict_solving.append(1)

                # assume conflict was solved
                cur_conflicts_new = [conflict for conflict in cur_conflicts if
                                     cur_agent_id not in conflict]  # taken out conflicts containing cur_agent_id
                del agent_dict_freq_conflicts[cur_agent_id]  # delete respective counter element
                if cur_conflicts_new:  # if there are still conflicts, update freq dict
                    removed_conflicts = [conflict for conflict in cur_conflicts if
                                         cur_agent_id in conflict]  # decrease counter of other agents invovled; important! And delete it if it is zero
                    agents_involved_in_removed_conflicts = list(
                        set([item for sublist in removed_conflicts for item in sublist]) - {cur_agent_id})
                    for agent_involved in agents_involved_in_removed_conflicts:
                        agent_dict_freq_conflicts[agent_involved] = agent_dict_freq_conflicts[agent_involved] - 1
                cur_conflicts = cur_conflicts_new

            batch_solved_global_actions.append(cur_solved_action)
        return batch_solved_global_actions, batch_num_conflicts_per_action

    def rewrite_old(self, dm_list_local, batch_solved_global_actions, local_batch_rewriting_info, take_x_y):
        rewritten_dm_list_local = []  # of one agent
        for i, dm in enumerate(
                dm_list_local):  # for each vrp (local since contains local perspective´, 64 dms contained in dm_list_local of respective agent)
            new_dm = self.model.rewriter.move(dm, batch_solved_global_actions[i], take_x_y)  # dm with corresponding global action
            # 2) append solved_global_action  i.e. if cur global action= batch_solved_global_actions[i]
            if local_batch_rewriting_info != None:   # in val case it's none; maybe also when agent has no node?
                local_batch_rewriting_info[i].append(batch_solved_global_actions[i])  # append global action to info --> call by reference! Will change corr_batch_rewriting_info_agents

            ""
            if local_batch_rewriting_info != None:  # inserted due to eval flag; there no rewriting info
                if batch_solved_global_actions[i][dm.agent_id][0][1] != local_batch_rewriting_info[i][0]:   # if action was changed due to conflict solving; can be seen by comparing region node
                if not batch_solved_global_actions[i][dm.agent_id]:  # if action of agent is empty; check whether intentionally or due to conflict solving (if not empty, then the info written in rewriting info is still correct)
                    if local_batch_rewriting_info[i][0]:  # if chosen region was not empty before conflict solving (i.e. if agent's action was discarded); then update info
                        local_batch_rewriting_info[i] = [[] for _ in range(
                            9)]  # then agent did not move his region node (and it was not planned; o/w there would be values for pred_reward)
                        # 8 since [chosen_rewrite_pos, pred_reward, pred_reward_tensor, neighbor_idx_tuple, ac_logprobs, chosen_candidate_acs, ac_probs, candidate_neighbor_idxes, rule_picking_entropy]

                        # compute pred_reward for "do nothing" to have a Q-score to learn from in loss (happening afterwards now after conflict solving)
                        pred_reward_tensor = self.model.compute_Q(dm, [batch_solved_global_actions[i]])[
                            0]  # compute Q for solved global action in which agent's action was discarded (value); result is a list with one elem that's why index 0
                        local_batch_rewriting_info[i][2] = pred_reward_tensor
                        local_batch_rewriting_info[i][1] = pred_reward_tensor.tolist()[0]

                local_batch_rewriting_info[i].append(batch_solved_global_actions[
                                                         i])  # append global action to info --> call by reference! Will change batch_rewriting_info_agents
             ""
            rewritten_dm_list_local.append(new_dm)
        return rewritten_dm_list_local






def train(self, batch_data):
        self.model.dropout_rate = self.dropout_rate
        self.model.optimizer.zero_grad()
        avg_loss, avg_reward, dm_rec = self.model(batch_data)  # vrpModel --> calls forward function
        self.global_step += 1
        if type(avg_loss) != float:
            avg_loss.backward()
            self.model.train()
        return avg_loss.item(), avg_reward"""

"""
    if local_batch_rewriting_info != None:  # inserted due to eval flag; there no rewriting info
        if batch_solved_global_actions[i][dm.agent_id][0][1] != local_batch_rewriting_info[i][0]:   # if action was changed due to conflict solving; can be seen by comparing region node
        if not batch_solved_global_actions[i][dm.agent_id]:  # if action of agent is empty; check whether intentionally or due to conflict solving (if not empty, then the info written in rewriting info is still correct)
            if local_batch_rewriting_info[i][0]:  # if chosen region was not empty before conflict solving (i.e. if agent's action was discarded); then update info
                    local_batch_rewriting_info[i] = [[] for _ in range(
                        9)]  # then agent did not move his region node (and it was not planned; o/w there would be values for pred_reward)
                    # 8 since [chosen_rewrite_pos, pred_reward, pred_reward_tensor, neighbor_idx_tuple, ac_logprobs, chosen_candidate_acs, ac_probs, candidate_neighbor_idxes, rule_picking_entropy]

                    # compute pred_reward for "do nothing" to have a Q-score to learn from in loss (happening afterwards now after conflict solving)
                    pred_reward_tensor = self.model.compute_Q(dm, [batch_solved_global_actions[i]])[
                        0]  # compute Q for solved global action in which agent's action was discarded (value); result is a list with one elem that's why index 0
                    local_batch_rewriting_info[i][2] = pred_reward_tensor
                    local_batch_rewriting_info[i][1] = pred_reward_tensor.tolist()[0]

            local_batch_rewriting_info[i].append(batch_solved_global_actions[
                                                     i])  # append global action to info --> call by reference! Will change batch_rewriting_info_agents
"""

"""
    batch_idx, dm_list_local, dm_ids_pool_filled, dm_pool_node_assignments, allow_no_change_in_rule, rew_all_possible_embeddings, take_x_y,
    only_rule_can_do_nothing, epsilon_greedy, epsilon, eval_flag, batch_count_random = None, Z = None,
    batch_cand_global_actions = None, softmax_Q = None, batch_l_cand_actions_rule_entropies = None,
    batch_l_cand_actions_ac_logprobs = None, batch_l_cand_actions_ac_probs = None,
    batch_fictional_actions_chosen_candidate_acs = None,
    batch_fictional_actions_candidate_neighbor_idxes = None,
    candidate_flag = False

        #self.model(batch_idx, dm_list_local, None,None, allow_no_change_in_rule,rew_all_possible_embeddings, take_x_y,only_rule_can_do_nothing,epsilon_greedy, epsilon,eval_flag, batch_count_random=batch_count_random,
                                                                                                batch_cand_global_actions=batch_cand_global_actions,
                                                                                                softmax_Q=self.softmax_Q, batch_l_cand_actions_rule_entropies=batch_fictional_actions_rule_entropies, batch_fictional_actions_ac_logprobs= batch_fictional_actions_ac_logprobs, batch_fictional_actions_ac_probs=batch_fictional_actions_ac_probs, batch_fictional_actions_chosen_candidate_acs=batch_fictional_actions_chosen_candidate_acs, batch_fictional_actions_candidate_neighbor_idxes=batch_fictional_actions_candidate_neighbor_idxes)  # vrpModel --> calls forward function
    # pred rewards correspond to chosen region nodes in batch_chosen_local_actions
    # here tracking info : ac_logprobs_pred_rewards, ...
    if eval_flag:
        return batch_chosen_local_actions
    else:
        return batch_chosen_local_actions, batch_rewriting_info, batch_rule_picking_entropies, batch_action_entropies,batch_count_random, batch_candidates_Qs, batch_step_bool_eps_case
"""