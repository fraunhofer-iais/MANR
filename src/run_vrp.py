import random
import numpy as np
import pandas as pd
import torch
import arguments
import models.data_utils.data_utils as data_utils
import models.model_utils as model_utils
from models.vrpModel import vrpModel
from visdom import Visdom
import time
import copy
import ray
from ray import tune
import pdb
import pickle
import sys


class VisdomPlotter(object):  
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom()  
        self.env = env_name
        self.plots = {}
        self.texts = {}
        self.text_windows = {}

    def plot(self, var_name, split_name, title_name, x, y, x_label):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel=x_label,
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                          update='append')

    def scatter(self, var_name, split_name, title_name, x, y, x_label):
        # x and y are floats each
        # np.array([[x,y]]).ndim = 2
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.scatter(X=np.array([[x, y]]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel=x_label,
                ylabel=var_name,
                opacity=0.5
            ))
        else:
            self.viz.scatter(X=np.array([[x, y]]), env=self.env, win=self.plots[var_name], name=split_name,
                             update='append')

    def scatter_old(self, var_name, split_name, title_name, x, y, x_label):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.scatter(X=np.array([x]), Y=np.array([y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel=x_label,
                ylabel=var_name
            ))
        else:
            self.viz.scatter(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                             update='append')

    def text(self, var_name, key, value):
        if var_name not in self.texts:
            self.texts[var_name] = {}
            self.texts[var_name][key] = value
        else:
            self.texts[var_name][key] = value

    def print_texts(self):
        for varname, text_dict in self.texts.items():
            self.print_text(varname, text_dict)

    def print_text(self, var_name, text_dict):
        text = var_name + ':<br>'
        for key, value in text_dict.items():
            if 'time' in key.split(' '):
                text += key + ': ' + self.convert_time(value) + '<br>'
            else:
                text += key + ': ' + str(value) + '<br>'
        if var_name not in self.text_windows:
            self.text_windows[var_name] = self.viz.text(text, env=self.env)
        else:
            self.viz.text(text, win=self.text_windows[var_name], append=False, env=self.env)

    def convert_time(self, time):
        time = int(np.round(time))
        hours = time // 3600
        minutes = (time % 3600) // 60
        seconds = time % 60
        if hours > 0:
            return str(hours).zfill(2) + ':' + str(minutes).zfill(2) + ':' + str(seconds).zfill(2)
        elif minutes > 0:
            return str(minutes).zfill(2) + ':' + str(seconds).zfill(2)
        return str(seconds)


def create_model(args, plotter, num_agents, teacher=None):
    # model init: in Base it prints "current learning rate is..."
    model = vrpModel(args, num_agents)

    if model.cuda_flag:
        model = model.cuda()
    model.share_memory()
    model_supervisor = model_utils.vrpSupervisor(model, args, plotter)

    if teacher:  # then load and quit function
        model_supervisor.load_pretrained(args.tl_teacher_model_path)
        print("loaded teacher model from checkpoint")
        return model_supervisor

    # the rest for the "normal model"

    if args.load_model:  # load complete model
        print("loaded model from checkpoint")
        model_supervisor.load_pretrained(args.load_model)

    elif args.load_partially:
        # first init randomly and then only overwrite respective model components
        model_supervisor.model.init_weights(args.param_init)
        model_dict = model_supervisor.model.state_dict()

        print(f"load parts of model from checkpoint {args.load_partially_model_file}")
        # load init model
        init_checkpoint = torch.load(args.load_partially_model_file)


        if args.warm_start:
            # sampled once since it needs to be equal for all parameters
            np.random.seed(42)
            noise = np.random.normal(0, args.perturb_sigma, 1)[0]

        # overwrite random init with respective loaded model components
        if args.init_policy:  # load rule policy
            rel_dict_part = {k: v for k, v in init_checkpoint.items() if "policy" in k}

            if args.warm_start:
                # shrink and perturb parameters
                rel_dict_part.update((x, y*args.shrink_lambda + noise) for x, y in rel_dict_part.items())
                print("loaded rule policy from checkpoint + perturbation")
            else:
                print("loaded rule policy from checkpoint")

            model_dict.update(rel_dict_part)
            model_supervisor.model.load_state_dict(model_dict)

            # 12 (policy) + 26 (encoder) + 8 (Q)

        if args.init_decoders:  # load decoders
            rel_dict_part = {k: v for k, v in init_checkpoint.items() if "encoder" in k}

            if args.warm_start:
                # shrink and perturb parameters
                rel_dict_part.update((x, y*args.shrink_lambda + noise) for x, y in rel_dict_part.items())
                print("loaded decoders from checkpoint + perturbation")
            else:
                print("loaded decoders from checkpoint")

            model_dict.update(rel_dict_part)
            model_supervisor.model.load_state_dict(model_dict)


        if args.init_Q_layers:  # load Q-layers (except for first one)
            rel_dict_part = {k: v for k, v in init_checkpoint.items() if
                             "value_estimator" in k and not "0.0.0.0.weight" in k and not "0.0.0.0.bias" in k}

            if args.warm_start:
                # shrink and perturb parameters
                rel_dict_part.update((x, y*args.shrink_lambda + noise) for x, y in rel_dict_part.items())
                print("loaded Q-layers (except first one) from checkpoint + perturbation")
            else:
                print("loaded Q-layers (except first one) from checkpoint")

            model_dict.update(rel_dict_part)
            model_supervisor.model.load_state_dict(model_dict)

    elif args.resume:
        pretrained = 'ckpt-' + str(args.resume).zfill(8)
        print('Resume from {} iterations.'.format(args.resume))
        # model_supervisor.load_pretrained(args.model_dir + '/' + pretrained)
        model_supervisor.load_pretrained('/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/logs/NewData_SameTops_5ag10n_7b465_00000_Z5_rewr30_gradClip0.05_eps0.15_alpha9e-06_lr0.0005_maxStepPoolFill6_penalVal10.0/' + pretrained)

           # '/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/logs/paper_00_central_5agents20n_c0c1c_00000_Z10_rewr40_gradClip0.05_eps0.15_alpha1e-06_lr0.0005_dstep200_drate0.9_maxStepsPoolFilled6_penaltyVal10.0' + '/' + pretrained)

    else:
        print('Created model with fresh parameters.')
        model_supervisor.model.init_weights(args.param_init)
    return model_supervisor


#@profile
def init_state(DataProcessor, model_supervisor, data, batch_idx, args, eval_flag, rel_dm_ids=None,
               init_route_nodes=None):
    # called for training and validation
    # contains one dm per vrp in batch

    # passed via args: heuristic, take_x_y, batch_size
    dm_list = DataProcessor.get_batch(data, args, init_route_nodes, batch_idx)


    num_agents = dm_list[0].num_agents

    # init variables for info tracking
    if eval_flag:
        team_avg_cost_rec = []  # not like in train tracking all solutions; but the last feasible one
        # rewrite rec enough if for one agent, since only interested in global action (naming out of simplicity kept)
        rewrite_rec_agents = [[] for _ in range(
            len(rel_dm_ids))]  # one list per vrp or interest --> later only save global actions to it
        dm_rec = [[] for _ in range(len(rel_dm_ids))]
        # if rel_dm_ids empty they will just be like []
    else:
        team_avg_cost_rec = [[] for _ in range(
            len(dm_list))]  # tracks team average costs for all solutions within rewriting episode
        rewrite_rec_agents = [[[] for _ in range(len(dm_list))] for _ in range(
            num_agents)]  # one global rewrite rec: information of rewriting history for each vrp in batch
        dm_rec = [[] for _ in
                  range(len(dm_list))]  # one global dm rec: rewriting history (=rewritten dms) for each vrp in batch

    #  compute node representations for start dms and add them to current list and history
    dm_list = model_supervisor.update_node_representations(dm_list, args.take_x_y)

    #  track local costs and team avg cost --> write to rewriting history of respective agent
    for dm_idx in range(len(dm_list)):
        # local_tot_cost_rec[dm_idx].append([dm_list[dm_idx].tot_costs[agent_id][-1] for agent_id in range(num_agents)])
        if eval_flag:
            team_avg_cost_rec.append(dm_list[dm_idx].team_avg_cost)  # init solution is always feasible
            if dm_idx in rel_dm_ids:
                dm_rec[rel_dm_ids.index(dm_idx)].append(dm_list[dm_idx])
        else:
            dm_rec[dm_idx].append(dm_list[dm_idx])
            team_avg_cost_rec[dm_idx].append(dm_list[dm_idx].team_avg_cost)

    return dm_list, team_avg_cost_rec, rewrite_rec_agents, dm_rec  # local_tot_cost_rec,


#@profile
def global_action_candidates(model_supervisor, dm_list, batch_idx, dm_ids_pool_filled, dm_pool_node_assignments,
                             num_agents, args, eval_flag):
    # needed for train and val
    batch_cand_l_actions_agents = []
    batch_cand_l_actions_rule_entropies_agents = []
    batch_cand_l_actions_rule_probs_agents = []
    batch_cand_l_actions_rule_logprobs_agents = []

    # check if temperature scaling on rule policy shall be applied (if wanted and if training process not too much progressed)
    if args.rule_temperature_scaling and model_supervisor.global_step < args.rule_temperature_scaling_batch_thresh:
        rule_temperature_scaling = True

        if args.rule_temperature_adaptive:
            rule_temperature = args.rule_temperature + model_supervisor.global_step * (
                        (1 - args.rule_temperature) / args.rule_temperature_scaling_batch_thresh)
        else:
            # always same temperature
            rule_temperature = args.rule_temperature

    else:
        rule_temperature_scaling = False
        rule_temperature = 1

    # each agent generates Z local candidate actions by calling the same rule policy
    for agent_id in range(num_agents):
        batch_cand_l_actions, batch_cand_l_actions_rule_entropies, batch_cand_l_actions_rule_probs, batch_cand_l_actions_rule_logprobs = \
            model_supervisor.candidate_local_actions(agent_id, batch_idx, dm_list, dm_ids_pool_filled,
                                                     dm_pool_node_assignments, args.allow_no_change_in_rule,
                                                     args.rew_all_possible_embeddings, args.take_x_y,
                                                     rule_temperature_scaling, rule_temperature,
                                                     eval_flag=eval_flag)
        batch_cand_l_actions_agents.append(batch_cand_l_actions)
        batch_cand_l_actions_rule_entropies_agents.append(batch_cand_l_actions_rule_entropies)
        batch_cand_l_actions_rule_probs_agents.append(batch_cand_l_actions_rule_probs)
        batch_cand_l_actions_rule_logprobs_agents.append(batch_cand_l_actions_rule_logprobs)

    # Z local candidate actions are put together to Z global candidate actions
    batch_cand_global_actions = []
    for j in range(len(dm_list)):  # receive local candidate actions from others
        # get respective vrp
        to_be_zipped = []  # len = 5, one entry: list of Z actions of respective agent
        for i in range(num_agents):
            to_be_zipped.append(batch_cand_l_actions_agents[i][
                                    j])  # agent idx, vrp index --> 10 possible actions by agent i
        global_action_z_list = list(
            map(list, zip(*to_be_zipped)))  # 10 action entries, each of size 5

        batch_cand_global_actions.append(global_action_z_list)

    return batch_cand_global_actions, batch_cand_l_actions_rule_entropies_agents, batch_cand_l_actions_rule_probs_agents, batch_cand_l_actions_rule_logprobs_agents


#@profile
def voting(model_supervisor, dm_list, batch_idx, batch_cand_global_actions, num_agents, args, batch_random_votes,
           eval_flag):
    # each global candidate action is centrally evaluated with global Q

    chosen_action_idxs_per_vrp, batch_pred_reward_list, batch_random_vote = model_supervisor.global_action_voting(
        batch_idx,
        batch_cand_global_actions,
        dm_list,
        args.epsilon_greedy,
        args.epsilon,
        eval_flag=eval_flag)
    # final decisions already
    if not eval_flag:
        batch_random_votes.append(batch_random_vote)

    return chosen_action_idxs_per_vrp, batch_pred_reward_list  # batch_random_votes doesn't need to be returned; changed in-place


#@profile
def rewriting(model_supervisor, dm_list, chosen_action_idxs_per_vrp, batch_cand_global_actions, batch_pred_reward_list,
              batch_cand_l_actions_rule_entropies_agents, batch_cand_l_actions_rule_probs_agents,
              batch_cand_l_actions_rule_logprobs_agents, dm_rec, rewrite_rec_agents, episode_rule_entropies,
              team_avg_cost_rec, num_agents, args, reduce_steps, eval_flag, val_rel_dm_ids=None,
              val_pool_usage_count=None, val_indices_last_feasible=None,
              val_given_to_pool_count=None):  # local_tot_cost_rec

    rewritten_dm_list, batch_rewriting_info_agents = model_supervisor.rewrite(
        dm_list,
        chosen_action_idxs_per_vrp,
        batch_cand_global_actions,
        batch_pred_reward_list,
        batch_cand_l_actions_rule_entropies_agents,
        batch_cand_l_actions_rule_probs_agents,
        batch_cand_l_actions_rule_logprobs_agents,
        args.take_x_y,
        eval_flag)
    # update node representations for rewritten vrps
    rewritten_dm_list = model_supervisor.update_node_representations(rewritten_dm_list, args.take_x_y)

    if eval_flag:
        val_team_avg_cost_last_feasible = team_avg_cost_rec

    # tracking variables
    for dm_idx in range(len(dm_list)):
        if eval_flag:

            # update last feasible val cost, if team avg cost is not nan
            if not np.isnan(rewritten_dm_list[dm_idx].team_avg_cost):  # i.e. pool is empty
                val_team_avg_cost_last_feasible[dm_idx] = rewritten_dm_list[dm_idx].team_avg_cost
                val_indices_last_feasible[dm_idx] = reduce_steps  # it's the rewriting step
            else:  # then nan, i.e. pool is filled
                # increase counter for pool usage of respective vrp
                # counting how often the pool is filled
                val_pool_usage_count[dm_idx] += 1

            # other pool usage statistic: don't count in how many states pool is filled, but how often a node is given to the pool
            chosen_global_action = batch_cand_global_actions[dm_idx][chosen_action_idxs_per_vrp[dm_idx]]
            for chosen_local_action in chosen_global_action:
                if chosen_local_action:  # if not empty
                    if chosen_local_action[0][0] != num_agents and chosen_local_action[1][
                        0] == num_agents:  # then action = giving local node to pool (and not rejecting a pool node offer)
                        val_given_to_pool_count[dm_idx] += 1

            # only for interesting val dms, rewrite_rec_agents only for one agent. Save only global action to it
            if dm_idx in val_rel_dm_ids:
                dm_rec[val_rel_dm_ids.index(dm_idx)].append(rewritten_dm_list[dm_idx])
                # batch rewriting info already only contains the global actions for the dms (rec for ONE agent)
                # record global candidate actions, corresponding Q values and chosen action

                if args.Z == 1:  # pred reward is none; old before:       # 0d array error, if 0-d array converted to list in case of Z = 1
                    cur_rel_info = [batch_cand_global_actions[dm_idx], batch_pred_reward_list[dm_idx],
                                    batch_rewriting_info_agents[dm_idx]]
                else:
                    cur_rel_info = [batch_cand_global_actions[dm_idx],
                                    list(batch_pred_reward_list[dm_idx].squeeze().data.cpu().numpy()),
                                    batch_rewriting_info_agents[dm_idx]]
                rewrite_rec_agents[val_rel_dm_ids.index(dm_idx)].append(
                    cur_rel_info)  # batch_rewriting_info_agents[dm_idx])
        else:
            # tracking during training since needed for loss computations
            team_avg_cost_rec[dm_idx].append(rewritten_dm_list[dm_idx].team_avg_cost)
            dm_rec[dm_idx].append(rewritten_dm_list[dm_idx])
            for agent_id in range(num_agents):
                rewrite_rec_agents[agent_id][dm_idx].append(batch_rewriting_info_agents[agent_id][dm_idx])
                episode_rule_entropies[agent_id][dm_idx].append(
                    batch_rewriting_info_agents[agent_id][dm_idx][-1])  # last entry in rewriting info is rule entropy

    return rewritten_dm_list

def update_pool_node_dicts(dm_list, pool_node_dicts, batch_cand_global_actions, chosen_action_idxs_per_vrp):
    # currenty only adding nodes; need to reset it somewhere if pool is empty
    pool_idx = dm_list[
        0].num_agents  # in global vehicle state; what's written in an action (index 2 for 2 agents which have 0 and 1)
    for vrp_idx, vrp_cand_actions in enumerate(batch_cand_global_actions):
        chosen_action = vrp_cand_actions[chosen_action_idxs_per_vrp[vrp_idx]]

        for agent_id, local_action in enumerate(chosen_action):
            if local_action:
                region_spec = local_action[0]
                rule_spec = local_action[1]
                region_node_id = dm_list[vrp_idx].vehicle_states[region_spec[0]][region_spec[1]]
                if rule_spec[0] == pool_idx:  # then add node id to the dict
                    pool_node_dicts[vrp_idx][region_node_id].add(
                        agent_id)  # add agent id who gave node to the pool (either giving it away or declining offer from pool)
                if region_spec[0] == pool_idx and rule_spec[
                    0] != pool_idx:  # if node from pool integrated by some agent, reset the respective node's entry
                    pool_node_dicts[vrp_idx][region_node_id] = set()

    return pool_node_dicts


#@profile
def get_team_avg_costs_per_vrp(team_avg_cost_rec):
    # team_avg_cost_rec: list per vrp. For one vrp: list of team avg costs of all solutions within rewriting episode
    # first select team avg cost from the rewriting episode per vrp: don't take best solution as before but the last feasible one
    avg_team_costs_per_vrp = []
    indices_last_feasible = []
    for vrp_avg_team_cost_history in team_avg_cost_rec:
        cur_hist = vrp_avg_team_cost_history.copy()
        count = 1  # since indexing starts at 0
        feas_sol_found = False
        while not feas_sol_found:
            if np.isnan(cur_hist[-1]):  # if last solution is not feasible since pool filled
                cur_hist = cur_hist[:-1].copy()
                count += 1
            else:
                avg_team_costs_per_vrp.append(cur_hist[-1])
                feas_sol_found = True
                last_feasible_index = int(len(vrp_avg_team_cost_history) - count)
                indices_last_feasible.append(last_feasible_index)
    return avg_team_costs_per_vrp, indices_last_feasible


def generate_rewriting_episode(data, batch_idx, DataProcessor, model_supervisor, args, eval_flag,
                               batch_interesting_dm_ids=None, init_route_nodes=None):
    # INITIALIZATION
    # init routes
    dm_list, team_avg_cost_rec, rewrite_rec_agents, dm_rec = init_state(
        DataProcessor, model_supervisor, data, batch_idx, args,
        eval_flag=eval_flag, rel_dm_ids=batch_interesting_dm_ids, init_route_nodes=init_route_nodes)

    # init team average costs
    init_team_avg_costs_batch = [dm.team_avg_cost for dm in dm_list]

    # init pool node dicts (one dict per vrp, no node in pool in init)
    num_agents = dm_list[0].num_agents
    pool_node_dict = {k: set() for k in range(
        dm_list[0].num_nodes - num_agents - 1)}  # get customer node IDS (-1 due to pool node)
    pool_node_dicts = [copy.deepcopy(pool_node_dict) for _ in range(args.batch_size)]

    # start rewriting episode
    active = True
    reduce_steps = 0  # rewriting step

    if eval_flag:
        batch_random_votes = None
        episode_rule_entropies = None
        eval_pool_usage_count_in_episode_batch = [0] * args.batch_size
        eval_given_to_pool_count_in_episode_batch = [0] * args.batch_size
        val_indices_last_feasible = [0] * args.batch_size  # init solution always feasible
    else:  # only relevant for training
        batch_random_votes = []  # track when eps-case was entered (in which an agent voted for a random global action from the candidate pool)
        episode_rule_entropies = [[[] for _ in range(args.batch_size)] for _ in range(num_agents)]
        eval_pool_usage_count_in_episode_batch = None
        eval_given_to_pool_count_in_episode_batch = None
        val_indices_last_feasible = None

    while active and (args.max_reduce_steps is None or reduce_steps < args.max_reduce_steps):
        reduce_steps += 1
        print(f"rewriting step {reduce_steps}")

        tic = time.perf_counter()
        # check status of pool
        dm_ids_pool_filled, dm_pool_node_assignments = model_supervisor.check_pool(dm_list, args.Z, pool_node_dicts)
        # generate global candidate actions
        batch_cand_global_actions, batch_cand_l_actions_rule_entropies_agents, batch_cand_l_actions_rule_probs_agents, batch_cand_l_actions_rule_logprobs_agents = global_action_candidates(
            model_supervisor, dm_list, batch_idx, dm_ids_pool_filled,
            dm_pool_node_assignments, num_agents, args, eval_flag=eval_flag)  # batch_idx not used in function

        # choose global action from candidates with Q; if there are candidates to choose from
        if args.Z != 1:
            chosen_action_idxs_per_vrp, batch_pred_reward_list = voting(model_supervisor, dm_list,
                                                                        batch_idx,
                                                                        batch_cand_global_actions,
                                                                        num_agents, args, batch_random_votes,
                                                                        eval_flag=eval_flag)
        else:
            # then only one candidate there; Q-values are not computed
            chosen_action_idxs_per_vrp = [0] * len(batch_cand_global_actions)
            batch_pred_reward_list = [[None]] * len(batch_cand_global_actions)

        # update pool node dicts (remember which agents are in contact with a pool from the node in the chosen action)
        pool_node_dicts = update_pool_node_dicts(dm_list, pool_node_dicts, batch_cand_global_actions,
                                                 chosen_action_idxs_per_vrp)
        # rewrite vrps
        rewritten_dm_list = rewriting(model_supervisor, dm_list,
                                      chosen_action_idxs_per_vrp,
                                      batch_cand_global_actions,
                                      batch_pred_reward_list,
                                      batch_cand_l_actions_rule_entropies_agents,
                                      batch_cand_l_actions_rule_probs_agents,
                                      batch_cand_l_actions_rule_logprobs_agents,
                                      dm_rec,
                                      rewrite_rec_agents, episode_rule_entropies,
                                      team_avg_cost_rec,
                                      num_agents, args, reduce_steps, eval_flag=eval_flag,
                                      val_rel_dm_ids=batch_interesting_dm_ids,
                                      val_pool_usage_count=eval_pool_usage_count_in_episode_batch,
                                      val_indices_last_feasible=val_indices_last_feasible,
                                      val_given_to_pool_count=eval_given_to_pool_count_in_episode_batch)
        # in train: batch_interesting_dm_ids=eval_pool_usage_count_in_episode_batch=None
        dm_list = rewritten_dm_list.copy()
    # done with rewriting episode
    if eval_flag:
        # for val:
        return dm_rec, rewrite_rec_agents, team_avg_cost_rec, eval_pool_usage_count_in_episode_batch, val_indices_last_feasible, init_team_avg_costs_batch, eval_given_to_pool_count_in_episode_batch
    else:
        return dm_rec, rewrite_rec_agents, team_avg_cost_rec, episode_rule_entropies


def train(args):

    print('Training:')

    if args.visdom:
        plotter = VisdomPlotter(env_name=args.experiment_name)  # created plotter, given to all agents' supervisors
    else:
        plotter = None

    global_time = time.time()

    # load training data
    train_data = data_utils.load_dataset(args.train_dataset, args)  # list of vrps; one vrp=list of agent dicts
    # contains 80.000 --> cuda memory
    train_data = train_data[:5000].copy()#[:5024].copy()
    train_data_size = len(train_data)

    num_agents = len(train_data[0])  # number of dicts for first vrp

    if args.train_proportion < 1.0:  # default 1.0; train on less samples than training set
        random.shuffle(train_data)
        train_data_size = int(train_data_size * args.train_proportion)
        train_data = train_data[:train_data_size]

    print(f'Number of train data samples: {train_data_size}')
    # load validation data
    if args.val_tracking:
        val_data = data_utils.load_dataset(args.val_dataset, args)  # contains 10.000
        val_data = val_data[:628].copy()  #val_data[5000:].copy()
        val_data_size = len(val_data)
        print(f'Number of val data samples: {val_data_size}')

    # take one global DataProcessor (provides batch data; one sample contains list of VRPManagers per agent)
    DataProcessor = data_utils.vrpDataProcessor()
    # a supervisor has a model, in central setting: only one supervisor needed  (num_agents needed for network dimensions)
    model_supervisor = create_model(args, plotter, num_agents)

    # TrainedFiveAg_10n = torch.load("/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/logs/paper_00_central_5agents10n_61429_00000_Z5_rewr30_gradClip0.05_eps0.15_alpha1e-05_lr0.0005_dstep200_drate0.9_maxStepsPoolFilled6_penaltyVal10.0/ckpt-00004700")

    if args.tl_entropy_distill:
        model_supervisor_teacher = create_model(args, plotter, args.tl_teacher_num_agents, teacher=True)
    else:
        model_supervisor_teacher = None

    # resume training or not
    if args.resume:
        resume_step = True
        val_team_avg_cost = 0  # necessary when starting from checkpoint if global step % args eval every n != 0; since at the end  tune reports this metric
    else:
        resume_step = False
    resume_idx = args.resume * args.batch_size

    # state and actions tracked currently only by agent 0 (not individually)
    # loggers = [model_utils.Logger(args, i) for i in range(num_agents)]
    logger = model_utils.Logger(args)  # here: logging folder is created

    # if training is continued from a checkpoint
    # ############!!!!!!!!!!!!!!!!needs ot be fixed later (log summary per agent) loggers[i]
    if args.resume:
        print("Fixing reading logs later - not needed now")


    avg_batch_time = 0
    batch_counter = 0

    if args.val_tracking:
        # compute OR Tools solution for val data: not changing over time so can be done once
        if num_agents > 1:
            print("Computing OR Tools solution for validation set")
            tic = time.perf_counter()
            val_ortools_team_avg_costs, val_or_tools_routes = Benchmark_nAgents.solve_or_tools(val_data)
            toc = time.perf_counter()
            inf_time_or_tools_val_avg_one_vrp = (float(toc - tic)) / len(val_data)

            interesting_val_dm_ids = [0]  # [1, 5, 50, 51, 52]
            for interesting_id in interesting_val_dm_ids:
                logger.track_state_actions(None, None, None, None, None, interesting_id, eval_flag=True,
                                           or_tools_flag=True, or_tools_route=val_or_tools_routes[interesting_id],
                                           or_tools_cost=val_ortools_team_avg_costs[interesting_id])

            # translate interesting val_dm_ids to batch
            batch_translated_interesting_val_dm_ids = [
                [int(int_id / args.batch_size) * args.batch_size, int_id % args.batch_size] for
                int_id in
                interesting_val_dm_ids]  # translate from interesting_eval_dm_ids to tuple (batch idx, id in batch)
            batch_idxs_with_interesting_val_dms = list(np.array(batch_translated_interesting_val_dm_ids).T[0])
            ids_in_batch_with_interesting_val_dms = list(np.array(batch_translated_interesting_val_dm_ids).T[1])

        else:
            raise ValueError('case not coded')

        # INITIALIZATION OF VAL ROUTES: for val important that they are always equally initialized. That's why done here once and then copies are used later one
        # only sample the node distributions, everything else is deterministic and we don't need to save all the dms
        tic = time.perf_counter()
        num_agents = len(val_data[0])
        customers = val_data[0][0]["customers"]
        num_customers = len(customers)
        num_customers_per_agent_min = int(num_customers / num_agents)
        init_route_nodes_val_vrps = [[] for _ in range(val_data_size)]
        for idx, vrp in enumerate(val_data):
            # init solution per vrp: random distribution of nodes first
            cur_customers = customers.copy()
            for i in range(num_agents):
                agent_nodes = random.sample(cur_customers, num_customers_per_agent_min)
                init_route_nodes_val_vrps[idx].append(agent_nodes)
                cur_customers = list(set(cur_customers) - set(agent_nodes))
            for j, left_customer in enumerate(cur_customers):
                # fill up agent routes: first left node to agent 0, second to agent 1, ...
                init_route_nodes_val_vrps[idx][j].append(left_customer)

        toc = time.perf_counter()
        init_vrp_inf_time_model_val_avg_one_vrp = (float(toc - tic)) / len(val_data)  # add this time later on always

    # actual training process starts: all in run_vrp.py since agents have to communicate and this has to happen here
    for epoch in range(resume_idx // train_data_size, args.num_epochs):
        random.shuffle(train_data)
        if args.visdom:
            plotter.text('Training', 'epoch', epoch)

        for batch_idx in range(0 + resume_step * resume_idx % train_data_size, train_data_size, args.batch_size):

            # batch_idx: then problems batch_idx - (batch_idx + batch_size are handled); 0: 0-32; 32: 32-64

            if (batch_idx+args.batch_size) > train_data_size:   # then less than batch-size many samples left
                cur_batch_size = train_data_size - batch_idx
            else:
                cur_batch_size = args.batch_size

            print(f"batch_idx: {batch_idx}, cur batchsize: {cur_batch_size}")
            # if val time
            if args.val_tracking:
                if model_supervisor.global_step % args.eval_every_n == 0:  # if model shall be evaluated on validation data
                    print("Validating")
                    # plot or-tools solution from above: just always newly plotted with global step such that line is plotted through (and can be easier compared with model)
                    if args.visdom:
                        if model_supervisor.global_step % 15 == 0:  # just take first vrp in batch (in val always the same, in train always different)
                            # plot always in advance computed or tools solution for val
                            model_supervisor.plotter.plot('team avg costs', 'val OR-Tools mean over batch',
                                                          'Team avg costs (team avg cost of one vrp = of last feasible sol within RE)',
                                                          model_supervisor.global_step,
                                                          np.mean(val_ortools_team_avg_costs),
                                                          "global step")
                            model_supervisor.plotter.plot('team avg costs', 'val OR-Tools max over batch',
                                                          'Team avg costs (team avg cost of one vrp = of last feasible sol within RE)',
                                                          model_supervisor.global_step,
                                                          np.max(val_ortools_team_avg_costs),
                                                          "global step")
                            model_supervisor.plotter.plot('run times', 'OR Tools val',
                                                          'inf run time avg one vrp(s)',
                                                          model_supervisor.global_step,
                                                          inf_time_or_tools_val_avg_one_vrp,
                                                          "global step")

                    if len(val_data) < 1000:  # if more then don't do it and don't compute correlation
                        val_team_avg_cost_list = []  # collect all solutions in order to compute the correlation

                    # batch-wise val
                    max_val_team_avg_cost_batch = 0
                    sum_val_team_avg_cost_all_batches = 0
                    sum_val_pool_usage_frac_per_step_all_batches = 0
                    sum_val_given_to_pool_all_batches = 0
                    sum_val_indices_last_feasible_all_batches = 0
                    inf_time_model_val_all_batches = 0.0  # inference time of model initialized

                    for val_batch_idx in range(0, val_data_size, args.batch_size):
                        # for each val batch
                        print(f"Processed {val_batch_idx} val vrp instances out of {val_data_size}")

                        batch_interesting_val_dm_ids = []
                        if val_batch_idx in batch_idxs_with_interesting_val_dms:
                            # run through since it could be present several times
                            for i, batch_i in enumerate(batch_idxs_with_interesting_val_dms):
                                if batch_i == val_batch_idx:
                                    batch_interesting_val_dm_ids.append(ids_in_batch_with_interesting_val_dms[i])

                        tic = time.perf_counter()
                        # val team avg cost = last feasible of vrp within re
                        val_dm_rec_int_dms, val_rewrite_rec_agent_int_dms, val_team_avg_cost_batch, val_pool_usage_count_in_episode_batch, val_indices_last_feasible, val_init_team_avg_costs_batch, val_given_to_pool_count_in_episode_batch = generate_rewriting_episode(
                            val_data, val_batch_idx, DataProcessor, model_supervisor, args,
                            eval_flag=True, batch_interesting_dm_ids=batch_interesting_val_dm_ids,
                            init_route_nodes=init_route_nodes_val_vrps)
                        toc = time.perf_counter()
                        inf_time_val_batch = toc - tic
                        inf_time_model_val_all_batches += inf_time_val_batch

                        for idx, interesting_id_in_batch in enumerate(batch_interesting_val_dm_ids):
                            # get original id in eval
                            interesting_id = val_batch_idx + interesting_id_in_batch
                            # here log batch idx of train
                            logger.track_state_actions(model_supervisor.global_step, epoch, batch_idx,
                                                       val_dm_rec_int_dms[idx],
                                                       val_rewrite_rec_agent_int_dms[idx], interesting_id,
                                                       eval_flag=True,
                                                       or_tools_flag=False)

                        # done with batch
                        max_val_team_avg_cost = max(max(val_team_avg_cost_batch), max_val_team_avg_cost_batch)
                        sum_val_team_avg_cost_all_batches += sum(val_team_avg_cost_batch)
                        sum_val_indices_last_feasible_all_batches += sum(val_indices_last_feasible)
                        sum_val_pool_usage_frac_per_step_all_batches += sum(
                            [float(counter) / args.max_reduce_steps for counter in
                             val_pool_usage_count_in_episode_batch])
                        sum_val_given_to_pool_all_batches += sum(val_given_to_pool_count_in_episode_batch)
                        if len(val_data) < 1000:
                            val_team_avg_cost_list.extend(val_team_avg_cost_batch)

                    # done with all batches
                    inf_time_model_val_avg_one_vrp = inf_time_model_val_all_batches / val_data_size
                    val_team_avg_cost = sum_val_team_avg_cost_all_batches / val_data_size
                    val_indices_last_feasible_mean = sum_val_indices_last_feasible_all_batches / val_data_size
                    avg_val_frac_pool_usage_in_episode = sum_val_pool_usage_frac_per_step_all_batches / val_data_size  # tells me: in x% of the episode the pool was filled (in 8 out of 30 steps on average...)
                    avg_val_given_to_pool_count_in_episode = sum_val_given_to_pool_all_batches / val_data_size  # on average per vrp: x times given something to pool within the episode (not normalized)

                    if args.visdom:
                        # plot avg team avg costs on validation data
                        model_supervisor.plotter.plot('team avg costs', 'val model mean over batch',
                                                      'Team avg costs (team avg cost of one vrp = of last feasible sol within RE)',
                                                      model_supervisor.global_step,
                                                      val_team_avg_cost,
                                                      # average (team avg) costs over batch  #val_avg_team_costs_per_vrp
                                                      "global step")
                        model_supervisor.plotter.plot('team avg costs', 'val model max over batch',
                                                      'Team avg costs (team avg cost of one vrp = of last feasible sol within RE)',
                                                      model_supervisor.global_step,
                                                      max_val_team_avg_cost,
                                                      # max (team avg) cost over batch   # val_avg_team_costs_per_vrp
                                                      "global step")

                        # plot idx of last feasible solution (mean over batch); in the end: this should increase, in the meantime can be low since pool not emptied and only unfeasible solutions constructed
                        model_supervisor.plotter.plot('idx of last feasible solution', 'val model mean over batch',
                                                      'idx of last feasible solution',
                                                      model_supervisor.global_step, val_indices_last_feasible_mean,
                                                      # it's the mean over batch (dynamically comp)
                                                      # in between [0, num_rewriting_steps]
                                                      "global step")

                        # plot correlation between costs of val or tools and val model and show median of val costs (to see how many outliers there are)
                        if len(val_data) < 1000:
                            # compute corr between or-tools and model results
                            val_corr = np.corrcoef(val_team_avg_cost_list, val_ortools_team_avg_costs)[0][1]
                            model_supervisor.plotter.plot('corr team avg cost model or-tools', 'val',
                                                          'corr model or-tools)',
                                                          model_supervisor.global_step,
                                                          val_corr,
                                                          "global step")

                            model_supervisor.plotter.plot('team avg costs', 'val model median over batch',
                                                          'Team avg costs (team avg cost of one vrp = of last feasible sol within RE)',
                                                          model_supervisor.global_step,
                                                          np.median(val_team_avg_cost_list),
                                                          "global step")

                            # diff_model_orTools = list(np.array(val_team_avg_cost_list)-np.array(val_ortools_team_avg_costs))
                            # for diff_val in diff_model_orTools:
                            #    model_supervisor.plotter.scatter('diff model - ortools', 'val',
                            #                                  'Diff: Model - OrTools',
                            #                                  model_supervisor.global_step,
                            #                                  diff_val,
                            #                                  "global step")

                        # plot frac of pool usage within episode averaged over batch in val (i.e. count nans in val_team_avg_cost_rec)
                        model_supervisor.plotter.plot('frac pool usage', 'val',
                                                      'frac pool usage per episode avg over batch',
                                                      model_supervisor.global_step,
                                                      avg_val_frac_pool_usage_in_episode,
                                                      "global step")

                        model_supervisor.plotter.plot('avg num give-to-pool-actions within episode', 'val',
                                                      'avg num give-to-pool-actions within episode',
                                                      model_supervisor.global_step,
                                                      avg_val_given_to_pool_count_in_episode,
                                                      "global step")

                        # plot inf time of model average per vrp on val dataset
                        model_supervisor.plotter.plot('run times', 'model val',
                                                      'inf run time avg one vrp(s)',
                                                      model_supervisor.global_step,
                                                      inf_time_model_val_avg_one_vrp,
                                                      "global step")

            # training process: one batch
            # compute or-tools results for vrps in batch
            print("Computing OR-Tools Results For Train Batch")
            if num_agents > 1:
                batch_ortools_team_avg_costs, _ = Benchmark_nAgents.solve_or_tools(
                    train_data[batch_idx:min((batch_idx + args.batch_size), train_data_size)])
                if args.visdom:
                    model_supervisor.plotter.plot('team avg costs', 'train OR-Tools mean over batch',
                                                  'Team avg costs (team avg cost of one vrp = of last feasible sol within RE)',
                                                  model_supervisor.global_step,
                                                  np.mean(batch_ortools_team_avg_costs),
                                                  "global step")
                    model_supervisor.plotter.plot('team avg costs', 'train OR-Tools max over batch',
                                                  'Team avg costs (team avg cost of one vrp = of last feasible sol within RE)',
                                                  model_supervisor.global_step,
                                                  np.max(batch_ortools_team_avg_costs),
                                                  "global step")

            # at the end: total_loss and total_reward for this batch
            print(f"epoch {epoch}, batch_idx {batch_idx}")
            train_time_start = time.time()

            dm_rec, rewrite_rec_agents, team_avg_cost_rec, episode_rule_entropies = generate_rewriting_episode(
                train_data, batch_idx, DataProcessor, model_supervisor, args, eval_flag=False)

            # now: EPISODES OF REWRITING STEPS GENERATED for all vrps in batch
            # compute rule entropy per vrp in batch (= avg over rewriting steps and avg over agents)
            agent_batch_rule_entropies = []
            for agent_id in range(num_agents):
                # take mean of entropies per rewriting step for each vrp
                cur_agent_batch_rule_entropies = []
                for j in range(cur_batch_size):
                    cur_agent_batch_rule_entropies.append(np.nanmean(episode_rule_entropies[agent_id][j]))
                agent_batch_rule_entropies.append(cur_agent_batch_rule_entropies)
            # one entropy per vrp (aggregated over rewriting steps and agents)
            batch_rule_entropies = list(np.mean(agent_batch_rule_entropies, axis=0))

            # get team avg cost per vrp from rewriting episode by taking the value (=team avg cost) of last feasible solution
            avg_team_costs_per_vrp, indices_last_feasible = get_team_avg_costs_per_vrp(team_avg_cost_rec)
            # average (team avg) costs over batch
            batch_avg_team_avg_costs = np.mean(avg_team_costs_per_vrp)

            counters_init_better = []
            # if final (= last feasible) solution is better than init solution, then count
            for dm_idx in range(cur_batch_size):
                counter = 0
                if team_avg_cost_rec[dm_idx][0] <= avg_team_costs_per_vrp[dm_idx]:  # then init better than final
                    counter += 1
                counters_init_better.append(counter)

            if args.visdom:
                # plot mean (over batch) team avg costs
                model_supervisor.plotter.plot('team avg costs', 'train model mean over batch',
                                              'Team avg costs (team avg cost of one vrp = of last feasible sol within RE)',
                                              model_supervisor.global_step, batch_avg_team_avg_costs,
                                              "global step")
                # plot idx of last feasible solution (mean over batch); in the end: this should increase, in the meantime can be low since pool not emptied and only unfeasible solutions constructed
                model_supervisor.plotter.plot('idx of last feasible solution', 'train model mean over batch',
                                              'idx of last feasible solution',
                                              model_supervisor.global_step, np.mean(indices_last_feasible),
                                              # in between [0, num_rewriting_steps]
                                              "global step")

                # plot fraction how often init solution is better than the final solution (= last feasible one) within batch
                model_supervisor.plotter.plot('frac init is better than or same as final (= last feas.) in batch',
                                              'train ',
                                              'frac init better than final',
                                              model_supervisor.global_step, np.mean(counters_init_better) * 100,
                                              "global step")


            # global training step - compute loss
            model_supervisor.model.dropout_rate = model_supervisor.dropout_rate
            model_supervisor.model.optimizer.zero_grad()  # all gradients zero
            avg_total_loss, avg_rule_loss, avg_Q_loss, track_pred_reward, track_obs_reward, A_track_1st_batch_vrp, batch_advantage_track_probs, batch_advantage_track_Qs, batch_pool_penalties, batch_normal_rewards, avg_final_policy_loss, avg_transfer_loss, batch_H_track, batch_R_track = \
                model_supervisor.train_batch(dm_rec,
                                             team_avg_cost_rec,
                                             rewrite_rec_agents, epoch, args, batch_rule_entropies,
                                             model_supervisor_teacher)

            if args.tl_entropy_distill and model_supervisor.global_step < args.tl_entropy_distill_batch_thresh:
                sum_Hs = 0
                sum_Rs = 0

                count_n = 0
                # H and R computed for same actions (sum over all agent values for the whole episode for all vrps in batch IF it's a local re-ordering action / pool action)
                for vrp_id, vrp in enumerate(batch_H_track):
                    for t_step, H_t_step in enumerate(vrp):
                        if H_t_step:
                            count_n += 1
                            sum_Hs += sum(H_t_step)
                            sum_Rs += sum(batch_R_track[vrp_id][t_step])


                if args.visdom:
                    if count_n > 0:
                        sum_Hs = torch.tensor(sum_Hs, dtype=torch.float).cuda()
                        sum_Rs = torch.tensor(sum_Rs, dtype=torch.float).cuda()

                        model_supervisor.plotter.plot('avg H entropy value in batch', 'global',
                                                      'avg H entropy value in batch - Training',
                                                      model_supervisor.global_step, sum_Hs.item() / count_n,
                                                      "global step")

                        model_supervisor.plotter.plot('avg R entropy value in batch', 'global',
                                                      'avg R entropy value in batch - Training',
                                                      model_supervisor.global_step, sum_Rs.item() / count_n,
                                                      "global step")

                    model_supervisor.plotter.plot('avg rule loss', f'{args.tl_loss_weight}* transfer loss',
                                                  'Avg rule policy loss - Training',
                                                  model_supervisor.global_step,
                                                  args.tl_loss_weight * avg_transfer_loss.item(),
                                                  "global step")
                    # summed up transfer and pure model
                    model_supervisor.plotter.plot('avg rule loss', 'final loss', 'Avg rule policy loss - Training',
                                                  model_supervisor.global_step, avg_final_policy_loss.item(),
                                                  "global step")

            # y-axis, label, titel of plot
            if args.visdom:
                # total loss = rule + Q (if TL distill enabled, then it includes final rule policy loss including the transfer loss)
                model_supervisor.plotter.plot('avg total loss', f'global', 'Avg total loss - Training',
                                              model_supervisor.global_step, avg_total_loss.item(),
                                              "global step")
                model_supervisor.plotter.plot('avg rule loss', 'pure model', 'Avg rule policy loss - Training',
                                              model_supervisor.global_step, avg_rule_loss.item(),
                                              "global step")

                # model_supervisors[i].plotter.plot('avg rule loss', f'agent{i} with entro penalty',
                #                                  'Avg rule policy loss - Training',
                #                                  model_supervisors[i].global_step, avg_penalty_rule_loss.item(),
                #                                  "global step")
                model_supervisor.plotter.plot('avg Q action loss', f'global', 'Avg action loss - Training',
                                              model_supervisor.global_step, avg_Q_loss.item(),
                                              "global step")  # not region; whole action

                # plot rule picking entropies per agent
                model_supervisor.plotter.plot('rule entropy', f'global', 'Rule entropy avg batch (train)',
                                              model_supervisor.global_step,
                                              np.mean(batch_rule_entropies),
                                              "global step")  # these mean values could be tracked; currently not done (just computed and plotted)

                # plot frac occurence of penalties in episode avg over batch
                batch_pool_penalties_01 = [(1 / args.penalty_val) * elem for vrp in batch_pool_penalties for elem in
                                           vrp]
                avg_batch_pool_penalties = np.mean(batch_pool_penalties_01)
                model_supervisor.plotter.plot('frac pool penalties', f'train',
                                              'frac pool penalties within re-ep. avg over batch',
                                              model_supervisor.global_step,
                                              avg_batch_pool_penalties,
                                              "global step")

                if model_supervisor.global_step % 30 == 0:
                    # plot value of advantage function for first vrp in batch
                    # log corresponding rewriting rec
                    logger.track_state_actions(model_supervisor.global_step, epoch, batch_idx, dm_rec[0],
                                               rewrite_rec_agents[0][0], 0, eval_flag=False, or_tools_flag=False)


            model_supervisor.global_step += 1
            # loss backward (before: check that avg total loss != float; only then training update)
            avg_total_loss.backward(retain_graph=True)

            all_params = list(model_supervisor.model.parameters())
            l1 = len(all_params)
            all_params_2 = [p for p in all_params if p.grad is not None]
            l2 = len(all_params_2)

            # plot gradient norms per layer per model
            if args.visdom:
                # norm used for clip grad norm
                all_params = list(model_supervisor.model.parameters())
                l1 = len(all_params)
                all_params = [p for p in all_params if p.grad is not None]
                l2 = len(all_params)
                device = all_params[0].grad.device
                total_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in all_params]),
                    2.0)
                model_supervisor.plotter.plot('whole model grad norm',
                                              f'grad norm before clipping',
                                              'whole model grad norm',
                                              model_supervisor.global_step,
                                              total_norm.item(),
                                              "global step")

                # check new model is training
                for k, cur_lay_ps in enumerate(list(model_supervisor.model.state_encoder.parameters())):
                    cur_lay_norm = cur_lay_ps.grad.data.norm(2).item()
                    model_supervisor.plotter.plot('state encoder layer grad norms',
                                                      f'layer/bias{k}',
                                                      'state encoder layers grad norms',
                                                      model_supervisor.global_step,
                                                      cur_lay_norm,
                                                      "global step")
                for k, cur_lay_ps in enumerate(list(model_supervisor.model.state_encoder_no_cost.parameters())):
                    cur_lay_norm = cur_lay_ps.grad.data.norm(2).item()
                    model_supervisor.plotter.plot('state encoder no cost layer grad norms',
                                                      f'layer/bias{k}',
                                                      'state encoder no cost layers grad norms',
                                                      model_supervisor.global_step,
                                                      cur_lay_norm,
                                                      "global step")

            # global training update / model update:
            new_norm = model_supervisor.model.train()

            if args.lr_decay_steps and model_supervisor.global_step % args.lr_decay_steps == 0:
                model_supervisor.model.lr_decay(args.lr_decay_rate)

            if model_supervisor.global_step % 100 == 0:  # save model every 100 global steps
                model_supervisor.save_model()

            if args.visdom:
                model_supervisor.plotter.plot('whole model grad norm',
                                              f'grad norm after clipping',
                                              'whole model grad norm',
                                              model_supervisor.global_step,
                                              new_norm.item(),
                                              "global step")
                plotter.text('Training', 'global step', model_supervisor.global_step)  # added by markus
                plotter.text('Training', 'data samples',
                             str(batch_idx + args.batch_size) + '/' + str(train_data_size))  # added by markus
                plotter.text('Training', 'global time', time.time() - global_time)  # added by markus
                batch_time = time.time() - train_time_start  # added by markus
                batch_counter += 1  # added by markus
                avg_batch_time += (batch_time - avg_batch_time) / batch_counter  # moving average  #added by markus
                plotter.text('Training', 'last batch time', batch_time)  # added by markus
                plotter.text('Training', 'avg batch time', avg_batch_time)  # added by markus
                plotter.print_texts()

            if args.ray:
                if args.val_tracking:
                    tune.report(val_avg_cost=val_team_avg_cost)
                else:
                    tune.report(val_avg_cost=0)

        # when done with resuming epoch; reset resume_step such that all batches are considered for one epoch

        if resume_step:
            resume_step = False

    # when complete training is done; save plots

    if args.visdom:
        plotter.viz.save([args.experiment_name])


def evaluate(args):  # runs inference of a checkpointed model

    print('Evaluation:')
    if args.visdom:
        print("Visualized in Visdom")
        plotter = VisdomPlotter(env_name=args.experiment_name)  # created plotter, given to all agents' supervisors
    else:  # default: visdom false
        print("No Visdom")
        plotter = None

    test_data = data_utils.load_dataset(args.test_dataset, args)
    test_data = test_data[:628].copy()  #test_data[5000:].copy()
    test_data_size = len(test_data)
    print(f"test data size: {test_data_size}")
    num_agents = len(test_data[0])

    DataProcessor = data_utils.vrpDataProcessor()

    if not args.alternative_scenario: # alternative scenario just computes tsp solutions of the final model node assignment solutions; no model to be loaded

        # loads model from args.load_model
        model_supervisor = create_model(args, plotter, num_agents)  #usually: num_agents (for generalizing agents adapt number here)

        if not args.debug:   # no logging in debug mode
            logger = model_utils.Logger(args) # here: logging folder created

    # logged vrps HERE SET!
    interesting_eval_dm_ids = list(
        np.arange(test_data_size))  # 20 testdatasize #[7, 8]#, 103, 233, 239, 268, 283, 385, 387, 439, 472, 515, 559]
    log_dm_ids = interesting_eval_dm_ids  # list(np.arange(300))
    # translate to batch_idx and idx in batch
    batch_translated_interesting_eval_dm_ids = [
        [int(int_id / args.batch_size) * args.batch_size, int_id % args.batch_size] for
        int_id in
        interesting_eval_dm_ids]  # translate from interesting_eval_dm_ids to tuple (batch idx, id in batch)
    # 10 Probleme, bs=2: [0,4,5,7] --> batch id int(:2) und id in batch % 2, da nicht batch gezählt sondern in args.batchsize schritten: batch idx = int(:2)*2
    batch_idxs_with_interesting_eval_dms = list(np.array(batch_translated_interesting_eval_dm_ids).T[0])
    ids_in_batch_with_interesting_eval_dms = list(np.array(batch_translated_interesting_eval_dm_ids).T[1])

    # init of eval routes: needs to be equally initialized for each run (and also for OR-Tools); here: distribute nodes to agents
    tic = time.perf_counter()
    customers = test_data[0][0]["customers"]
    num_customers = len(customers)
    num_customers_per_agent_min = int(num_customers / num_agents)

    if args.path_init_route_nodes_eval:
        print("Reading init nodes from file")
        init_route_nodes_eval_vrps = pickle.load(open(args.path_init_route_nodes_eval, "rb"))
    else:
        print("Randomly distributing init nodes")
        init_route_nodes_eval_vrps = [[] for _ in range(test_data_size)]
        for idx, vrp in enumerate(test_data):
            # init solution per vrp: random distribution of nodes first
            cur_customers = customers.copy()
            for i in range(num_agents):
                agent_nodes = random.sample(cur_customers, num_customers_per_agent_min)
                init_route_nodes_eval_vrps[idx].append(agent_nodes)
                cur_customers = list(set(cur_customers) - set(agent_nodes))
            for j, left_customer in enumerate(cur_customers):
                # fill up agent routes: first left node to agent 0, second to agent 1, ...
                init_route_nodes_eval_vrps[idx][j].append(left_customer)

        if not args.debug:  # no logging in debug mode
            logger.save_init_nodes(init_route_nodes_eval_vrps)

    toc = time.perf_counter()
    init_vrp_inf_time_model_eval_avg_one_vrp = (float(toc - tic)) / len(test_data)  # add this time later on always

    # for alternative scenario: solve TSP per agent based on initial node assignment
    if args.alternative_scenario:

        # new: check whether local rewriting worked in final solutions
        # only thing which needs to be set: final solutions + test data (model etc. irrelevant; not needed here)
        with open('../new_data_localRewritingSuccess/final_solutions_5agents10n_TL_WeighPool_9021d_00001.pkl', 'rb') as f:
            final_solutions = pickle.load(f)
        # evaluate costs of individual agent routes produced by MANR
        final_solutions_agent_costs = []
        for vrp_id, sol in enumerate(final_solutions):
            agent_costs = []
            for a_id, agent_route in enumerate(sol[:-1]):  # exclude pool state
                if len(agent_route) < 3:  # then no node visited
                    agent_costs.append(0)
                else:
                    # compute cost
                    agent_cost = 0
                    for n_id, node in enumerate(agent_route[:-1]):  # get cost to successor
                        suc_node = agent_route[n_id + 1]
                        vrp_agent_info = test_data[vrp_id][a_id]
                        if node in vrp_agent_info["depots"]:
                            node_cost_matrix_id = len(
                                vrp_agent_info["customers"])  # depot of agent is always last entry in cost matrix
                        else:
                            node_cost_matrix_id = node

                        if suc_node in vrp_agent_info["depots"]:
                            suc_node_cost_matrix_id = len(
                                vrp_agent_info["customers"])  # depot of agent is always last entry in cost matrix
                        else:
                            suc_node_cost_matrix_id = suc_node

                        agent_cost += vrp_agent_info["costs"][node_cost_matrix_id][suc_node_cost_matrix_id]

                    agent_costs.append(agent_cost)
            final_solutions_agent_costs.append(agent_costs)

        # compute TSP solution with assigned nodes
        tsp_solutions = []
        tsp_solutions_agent_costs = []
        for vrp_id, final_solution in enumerate(final_solutions):
            # for one vrp in the test data
            vrp_tsp_solutions = []
            vrp_tsp_costs = []
            for agent_id, agent_solution in enumerate(final_solution[:-1]):  # exclude pool state
                # call tsp for each agent route in the current vrp
                if len(agent_solution) < 3:  # then no node visited, i.e. no tsp to be computed
                    vrp_tsp_costs.append(0)
                    vrp_tsp_solutions.append([])
                elif len(agent_solution) == 3:  # one node visited; no order to optimize
                    vrp_tsp_costs.append(np.nan)
                    vrp_tsp_solutions.append([np.nan])
                else:
                    # call tsp
                    cost, route = Benchmark_1Agent_TSP_AltScenario.solve_or_tools(test_data[vrp_id][agent_id],
                                                                                  agent_solution[
                                                                                  1:-1])  # exclude start and end depot
                    vrp_tsp_costs.append(cost[0])  # cost returned in a list
                    vrp_tsp_solutions.append(route[0])  # route returned in a list

            tsp_solutions.append(vrp_tsp_solutions)
            tsp_solutions_agent_costs.append(vrp_tsp_costs)

        with open('../new_data_localRewritingSuccess/final_solutions_5agents10n_TL_WeighPool_9021d_00001_costs.pkl', 'wb') as f:
            pickle.dump(final_solutions_agent_costs, f)
        with open('../new_data_localRewritingSuccess/tsp_routes_of_final_solutions_5agents10n_TL_WeighPool_9021d_00001.pkl', 'wb') as f:
            pickle.dump(tsp_solutions, f)
        with open('../new_data_localRewritingSuccess/tsp_routes_of_final_solutions_5agents10n_TL_WeighPool_9021d_00001_costs.pkl', 'wb') as f:
            pickle.dump(tsp_solutions_agent_costs, f)

        # no collab setting in paper:
        alt_scenario_agent_costs_all_agents = []

        for agent_id in range(num_agents):
            agent_data = [vrp[agent_id] for vrp in test_data]
            # nodes from initial assignment relevant for the TSP
            agent_nodes = [vrp_init[agent_id] for vrp_init in init_route_nodes_eval_vrps]
            alt_scenario_agent_costs = Benchmark_1Agent_TSP_AltScenario.solve_or_tools(agent_data, agent_nodes)
            alt_scenario_agent_costs_all_agents.append(alt_scenario_agent_costs)

        mean_agent_costs_no_collaboration_init_given = np.mean(alt_scenario_agent_costs_all_agents, axis=0).tolist()
        mean_agent_costs_no_collaboration_test_mean = np.mean(mean_agent_costs_no_collaboration_init_given)
        # done for paper: just stop debugger here and write out mean_agent_costs_no_collaboration_test_mean
        dummy_test = 4

    # save results to df (saved under logs / experiment name)
    eval_results = pd.DataFrame(
        columns=['team avg cost - mean test data', 'team avg cost - max test data', 'run time avg one vrp',
                 'init team avg cost - mean test data', 'percentage cost reduction from init - mean test data',
                 'percentage cost reduction from init - median test data',
                 'percentage cost reduction from init - 30%-quantile test data',
                 'percentage cost reduction from init - 70%-quantile test data', 'frac_oneAgentEverything', 'corr',
                 'frac model better', 'vrp ids model better', 'Num_steps_Done_Nothing - mean test',
                 'Frac_steps_Done_Nothing', 'Frac_last_infeasible'],
        index=['or-tools'] + [f'model_run_{i}' for i in range(args.inf_num_runs)])

    # done with model inference; now OR-Tools: compute OR Tools solution for eval data:
    if num_agents > 1:
        # get init routes based on the init node distribution; same as for model
        initial_routes = []
        for batch_idx in range(0, test_data_size, args.batch_size):
            dm_list_init = DataProcessor.get_batch(test_data, args, init_route_nodes_eval_vrps, batch_idx)
            for dm in dm_list_init:
                cur_init = []
                for agent_route in dm.vehicle_states[:-1]:  # exclude pool
                    cur_init.append(agent_route[1:-1])  # exclude depots since OR-Tools syntax requires that
                initial_routes.append(cur_init)

        print("Computing OR Tools solution for evaluation set")
        tic = time.perf_counter()
        eval_ortools_team_avg_costs, eval_or_tools_routes = Benchmark_nAgents.solve_or_tools(test_data, initial_routes)
        toc = time.perf_counter()
        inf_time_or_tools_eval_avg_one_vrp = (float(toc - tic)) / len(test_data)

        count_oneAgentEverything = 0
        for sol in eval_or_tools_routes:
            route_lengths = [len(route) for route in sol]
            if route_lengths.count(2) == num_agents - 1:
                count_oneAgentEverything += 1

        frac_oneAgentEverything = count_oneAgentEverything / len(eval_or_tools_routes)

        # save OR-tools results
        eval_results.loc['or-tools'] = [np.mean(eval_ortools_team_avg_costs),
                                        np.max(eval_ortools_team_avg_costs), inf_time_or_tools_eval_avg_one_vrp,
                                        None, None, None, None, None,  # percentage cost reduction from init to final
                                        frac_oneAgentEverything,
                                        None, None, None, None, None, None]

        if not args.debug:  # no logging in debug mode
            for interesting_id in interesting_eval_dm_ids:
                if interesting_id in log_dm_ids:
                    logger.track_state_actions(None, None, None, None, None, interesting_id, eval_flag=True,
                                               or_tools_flag=True,
                                               or_tools_route=eval_or_tools_routes[interesting_id],
                                               or_tools_cost=eval_ortools_team_avg_costs[interesting_id])


    else:
        raise ValueError('case not coded')

    if args.visdom:
        # plot computed or tools solution for eval
        model_supervisor.plotter.plot('team avg costs', 'eval OR-Tools mean',
                                      'Team avg costs (team avg cost of one vrp = of last feasible sol within RE)',
                                      0,
                                      np.mean(eval_ortools_team_avg_costs),
                                      "one time evaluation")
        model_supervisor.plotter.plot('team avg costs', 'eval OR-Tools max',
                                      'Team avg costs (team avg cost of one vrp = of last feasible sol within RE)',
                                      0,
                                      np.max(eval_ortools_team_avg_costs),
                                      "one time evaluation")
        model_supervisor.plotter.plot('run times', 'OR Tools eval',
                                      'inf run time avg one vrp(s)',
                                      0,
                                      inf_time_or_tools_eval_avg_one_vrp,
                                      "one time evaluation")

    for run_id in range(args.inf_num_runs):
        if len(test_data) < 1000:  # if more then don't do it and don't compute correlation
            eval_team_avg_cost_list = []  # collect all solutions in order to compute the correlation and improvement over init solution
            eval_init_team_avg_cost_list = []

        max_eval_team_avg_cost = 0
        sum_eval_team_avg_cost_all_batches = 0  # always last feasible
        sum_eval_pool_usage_frac_per_step_all_batches = 0
        sum_eval_indices_last_feasible_all_batches = 0
        inf_time_model_eval_all_batches = 0.0  # inference time of model initialized

        count_oneAgentEverything_m = 0  # count for all vrps in test data
        count_AllDoneNothing_Within_Episode = 0  # count for all vrps in test data
        count_lastInfeasible = 0  # count for all vrps in test data

        for batch_idx in range(0, test_data_size, args.batch_size):
            print(f"Run {run_id}: Processed {batch_idx} vrp instances out of {len(test_data)}")

            batch_interesting_eval_dm_ids = []
            if batch_idx in batch_idxs_with_interesting_eval_dms:
                # run through since it could be present several times
                for i, batch_i in enumerate(batch_idxs_with_interesting_eval_dms):
                    if batch_i == batch_idx:
                        batch_interesting_eval_dm_ids.append(ids_in_batch_with_interesting_eval_dms[i])

            tic = time.perf_counter()
            # rewriting episode for vrps in batch
            eval_dm_rec_int_dms, eval_rewrite_rec_agent_int_dms, eval_team_avg_cost_batch, eval_pool_usage_count_in_episode_batch, eval_indices_last_feasible, eval_init_team_avg_costs_batch, eval_given_to_pool_count_in_episode_batch = generate_rewriting_episode(
                test_data, batch_idx, DataProcessor, model_supervisor, args, eval_flag=True,
                batch_interesting_dm_ids=batch_interesting_eval_dm_ids, init_route_nodes=init_route_nodes_eval_vrps)
            toc = time.perf_counter()
            inf_time_eval_batch = toc - tic
            inf_time_model_eval_all_batches += inf_time_eval_batch

            if len(interesting_eval_dm_ids) == test_data_size:  # if all vrp recs tracked
                # how often one agent everything AND
                # how often all done nothing within episode
                for vrp in eval_dm_rec_int_dms:
                    last_solution = vrp[30].vehicle_states  # assumes that it's feasible...
                    route_lengths_m = [len(route) for route in last_solution]
                    if route_lengths_m[-1] != 1:  # then pool filled
                        count_lastInfeasible += 1
                    if route_lengths_m.count(2) == num_agents - 1:  # if 2 in route_lengths_m:
                        count_oneAgentEverything_m += 1

                    for rewritten_id, rewritten_vrp in enumerate(vrp[:-1]):  # exclude last one since id+1 considered
                        if vrp[rewritten_id].vehicle_states == vrp[rewritten_id + 1].vehicle_states:
                            count_AllDoneNothing_Within_Episode += 1

            if not args.debug:  # no logging in debug mode
                for idx, interesting_id_in_batch in enumerate(batch_interesting_eval_dm_ids):
                    # get original id in eval
                    interesting_id = batch_idx + interesting_id_in_batch
                    if interesting_id in log_dm_ids:
                        logger.track_state_actions("inf", "inf", "inf", eval_dm_rec_int_dms[idx],
                                                   eval_rewrite_rec_agent_int_dms[idx], interesting_id, eval_flag=True,
                                                   or_tools_flag=False)  # global step, epoch, batch_idx = "inf"

            # done with batch
            max_eval_team_avg_cost = max(max(eval_team_avg_cost_batch), max_eval_team_avg_cost)
            sum_eval_team_avg_cost_all_batches += sum(eval_team_avg_cost_batch)
            sum_eval_indices_last_feasible_all_batches += sum(eval_indices_last_feasible)
            sum_eval_pool_usage_frac_per_step_all_batches += sum(
                [float(counter) / args.max_reduce_steps for counter in eval_pool_usage_count_in_episode_batch])
            if len(test_data) < 1000:
                eval_team_avg_cost_list.extend(eval_team_avg_cost_batch)
                eval_init_team_avg_cost_list.extend(eval_init_team_avg_costs_batch)

        # done with all batches
        inf_time_model_eval_avg_one_vrp = inf_time_model_eval_all_batches / len(test_data)
        # add inference time for initial distribution of nodes to agents
        inf_time_model_eval_avg_one_vrp += init_vrp_inf_time_model_eval_avg_one_vrp

        eval_team_avg_cost = sum_eval_team_avg_cost_all_batches / len(test_data)
        eval_indices_last_feasible_mean = sum_eval_indices_last_feasible_all_batches / len(test_data)
        avg_eval_frac_pool_usage_in_episode = sum_eval_pool_usage_frac_per_step_all_batches / len(test_data)

        if len(test_data) < 1000:
            # compute corr between or-tools and model results
            eval_corr = np.corrcoef(eval_team_avg_cost_list, eval_ortools_team_avg_costs)[0][1]
            diff_model_orTools = np.round(np.array(eval_team_avg_cost_list), 4) - np.round(
                np.array(eval_ortools_team_avg_costs), 4)
            vrp_ids_model_better = np.where(diff_model_orTools < 0)[0]  # syntax; by zero index we get the list
            frac_model_better = len(vrp_ids_model_better) / len(eval_team_avg_cost_list)

            # compute percentage change from init to final solution
            perc_cost_reduction = (np.array(eval_init_team_avg_cost_list) - np.array(eval_team_avg_cost_list)) / (
                np.array(eval_init_team_avg_cost_list))
            mean_eval_perc_cost_reduction = np.mean(perc_cost_reduction)
            median_eval_perc_cost_reduction = np.median(perc_cost_reduction)
            quantile_30_eval_perc_cost_reduction = np.quantile(perc_cost_reduction, 0.3)
            quantile_70_eval_perc_cost_reduction = np.quantile(perc_cost_reduction, 0.7)

            mean_eval_init_costs = np.mean(eval_init_team_avg_cost_list)
        else:
            eval_corr = None
            frac_model_better = None
            vrp_ids_model_better = None

        if len(interesting_eval_dm_ids) == test_data_size:  # if all vrp recs tracked:
            frac_oneAgentEverything_m = count_oneAgentEverything_m / test_data_size
            frac_lastInfeasible = count_lastInfeasible / test_data_size
            mean_num_steps_doneNothing_withinEpisode = count_AllDoneNothing_Within_Episode / test_data_size
            frac_AllDoneNothing = mean_num_steps_doneNothing_withinEpisode / args.max_reduce_steps
        else:
            frac_oneAgentEverything_m = None
            frac_lastInfeasible = None
            mean_num_steps_doneNothing_withinEpisode = None
            frac_AllDoneNothing = None

        # save model results to df
        eval_results.loc[f'model_run_{run_id}'] = [eval_team_avg_cost, max_eval_team_avg_cost,
                                                   inf_time_model_eval_avg_one_vrp, mean_eval_init_costs,
                                                   mean_eval_perc_cost_reduction, median_eval_perc_cost_reduction,
                                                   quantile_30_eval_perc_cost_reduction,
                                                   quantile_70_eval_perc_cost_reduction, frac_oneAgentEverything_m,
                                                   eval_corr, frac_model_better, vrp_ids_model_better,
                                                   mean_num_steps_doneNothing_withinEpisode, frac_AllDoneNothing,
                                                   frac_lastInfeasible]

    # save df
    if not args.debug:  # no logging in debug mode
        logger.save_df_results(eval_results)

    if args.visdom:
        # plot avg team avg costs on test data
        model_supervisor.plotter.plot('team avg costs', 'eval model mean',
                                      'Team avg costs (team avg cost of one vrp = of last feasible sol within RE)',
                                      0,
                                      eval_team_avg_cost,
                                      "one time evaluation")
        model_supervisor.plotter.plot('team avg costs', 'eval model max',
                                      'Team avg costs (team avg cost of one vrp = of last feasible sol within RE)',
                                      0,
                                      max_eval_team_avg_cost,
                                      "one time evaluation")
        if len(test_data) < 1000:
            # plot correlation between costs of val or tools and val model
            model_supervisor.plotter.plot('corr team avg cost model or-tools', 'eval',
                                          'corr model or-tools)',
                                          0,
                                          eval_corr,
                                          "one time evaluation")

            model_supervisor.plotter.plot('team avg costs', 'eval model median',
                                          'Team avg costs (team avg cost of one vrp = of last feasible sol within RE)',
                                          0,
                                          np.median(eval_team_avg_cost_list),
                                          "global step")

        model_supervisor.plotter.plot('idx of last feasible solution', 'val model mean over batch',
                                      'idx of last feasible solution',
                                      0, eval_indices_last_feasible_mean,
                                      # it's the mean over batch (dynamically comp)
                                      # in between [0, num_rewriting_steps]
                                      "one time evaluation")

        model_supervisor.plotter.plot('frac pool usage', 'eval',
                                      'frac pool usage per episode avg',
                                      0,
                                      avg_eval_frac_pool_usage_in_episode,
                                      "one time evaluation")
        # plot inf time of model average per vrp on val dataset
        model_supervisor.plotter.plot('run times', 'model eval',
                                      'inf run time avg one vrp(s)',
                                      0,
                                      inf_time_model_eval_avg_one_vrp,
                                      "one time evaluation")

    print("Done with evaluation")


def ray_training_function(config):
    args.lr = config["lr"]
    args.value_loss_coef = config["alpha"]
    args.epsilon = config["eps"]
    args.gradient_clip = config["gradient_clip"]

    args.lr_decay_steps = config["lr_decay_steps"]
    args.lr_decay_rate = config["lr_decay_rate"]

    args.penalty_val = config["penalty_val"]
    args.max_steps_pool_filled = config["max_steps_pool_filled"]
    args.max_reduce_steps = config["max_reduce_steps"]

    args.Z = config["Z"]

    args.experiment_name = f"NewData_2ag10n_{tune.get_trial_id()}_NoDropout_DifQPolCntxt_lrDecay{args.lr_decay_rate}_Z{args.Z}_rewr{args.max_reduce_steps}_gradClip{args.gradient_clip}_eps{args.epsilon}_alpha{args.value_loss_coef}_lr{args.lr}_maxPoolFill{args.max_steps_pool_filled}_penalVal{args.penalty_val}"  # _2Zpool_Zrandom_Zrule_sameRegion {tune.get_trial_id()}"
    # args.advantage_file_name = f"advantage_{tune.get_trial_id()}.txt"
    args.visdom = True
    args.model_dir = "/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/logs/" + args.experiment_name
    train(args)


if __name__ == "__main__":
    argParser = arguments.get_arg_parser()
    args = argParser.parse_args()
    args.cuda = not args.cpu and torch.cuda.is_available()
    # random.seed(args.seed)
    # np.random.seed(args.seed)

    if args.eval:
        args.visdom = False
        args.ray = False

        #from OR_Tools.OR_Tools import Benchmark_nAgents
        from OR_Tools import Benchmark_nAgents

        #from OR_Tools.OR_Tools import Benchmark_1Agent_TSP_AltScenario

        test_data_str = "20n_5a_diffTops_Vel05EdgeVel05_600s_test"

        args.test_dataset = f"/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/data/vrp/newAgVelAgEdgeVel/{test_data_str}.p"
        # MODEL2: Transfer_InitPolicyDecoders2Ag_tempScalingAdaptive_weighLocalReord_central_5agents10n_6ae83_00000_ruleTemp0.1_ruleTempBatchThresh500_weightLocReAct0.01_Z5_rewr30_gradClip0.05_eps0.15_alpha1e-05_lr0.0005_dstep200_drate0.9_maxStepsPoolFilled6_penaltyVal10.0
        exp_run_str = "NewData_5ag20n_12aea_00000_DiffQLayers_PolicyWithContextInfo_Z10_rewr40_gradClip0.05_eps0.15_alpha5e-06_lr0.0005_maxStepPoolFill6_penalVal10.0"
        # "paper_00_central_2agents10n_e742e_00000_Z5_rewr30_gradClip0.05_eps0.15_alpha1e-05_lr0.0005_dstep200_drate0.9_maxStepsPoolFilled3_penaltyVal10.0"#"paper_01_central_5agents20n_hihgerAlpha_22259_00001_Z10_rewr40_gradClip0.05_eps0.15_alpha5e-06_lr0.0005_dstep200_drate0.9_maxStepsPoolFilled6_penaltyVal10.0"
        ckpt_str = "/ckpt-00004700"  # for 22 epochs: 3400  #"/ckpt-00004700" # "/ckpt-00003500" #"ckpt-00002900" # "ckpt-00001200"
        # args.load_model = f"../checkpoints/model_0_central/{ckpt_str}"
        args.load_model = f"../logs/{exp_run_str + ckpt_str}"
        # f"../logs/model_0_central/{ckpt_str}"
        args.experiment_name = f"inf_{exp_run_str + ckpt_str}_{test_data_str}_InitLoaded_Z1_100re" #_generalizing_agents"
                        #_InitLoaded_Z1_100re" .split('_Z5_')[0]
        #exp_run_str.split('_Z5_')[0]


        args.max_reduce_steps = 100  # 60  #40 # 30
        args.Z = 1  # 1 # 10 # 5
        # get same init routes (determined by test data)
        inf_run_with_nodes = "inf_NewData_SameTops_5ag20n_7cac5_00002_Z10_rewr40_gradClip0.05_eps0.15_alpha1e-06_lr0.0005_maxStepPoolFill6_penalVal10.0/ckpt-00004700_20n_5a_diffTops_Vel05EdgeVel05_600s_test_InitSaved_Z1_100re"

        args.path_init_route_nodes_eval = "../logs/" + inf_run_with_nodes + "/init_nodes_eval.p"  # ""

        evaluate(args)


    elif args.ray:
        # made as package
        from OR_Tools import Benchmark_nAgents

        # from OR_Tools.OR_Tools import Benchmark_nAgents

        analysis = tune.run(ray_training_function, config={"eps": tune.grid_search([0.15]),
                                                           "lr_decay_steps": tune.grid_search([200]),
                                                           "lr_decay_rate": tune.grid_search([0.9]),#, 1.0]),
                                                           "gradient_clip": tune.grid_search([0.05]),
                                                           "lr": tune.grid_search([5e-4]),
                                                           "penalty_val": tune.grid_search([10.0]),

                                                           "max_reduce_steps": tune.grid_search([30]),  #30, 40

                                                           "alpha": tune.grid_search([1e-6]),#[1e-6, 5e-7, 1e-5]),  #, 1e-6  1e-5; 1e-6 , 5e-6, 9e-6
                                                           "max_steps_pool_filled": tune.grid_search([3]),
                                                           "Z": tune.grid_search([5])#,#5  #10
                                    
                                                           },
                            mode="min", resources_per_trial={"cpu": 32, "gpu": 1}, keep_checkpoints_num=2,
                            checkpoint_score_attr='val_avg_cost')  # , log_to_file=True) #, False, False  "gamma": tune.grid_search([0.3, 0.7, 0.9])


        print("Best config: ", analysis.get_best_config(metric="val_avg_cost", mode="min"))  # 32
        df = analysis.results_df
        # Init2Ag_Policy_Decoders_TempScalingAdaptive_WeighLocalReord.
        df.to_pickle(
            '/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/exp_results/central_2agents_10nodes_newdata_SameTops_DiffQLayers_PolicyWithContext_NoDropout.pkl')


    else:  # running in pycharm
        if args.visdom:
            from OR_Tools import Benchmark_nAgents
        else:
            import sys

            sys.path.append("../marl/")
            sys.path.append("/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/")
            from OR_Tools.OR_Tools import Benchmark_nAgents  #.OR_Tools

            import os
            os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

        train(args)
