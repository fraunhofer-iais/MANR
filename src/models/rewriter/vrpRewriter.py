import numpy as np
import operator
import random
import time
import copy
from ..data_utils import data_utils


class vrpRewriter(object):
    """ Rewriter for vehicle routing. """

    def move(self, dm, global_action,
             take_x_y):  # updates global_vehicle_states and if agent himself also route and tot_cost; returns rewritten_dm
        # check if someone is acting
        someone_acts = 0
        for local_action in global_action:
            if local_action:  # can be empty if some agent visits all nodes
                if not (local_action[0][0] == local_action[1][0] and local_action[0][1] - local_action[1][
                    1] == 1):  # region agent = rule agent and region-rule = 1 then it's do nothing
                    someone_acts = 1

        if someone_acts == 1:
            rewritten_dm = dm.clone(take_x_y)
            # update the state of each agent (+ dummy pool agent) in global_vehicle_states
            # Before: each agent route in global vehicle state rewritten for itself; now everything together (((for agent_id in range(dm.num_agents): self.rewrite_agent = rewrite:global_vehicle_state
            rewritten_vehicle_state = self.rewrite_global_vehicle_state(dm,
                                                                        global_action)  # currently only state updated
            rewritten_dm.vehicle_states = rewritten_vehicle_state

            # update route, tot_cost for each agent whose state changed
            for agent_id in range(dm.num_agents):
                old_local_state = dm.vehicle_states[agent_id]
                new_local_state = rewritten_vehicle_state[agent_id]
                if old_local_state != new_local_state:
                    # get min_index from which on route changed (works also with uneven state sizes since start and end are always the depot)
                    min_length = min(len(old_local_state), len(new_local_state))
                    for j in range(min_length):
                        if old_local_state[j] != new_local_state[j]:
                            min_update_idx = j
                            break

                    # now locally keep everything which stayed unchanged
                    rewritten_dm.vehicle_states[agent_id] = rewritten_dm.vehicle_states[agent_id][:min_update_idx]
                    rewritten_dm.routes[agent_id] = dm.routes[agent_id][:min_update_idx]
                    rewritten_dm.tot_costs[agent_id] = dm.tot_costs[agent_id][:min_update_idx]
                    # for changes: re-build local route from this point on (to automatically update route and total cost)
                    for new_visited_node_id in new_local_state[min_update_idx:]:
                        rewritten_dm.add_route_node(new_visited_node_id, agent_id, take_x_y)  # updates global_vehicle_state, route and tot_cost

            # update team avg cost
            rewritten_dm.update_team_avg_cost()
            return rewritten_dm
        else:
            return dm

    def rewrite_global_vehicle_state(self, dm, global_action):
        # 2 two cases:
        # A region node from agent itself --> rule in own route or rule is pool node
        # B region node from pool --> rule in own route or rule is pool node

        # in contrast to before: an agent id cannot appear in another agent's local route --> can just execute, not as complicated anymore

        # each local action can be independently treated and states adapted --> always rewrite rewritten_vehicle_state
        # only exception: pool --> if both take something from pool (one inserting and one taking out not possible due to workflow)

        rewritten_vehicle_state = copy.deepcopy(dm.vehicle_states) # dm.global_vehicle_states.copy()

        nodes_remove_from_pool = []
        nodes_add_to_pool = []

        for i, local_action in enumerate(global_action):
            if local_action:  # can be empty if some agent visits all nodes or if pool contains less than nodes agents
                region_spec = local_action[0]  # region tuple = (agent id, id in state)
                region_agent = region_spec[0]
                region_idx = region_spec[1]  # in state
                rule_spec = local_action[1]  # rule tuple = (agent id, id in state)
                rule_agent = rule_spec[0]
                rule_idx = rule_spec[1]  # in state

                if region_agent == rule_agent:  # either agent re-ordering its own route or doing nothing ( = own route doing nothing OR rejecting node from pool)
                    # only case where change is needed is if true agent is re-ordering its own route
                    if region_agent != dm.num_agents:  # if it's not the dummy pool agent
                        if region_idx - rule_idx != 1:  # if the true agent is truly acting (not doing nothing)
                            min_idx = min(region_idx, rule_idx)
                            # exclude region from region agent
                            # rewritten state can be updated; there cannot be any other interfering local action
                            if min_idx == region_idx:
                                # exclude region; go after rule and insert region there
                                rewritten_vehicle_state[region_agent] = dm.vehicle_states[region_agent][
                                                                        :region_idx] + \
                                                                        dm.vehicle_states[region_agent][
                                                                        region_idx + 1: rule_idx + 1] + \
                                                                        [dm.vehicle_states[region_agent][
                                                                             region_idx]] + \
                                                                        dm.vehicle_states[rule_agent][
                                                                        rule_idx + 1:]
                            else:  # then rule idx smaller
                                rewritten_vehicle_state[region_agent] = dm.vehicle_states[region_agent][
                                                                        :rule_idx + 1] + \
                                                                        [dm.vehicle_states[region_agent][
                                                                             region_idx]] + \
                                                                        dm.vehicle_states[region_agent][
                                                                        rule_idx + 1: region_idx] + \
                                                                        dm.vehicle_states[rule_agent][
                                                                        region_idx + 1:]
                else:  # region agent and rule agent are different (i.e. either a pool node is integrated or a node is given to the pool)
                    if region_agent == dm.num_agents:  # if a node from the pool is integrated locally
                        # local route of rule agent can be updated (no interference) --> integrate region from pool (read pool from cur state)
                        rewritten_vehicle_state[rule_agent] = dm.vehicle_states[rule_agent][:rule_idx + 1] + \
                                                              [dm.vehicle_states[region_agent][region_idx]] + \
                                                              dm.vehicle_states[rule_agent][rule_idx + 1:]
                        # remove nodes from pool at the end
                        nodes_remove_from_pool.append(region_idx)
                    else:  # if region is from own route and rule node is the pool, i.e. something given to pool
                        # local route of region agent can be updated (no interference)
                        rewritten_vehicle_state[region_agent] = dm.vehicle_states[region_agent][:region_idx] + \
                                                                dm.vehicle_states[region_agent][region_idx + 1:]

                        # node can also be integrated to pool without interference since there's no order in the pool; always place inserted node after pool node
                        # since it is not working add them later on to pool
                        nodes_add_to_pool.append(dm.vehicle_states[region_agent][region_idx])
                        #rewritten_vehicle_state[rule_agent] = rewritten_vehicle_state[rule_agent][:rule_idx + 1] + \
                        #                                      [dm.global_vehicle_states[region_agent][region_idx]] + \
                        #                                      rewritten_vehicle_state[rule_agent][rule_idx + 1:]

        # nodes_remove_from_pool and nodes_add_to_pool cannot be filled at the same time
        if len(nodes_add_to_pool) > 0 and len(nodes_remove_from_pool) > 0:
            raise NameError("Something is wrong")

        if nodes_remove_from_pool:  # remove nodes from pool with specified indices
            for idx in sorted(nodes_remove_from_pool, reverse=True):  # delete highest indices first, then no problem
                del rewritten_vehicle_state[dm.num_agents][idx]

        if nodes_add_to_pool:  # since order of insertion irrelevant, here are the true nodeIds (and not some indices)
            for nodeId in nodes_add_to_pool:
                rewritten_vehicle_state[dm.num_agents].append(nodeId)

        return rewritten_vehicle_state



    def rewrite_agent_old(self, dm, global_action, agent_id):
        # collect region and rule nodes of respective agent (first: rules with corresponding regions)

        # 2 two cases:
        # A region node from agent itself --> rule in own route or rule is pool node
        # B region node from pool --> rule in own route or rule is pool node
        # in contrast to before: an agent id cannot appear in another agent's local route

        rule_idxs = []
        region_tuples = []
        own_region_idx = []
        for local_action in global_action:
            if local_action:  # can be empty if some agent visits all nodes
                if not (local_action[0][0] == local_action[1][0] and local_action[0][1] - local_action[1][
                    1] == 1):  # if local action means proper acting (not doing nothing)
                    if local_action[1][0] == agent_id:  # check if agent is involved in rule
                        rule_idxs.append(local_action[1][1])  # collect rule node in agent's own route
                        region_tuples.append(local_action[0])  # collect corresponding region tuple
            # collected so far: local actions which involved a rule node within the agent's route

        # now check if there's a region node for the agent (proper one, i.e. in a not doing nothing action)
        if global_action[agent_id]:  # can be empty if some agent visits all nodes
            if not (global_action[agent_id][0][0] == global_action[agent_id][1][0] and global_action[agent_id][0][1] -
                    global_action[agent_id][1][1] == 1):
                own_region_idx.append(global_action[agent_id][0][
                                          1])  # can already be included in region_tuples if rule in same route, but needed to get correct slicing order

        all_indices = rule_idxs + own_region_idx  # collect all indices within local agent's route where something will be changed; if own_region_idx is empty no problem
        all_indices = np.sort(all_indices)  # min one comes first
        if list(all_indices):  # if not empty
            min_update_idx = all_indices[0]
        else:
            min_update_idx = []
        rewritten_vehicle_state = []  # containing single parts of the new state
        cur_start_idx = 0
        for ind in all_indices:  # i: number of nodes in region_tuples and rule_idxs
            if ind in own_region_idx:  # if index is region node; extract state until region node (since region node is removed), delete region node
                cur_part = dm.vehicle_states[agent_id][cur_start_idx: ind]  # region node exclusively
                rewritten_vehicle_state.append(cur_part)
                cur_start_idx = ind + 1  # next relevant node=successor of region node
            else:  # then rule node; get state until route node (including route node) and insert corresponding region node (region node can be within route or from someone else)
                cur_part = dm.vehicle_states[agent_id][cur_start_idx: (ind + 1)]  # rule node inclusively
                i = rule_idxs.index(ind)  # get corresponding region node; can be own region node --> here inserted
                cur_part = cur_part + [dm.vehicle_states[region_tuples[i][0]][region_tuples[i][
                    1]]]  # region_tuples[i][0]=  agents id (can be = own_region_idx[0]), region_tuples[i][1]=idx of region node in that agent's route
                rewritten_vehicle_state.append(cur_part)
                cur_start_idx = ind + 1  # next relevant node=successor of rule node
        # add remaining (unchanged) part of state to rewritten_state
        rewritten_vehicle_state.append(dm.vehicle_states[agent_id][cur_start_idx:])
        rewritten_vehicle_state = [item for sublist in rewritten_vehicle_state for item in
                                   sublist]  # rewritten own vehicle state
        return rewritten_vehicle_state, min_update_idx
