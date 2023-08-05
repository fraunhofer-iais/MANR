import numpy as np
import os
import sys
import argparse
import pyparsing as pyp
import random
from .Seq import *
from .utils import *


class vrpParser(object):
    """
    NEW Class to
    vrpParser for one problem= one vrp dict (of one agent)
    
    #creates vrpManagers for each agent
    #init solution has to be communciated
    # define global_parse function which calls everything locallyr
"""

    def parse(self, problem, heuristic, take_x_y, init_route_nodes):  # parsing one vrp;  one vrp problem = list of agent dicts
        # return only one global vrpManager

        num_agents = len(problem)

        # for init solution: random allocation of nodes to agents and agents optimize their route given the random node allocation (same heuristic; always visit closest node)
        customers = problem[0]["customers"]
        num_customers = len(customers)
        num_customers_per_agent_min = int(num_customers / num_agents)

        if not init_route_nodes:
            init_route_nodes = []

            # init solution per vrp: random distribution of nodes first
            for i in range(num_agents):
                agent_nodes = random.sample(customers, num_customers_per_agent_min)
                init_route_nodes.append(agent_nodes)
                customers = list(set(customers) - set(agent_nodes))
            # distribute nodes left in customers to first agents
            for i, customer in enumerate(customers):
                init_route_nodes[i].append(customer)

        # parse problem globally (done by one agent)
        dm = self.global_parsing(problem, init_route_nodes, num_agents, heuristic, take_x_y)
        # self.global_parsing(local_share, init_route_nodes[i], num_agents, heuristic, take_x_y, debug=False)
        return dm

    def global_parsing(self, problem, init_route_nodes, num_agents, heuristic,take_x_y):
        # problem definition
        dm = VrpManager([local_share['costs'] for local_share in problem], num_agents)
        # for all nodes: predecessor is node itself since route not yet exists; thus cost between node and predecessor zero

        # first add customer nodes
        for customer in problem[0]['customers']:  # here nodeId = idx in nodes
            if take_x_y:
                dm.nodes.append(VrpNode_xy(nodeId=customer,  node_x=problem[0]['customers_coord'][customer][0], node_y=problem[0]['customers_coord'][customer][1], preNodeId=customer, preNode_x=problem[0]['customers_coord'][customer][0], preNode_y=problem[0]['customers_coord'][customer][1],cost=0.0))
            else:
                dm.nodes.append(VrpNode(nodeId=customer, preNodeId=customer, cost=0.0))
        # then add depot nodes of all agents
        for id, depot in enumerate(problem[0]['depots']):
            if take_x_y:
                dm.nodes.append(VrpNode_xy(nodeId=depot, node_x=problem[0]['depot_coords'][id][0], node_y=problem[0]['depot_coords'][id][1], preNodeId=depot, preNode_x=problem[0]['depot_coords'][id][0], preNode_y=problem[0]['depot_coords'][id][1], cost=0.0))
            else:
                dm.nodes.append(VrpNode(nodeId=depot, preNodeId=depot, cost=0.0))

        # add pool node
        if take_x_y:
            dm.nodes.append(VrpNode_xy(nodeId=problem[0]['pool'], node_x=0,
                                       node_y=0, preNodeId=problem[0]['pool'],
                                       preNode_x=0,
                                       preNode_y=0, cost=0.0))
        else:
            dm.nodes.append(VrpNode(nodeId=problem[0]['pool'], preNodeId=problem[0]['pool'], cost=0.0))

        dm.num_nodes = len(dm.nodes)  # num_nodes =customers + depots (number of nodes visited by the agent)
        # create initial solution via heuristic --> needs to be done for each agent
        for agent_id in range(num_agents):
            # create init route per agent
            pending_nodes = init_route_nodes[agent_id]
            dm.add_route_node(problem[0]["depots"][agent_id], agent_id, take_x_y)  # add agent's depot to its route --> it's node idx = local_share["depots"][dm.agent_id] im dm.nodes list! (for agent 0: 15; for agent 1: 16, ...

            while len(pending_nodes) > 0:  # before 1; probably because of depot..here all nodes need to be distributed
                cost = []
                pre_node_idx = dm.vehicle_states[agent_id][-1]  # vehicle state at the beginning contains added depot
                pre_node = dm.get_node(pre_node_idx)
                # get cost of pre_node to all other nodes
                for i in pending_nodes:
                    cur_node = dm.get_node(i)
                    cost.append(dm.get_cost(pre_node, cur_node, agent_id))

                if heuristic == "closest":
                    # sort pending nodes --> ascending cost (the one with lowest cost is visited)
                    for i in range(len(pending_nodes)):
                        for j in range(i + 1, len(pending_nodes)):
                            if cost[i] > cost[j]:
                                pending_nodes[i], pending_nodes[j] = pending_nodes[j], pending_nodes[i]
                                cost[i], cost[j] = cost[j], cost[i]
                elif heuristic == "farthest":
                    # sort pending nodes --> descending cost (the one with highest cost is visited)
                    for i in range(len(pending_nodes)):
                        for j in range(i + 1, len(pending_nodes)):
                            if cost[i] < cost[j]:
                                pending_nodes[i], pending_nodes[j] = pending_nodes[j], pending_nodes[i]
                                cost[i], cost[j] = cost[j], cost[i]

                # create solution
                for i in pending_nodes:
                    cur_node = dm.get_node(i)
                    dm.add_route_node(i, agent_id, take_x_y)
                    pending_nodes.remove(i)
                    break
            dm.add_route_node(problem[0]["depots"][agent_id], agent_id, take_x_y)  # add final depot node

        dm.update_team_avg_cost()
        return dm


"""
# np.random.seed(3112)
        # hard coded for 2 agents and same vrp case
        agent_nodes_0 = [node for node in customers if (node + 1) % 2 == 0]
        init_route_nodes.append(agent_nodes_0)
        agent_nodes_1= list(set(customers) - set(agent_nodes_0))
        init_route_nodes.append(agent_nodes_1)

dm.update_team_avg_cost() for not feasible deleted
"""