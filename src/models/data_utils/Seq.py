import numpy as np
import torch
from torch.autograd import Variable
from .utils import *
import copy

class VrpNode_xy(object):
    """
    NEW Class to represent each node for vehicle routing.
    Not needed anymore: x,y; because cost matrix integrated
    """

    def __init__(self, nodeId, node_x, node_y, preNodeId,preNode_x,preNode_y, cost, embedding=None):
        self.nodeId = nodeId  # 0.0, 0.1, 0.2 for depot!
        self.node_x = node_x
        self.node_y = node_y
        self.preNodeId = preNodeId
        self.preNode_x = preNode_x
        self.preNode_y = preNode_y
        self.cost = cost  # cost of edge between previous and current node
        if embedding is None:  # what is this node embedding actually???
            self.embedding = None
        else:
            self.embedding = embedding.copy()



class VrpNode(object):
    """
    NEW Class to represent each node for vehicle routing.
    Not needed anymore: x,y; because cost matrix integrated
    """

    def __init__(self, nodeId, preNodeId, cost, embedding=None):
        self.nodeId = nodeId  # 0.0, 0.1, 0.2 for depot!
        self.preNodeId = preNodeId
        self.cost = cost  # cost of edge between previous and current node
        if embedding is None:
            self.embedding = None
        else:
            self.embedding = embedding.copy()


class SeqManager(object):
    """
    Base class for sequential input data. Can be used for vehicle routing.
    NOT CHANGED
    """

    def __init__(self):
        self.nodes = []
        self.num_nodes = 0

    def get_node(self, idx):
        return self.nodes[idx]



class VrpManager(SeqManager):
    """
    The NEW class to maintain the state for vehicle routing.
    """

    def __init__(self, costs, num_agents):
        super(VrpManager, self).__init__()
        self.costs = costs  # list of cost matrices; one per agent
        self.routes = [[] for i in range(num_agents)]   # list of routes; one per agent
        self.tot_costs = [[] for i in range(num_agents)]  # list of local_costs; one per agent
        # self.vehicle_state = []
        self.team_avg_cost = -1
        # self.encoder_outputs = None  # embedding output based on route [pre node, node, cost]
        self.num_agents = num_agents  # locally you need to know how many others there are
        # + 1 for dummy pool route
        self.vehicle_states = [[] for i in range(num_agents + 1)]  # contains vehicle states for all agents
        self.encoder_outputs_with_cost = [[] for i in range(num_agents + 1)] # contains encoder outputs for all agents; computed separately by each agent by using local cost information with LSTM incorporating sequential route info
        self.encoder_outputs_no_cost = [[] for i in range(num_agents + 1)]  # contains encoder outputs for all agents solely based on x-y coordinates (computed per agent route with LSTM incorporating sequential route info)

        # OLD: own agent's one contains more information (cost included), the ones of others less: there embedding output based on global stat [pre node, node] (cost unkown of other agents) --> information of all other agents

    def clone(self, take_x_y):
        res = VrpManager(self.costs, self.num_agents)
        res.nodes = []
        for node in self.nodes:
            if take_x_y:
                res.nodes.append(VrpNode_xy(nodeId=node.nodeId, node_x=node.node_x, node_y=node.node_y,preNodeId=node.preNodeId, preNode_x = node.preNode_x,preNode_y = node.preNode_y, cost=node.cost, embedding=node.embedding))
            else:
                res.nodes.append(VrpNode(nodeId=node.nodeId, preNodeId=node.preNodeId, cost=node.cost, embedding=node.embedding))

        res.num_nodes = self.num_nodes
        res.costs = self.costs[:]
        res.routes = self.routes[:]
        res.tot_costs = self.tot_costs[:]
        res.team_avg_cost = self.team_avg_cost
        res.num_agents = self.num_agents
        res.vehicle_states = self.vehicle_states[:]
        res.encoder_outputs_with_cost = self.encoder_outputs_with_cost[:]
        res.encoder_outputs_no_cost = self.encoder_outputs_no_cost[:]
        return res

    def get_cost(self, node_1, node_2, agent_id):  # trouble because of depot floats
        idx1 = node_1.nodeId
        idx2 = node_2.nodeId
        # if depot node contained, need to adjust index --> it's the last one in the matrix

        # -1 due to added pool node in nodes
        if idx1 >= (self.num_nodes-1) - self.num_agents:  # then it's the agent's depot node! since depot node indices come after the customers node Ids; (num_nodes = number custom + all depots)
            idx1 = (self.num_nodes-1) - self.num_agents  # it's always last cost row/col for each agent
        if idx2 >= (self.num_nodes-1) - self.num_agents:
            idx2 = (self.num_nodes-1) - self.num_agents
        cost = self.costs[agent_id][idx1, idx2]
        return cost

    def get_neighbor_idxes(self, rewrite_pos, allow_no_change_in_rule):  # returns list of possible rule nodes! possible rule nodes within same agent's route: all except region node and precedessor (because then no change); and possible rule nodes within other agent's routes: all except last depot node
        # rewrite_pos now a tuple (agent_id, idx in agent's route) --> region_node now global and get all candidate rule nodes
        neighbor_idxes = []  # tuple (agent_id, route_idx in agent's vehicle state)
        for i in range(self.num_agents):  # go through each agent's state
            for k in range(len(self.vehicle_states[i]) - 1):  # exclude last depot of current agent's route
                if i == rewrite_pos[0]:  # if cur_agent = agent owning region node
                    if allow_no_change_in_rule:
                        if k != rewrite_pos[1]:  # take every index except region one (you can take the one before which means no change)
                            neighbor_idxes.append([i, k])
                    else:
                        if k != rewrite_pos[1] and k != (rewrite_pos[1] - 1):  # take every index except region one and the node before region
                            neighbor_idxes.append([i, k])
                else:  # each node possible
                    neighbor_idxes.append([i, k])

        return neighbor_idxes

    def get_feasible_node_idxes(self, rewrite_pos, allow_no_change_in_rule, agent_id):  # returns list of possible rule nodes! possible rule nodes within same agent's route: all except region node and precedessor (because then no change); and possible rule nodes within other agent's routes: all except last depot node
        # rewrite_pos now a tuple (agent_id, idx in agent's route)
        # feasible indices are own local nodes + pool node
        feasible_node_idxes = []  # indices in dm.nodes = nodeIds in global vehicle states

        if rewrite_pos[0] == agent_id:  # then region node from own route
            # all local nodes except region and predecessor of region can be rule nodes
            # go through own route
            for k in range(len(self.vehicle_states[agent_id]) - 1):  # exclude last depot of current agent's route
                if allow_no_change_in_rule:
                    if k != rewrite_pos[1]:  # take every index except region one (you can take the one before which means no change)
                        feasible_node_idxes.append(self.vehicle_states[agent_id][k])
                else:
                    if k != rewrite_pos[1] and k != (rewrite_pos[1] - 1):  # take every index except region one and the node before region
                        feasible_node_idxes.append(self.vehicle_states[agent_id][k])
        else:  # region from pool
            # all local nodes are feasible
            feasible_node_idxes = self.vehicle_states[agent_id][:-1]  # exclude last depot

        # pool node always a feasible rule node
        feasible_node_idxes.append(self.num_nodes - 1)

        return feasible_node_idxes

    def get_route_dIdx(self, nodeId):
        # nodeId in dm.nodes
        # return: tuple-index in global_vehicle_state (agent_id, index in route)
        ag_route = [ag_route for ag_route in self.vehicle_states if nodeId in ag_route][
            0]  # route of agent which contains node
        agent_id = self.vehicle_states.index(ag_route)
        id_in_vehicle_state = ag_route.index(nodeId)  # if depot node then contained twice --> always lower index taken, so ok (since successor of depot might be needed)
        return agent_id, id_in_vehicle_state


    def add_route_node(self, node_idx, agent_id, take_x_y):  # add node node_idx to route of agent agent_id (saved in global_vehicle_states)
        node = self.get_node(node_idx)
        if len(self.vehicle_states[agent_id]) == 0:  # if no node visited yet by agent vehicle (can not happen later on; so ok if being called then), then it's the depot node
            pre_node_idx = node_idx
        else:
            pre_node_idx = self.vehicle_states[agent_id][-1]  # get last visited node idx
        pre_node = self.get_node(pre_node_idx)
        self.vehicle_states[agent_id].append(node_idx)  # vehicle state contains node idx (position of node in nodes; depot node has index 0)

        cur_cost = self.get_cost(node, pre_node, agent_id)

        if len(self.tot_costs[agent_id]) == 0:
            self.tot_costs[agent_id].append(cur_cost)
        else:
            self.tot_costs[agent_id].append(self.tot_costs[agent_id][-1] + cur_cost)
        if take_x_y:
            new_node = VrpNode_xy(nodeId=node.nodeId, node_x=node.node_x, node_y=node.node_y, preNodeId=pre_node.nodeId, preNode_x=pre_node.node_x, preNode_y=pre_node.node_y, cost=cur_cost)
            new_node.embedding = [new_node.node_x, new_node.node_y, new_node.preNode_x, new_node.preNode_y, new_node.cost]
        else:
            new_node = VrpNode(nodeId=node.nodeId, preNodeId=pre_node.nodeId, cost=cur_cost)
            new_node.embedding = [new_node.nodeId, new_node.preNodeId, new_node.cost]
        self.nodes[node_idx] = new_node
        self.routes[agent_id].append(new_node.embedding[:])

    def update_team_avg_cost(self):
        # if pool filled, then nan; otherwise feasible solution
        if len(self.vehicle_states[-1]) > 1:
            self.team_avg_cost = np.nan
        else:  # feasible solution
            to_be_averaged = []
            for local_costs in self.tot_costs:  # for each local agent cost
                if local_costs:  # can be empty if agent is not visiting any mode
                    to_be_averaged.append(local_costs[-1])
            self.team_avg_cost = np.mean(to_be_averaged)
