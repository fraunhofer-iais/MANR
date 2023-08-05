import os
import sys
import numpy as np
import argparse
# import json
import pickle
from sklearn import metrics
from tqdm import tqdm
from scipy.stats import truncnorm
import random

# to call: python vrpDatagen.py --num_agents 5 --num_customers 15 --num_samples 100


argParser = argparse.ArgumentParser()
#10n_2a_diffTops_Vel05EdgeVel05_5000s_train
#10n_2a_30SameTops_Vel05EdgeVel05_600s_test
#20n_2a_30SameTops_Vel05EdgeVel05_600s_val
argParser.add_argument('--res_file', type=str, default='10n_5a_OneTop_5600CostMatrices_train_val.p')

argParser.add_argument('--num_samples', type=int, default=5600) #600, 5000
argParser.add_argument('--num_topologies', type=int, default=1)#30, 100 #600, 5000

argParser.add_argument('--num_agents', type=int, default=5)
argParser.add_argument('--gen_vel', type=list, default=[0.5, 1])
argParser.add_argument('--edge_vel', type=list, default=[0.5, 1.5])

argParser.add_argument('--num_customers', type=int, default=10)

argParser.add_argument('--seed', type=int, default=3211)

args = argParser.parse_args()


def gen_vrp_new(num_customers, num_agents, gen_vel, edge_vel,
                topology):  # generates multiple vrp problems: one topology with different velocities (i.e. different cost matrices)

    depots = topology["depots"]
    customers = topology["customers"]
    # for agent-specific costs:
    # sample general agent-specific velocity within [0.5,1]
    # for each edge: sample additional agent-specific factor in [0.8,1.2]
    num_agent_nodes = num_customers + 1  # +1 for depot
    num_agent_edges = (num_agent_nodes * (num_agent_nodes - 1)) // 2

    vrp = []
    for i in range(num_agents):
        cur_general_agent_velocity = np.round(np.random.uniform(gen_vel[0], gen_vel[1]), 4)
        cur_agent_edge_velocities = [np.round(np.random.uniform(edge_vel[0], edge_vel[1]), 4) for _ in
                                     range(num_agent_edges)]
        agent_dict = {}
        # n customers named 1, ..., n
        agent_dict["agent_id"] = i
        agent_dict["general_agent_velocity"] = cur_general_agent_velocity
        agent_dict["agent_edge_velocities"] = cur_agent_edge_velocities
        agent_dict["customers"] = list(np.arange(0, num_customers))
        agent_dict["depots"] = list(np.arange(num_customers, num_customers + num_agents))  # depot of agent i = entry i in depots
        agent_dict["pool"] = num_customers + num_agents
        agent_dict["customers_coord"] = customers
        agent_dict["depot_coords"] = depots
        ############

        depot = depots[i]

        customersWithDepot = customers + [
            depot]  # first customer nodes; last one depot  such that for customer nodes the indices are always correct; and for depot nodes 12, 13, 14,...you need to take the last col
        costs = metrics.pairwise.euclidean_distances(customersWithDepot, customersWithDepot)

        # create agent specific costs by using velocities
        id_in_vel = 0
        for i in range(len(costs)):
            for j in range(i + 1, len(costs)):  # i+1 to exclude diagonal
                costs[i, j] = costs[i, j] * (1 / cur_agent_edge_velocities[id_in_vel]) * (
                            1 / cur_general_agent_velocity)
                costs[j, i] = costs[j, i] * (1 / cur_agent_edge_velocities[id_in_vel]) * (
                        1 / cur_general_agent_velocity)
                id_in_vel += 1

        agent_dict["costs"] = np.round(costs, 5)
        vrp.append(agent_dict)

    return vrp


def generate_topology(num_agents, num_customers):

    def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    cur_topology = {"depots": None, "customers": None}
    # depots x,y uniformly over [0,1]
    depots = [[np.random.rand(), np.random.rand()] for i in range(num_agents)]
    cur_topology["depots"] = [list(elem) for elem in np.round(depots,4)]  #depots
    # fraction of uniformly drawn customer nodes
    frac_uniform_customers = np.random.rand()
    num_uniform_customers = int(frac_uniform_customers * num_customers)
    uniform_customers = [[np.random.rand(), np.random.rand()] for i in range(
        num_uniform_customers)]  # customers = [[np.random.rand(), np.random.rand()] for i in range(num_customers)]
    # rest:
    trunc_norm_customers = []
    num_customers_left = num_customers - num_uniform_customers
    for agent_id in range(num_agents):
        if agent_id == num_agents - 1:
            cur_num_cust = num_customers_left
        else:
            cur_num_cust = int(num_customers_left / num_agents)
            num_customers_left -= cur_num_cust

        cur_x_distribution = get_truncated_normal(mean=depots[agent_id][0], sd=0.1, low=0, upp=1)
        cur_y_distribution = get_truncated_normal(mean=depots[agent_id][1], sd=0.1, low=0, upp=1)

        cur_customers_x = cur_x_distribution.rvs(cur_num_cust)
        cur_customers_y = cur_y_distribution.rvs(cur_num_cust)
        trunc_norm_customers.extend(list(map(list, zip(cur_customers_x, cur_customers_y))))

    customers = trunc_norm_customers + uniform_customers
    if len(customers) != num_customers:
        raise ValueError('error')

    cur_topology["customers"] = [list(elem) for elem in np.round(customers,4)]  #customers
    return cur_topology


def main():

    num_costMatrices_per_topology = args.num_samples // args.num_topologies

    topologies = []
    for _ in range(args.num_topologies):
        cur_topology = generate_topology(args.num_agents,args.num_customers)
        topologies.append(cur_topology)

    data = []
    with tqdm(total=args.num_topologies) as pbar:
        for topology in topologies:
            for _ in range(num_costMatrices_per_topology):
                one_vrp = gen_vrp_new(args.num_customers, args.num_agents, args.gen_vel, args.edge_vel,
                            topology)
                data.append(one_vrp)
            pbar.update(1)

    path = '../data/vrp/newAgVelAgEdgeVel/'
    if not os.path.exists(path):
        os.makedirs(path)

    # file containing all data
    data_size = len(data)
    print(data_size)

    fout_res = open(path + args.res_file, 'wb')
    pickle.dump(data, fout_res)

main()
