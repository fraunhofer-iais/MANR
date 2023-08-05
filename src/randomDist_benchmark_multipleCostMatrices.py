import models.data_utils.data_utils as data_utils
#from OR_Tools.OR_Tools import Benchmark_1Agent_TSP_AltScenario
from OR_Tools import Benchmark_1Agent_TSP_AltScenario
import random
import numpy as np
import copy
from sklearn import metrics
import sys
import time


def run_randomDist_benchmark(test_dataset):
    test_data = data_utils.load_dataset(test_dataset, "dummy")
    test_data = test_data[:628].copy()# test_data[5000:].copy()
    test_data_size = len(test_data)
    print(f"test data size: {test_data_size}")

    num_agents = len(test_data[0])
    customers = test_data[0][0]["customers"]
    num_customers = len(customers)
    num_agent_nodes = num_customers + 1  # +1 for depot
    num_agent_edges = (num_agent_nodes * (num_agent_nodes - 1)) // 2

    gen_vel =[0.5, 1]
    edge_vel =[0.5, 1.5]

    all_runs_tsp_team_avg_cost_per_vrp_median50CostMatrices = []
    all_runs_init_route_nodes_per_vrp = []

    for run_id in range(100):
        print(f"Run {run_id}: Randomly distributing init nodes")
        init_route_nodes_eval_vrps = [[] for _ in range(test_data_size)]
        cur_run_team_avg_cost_per_vrp_median50CostMatrices = []
        for idx, vrp in enumerate(test_data):
            # one random assignment of nodes
            cur_customers = customers.copy()
            for i in range(num_agents-1):
                cur_num_nodes = random.randint(0, len(cur_customers))
                agent_nodes = random.sample(cur_customers, cur_num_nodes)
                init_route_nodes_eval_vrps[idx].append(agent_nodes)
                cur_customers = list(set(cur_customers) - set(agent_nodes))
            # give last agent what's left
            init_route_nodes_eval_vrps[idx].append(cur_customers)

            # evaluate problem with assigned nodes for 50 cost matrices
            fifty_newCostVrps_forOneVrp = []
            customer_coords = vrp[0]["customers_coord"]

            # sample 50 new vrps for the given one (only cost matrices changed through velocities; coords fix)
            for cost_matrix_id in range(50):
                cur_vrp = copy.deepcopy(vrp)
                for agent_id in range(len(vrp)):
                    depot_coords = vrp[0]["depot_coords"][agent_id]
                    customersWithDepot_coords = customer_coords + [depot_coords]  # first customer nodes; last one depot  such that for customer nodes the indices are always correct; and for depot nodes 12, 13, 14,...you need to take the last col
                    # raw Euclidean costs
                    costs = metrics.pairwise.euclidean_distances(customersWithDepot_coords, customersWithDepot_coords)

                    cur_general_agent_velocity = np.round(np.random.uniform(gen_vel[0], gen_vel[1]), 4)
                    cur_agent_edge_velocities = [np.round(np.random.uniform(edge_vel[0], edge_vel[1]), 4) for _ in
                                                 range(num_agent_edges)]
                    id_in_vel = 0
                    for i in range(len(costs)):
                        for j in range(i + 1, len(costs)):  # i+1 to exclude diagonal
                            costs[i, j] = costs[i, j] * (1 / cur_agent_edge_velocities[id_in_vel]) * (
                                    1 / cur_general_agent_velocity)
                            costs[j, i] = costs[j, i] * (1 / cur_agent_edge_velocities[id_in_vel]) * (
                                    1 / cur_general_agent_velocity)
                            id_in_vel += 1

                    # replace agent cost with newly sampled one
                    cur_vrp[agent_id]["costs"] = np.round(costs, 5)
                    cur_vrp[agent_id]["general_agent_velocity"] = cur_general_agent_velocity
                    cur_vrp[agent_id]["agent_edge_velocities"] = cur_agent_edge_velocities
                fifty_newCostVrps_forOneVrp.append(cur_vrp)

            # compute TSP solution for given assigned nodes for 50 cost matrices for current test vrp
            tsp_solutions_50CostMatrices_oneVrp = []
            tsp_solutions_agent_costs_50CostMatrices_oneVrp = []
            node_assignment = init_route_nodes_eval_vrps[idx]  # get assigned nodes from above
            for cost_id,cur_costM_vrp in enumerate(fifty_newCostVrps_forOneVrp):  # for each cost matrix
                vrp_tsp_solutions = []
                vrp_tsp_costs = []
                for agent_id, agent_nodes in enumerate(node_assignment):  # exclude pool state
                    # call tsp for each agent route in the current costM vrp
                    if len(agent_nodes) == 0:  # then no node visited, i.e. no tsp to be computed
                        vrp_tsp_costs.append(0)
                        vrp_tsp_solutions.append([])
                    else: # call tsp
                        cost, route = Benchmark_1Agent_TSP_AltScenario.solve_or_tools(cur_costM_vrp[agent_id],
                                                                                      agent_nodes)
                        vrp_tsp_costs.append(cost[0])  # cost returned in a list
                        vrp_tsp_solutions.append(route[0])  # route returned in a list

                tsp_solutions_50CostMatrices_oneVrp.append(vrp_tsp_solutions)
                tsp_solutions_agent_costs_50CostMatrices_oneVrp.append(vrp_tsp_costs)

            tsp_solutions_50CostMatrices_oneVrp_team_avg_costs = [sum(agent_costs) / num_agents for agent_costs in tsp_solutions_agent_costs_50CostMatrices_oneVrp]
            # assess current node assignment (=cur run) for cur vrp by looking at the median team avg costs (of the 50 sampled cost matrices vrps)
            performance_of_cur_node_assignment = np.median(tsp_solutions_50CostMatrices_oneVrp_team_avg_costs)
            cur_run_team_avg_cost_per_vrp_median50CostMatrices.append(performance_of_cur_node_assignment)

        all_runs_tsp_team_avg_cost_per_vrp_median50CostMatrices.append(cur_run_team_avg_cost_per_vrp_median50CostMatrices)
        all_runs_init_route_nodes_per_vrp.append(init_route_nodes_eval_vrps)
    # all runs done. For each vrp we have tested 100 node assignments, which are assessed with the median team avg cost performance over 50 vrp problems where velocities given the current vrp topology were sampled for new costs.
    # For each test vrp, get the node assignment with the lowest team avg (median of 50 costs matrices) cost
    node_assignments_per_vrp = np.transpose(np.array(all_runs_init_route_nodes_per_vrp), axes=(1,0,2)).tolist()
    median_team_avg_costs_per_vrp = np.array(all_runs_tsp_team_avg_cost_per_vrp_median50CostMatrices).T.tolist()
    min_medianTeamAvgcosts_per_vrp_index = [np.argmin(vrp) for vrp in median_team_avg_costs_per_vrp]
    best_assignments_per_vrp = []
    for vrp_id, cur_cand_assignments in enumerate(node_assignments_per_vrp):
        best_idx = min_medianTeamAvgcosts_per_vrp_index[vrp_id]
        best_assignment = cur_cand_assignments[best_idx]
        best_assignments_per_vrp.append(best_assignment)

    # evaluate the final best assignments per vrp with the actual underlying cost matrix in the test dataset
    # compute final TSP solutions with best node assignment
    final_tsp_solutions = []
    final_tsp_solutions_agent_costs = []
    for vrp_id, best_node_assignment in enumerate(best_assignments_per_vrp):
        # for one vrp in the test data
        vrp_tsp_solutions = []
        vrp_tsp_costs = []
        for agent_id, agent_nodes in enumerate(best_node_assignment):  # exclude pool state
            # call tsp for each agent route in the current vrp
            if len(agent_nodes) == 0:  # then no node visited, i.e. no tsp to be computed
                vrp_tsp_costs.append(0)
                vrp_tsp_solutions.append([])
            else:
                # call tsp solver with original test data cost matrix
                cost, route = Benchmark_1Agent_TSP_AltScenario.solve_or_tools(test_data[vrp_id][agent_id],
                                                                              agent_nodes)
                vrp_tsp_costs.append(cost[0])  # cost returned in a list
                vrp_tsp_solutions.append(route[0])  # route returned in a list

        final_tsp_solutions.append(vrp_tsp_solutions)
        final_tsp_solutions_agent_costs.append(vrp_tsp_costs)

    final_tsp_team_avg_cost_per_vrp = [sum(agent_costs)/num_agents for agent_costs in final_tsp_solutions_agent_costs]
    final_avg_test_set_performance_team_avg_costs = np.mean(final_tsp_team_avg_cost_per_vrp)

    return final_avg_test_set_performance_team_avg_costs




if __name__ == "__main__":
    #### to vary
    test_data_str = "20n_2a_diffTops_Vel05EdgeVel05_600s_test" #"10n_5a_OneTop_5600CostMatrices_train_val"
    ########

    test_dataset = f"/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/data/vrp/newAgVelAgEdgeVel/{test_data_str}.p"

    start = time.time()
    print(f"start {start}")
    print(test_data_str)
    final_performance = run_randomDist_benchmark(test_dataset)
    end = time.time()
    print(test_data_str)
    print(final_performance)
    print(f"run time in seconds for 600 vrps: {end-start}")