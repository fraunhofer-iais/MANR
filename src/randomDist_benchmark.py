import models.data_utils.data_utils as data_utils
#from OR_Tools.OR_Tools import Benchmark_1Agent_TSP_AltScenario
from OR_Tools import Benchmark_1Agent_TSP_AltScenario
import random
import numpy as np
import pickle


def run_randomDist_benchmark(test_dataset, test_data_str):
    test_data = data_utils.load_dataset(test_dataset, "dummy")
    test_data = test_data[:628].copy()
    test_data_size = len(test_data)
    print(f"test data size: {test_data_size}")

    num_agents = len(test_data[0])
    customers = test_data[0][0]["customers"]
    num_customers = len(customers)

    all_runs_tsp_team_avg_cost_per_vrp = []
    all_runs_tsp_solutions = []

    for run_id in range(100):
        print(f"Run {run_id}: Randomly distributing init nodes")
        init_route_nodes_eval_vrps = [[] for _ in range(test_data_size)]
        for idx, vrp in enumerate(test_data):
            # init solution per vrp: random distribution of nodes first
            cur_customers = customers.copy()
            for i in range(num_agents-1):
                cur_num_nodes = random.randint(0, len(cur_customers))
                agent_nodes = random.sample(cur_customers, cur_num_nodes)
                init_route_nodes_eval_vrps[idx].append(agent_nodes)
                cur_customers = list(set(cur_customers) - set(agent_nodes))
            # give last agent what's left
            init_route_nodes_eval_vrps[idx].append(cur_customers)

        # compute TSP solution with assigned nodes
        tsp_solutions = []
        tsp_solutions_agent_costs = []
        for vrp_id, node_assignment in enumerate(init_route_nodes_eval_vrps):
            # for one vrp in the test data
            vrp_tsp_solutions = []
            vrp_tsp_costs = []
            for agent_id, agent_nodes in enumerate(node_assignment):  # exclude pool state
                # call tsp for each agent route in the current vrp
                if len(agent_nodes) == 0:  # then no node visited, i.e. no tsp to be computed
                    vrp_tsp_costs.append(0)
                    vrp_tsp_solutions.append([])
                #elif len(agent_nodes) == 1:  # one node visited; no order to optimize
                #    vrp_tsp_costs.append(np.nan)
                #    vrp_tsp_solutions.append([np.nan])
                else:
                    # call tsp
                    cost, route = Benchmark_1Agent_TSP_AltScenario.solve_or_tools(test_data[vrp_id][agent_id],
                                                                                  agent_nodes)
                    vrp_tsp_costs.append(cost[0])  # cost returned in a list
                    vrp_tsp_solutions.append(route[0])  # route returned in a list

            tsp_solutions.append(vrp_tsp_solutions)
            tsp_solutions_agent_costs.append(vrp_tsp_costs)

        tsp_team_avg_cost_per_vrp = [sum(agent_costs)/num_agents for agent_costs in tsp_solutions_agent_costs]
        all_runs_tsp_team_avg_cost_per_vrp.append(tsp_team_avg_cost_per_vrp)
        all_runs_tsp_solutions.append(tsp_solutions)

    runs_per_vrp = np.array(all_runs_tsp_team_avg_cost_per_vrp).T.tolist()
    best_run_per_vrp = [min(vrp_run_costs) for vrp_run_costs in runs_per_vrp]
    final_avg_test_set_performance_team_avg_costs = np.mean(best_run_per_vrp)

    with open(f'benchmark_results/all_runs_tsp_solutions_{test_data_str}.pkl', 'wb') as f:
        pickle.dump(all_runs_tsp_solutions, f)
    with open(f'benchmark_results/all_runs_team_avg_costs_{test_data_str}.pkl', 'wb') as f:
        pickle.dump(all_runs_tsp_team_avg_cost_per_vrp, f)

    return final_avg_test_set_performance_team_avg_costs

if __name__ == "__main__":
    #### to vary
    test_data_str = "10n_5a_diffTops_Vel05EdgeVel05_600s_test"
    ########

    test_dataset = f"/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/data/vrp/newAgVelAgEdgeVel/{test_data_str}.p"
    final_performance = run_randomDist_benchmark(test_dataset, test_data_str)
    print(test_data_str)
    print(final_performance)