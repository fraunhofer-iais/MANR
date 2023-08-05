"""Vehicles Routing Problem (VRP).
Adapted from https://developers.google.com/optimization/routing/vrp
"""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np


def get_routes(solution, routing, manager):
    """Get vehicle routes from a solution and store them in an array."""
    # Get vehicle routes and store them in a two dimensional array whose
    # i,j entry is the jth location visited by vehicle i along its route.
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    return routes

def solve_or_tools(data, relevant_nodes):  # one batch data
    # function solves TSP based on vrp data with an arbitrary subset of nodes to be considered (relevant nodes)

    if type(data) == dict:
        num_vrps = 1
        cur_agent_id = data["agent_id"]
        relevant_nodes = [relevant_nodes]   # list per vrp
        data = [data]   # list per vrp
    else:
        num_vrps = len(data)
        cur_agent_id = data[0]["agent_id"]

    ortools_agent_tot_costs = []
    routes = []

    for i in range(num_vrps):  # solve each vrp in train data
        # Instantiate the data problem.
        cur_data = {}
        # restrict problem to nodes which are in the initial assignment = relevant nodes
        # order in matrix according to order in relevant_nodes, at the end attached: depot

        cost_matrix_dim = len(relevant_nodes[i]) + 1
        cost_matrix = np.zeros((cost_matrix_dim,cost_matrix_dim))  # +1 for depot

        for new_row in range(cost_matrix_dim):
            if new_row != cost_matrix_dim-1: # if not depot
                orig_node_row = relevant_nodes[i][new_row]
            else:   # then depot; always last row/col in orig cost matrix
                orig_node_row = len(data[i]["costs"])-1
            for new_col in range(cost_matrix_dim):
                if new_col != cost_matrix_dim-1:  # if not depot
                    orig_node_col = relevant_nodes[i][new_col]
                else:   # then depot
                    orig_node_col = len(data[i]["costs"])-1

                cost_matrix[new_row][new_col] = (data[i]["costs"][orig_node_row][orig_node_col]*100000).astype(int)

        #for new_row, orig_node_row in enumerate(relevant_nodes[i]):
        #    for new_col, orig_node_col in enumerate(relevant_nodes[i]):
        #        cost_matrix[new_row][new_col] = (data[i]["costs"][orig_node_row][orig_node_col]*100000).astype(int)
        # add depot costs
        #cost_matrix[cost_matrix_dim-1][cost_matrix_dim-1] =

        cur_data['distance_matrix'] = cost_matrix #(data[i][0]["costs"] * 100000).astype(int)
        cur_data['num_vehicles'] = 1
        # depot is last row/col in distance matrix
        cur_data['depot'] = cost_matrix_dim-1


        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(cur_data['distance_matrix']),
                                               cur_data['num_vehicles'], cur_data['depot'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)


        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return cur_data['distance_matrix'][from_node][to_node]


        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        # Set metaheuristic to escape local minima
        #search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        #search_parameters.time_limit.seconds = 30

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
        route = get_routes(solution, routing, manager)[0] # comes in a list

        total_cost = solution.ObjectiveValue() / float(100000)
        ortools_agent_tot_costs.append(total_cost)

        orig_agent_depot = data[i]["depots"][cur_agent_id]
        orig_nodes_route = [orig_agent_depot] + [relevant_nodes[i][node] for node in route[1:-1]] + [orig_agent_depot]
        routes.append(orig_nodes_route)
        # returning routes was only meaningful in the case of the same vrp the whole time
    return ortools_agent_tot_costs, routes
