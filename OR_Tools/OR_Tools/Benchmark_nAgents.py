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

def solve_or_tools(data, init_routes = None):  # one batch data
    num_vrps = len(data)
    num_agents = len(data[0])
    num_customers = len(data[0][0]["customers"])
    ortools_team_avg_costs = []
    routes = []

    for i in range(num_vrps):  # solve each vrp in train data
        # Instantiate the data problem.
        cur_data = {}
        cur_data['num_vehicles'] = num_agents
        cur_data['starts'] = [int(elem) for elem in data[i][0]["depots"]]   #e.g. [10, 11]
        cur_data['ends'] = cur_data['starts']
        for k in range(num_agents):
            # create a "global" cost matrix per agent (with its agent specific customers nodes and a dummy zero row/col per depot node of a different agent)
            # insert dummy row/coö of zero costs to depot nodes of other agents (needed since in order to specific depot nodes we need a global node index system)
            tmp_dist_matrix = np.zeros((num_customers+num_agents,num_customers+num_agents)).astype(int)
            agent_custom_dists = (data[i][k]["costs"] * 100000).astype(int)[:num_customers, :num_customers]
            agent_depot_dist_row = (data[i][k]["costs"] * 100000).astype(int)[num_customers, :]  # row=col, last entry is always zero!
            tmp_dist_matrix[:num_customers, :num_customers] = agent_custom_dists
            # now insert depot cost of agent at right position (first agent belongs first empty row, 2nd the 2nd,...)

            # first bring depot col/row to correct size by inserting a zero per agent at the end (works since costs to own depot node and others are all zero; only important thing are the distances to the customer nodes are correct in the front indices
            agent_depot_dist_row_new = np.hstack((agent_depot_dist_row, np.zeros(num_agents-1)))
            # depot of agent k inserted at position num_customers+k
            tmp_dist_matrix[num_customers + k,:] = agent_depot_dist_row_new  # as row
            tmp_dist_matrix[:, num_customers + k] = agent_depot_dist_row_new  # as col
            cur_data[f'distance_matrix_{k}'] = tmp_dist_matrix

        if init_routes:
            # The initial routes do not include the depot.
            cur_data['initial_routes'] = init_routes[i]

        """
        # first zero after --- for depot is always taken from agent_custom_dists
        
        # for agent 1 with depot 1 it looks like: 
        
                  custom0  custom1 ... customN  depot0  depot1  depot2
        custom0   -------------------------       0        |       0
        custom1   |                       |       0        |       0
        ...       |                       |       0        |       0
        customN   -------------------------       0        |       0
        depot0    0          0     ...    0       0        0       0
        depot1    -------------------------       0        0       0
        depot2    0          0     ...    0       0        0       0
        """

        # include all depots in cost matrix (needed to say which agent is starting from there)
        # zero cost between depots of agents is not a problem (might think that inf is necessary or that we need to set constraints that e.g. agent 0 cannot visit depot of agent 1
        # but; since you cannot visit a node multiple times and each agent needs to (start and) end at its depot, it should work just like that

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(cur_data['distance_matrix_0']), cur_data['num_vehicles'], cur_data['starts'], cur_data['ends'])
        # init: num_nodes, num_vehicles

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # integrate agent specific cost matrices

        callback_indices = []

        for k in range(cur_data['num_vehicles']):  # for each agent

            def vehicle_callback(from_index, to_index, vehicle_id=k):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return cur_data[f'distance_matrix_{vehicle_id}'][from_node][to_node]

            callback_index = routing.RegisterTransitCallback(vehicle_callback)
            callback_indices.append(callback_index)

        for k in range(cur_data['num_vehicles']):  # for each agent
            # Define cost of each arc.
            routing.SetArcCostEvaluatorOfVehicle(callback_indices[k], k)



        #routing.AddDimensionWithVehicleTransits(
        #    callback_indices,
        #    0,
        #    max,
        #    False,
        #    'DimensionName')


        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        ######search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)


        # Set metaheuristic to escape local minima
        #search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        #search_parameters.time_limit.seconds = 30

        # Solve the problem.
        if init_routes:
            initial_solution = routing.ReadAssignmentFromRoutes(cur_data['initial_routes'], True)
            solution = routing.SolveFromAssignmentWithParameters(initial_solution, search_parameters)
        else:
            solution = routing.SolveWithParameters(search_parameters)

        route = get_routes(solution, routing, manager)

        total_cost = solution.ObjectiveValue() / float(100000)
        team_avg_cost = total_cost / num_agents
        ortools_team_avg_costs.append(team_avg_cost)
        routes.append(route)
        # returning routes was only meaningful in the case of the same vrp the whole time
    return ortools_team_avg_costs, routes


"""
        def distance_callback(from_index, to_index):
            #Returns the distance between the two nodes.
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return cur_data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        
        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


"""