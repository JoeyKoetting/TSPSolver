#!/usr/bin/python3

from which_pyqt import PYQT_VER
import cProfile

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

from TSPClasses import *
import heapq


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None
        self._bssf = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution, 
        time spent to find solution, number of permutations tried during search, the 
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                self._bssf = bssf
                foundTour = True

        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for 
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def greedy(self, time_allowance=60.0):
        pass

    """
        Generates the base cost matrix. O(n^2)
    
    """
    def _genBaseMatrix(self, cities):
        ncities = len(cities)
        cost_matrix = np.full((ncities, ncities), np.inf)

        lower_bound = 0

        for i in range(ncities):
            for j in range(ncities):
                cost_matrix[i, j] = cities[i].costTo(cities[j])

        matrix = cost_matrix
        # Reduce rows
        for i in range(ncities):
            row_min = matrix[i].min()
            if row_min != np.inf:
                lower_bound += row_min
                for j in range(ncities):
                    matrix[i,j] = matrix[i,j] - row_min if matrix[i, j] != np.inf else matrix[i, j]

        # Reduce rows
        for j in range(ncities):
            col_min = matrix[:, j].min()
            if col_min != np.inf:
                lower_bound += col_min
                for j in range(ncities):
                    matrix[i, j] = matrix[i, j] - col_min if matrix[i, j] != np.inf else matrix[i, j]

        return cost_matrix, lower_bound

    # Updates the cost matrix and lower bound and returns them. O(n^2)
    def _getLowerBound(self, matrix, state_list, parent_bound, next):
        bound = parent_bound
        ncities = matrix.shape[0]
        matrix = matrix.copy()

        bound += matrix[state_list[-1], next]

        cities_left = self._getNextPos(state_list, ncities)

        # Set source of edge row to inf
        matrix[state_list[-1]].fill(np.inf)

        # Set dest of edge col to inf
        matrix[:, next].fill(np.inf)

        # Set edge from dest to src to inf
        matrix[next, state_list[-1]] = np.inf

        # Row reduction and add to bound, only need to check cities left rows and cols
        for row in cities_left:
            row_min = np.min(matrix[row, cities_left])
            if row_min != np.inf:
                bound += row_min
                matrix[row, cities_left] -= row_min

        for col in cities_left:
            col_min = np.min(matrix[cities_left, col])
            if col_min != np.inf:
                bound += col_min
                matrix[cities_left, col] -= col_min

        return bound, matrix


    # Returns list of cities that haven't been visited yet O(n)
    def _getNextPos(self, state, ncities):
        return [i for i in range(ncities) if i not in state]

    """
     Main function for branch and bound
     --Time Complexity can theoretically be up to O(n-1!) however in practice
     since we are optimizing rather than exhaustively searching it will generally be less. 
     --Space complexity is O(n^2) for each cost matrix * the number of nodes which can also get up to O(n-1!) theoretically 
     but is reduced in practice. 
    """
    def branchAndBound(self, time_allowance=60.0):
        cities = self._scenario.getCities()
        ncities = len(cities)
        queue = []
        start_time = time.time()
        count = 0
        sols = 1
        pruned = 0
        best_updates = 0
        max_queue_len = 1
        edges = self._scenario._edge_exists
        base_matrix, base_bound = self._genBaseMatrix(cities)

        # Initialize bssf
        n_tries = 100
        self.defaultRandomTour()
        bssf = self._bssf

        # Try many combinations of random and take the best one O(n^2)
        for _ in range(n_tries):
            self.defaultRandomTour()
            if self._bssf.cost < bssf.cost:
                bssf = self._bssf

        # Insert states and lower bound for each city onto queue that's lower_bound is less than the cost O(n)
        for i in range(ncities):
            heapq.heappush(queue, (base_bound, [i], base_matrix)) # O(log n)


        # Could be up to (n-1)! factorial permutations, in practice will
        # be less as we prune branches that are less promising.
        # Each loop will be O(n^3)
        while len(queue) is not 0 and time.time() - start_time < time_allowance:

            parent_bound, state_list, parent_matrix = heapq.heappop(queue) # O(1) since it's a heap queue
            bound_mod = (1.0-0.02*(ncities-len(state_list)))
            if parent_bound > bssf.cost*bound_mod:
                pruned += 1
                continue

            if len(state_list) == ncities:
                route = [cities[i] for i in state_list]
                sol = TSPSolution(route)
                sols += 1
                if sol.cost < bssf.cost:
                    bssf = sol
                    best_updates += 1
                    continue

            # Else if it hasn't yet...
            next_poss_cities = self._getNextPos(state_list, ncities)
            for i in next_poss_cities: # O(n^3)
                if(edges[state_list[-1]][i]):
                    lower_bound, child_matrix = self._getLowerBound(parent_matrix, state_list, parent_bound, i)  # O(n^2)
                    if lower_bound < bssf.cost:
                        next_state_list = state_list + [i]
                        count += 1
                        heapq.heappush(queue, (lower_bound, next_state_list, child_matrix)) # O(log(n)) for heap queue
                    else:
                        pruned += 1

            max_queue_len = max(max_queue_len, len(queue))

        pruned += len(queue)
        results = {}
        end_time = time.time()
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = pruned

        print("N cities is ", ncities)
        print("Running time is ", end_time - start_time)
        print("Cost of best tour is ", bssf.cost)
        print("Max stored states is ", max_queue_len)
        print("Total states created is ", count)
        print("Bssf updates is ", best_updates)
        print("Total pruned is ", pruned)
        print("Total sols is ", sols)

        print(" ")
        return results

    # Generates all possible sets of k length in an arr # O(n^k)
    def generatePerms(self, arr, i, k_perm, k, perms):

        if len(k_perm) is k:
            perms.append(k_perm)
            return
        elif i is len(arr):
            return
        else:
            # Two possible actions, add arr[i] to perm
            k_perm1 = k_perm[:]
            k_perm1.append(arr[i])
            self.generatePerms(arr, i+1, k_perm1, k, perms)

            # Or don't add i
            k_perm2 = k_perm[:]
            self.generatePerms(arr, i+1, k_perm2, k, perms)

    #  Returns a list representing cities if it's a valid directed route, none otherwise O(k+n)
    def directedRoute(self, route_dict, removed_edges, added_edges):

        # Update route_dict
        for edge in removed_edges:
            route_dict[edge[0]].remove(edge[1])
            route_dict[edge[1]].remove(edge[0])
        for edge in added_edges:
            route_dict[edge[0]].append(edge[1])
            route_dict[edge[1]].append(edge[0])

        # Check if the route is valid
        visited = np.zeros(len(route_dict), dtype=np.bool)
        current_city = 0
        directed_route = [current_city]

        for _ in range(len(route_dict)-1):
            if visited[current_city]:
                directed_route = None
                break
            else:
                visited[current_city] = True

            conn_cities = route_dict[current_city]
            if len(conn_cities) != 2:
                directed_route = None
                break

            if not visited[conn_cities[0]]:
                current_city = conn_cities[0]
            elif not visited[conn_cities[1]]:
                current_city = conn_cities[1]
            else:
                directed_route = None
                break

            directed_route.append(current_city)

        # Put route_dict back to how it was
        for edge in added_edges:
            route_dict[edge[0]].remove(edge[1])
            route_dict[edge[1]].remove(edge[0])
        for edge in removed_edges:
            route_dict[edge[0]].append(edge[1])
            route_dict[edge[1]].append(edge[0])

        return directed_route


    """ Returns a list of route re-orderings based on k for the number of possible neighbors of k-opt
        Complexity of the following code will include big O in terms of k, but k is essentially a constant
    """
    def getNeighbors(self, n_cities, k):  # O(n^(k+1))
        neighbors = set()

        # Used for building route with permutations
        route_dict = {}
        src_city = 0
        dst_city = 1
        for i in range(n_cities): # O(n)
            route_dict[src_city+i] = [(dst_city+i) % (n_cities)]
        for i in range(n_cities):
            route_dict[((dst_city+i) % n_cities)].append(src_city + i)

        # Each perm is a tuple containing the index of the src node of the edge to delete in the route
        src_node_perms = []
        self.generatePerms(list(range(n_cities)), 0, [], k, src_node_perms)  # O(n^k)

        # Generate the nodes available for new edges based on src nodes
        for rem_src_nodes in src_node_perms:  # (O(n^k))
            rem_dst_nodes = []
            removed_edges = []
            available_nodes = rem_src_nodes[:]

            for src_node_ind in rem_src_nodes:  # O(k)
                dst_node_ind = (src_node_ind+1) % n_cities
                rem_dst_nodes.append(dst_node_ind)
                removed_edges.append((src_node_ind, dst_node_ind))

                if dst_node_ind not in available_nodes:
                    available_nodes.append(dst_node_ind)

            possible_edges = []
            self.generatePerms(available_nodes, 0, [], 2, possible_edges)
            # There will be O(k^2) possible edges

            edge_combos = []
            self.generatePerms(possible_edges, 0, [], k, edge_combos)
            # There will be O(k^2k) possible edge_combos

            for added_edges in edge_combos:  # O(n(k^2k+1))
                new_route = self.directedRoute(route_dict, removed_edges, added_edges)  # O(n+k)
                if new_route is not None:
                    neighbors.add(tuple(new_route))

        return neighbors

    def getNeighborRoute(self, perm, old_route):
        route = []
        for i in range(len(perm)):
            route.append(old_route[perm[i]])

        return route

    def fancy(self, time_allowance=60.0):
        cProfile.runctx('self.fancyReal()', None, locals())
        return self._results
    
    # K-Opt algorithm 
    def fancyReal(self, time_allowance=60.0, k=3):
        cities = self._scenario.getCities()
        ncities = len(cities)
        count = 0
        best_updates = 0
        sols = 0

        if k > ncities:
            return "Error k must be less than or equal to the number of edges"

        start_time = time.time()
        edges = self._scenario._edge_exists

        # Initialize bssf: Try many combinations of random and take the best one O(n^2)
        n_tries = 1
        self.defaultRandomTour()
        bssf = self._bssf
        for _ in range(n_tries):
            self.defaultRandomTour()
            if self._bssf.cost < bssf.cost:
                bssf = self._bssf

        done = False
        # Generate all possible neighbor permutations for ncities and k
        neighbor_dicts = self.getNeighbors(ncities, k)  # Most expensive portion: O(n^(k+1))
        max_stored_states = len(neighbor_dicts)

        # Main k-opt search all routes in the queue for a better solution than bssf
        # Unknown runtime, could be non polynomial in worst case
        while not done and time.time() - start_time < time_allowance:
            # Find next best
            next_best = bssf
            neighborhood = neighbor_dicts
            for route_perm in neighborhood:
                count += 1
                neighbor_route = self.getNeighborRoute(route_perm, bssf.route)
                sol = TSPSolution(neighbor_route)
                if sol.cost != np.inf:
                    sols += 1
                if sol.cost < next_best.cost:
                    next_best = sol

            # If there is a better neighbor in neighborhood then update bssf and keep going
            if next_best.cost < bssf.cost:
                bssf = next_best
                best_updates += 1
            # If there isn't a better neighbor then we finish with our solution. 
            else:
                done = True

        results = {}
        end_time = time.time()
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = max_stored_states
        results['total'] = None
        results['pruned'] = None

        print("N cities is ", ncities)
        print("Running time is ", end_time - start_time)
        print("Cost of best tour is ", bssf.cost)
        print("Max stored states is ", max_stored_states)
        print("Total states created is ", count)
        print("Bssf updates is ", best_updates)
        print("Total sols is ", sols)

        self._results = results
        return results

if __name__ == '__main__':
    pass
