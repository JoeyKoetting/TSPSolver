#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
from TSPTree import *

import heapq
import cProfile



class TSPSolver:
    def __init__( self, gui_view ):
        self._scenario = None

    def setupWithScenario( self, scenario ):
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

    def defaultRandomTour( self, time_allowance=60.0 ):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time()-start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation( ncities )
            route = []
            # Now build the route using the random permutation
            for i in range( ncities ):
                route.append( cities[ perm[i] ] )
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
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

    def greedy( self,time_allowance=60.0 ):
        pass



    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''

    ###########################################
    # def brandAndBound
    # Use a branch and bound algorithm to find the shortest path circuit in a graph
    #
    # Time Complexity: Between worst case O(n! * n^3) and best case O(n^3)
    # Because each child creates at worst n - 1 children and each of those
    # children cost O(n^3) when they are visited/ called, the worst case scenario is
    # scary big number. However because we are using branch and bound, my
    # algorithm prunes bad paths that do not lead to a good solution. If I pruned
    # every bad solution then I would only need to account for the time it takes to visit
    # each child which is a small time cost respectively.
    # Space Complexity: See Complexity of Priority Queue
    ###########################################

    ###########################################
    # Initial Approach for BSSF
    # I used the built in default random tour, which has a
    # relatively small call cost, and found the smallest
    # cost in the number of times it was called.
    # Depending on the size of the problem, it is
    # called n ^ 3 amount of times. Because it scales
    # with the problem size, it is called a useful amount
    # of times regardless of the size of the problem
    # being tried to solve.
    #
    # Time Complexity: Teacher given Code
    # Space Complexity: Teacher given Code
    ###########################################

    ###########################################
    # Complexity of Priority Queue
    # I implemented with the priority queue using heapq
    # heapq is a built in priority queue that uses a heap
    # structure to store all the nodes
    #
    # Time Complexity: O(log n) push and O(log n) pop
    # Space Complexity: In worst case scenario this would be n!
    # because every child could create n - 1 children.
    # However because a lot of children get pruned it is more
    # like n^3 or n^4, from how my algorithm prunes the heap
    ###########################################
    def branchAndBound( self, time_allowance=60.0 ):

        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        max = 0
        total = 1
        solutions = 0
        pruned = 0
        cost = float('inf')
        bssf = None
        start_time = time.time()

        ###############################
        # DO DEFAULT REAL QUICK
        ###############################
        for i in range(ncities ^ 3):
            results = self.defaultRandomTour()
            if cost > results['cost']:
                cost = results['cost']


        ###############################
        # MY CODE STARTS HERE
        ###############################

        # Add initial tree to heap
        initial_tree = TSPTree(cities)
        heap = []
        heapq.heappush(heap, initial_tree)

        while len(heap) > 0 and time.time()-start_time < time_allowance:
            current_tree = heapq.heappop(heap)
            tree_list = current_tree.visit()
            total += len(tree_list)
            for i in range(len(tree_list)):

                # Found a solution
                if tree_list[i].path_length == ncities:
                    # print("found solution")
                    solutions += 1

                    if bssf is None or cost >= tree_list[i].cost:
                        bssf = TSPSolution(tree_list[i].path)
                        cost = tree_list[i].cost
                        foundTour = True
                    break

                if tree_list[i].cost <= cost:
                    heapq.heappush(heap, tree_list[i])
                else:
                    pruned += 1

                if len(heap) > max:
                    max = len(heap)

        ###############################
        # MY CODE ENDS HERE
        ###############################

        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = solutions
        results['soln'] = bssf
        results['max'] = max
        results['total'] = total
        results['pruned'] = pruned

        return results

    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''

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



