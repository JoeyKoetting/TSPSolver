from copy import deepcopy


###########################################
# TSP Tree
# Stores Tree Data to solve shortest path
#
# Each Tree stores a reduced matrix, either it is initialized
# when the root is created or it is inherited from its
# parent.
#
# Each tree contains a path of cities that each
# of its children add on to when they are visited.
# When the length of the route equals the cities,
# it is designated as a possible solution
#
# It has two main methods, reduce list and visit
#
# When reduce list is called it removes the
# lowest value from each row and column
# then reduces each other element of each
# respective row and column
#
# When visit is called, it visits each of the
# edges to the other cities and returns a list
# of new children for each city visited.
###########################################
class TSPTree:

    ###########################################
    # def __init__
    # Initialize Tree
    #
    # Time Complexity: O(n^2) Calls make_list method
    # Space Complexity: O(1) init some variables
    ###########################################
    def __init__(self, cities, cost=0, path=[], my_list=None, visit_row=0):

        # init variables
        self.cities = cities
        self.cities_length = len(self.cities)
        self.cost = cost
        self.path = path
        self.path_length = len(self.path)
        self.my_list = my_list
        self.visit_row = visit_row

        # make new list is no list
        if self.my_list is None:
            self.my_list = self.make_list()

        self.my_list, self.cost = self.reduce_list(self.my_list, cost)

    ###########################################
    # def make_list
    # Initialize cost list
    #
    # Time Complexity: O(n^2) Biggest nest is two loops
    # Space Complexity: O(n^2) Makes a 2 dimensional list
    ###########################################
    def make_list(self):

        # makes an empty list
        my_list = [[] for _ in range(self.cities_length)]

        # fills list with distances
        for i in range(self.cities_length):
            for j in range(self.cities_length):
                my_list[i].append(self.cities[i].costTo(self.cities[j]))

        return my_list

    ###########################################
    # def reduce_list
    # Reduce List
    #
    # Time Complexity: O(n^2) Biggest nest is two loops
    # Space Complexity: O(1) Just init a few variables
    ###########################################
    def reduce_list(self, my_list, cost):

        # init var
        lowest_index = None

        # for each row, reduce
        for i in range(self.cities_length):
            for j in range(self.cities_length):

                # if zero we good
                if self.my_list[i][j] == 0:
                    lowest_index = None
                    break

                # if inf, pass
                if self.my_list[i][j] == float('inf'):
                    continue

                # if lowest and not zero, set lowest
                if lowest_index is None or (my_list[i][j] != 0 and my_list[i][j] < my_list[i][lowest_index]):
                    lowest_index = j

            # calc cost
            if lowest_index is not None:
                tmp_cost = my_list[i][lowest_index]
                cost += my_list[i][lowest_index]

                # re-balance row
                for k in range(self.cities_length):

                    if my_list[i][k] == 0 or my_list[i][k] == float('inf'):
                        continue
                    my_list[i][k] -= tmp_cost

            lowest_index = None

        # for each column, reduce
        for j in range(self.cities_length):
            for i in range(self.cities_length):

                # if found zero already pass
                if self.my_list[i][j] == 0:
                    lowest_index = None
                    break

                # if inf, pass
                if self.my_list[i][j] == float('inf'):
                    continue

                # if lowest and not zero, set lowest
                if lowest_index is None or (my_list[i][j] != 0 and my_list[i][j] < my_list[lowest_index][j]):
                    lowest_index = i

                # calc cost
            if lowest_index is not None:
                tmp_cost = my_list[lowest_index][j]
                cost += my_list[lowest_index][j]

                # re-balance row
                for k in range(self.cities_length):

                    if my_list[k][j] == 0 or my_list[k][j] == float('inf'):
                        continue
                    my_list[k][j] -= tmp_cost

            lowest_index = None

        return my_list, cost

    ###########################################
    # def visit
    # Create a list of children tree components
    #
    # Time Complexity: O(n^3) Loops through all
    # cities n times, then calls the constructor
    # for TSP tree which is O(n^2)
    # Space Complexity: O(n^2) For each city create
    # at most n-1 other children
    ###########################################
    def visit(self):

        tree_list = []

        for j in range(self.cities_length):
            cost_to_visit = deepcopy(self.my_list[self.visit_row][j])

            if cost_to_visit == float('inf'):
                continue

            # add new city to path
            tmp_path = deepcopy(self.path)
            tmp_path.append(self.cities[j])

            # calculate new cost
            cost_to_visit += self.cost

            # reduce visited row to inf
            tmp_list = deepcopy(self.my_list)
            for i in range(self.cities_length):
                tmp_list[self.visit_row][i] = float('inf')

            # reduce visited column to inf
            for k in range(self.cities_length):
                tmp_list[k][j] = float('inf')

            tree_list.append(TSPTree(self.cities, cost_to_visit, tmp_path, tmp_list, j))

        return tree_list

    ###########################################
    # def __lt__
    # Comparison method to compare objects of TSP TREE for heapq
    # I chose to determine the priority by the cost of the reduced tree
    # and how many cities it has visited. This means, that I will
    # find a lot of leaf nodes, and I will prune a lot of trees.
    #
    # Time Complexity: O(1) Just a few comparison operators
    # Space Complexity: O(1) Stores a few variables
    ###########################################
    def __lt__(self, other):
        return self.cost - (self.path_length * 250) < \
            other.cost - (other.path_length * 250)

