
###########################################
# TSP Tree
# Stores Tree Data to solve shortest path
###########################################
class TSPGreedy:

    ###########################################
    # def __init__
    # Initialize Tree
    #
    # Time Complexity:
    # Space Complexity:
    ###########################################
    def __init__(self, scenario):
        self.cities = scenario.getCities()
        self.cities_length = len(self.cities)
        self.edge_exists = scenario.getEdgeExists()

