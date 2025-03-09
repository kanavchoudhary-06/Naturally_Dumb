import numpy as np
import copy

class Agent:
    def __init__(self, size, player_number, adv_number):
        self.size = size
        self.player_number = player_number
        self.adv_number = adv_number
        self._grid = np.zeros(shape=(self.size, self.size))
        self.name = "Agent"

    def step(self):
        """To be implemented by subclasses"""
        pass

    def update(self, move_other_player):
        """To be implemented by subclasses"""
        pass
    
    def get_grid_size(self):
        """Returns size of the grid"""
        return self.size
    
    def set_hex(self, player, coordinate):
        """Set a hexagon to a player"""
        self._grid[coordinate[1], coordinate[0]] = player
        
    def get_hex(self, coordinate):
        """Get the player number at a coordinate"""
        return self._grid[coordinate[1], coordinate[0]]
        
    def neighbors(self, coordinates):
        """Get valid neighboring coordinates"""
        neighbors = []
        directions = [[0, -1], [1, -1], [1, 0], [0, 1], [-1, 1], [-1, 0]]
        for dir in directions:
            new_coordinates = [coordinates[0]+dir[0], coordinates[1]+dir[1]]
            if 0 <= new_coordinates[0] < self.size and 0 <= new_coordinates[1] < self.size:
                neighbors.append(new_coordinates)
        return neighbors
    
    def check_win(self, player):
        """Check if a player has won"""
        if player == 1:
            start = [[0, y] for y in range(self.size)]
            end = self.size - 1
            axis = 0
        else:
            start = [[x, 0] for x in range(self.size)]
            end = self.size - 1
            axis = 1

        for s in start:
            if self._dfs(s, player, end, axis, set()):
                return True
        return False

    def _dfs(self, current, player, end, axis, visited):
        """DFS helper for win checking"""
        if current[axis] == end:
            return True
        visited.add(tuple(current))
        for neighbor in self.neighbors(current):
            if tuple(neighbor) not in visited and self.get_hex(neighbor) == player:
                if self._dfs(neighbor, player, end, axis, visited):
                    return True
        return False

    def free_moves(self):
        """Get all available moves"""
        return [[x, y] for x in range(self.size) for y in range(self.size) if self.get_hex([x, y]) == 0]
    
    def copy(self):
        """Create a deep copy of the agent"""
        new_agent = Agent(self.size, self.player_number, self.adv_number)
        new_agent._grid = copy.deepcopy(self._grid)
        return new_agent