import numpy as np

class Grid:
    def __init__(self, size):
        self._size = size
        self._grid = np.zeros(shape=(self._size, self._size))

    def get_size(self):
        return self._size

    def set_hex(self, player, coordinate):
        self._grid[coordinate[1], coordinate[0]] = player

    def get_hex(self, coordinate):
        return self._grid[coordinate[1], coordinate[0]]

    def neighbors(self, coordinates):
        neighbors = []
        directions = [[0, -1], [1, -1], [1, 0], [0, 1], [-1, 1], [-1, 0]]
        for dir in directions:
            new_coordinates = [coordinates[0]+dir[0], coordinates[1]+dir[1]]
            if 0 <= new_coordinates[0] < self._size and 0 <= new_coordinates[1] < self._size:
                neighbors.append(new_coordinates)
        return neighbors

    def check_win(self, player):
        if player == 1:
            start = [[0, y] for y in range(self._size) if self.get_hex([0, y]) == player]
            end = self._size - 1
            axis = 0
        else:
            start = [[x, 0] for x in range(self._size) if self.get_hex([x,0]) == player ]
            end = self._size - 1
            axis = 1

        for s in start:
            if self._dfs(s, player, end, axis, set()):
                    return True
        return False

    def _dfs(self, current, player, end, axis, visited):
        if current[axis] == end:
            return True
        visited.add(tuple(current))
        for neighbor in self.neighbors(current):
            if tuple(neighbor) not in visited and self.get_hex(neighbor) == player:
                if self._dfs(neighbor, player, end, axis, visited):
                    return True
        return False

    def free_moves(self):
        return [[x, y] for x in range(self._size) for y in range(self._size) if self.get_hex([x, y]) == 0]
