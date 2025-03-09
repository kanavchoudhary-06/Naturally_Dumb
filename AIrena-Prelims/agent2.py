import numpy as np
import copy
from agent import Agent

class MinimaxPlayer(Agent):
    def __init__(self, size, player_number, adv_number):
        super().__init__(size, player_number, adv_number)
        self.name = "Minimax"

    def step(self):
        best_move = self.free_moves()[0]
        best = -np.inf
        alpha = best
        for move in self.free_moves():
            new_node = self.copy()
            new_node.set_hex(self.player_number, move)
            value = self.alphaBeta(new_node, 1, alpha, np.inf, self.adv_number)
            if value > best:
                best = value
                best_move = move
            alpha = max(alpha, best)
        self.set_hex(self.player_number, best_move)
        return best_move

    def update(self, move_other_player):
        self.set_hex(self.adv_number, move_other_player)

    def alphaBeta(self, node, depth, alpha, beta, player):
        if node.check_win(self.player_number):
            return np.inf
        if node.check_win(self.adv_number):
            return -np.inf
        if depth == 0:
            return self.heuristic(node)

        if player == self.player_number:
            value = -np.inf
            for move in node.free_moves():
                new_node = node.copy()
                new_node.set_hex(self.player_number, move)
                value = max(value, self.alphaBeta(new_node, depth - 1, alpha, beta, self.adv_number))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = np.inf
            for move in node.free_moves():
                new_node = node.copy()
                new_node.set_hex(self.adv_number, move)
                value = min(value, self.alphaBeta(new_node, depth - 1, alpha, beta, self.player_number))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def heuristic(self, node):
        return self._value_player(node, self.player_number)

    def _value_player(self, node, player):
        coordinates = []
        value = 0
        for x in range(node.get_grid_size()):
            for y in range(node.get_grid_size()):
                if ([x, y] not in coordinates) and (node.get_hex([x, y]) == player):
                    n = self._number_connected(player, [x, y], node)
                    coordinates += n[1]
                    if n[0] > value:
                        value = n[0]
        return value

    def _number_connected(self, player, coordinate, node):
        neighbors = [coordinate]
        for neighbor in neighbors:
            n = node.neighbors(neighbor)
            for next_neighbor in n:
                if node.get_hex(next_neighbor) == player and (next_neighbor not in neighbors):
                    neighbors.append(next_neighbor)
        return len(neighbors), neighbors