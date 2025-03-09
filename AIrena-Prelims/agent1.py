import numpy as np
import copy
from agent import Agent
from heapq import heappush, heappop

class CustomPlayer(Agent):
    def __init__(self, size, player_number, adv_number):
        super().__init__(size, player_number, adv_number)
        self.name = "EnhancedMinimax"
        self._possible_moves = [[x, y] for x in range(self.size) for y in range(self.size)]
        self.history_table = {}  # Store history of good moves
        
        # Initialize pattern weights with gradients
        self.weights = {
            'center': 2.0,
            'edge': 1.5,
            'connection': 3.0,
            'blocking': 0.5,
            'path': 2.0,  # Weight for shortest path evaluation
            'block_path': 2.5  # Weight for blocking opponent's paths
        }
        self.gradients = {k: 0.0 for k in self.weights}
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.previous_gradients = {k: 0.0 for k in self.weights}
        
        # Game state for learning
        self.moves_history = []
        self.positions_history = []
        self.game_outcome = None
        
        # Pre-calculate center positions for move ordering
        self.center_positions = self._calculate_center_positions()
        
        # Cache for shortest paths and critical points
        self.path_cache = {}
        self.critical_points_cache = {}

    def _calculate_center_positions(self):
        """Calculate positions closer to center with their weights."""
        center = self.size // 2
        positions = {}
        for x in range(self.size):
            for y in range(self.size):
                dist_from_center = abs(x - center) + abs(y - center)
                positions[(x, y)] = 1.0 / (1.0 + dist_from_center)
        return positions

    def step(self):
        current_position = self._get_board_state()
        
        # Random move with 20% probability
        if np.random.random() < 0.2 and len(self._possible_moves) > 0:
            move = self._possible_moves[np.random.randint(len(self._possible_moves))]
            self._possible_moves.remove(move)
            self.set_hex(self.player_number, move)
            
            # Store move and position for learning
            self.moves_history.append(move)
            self.positions_history.append(current_position)
            return move

        # Use minimax with fixed depth of 2
        best_move = self.free_moves()[0]
        best = -np.inf
        alpha = -np.inf
        beta = np.inf
        
        # Order moves based on heuristics
        moves = self._order_moves(self.free_moves())
        
        for move in moves:
            new_node = self.copy()
            new_node.set_hex(self.player_number, move)
            value = self.alphaBeta(new_node, 2, alpha, beta, self.adv_number)
            
            if value > best:
                best = value
                best_move = move
            alpha = max(alpha, best)
            
            # Store move value in history table
            self.history_table[tuple(move)] = self.history_table.get(tuple(move), 0) + 2**value

        if best_move in self._possible_moves:
            self._possible_moves.remove(best_move)
        self.set_hex(self.player_number, best_move)
        
        # Store move and position for learning
        self.moves_history.append(best_move)
        self.positions_history.append(current_position)
        return best_move

    def _get_board_state(self):
        """Get current board state as a numpy array."""
        state = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                state[x, y] = self.get_hex([x, y])
        return state

    def update(self, move_other_player):
        self.set_hex(self.adv_number, move_other_player)
        if move_other_player in self._possible_moves:
            self._possible_moves.remove(move_other_player)
            
        # Check if game is over and update weights if necessary
        if self.check_win(self.player_number):
            self.game_outcome = 1.0
            self._update_weights()
        elif self.check_win(self.adv_number):
            self.game_outcome = -1.0
            self._update_weights()
        elif len(self._possible_moves) == 0:
            self.game_outcome = 0.0
            self._update_weights()

    def _update_weights(self):
        """Update weights using gradient descent based on game outcome."""
        if not self.moves_history:
            return

        for i, (move, position) in enumerate(zip(self.moves_history, self.positions_history)):
            discount = 0.95 ** (len(self.moves_history) - i - 1)
            
            node = self.copy()
            node._grid = position
            
            # Existing weight gradients...
            center_contribution = self.center_positions[tuple(move)]
            self.gradients['center'] += discount * self.game_outcome * center_contribution
            
            edge_contribution = self._edge_proximity_value(node)
            self.gradients['edge'] += discount * self.game_outcome * edge_contribution
            
            connection_contribution = self._connection_strength(node)
            self.gradients['connection'] += discount * self.game_outcome * connection_contribution
            
            blocking_contribution = -self._value_player(node, self.adv_number)
            self.gradients['blocking'] += discount * self.game_outcome * blocking_contribution
            
            # Path and blocking gradients
            player_path = self._shortest_path_value(node, self.player_number)
            opponent_path = self._shortest_path_value(node, self.adv_number)
            path_contribution = opponent_path - player_path
            self.gradients['path'] += discount * self.game_outcome * path_contribution
            
            # Calculate blocking path gradient
            blocking_value = self._calculate_blocking_value(node, move)
            self.gradients['block_path'] += discount * self.game_outcome * blocking_value

        # Update weights using gradient descent with momentum
        for key in self.weights:
            update = (self.learning_rate * self.gradients[key] + 
                     self.momentum * self.previous_gradients[key])
            self.weights[key] = max(0.1, min(5.0, self.weights[key] + update))
            self.previous_gradients[key] = update
            self.gradients[key] = 0.0

        # Reset history and caches
        self.moves_history = []
        self.positions_history = []
        self.game_outcome = None
        self.path_cache = {}
        self.critical_points_cache = {}

    def _order_moves(self, moves):
        """Order moves based on multiple heuristics including blocking value."""
        move_scores = []
        node = self.copy()
        
        for move in moves:
            score = 0
            # Center preference
            score += self.center_positions[tuple(move)] * self.weights['center']
            # History heuristic
            score += self.history_table.get(tuple(move), 0)
            # Adjacent to existing pieces
            if self._has_adjacent_pieces(move):
                score += 1.0
            # Blocking value
            blocking_value = self._calculate_blocking_value(node, move)
            score += blocking_value * self.weights['block_path']
            
            move_scores.append((score, move))
        
        return [move for _, move in sorted(move_scores, reverse=True)]

    def _has_adjacent_pieces(self, move):
        """Check if move is adjacent to existing pieces."""
        for neighbor in self.neighbors(move):
            if self.get_hex(neighbor) != 0:  # 0 represents empty cell
                return True
        return False

    def heuristic(self, node):
        """Enhanced heuristic function with path blocking evaluation."""
        # Base component size value
        base_value = self._value_player(node, self.player_number)
        
        # Edge proximity for both players
        edge_value = self._edge_proximity_value(node)
        
        # Connection strength
        connection_value = self._connection_strength(node)
        
        # Opponent blocking value
        blocking_value = -self._value_player(node, self.adv_number)
        
        # Shortest path evaluation
        player_path = self._shortest_path_value(node, self.player_number)
        opponent_path = self._shortest_path_value(node, self.adv_number)
        path_value = opponent_path - player_path
        
        # Calculate blocking effectiveness
        critical_points = self._find_critical_points(node, self.adv_number)
        player_pieces = [(x, y) for x in range(self.size) for y in range(self.size) 
                        if node.get_hex([x, y]) == self.player_number]
        blocking_points = sum(1 for piece in player_pieces if piece in critical_points)
        blocking_effectiveness = blocking_points * self.weights['block_path']
        
        return (base_value * self.weights['connection'] + 
                edge_value * self.weights['edge'] + 
                blocking_value * self.weights['blocking'] +
                path_value * self.weights['path'] +
                blocking_effectiveness)

    def _edge_proximity_value(self, node):
        """Calculate value based on proximity to target edges."""
        value = 0
        if self.player_number == 1:  # Vertical connection
            for x in range(node.get_grid_size()):
                for y in range(node.get_grid_size()):
                    if node.get_hex([x, y]) == self.player_number:
                        value += (1.0 / (1.0 + min(x, node.get_grid_size() - 1 - x)))
        else:  # Horizontal connection
            for x in range(node.get_grid_size()):
                for y in range(node.get_grid_size()):
                    if node.get_hex([x, y]) == self.player_number:
                        value += (1.0 / (1.0 + min(y, node.get_grid_size() - 1 - y)))
        return value

    def _connection_strength(self, node):
        """Evaluate the strength of connections between pieces."""
        strength = 0
        for x in range(node.get_grid_size()):
            for y in range(node.get_grid_size()):
                if node.get_hex([x, y]) == self.player_number:
                    connected = 0
                    for neighbor in node.neighbors([x, y]):
                        if node.get_hex(neighbor) == self.player_number:
                            connected += 1
                    strength += connected
        return strength

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

    def _shortest_path_value(self, node, player):
        """Calculate the shortest path value from one edge to another."""
        # Create a unique key for the current board state
        board_key = tuple(tuple(row) for row in node._grid)
        cache_key = (board_key, player)
        
        # Check cache first
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        if player == 1:  # Vertical connection (top to bottom)
            start_edge = [(0, y) for y in range(self.size)]
            end_edge = [(self.size - 1, y) for y in range(self.size)]
        else:  # Horizontal connection (left to right)
            start_edge = [(x, 0) for x in range(self.size)]
            end_edge = [(x, self.size - 1) for x in range(self.size)]
        
        # Find shortest path using Dijkstra's algorithm
        shortest_dist = self._dijkstra_search(node, player, start_edge, end_edge)
        
        # Cache the result
        self.path_cache[cache_key] = shortest_dist
        return shortest_dist

    def _dijkstra_search(self, node, player, start_points, end_points):
        """Implement Dijkstra's algorithm for finding shortest path."""
        # Initialize distances
        distances = {}
        pq = []  # Priority queue
        
        # Initialize start points
        for x, y in start_points:
            if node.get_hex([x, y]) in [0, player]:  # Empty or owned by player
                distances[(x, y)] = 0 if node.get_hex([x, y]) == player else 1
                heappush(pq, (distances[(x, y)], (x, y)))
            
        while pq:
            current_dist, current = heappop(pq)
            
            # Check if we've reached the end
            if current in end_points:
                return current_dist
            
            # Check all neighbors
            for next_pos in node.neighbors([current[0], current[1]]):
                x, y = next_pos
                
                # Calculate new distance (1 for empty cells, 0 for owned cells)
                cell_value = node.get_hex([x, y])
                if cell_value == player:
                    step_cost = 0
                elif cell_value == 0:  # Empty cell
                    step_cost = 1
                else:  # Opponent's cell
                    continue  # Can't pass through opponent's cells
                
                new_dist = current_dist + step_cost
                
                # Update distance if shorter path found
                if (x, y) not in distances or new_dist < distances[(x, y)]:
                    distances[(x, y)] = new_dist
                    heappush(pq, (new_dist, (x, y)))
        
        return float('inf')  # No path found

    def _find_critical_points(self, node, player):
        """Find critical points along the opponent's shortest paths."""
        board_key = tuple(tuple(row) for row in node._grid)
        cache_key = (board_key, player)
        
        if cache_key in self.critical_points_cache:
            return self.critical_points_cache[cache_key]
        
        if player == 1:  # Vertical connection
            start_edge = [(0, y) for y in range(self.size)]
            end_edge = [(self.size - 1, y) for y in range(self.size)]
        else:  # Horizontal connection
            start_edge = [(x, 0) for x in range(self.size)]
            end_edge = [(x, self.size - 1) for x in range(self.size)]
        
        # Find all shortest paths
        critical_points = self._find_all_shortest_paths(node, player, start_edge, end_edge)
        
        self.critical_points_cache[cache_key] = critical_points
        return critical_points

    def _find_all_shortest_paths(self, node, player, start_points, end_points):
        """Find all shortest paths and return critical points."""
        distances = {}
        predecessors = {}
        pq = []
        critical_points = set()
        
        # Initialize distances
        for x, y in start_points:
            if node.get_hex([x, y]) in [0, player]:
                distances[(x, y)] = 0 if node.get_hex([x, y]) == player else 1
                predecessors[(x, y)] = None
                heappush(pq, (distances[(x, y)], (x, y)))
        
        # Find shortest paths
        while pq:
            current_dist, current = heappop(pq)
            
            if current in end_points:
                # Backtrack to find path
                path = []
                pos = current
                while pos is not None:
                    path.append(pos)
                    pos = predecessors.get(pos)
                
                # Add empty cells in path to critical points
                for x, y in path:
                    if node.get_hex([x, y]) == 0:
                        critical_points.add((x, y))
            
            for next_pos in node.neighbors([current[0], current[1]]):
                x, y = next_pos
                cell_value = node.get_hex([x, y])
                
                if cell_value == player:
                    step_cost = 0
                elif cell_value == 0:
                    step_cost = 1
                else:
                    continue
                
                new_dist = current_dist + step_cost
                
                if (x, y) not in distances or new_dist < distances[(x, y)]:
                    distances[(x, y)] = new_dist
                    predecessors[(x, y)] = current
                    heappush(pq, (new_dist, (x, y)))
        
        return critical_points

    def _calculate_blocking_value(self, node, move):
        """Calculate the value of a move for blocking opponent's paths."""
        # Create a new node with the move applied
        test_node = node.copy()
        test_node.set_hex(self.player_number, move)
        
        # Get opponent's path length before and after the move
        original_path = self._shortest_path_value(node, self.adv_number)
        new_path = self._shortest_path_value(test_node, self.adv_number)
        
        # Calculate how much the move increases opponent's path length
        path_increase = new_path - original_path
        
        # Check if move is on opponent's critical points
        critical_points = self._find_critical_points(node, self.adv_number)
        is_critical = tuple(move) in critical_points
        
        return path_increase * (2 if is_critical else 1)