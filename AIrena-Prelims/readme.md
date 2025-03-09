## Introduction
AI-rena is a two-player strategy game played on a hexagonal grid. Players take turns capturing hexagons, with the goal of creating a connected path between opposing edges of the board. The first player aims to connect the top and bottom edges, while the second player tries to connect the left and right edges.

This repository provides a basic environment and a placeholder adversory for you to test against. Final evalation will happen against a more spohesticated adversory

## Installation

### Requirements
- Python 3.6+
- NumPy
- Tkinter

For a complete list of allowed libraries, see [allowed_libraries.md](allowed_libraries.md).

## Running the Game

There are multiple ways to run the game:

### GUI Mode
```
python main.py --players <player1> <player2> --size <board_size>
```
Example:

```
python main.py --players Agent minimax --size 5
```
**Non-GUI Mode**

```
python main.py --players <player1> <player2> --size <board_size> --gui False
```
Example:
```
python main.py --players minimax minimax --size 7 --gui False
```

**Evaluation Mode**
To run 25 games with random player assignments and gather performance statistics:
```
python main.py --eval
```

### Parameters
- `--players`: Specify the player types (choices: "Agent", "minimax")
- `--size`: Specify the board size (default: 9)
- `--gui`: Enable or disable the graphical interface (default: True)
- `--eval`: Run evaluation mode (25 games with various player configurations)

## Game Rules

1. The game is played on a hexagonal grid
2. Players take turns placing their pieces on empty hexagons
3. Player 1 (red) aims to connect the top and bottom edges
4. Player 2 (blue) aims to connect the left and right edges
5. The first player to create a connected path between their edges wins

## Creating Your Own Agent

To create a custom agent, you'll need to modify the `agent1.py` file. The file contains a `CustomPlayer` class that you should implement with your own strategy.

### Guidelines:
1. ~~Your agent must be capable of competing on a 21 x 21 board, with the total time for all moves not exceeding 1 minute.~~
2. You can import any library listed in [allowed_libraries.md](allowed_libraries.md)
3. From this codebase, you may only import from the `Agent` module

### Agent Base Class

Your agent will inherit from the `Agent` base class, which provides the following:

#### Attributes
- `size`: Size of the game board
- `player_number`: Your player number (1 or 2)
- `adv_number`: Opponent's player number (2 or 1)
- `name`: Name of your agent (customize this)

#### Methods
- `step()`: **You must implement this.** Called when it's your turn to make a move
- `update(move_other_player)`: **You must implement this.** Called after opponent's move
- `get_grid_size()`: Returns the size of the grid
- `set_hex(player, coordinate)`: Sets a hexagon to a player
- `get_hex(coordinate)`: Gets the player number at a coordinate
- `neighbors(coordinates)`: Gets valid neighboring coordinates
- `check_win(player)`: Checks if a player has won
- `free_moves()`: Gets all available moves

### Example Implementation
Here's the placeholder agent provided in [`agent1.py`](agent1.py):

```python
import numpy as np
from agent import Agent

class CustomPlayer(Agent):
    def __init__(self, size, player_number, adv_number):
        super().__init__(size, player_number, adv_number)
        self.name = "Custom Agent"  # Change this to your agent's name
        self._possible_moves = [[x, y] for x in range(self.size) for y in range(self.size)]

    def step(self):
        # This is a simple agent that plays random moves
        # Replace with your own strategy
        move = self._possible_moves.pop(np.random.randint(len(self._possible_moves)))
        self.set_hex(self.player_number, move)
        return move

    def update(self, move_other_player):
        self.set_hex(self.adv_number, move_other_player)
        if move_other_player in self._possible_moves:
            self._possible_moves.remove(move_other_player)
```

### Creating a Smarter Agent

To create a smarter agent:

1. Modify the `step()` method to implement your strategy
2. Update the `update()` method to handle the opponent's moves
3. Add any helper methods needed for your strategy
4. Consider using search algorithms, heuristics, or machine learning approaches
5. Ensure your agent makes decisions within the 2-second time limit

## Evaluating Your Agent

You can evaluate your agent against the built-in Minimax agent using the evaluation mode:

```bash
python main.py --eval
```

This will run 25 games with randomized player assignments and provide detailed performance statistics 

## Evaluation Details

Your agent will be tested in a ubuntu 22.04 environment with 8GB RAM. Ensure that it is supported in such a environment.

## Contact

**For issues in the code, kindly raise a Issue.**

## Edit 1: 9th March

1. The final evaluation will be on a board of size no greater than 9 x 9. 
2. The total time limit for each player is now increased to 90s
3. Submission Link is [here](https://forms.gle/FGQEyfE5g3NTKRzRA)
