from grid import Grid

class Controller:
    def __init__(self, size, player1, player2):
        self._grid = Grid(size)
        self._player1 = player1
        self._player2 = player2
        self._current_player = 1
        self._winner = 0

    def update(self):
        if self._current_player == 1:
            coordinates = self._player1.step()
            self._grid.set_hex(self._current_player, coordinates)
            self._current_player = 2
            self._player2.update(coordinates)
        else:
            coordinates = self._player2.step()
            self._grid.set_hex(self._current_player, coordinates)
            self._current_player = 1
            self._player1.update(coordinates)

        self._check_win()

    def _check_win(self):
        if self._grid.check_win(1):
            self._winner = 1
        elif self._grid.check_win(2):
            self._winner = 2
