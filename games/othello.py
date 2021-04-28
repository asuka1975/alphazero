import os
import sys
import itertools
import typing as tp

sys.path.append(os.path.abspath("."))

import game

EMPTY = 0
BLACK = 1
WHITE = 2

def get_enemy(cell):
    return WHITE if cell == BLACK else BLACK

class Othello:
    def __init__(self):
        self.board = [EMPTY] * 64
        self[3, 3] = WHITE
        self[3, 4] = BLACK
        self[4, 3] = BLACK
        self[4, 4] = WHITE
        self.turn = BLACK
        self.skip_flag = False

    def __getitem__(self, index):
        return self.board[index[0] + index[1] * 8]

    def __setitem__(self, index, v):
        self.board[index[0] + index[1] * 8] = v

    def __str__(self):
        symbol = [".", "O", "@"]
        return "\n".join(["".join([symbol[self[i, j]] for i in range(8)]) for j in range(8)])

    def __eq__(self, other: game.Game):
        return type(self) == type(other) and hash(self) == hash(other)

    def __hash__(self):
        hash = 0
        digit = 1
        for v in self.board:
            hash += v * digit
            digit *= 3
        return hash

    def is_empty(self, x, y):
        return self[x, y] == EMPTY

    def in_range(self, x, y):
        return 0 <= x < 8 and 0 <= y < 8

    def is_reversable(self, x, y, d):
        enemy = get_enemy(self.turn)
        _x = x + d[0]
        _y = y + d[1]
        if self.in_range(_x, _y) and self[_x, _y] == enemy:
            _x += d[0]
            _y += d[1]
            while self.in_range(_x, _y):
                if self[_x, _y] == self.turn:
                    return True
                elif self[_x, _y] == EMPTY:
                    return False
                _x += d[0]
                _y += d[1]
        return False
    
    def is_puttable(self, x, y):
        enemy = get_enemy(self.turn)
        if self[x, y] is not EMPTY:
            return False
        for d in list(itertools.permutations([-1, 0, 1], 2)) + [(1, 1), (-1, -1)]:
            if self.is_reversable(x, y, d):
                return True
        return False
    
    def puttables(self):
        return [(i, j) for j in range(8) for i in range(8) if self.is_puttable(i, j)]

    def put(self, x, y):
        enemy = get_enemy(self.turn)
        self[x, y] = self.turn
        for d in [d for d in list(itertools.permutations([-1, 0, 1], 2)) + [(1, 1), (-1, -1)] if self.is_reversable(x, y, d)]:
            _x = x + d[0]
            _y = y + d[1]
            while self.in_range(_x, _y):
                if self[_x, _y] == self.turn:
                    break
                self[_x, _y] = self.turn
                _x += d[0]
                _y += d[1]
        self.turn = enemy
        self.skip_flag = False

    def copy(self):
        o = Othello()
        o.board = [v for v in self.board]
        o.turn = self.turn
        o.skip_flag = self.skip_flag
        return o

    def updated(self, action: game.Action):
        o = self.copy()
        o.update(action)
        return o

    def update(self, action: game.Action):
        if type(action) is tuple:
            self.put(action[0], action[1])
        elif type(action) is bool and action:
            self.skip()

    def skip(self):
        self.turn = get_enemy(self.turn)
        self.skip_flag = True

    def skipped(self):
        o = self.copy()
        o.skip()
        return o

    def actions(self):
        if self.finished():
            return []
        puttables = self.puttables()
        return self.puttables() if len(puttables) > 0 else [True]
    
    def finished(self):
        return len(self.puttables()) == 0 and self.skip_flag

    def winner(self):
        return game.Winner.PASSIVE if self.board.count(WHITE) > self.board.count(BLACK) else (game.Winner.FIRST if self.board.count(WHITE) < self.board.count(BLACK) else game.Winner.DRAW)



if __name__ == '__main__':
    import random
    o = Othello()
    print(o, "\n")
    while not o.finished():
        puttables = o.puttables()
        action = random.choice(o.actions())
        o.update(action)
        print(action)
        print(o, "\n")
