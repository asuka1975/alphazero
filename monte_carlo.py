from __future__ import annotations

import math
import random

import numpy as np

import game

class Node:
    def __init__(self, board: game.Game):
        self.n_s = 0
        self.q = 0
        self.value: game.Game = board
        self.cost_fn = None
        self.children: dict[game.Action, Node] = {}
        self.parent = None

    def select(self):
        unselected = [action for action, child in self.children.items() if child.n_s == 0]
        action = None
        if len(unselected) == 0:
            scores = [child.q / child.n_s + self.cost_fn(self, action) for action, child in self.children.items()]
            m = max(scores)
            max_indices = [i for i, v in enumerate(scores) if v == m]
            index = random.choice(max_indices)
            action = self.children.keys()[index]
        else:
            action = random.choice(unselected)
        self.children[action].n_s += 1
        return self.children[action]

    def expand(self):
        self.children = { action : Node(self.value.updated(action)) for action in self.value.actions() }
        for child in self.children.values():
            child.parent = self
            child.cost_fn = self.cost_fn
            

class MonteCarloTree:
    def __init__(self, cost_fn):
        self.root: Node = None
        self.cost_fn = cost_fn
        self.expansion_threshold = 2

    def set_root(self, root: Node) -> None:
        self.root = root
        self.root.cost_fn = self.cost_fn

    def expand_if(self, node: Node) -> bool:
        return node.n_s >= self.expansion_threshold and len(node.children) == 0

    def simulate(self, epoch):
        node = self.root
        for i in range(epoch):
            if self.expand_if(node):
                node.expand()
            elif not self.expand_if(node) and len(node.children) == 0:
                point = self.evaluate(node.value.copy())
                self.backup(node, point)
                node = self.root
                continue
            node = node.select()
            if node.value.finished():
                point = self.evaluate(node.value)
                self.backup(node, point)
                node = self.root
    
    def evaluate(self, situation: game.Game):
        while not situation.finished():
            situation.update(random.choice(situation.actions()))
        winner = situation.winner()
        return 1 if winner == game.Winner.FIRST else (-1 if winner == game.Winner.PASSIVE else 0)

    def backup(self, node: Node, point):
        while node != None:
            node.q += point
            node = node.parent

    def play(self):
        node = max(self.root.children, key=lambda n: n.q)
        self.root = node
        self.root.parent = None
        return self.root.value

    def enemy(self, action):
        self.root = self.root.children[action]
        self.root.parent = None
        return self.root.value
