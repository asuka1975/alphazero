import threading
import math

import tkinter as tk

import numpy as np

from tensorflow import keras
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, BatchNormalization, Activation, add

import game
import monte_carlo as mc
from games import othello


class OthelloGUI(tk.Frame):
    def __init__(self, ai: mc.MonteCarloTree, master=None):
        super().__init__(master)
        self.pack()

        self.ai: mc.MonteCarloTree = ai
        self.ai.simulate(10000)
        self.othello = self.ai.play()

        self.canvas = tk.Canvas(master, bg="white", height=800, width=800)
        self.canvas.create_rectangle(0, 0, 800, 800, fill="green")
        for j in range (9):
            for i in range(9):
                self.canvas.create_line(i * 100, 0, i * 100, 800)
                self.canvas.create_line(0, i * 100, 800, i * 100)
        self.create()
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.pack()
        self.waiting = False

    def create(self):
        self.ids = []
        for j in range(8):
            for i in range(8):
                rect = [i * 100 + 10, j * 100 + 10, (i + 1) * 100 - 10, (j + 1) * 100 - 10]
                if self.othello[i, j] == othello.BLACK:
                    self.ids.append(self.canvas.create_oval(*rect, fill="black"))
                elif self.othello[i, j] == othello.WHITE:
                    self.ids.append(self.canvas.create_oval(*rect, fill="white"))
                else:
                    self.ids.append(self.canvas.create_oval(*rect, fill="green"))
        self.waiting_label = self.canvas.create_text(400, 400, text=u"", font=("", 45), fill="gray")

    def draw(self):
        for j in range(8):
            for i in range(8):
                if self.othello[i, j] == othello.BLACK:
                    self.canvas.itemconfig(self.ids[i + j * 8], fill="black")
                elif self.othello[i, j] == othello.WHITE:
                    self.canvas.itemconfig(self.ids[i + j * 8], fill="white")
                else:
                    self.canvas.itemconfig(self.ids[i + j * 8], fill="green")
        
    def click(self, event):
        if self.waiting:
            return
        action = None
        position = (event.x // 100, event.y // 100)
        actions = self.othello.actions()
        if len(actions) == 1 and type(actions[0]) == bool:
            action = True
        elif position in actions:
            action = position
        if action != None:
            self.othello = self.ai.enemy(action)
            self.draw()
            self.canvas.pack()
            self.thread1 = threading.Thread(target=self.simulate)
            self.thread1.start()
        if self.othello.finished():
            print(self.othello.winner())

    def simulate(self):
        self.waiting = True
        self.canvas.itemconfig(self.waiting_label, text=u"思考中")
        if not self.othello.finished():
            self.ai.simulate(10000)
            self.othello = self.ai.play()
            self.draw()
        self.canvas.itemconfig(self.waiting_label, text=u"")
        self.waiting = False

def resnet(x):
    y = Conv2D(128, (3, 3), padding="same")(x)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    z = add([x, y])
    return Activation("relu")(z)

def create_model():
    inputs = Input(shape=(3, 8, 8))
    x = Conv2D(128, (3, 3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = resnet(x)
    x = resnet(x)
    x = resnet(x)
    x = resnet(x)
    x = resnet(x)
    x = resnet(x)
    x = resnet(x)
    x = resnet(x)

    #policy
    y1 = Conv2D(2, (1, 1))(x)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = Flatten()(y1)
    policy_output = Dense(65, activation="softmax")(y1)

    #value
    y2 = Conv2D(1, (1, 1))(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)
    y2 = Flatten()(y2)
    y2 = Dense(128, activation="relu")(y2)
    value_output = Dense(1, activation="tanh")(y2)

    model = keras.Model(inputs=[inputs], outputs=[policy_output, value_output])
    model.compile(optimizer='rmsprop', loss="mean_squared_error", metrics=["accuracy"])
    return model

def devide_channels(situation: game.Game):
    board: othello.Othello = situation
    b = [1. if v == othello.BLACK else 0. for v in board.board]
    w = [1. if v == othello.WHITE else 0. for v in board.board]
    return [[b[i:i+8] for i in range(0, 64, 8)],
            [w[i:i+8] for i in range(0, 64, 8)],
            [[2 - board.turn] * 8 for _ in range(8)]]

def to_input(situation: game.Game):
    return np.array([devide_channels(situation)])

def mcts_evaluate(self, situation: game.Game):
    input = to_input(situation)
    output = self.model.predict(input)
    return output[1][0][0]

def p_map(node: mc.Node):
    m = [0] * 65
    if len(node.children) == 1 and True in node.children:
        m[64] = 1
    else:
        s = sum(map(lambda n: n.n_s, node.children.values()))
        for action, child in node.children.items():
            m[action[0] + action[1] * 8] = float(child.n_s) / s
    return m


def learn(model: keras.Model):
    def cost_fn(node: mc.Node, action):
        if type(action) is bool:
            return 1.5 * model.predict(to_input(node.value))[0][0][64] * math.sqrt(sum([child.n_s for child in node.children.values()])) / (1 + node.n_s)
        else:
            return 1.5 * model.predict(to_input(node.value))[0][0][action[0] + action[1] * 8] * math.sqrt(sum([child.n_s for child in node.children.values()])) / (1 + node.n_s)
    for i in range(10):
        datas = []
        for j in range(500):
            game = othello.Othello()
            root = mc.Node(game)
            tree: mc.MonteCarloTree = mc.MonteCarloTree(cost_fn)
            tree.expansion_threshold = 5
            tree.set_root(root)
            tree.model = model
            tmp = []
            while not tree.root.value.finished():
                tree.simulate(20)
                tmp.append((tree.root.value, p_map(tree.root), 0))
                tree.play()
                print(i, j, "self-play")
            for k in range(len(tmp)):
                tmp[k] = (tmp[k][0], tmp[k][1], [0, 1, -1][int(tree.root.value.winner())])
            datas.extend(tmp)
        input = np.array([devide_channels(data[0]) for data in datas])
        policy_output = np.array([data[1] for data in datas])
        value_output = np.array([data[2] for data in datas])
        model.fit(x=input, y=[policy_output, value_output])
        model.save("models/model.h5")


def main():
    mc.MonteCarloTree.evaluate = mcts_evaluate
    game = othello.Othello()
    model = create_model()
    learn(model)
    """root = mc.Node(game)
    tree = mc.MonteCarloTree(mc.default_cost(1.0))
    tree.expansion_threshold = 5
    tree.set_root(root)
    root = tk.Tk()
    root.geometry("800x800")
    root.title("Othello")
    app = OthelloGUI(tree, master=root)
    app.mainloop()"""

if __name__ == "__main__":
    main()
