import threading

import tkinter as tk

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

def main():
    game = othello.Othello()
    root = mc.Node(game)
    tree = mc.MonteCarloTree(mc.default_cost(1.0))
    tree.expansion_threshold = 5
    tree.set_root(root)
    root = tk.Tk()
    root.geometry("800x800")
    root.title("Othello")
    app = OthelloGUI(tree, master=root)
    app.mainloop()

if __name__ == "__main__":
    main()
