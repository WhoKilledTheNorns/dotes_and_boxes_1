import tkinter as tk
import torch
import numpy as np
from ai_model import model, ai_move

class DotAndBoxGame:
    def __init__(self, master):
        self.master = master
        self.master.title("5x5 点线成盒游戏")
        self.canvas = tk.Canvas(self.master, width=900, height=900)
        self.canvas.pack()

        self.rows = 5
        self.cols = 5
        self.grid_size = 150
        self.margin = 50
        self.player_turn = True
        self.lines = {}
        self.boxes = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.player_score = 0
        self.ai_score = 0
        self.buttons = {}
        self.create_widgets()

    def create_widgets(self):
        self.canvas.delete("all")
        self.draw_grid()
        self.create_scoreboard()
        self.create_buttons()
        self.update_scoreboard()

    def draw_grid(self):
        for row in range(self.rows + 1):
            for col in range(self.cols + 1):
                x = col * self.grid_size + self.margin
                y = row * self.grid_size + self.margin
                self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill='black')

    def create_buttons(self):
        for row in range(self.rows + 1):
            for col in range(self.cols):
                x1 = col * self.grid_size + self.margin
                y1 = row * self.grid_size + self.margin
                x2 = (col + 1) * self.grid_size + self.margin
                y2 = y1 + 10
                self.create_button(x1, y1, x2, y2, (col, row, col + 1, row))

        for row in range(self.rows):
            for col in range(self.cols + 1):
                x1 = col * self.grid_size + self.margin
                y1 = row * self.grid_size + self.margin
                y2 = (row + 1) * self.grid_size + self.margin
                x2 = x1 + 10
                self.create_button(x1, y1, x2, y2, (col, row, col, row + 1))

    def create_button(self, x1, y1, x2, y2, coords):
        button = tk.Button(self.canvas, command=lambda: self.click_handler(*coords), bg="white")
        self.buttons[coords] = button
        self.canvas.create_window((x1 + x2) // 2, (y1 + y2) // 2, window=button, width=x2 - x1, height=y2 - y1)

    def click_handler(self, x1, y1, x2, y2):
        if self.player_turn and (x1, y1, x2, y2) not in self.lines:
            self.add_line(x1, y1, x2, y2)
            if not self.check_boxes():
                self.player_turn = not self.player_turn
                self.master.after(500, self.ai_turn)
            self.update_scoreboard()
            self.check_winner()

    def ai_turn(self):
        ai_move(self)
        self.update_scoreboard()
        self.check_winner()
        if not self.check_boxes():  # 确保AI无法再得分
            self.player_turn = True

    def add_line(self, x1, y1, x2, y2):
        self.lines[(x1, y1, x2, y2)] = self.player_turn
        self.lines[(x2, y2, x1, y1)] = self.player_turn
        button = self.buttons[(x1, y1, x2, y2)]
        button.config(bg="red" if self.player_turn else "green", state=tk.DISABLED)

    def get_board_state(self):
        board_h = np.zeros((25, 25), dtype=int)
        board_v = np.zeros((25, 25), dtype=int)
        for (x1, y1, x2, y2), player in self.lines.items():
            if x1 == x2:
                board_v[y1, x1] = 1
            else:
                board_h[y1, x1] = 1
        return np.concatenate((board_h, board_v), axis=0)

    def get_available_moves(self):
        available_moves = []
        for row in range(self.rows):
            for col in range(self.cols):
                if (col, row, col, row + 1) not in self.lines:
                    available_moves.append((col, row, col, row + 1))
                if (col, row, col + 1, row) not in self.lines:
                    available_moves.append((col, row, col + 1, row))
        return available_moves

    def check_boxes(self):
        box_completed = False
        for row in range(self.rows):
            for col in range(self.cols):
                if self.boxes[row][col] is None:
                    if (col, row, col + 1, row) in self.lines and \
                            (col + 1, row, col + 1, row + 1) in self.lines and \
                            (col, row + 1, col + 1, row + 1) in self.lines and \
                            (col, row, col, row + 1) in self.lines:
                        self.boxes[row][col] = self.player_turn
                        self.canvas.create_rectangle(
                            col * self.grid_size + self.margin,
                            row * self.grid_size + self.margin,
                            (col + 1) * self.grid_size + self.margin,
                            (row + 1) * self.grid_size + self.margin,
                            fill="red" if self.player_turn else "green"
                        )
                        if self.player_turn:
                            self.player_score += 1
                        else:
                            self.ai_score += 1
                        box_completed = True
        return box_completed

    def create_scoreboard(self):
        self.scoreboard = tk.Label(self.master, text="", font=("Arial", 16))
        self.scoreboard.pack()

        self.restart_button = tk.Button(self.master, text="重新开始", command=self.restart_game, font=("Arial", 16))
        self.restart_button.pack()

    def update_scoreboard(self):
        self.scoreboard.config(text=f"玩家得分: {self.player_score}  AI得分: {self.ai_score}")

    def restart_game(self):
        self.player_turn = True
        self.lines.clear()
        self.boxes = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.player_score = 0
        self.ai_score = 0
        self.create_widgets()

    def check_winner(self):
        if self.player_score + self.ai_score == self.rows * self.cols:
            if self.player_score > self.ai_score:
                self.show_winner("玩家获胜！")
            else:
                self.show_winner("AI获胜！")

    def show_winner(self, message):
        self.canvas.create_text(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            text=message,
            font=("Arial", 24),
            fill="black"
        )
        self.canvas.unbind("<Button-1>")
        for button in self.buttons.values():
            button.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    game = DotAndBoxGame(root)
    root.mainloop()
