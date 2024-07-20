import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple
import time as time

class DotsAndBoxesNet(nn.Module):
    def __init__(self):
        super(DotsAndBoxesNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 25 * 25, 256)
        self.fc2 = nn.Linear(256, 2 * 25 * 25)

    def forward(self, x):
        x = x.view(-1, 2, 25, 25)  # 将输入调整为卷积层需要的形状
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 25 * 25)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型和优化器
model = DotsAndBoxesNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 定义经验池
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

memory = ReplayMemory(10000)
BATCH_SIZE = 128
GAMMA = 0.99

def move_completes_box(game, move):
    x1, y1, x2, y2 = move
    if x1 == x2:  # vertical move
        # Check the box to the right
        right_complete = (
            (x1, y1, x1 + 1, y1) in game.lines and
            (x1, y1 + 1, x1 + 1, y1 + 1) in game.lines and
            (x1 + 1, y1, x1 + 1, y1 + 1) in game.lines
        )
        # Check the box to the left
        left_complete = (
            (x1 - 1, y1, x1, y1) in game.lines and
            (x1 - 1, y1 + 1, x1, y1 + 1) in game.lines and
            (x1 - 1, y1, x1 - 1, y1 + 1) in game.lines
        )
        return right_complete or left_complete
    else:  # horizontal move
        # Check the box above
        above_complete = (
            (x1, y1, x1, y1 + 1) in game.lines and
            (x1 + 1, y1, x1 + 1, y1 + 1) in game.lines and
            (x1, y1 + 1, x1 + 1, y1 + 1) in game.lines
        )
        # Check the box below
        below_complete = (
            (x1, y1 - 1, x1, y1) in game.lines and
            (x1 + 1, y1 - 1, x1 + 1, y1) in game.lines and
            (x1, y1 - 1, x1 + 1, y1 - 1) in game.lines
        )
        return above_complete or below_complete

def move_creates_three_sides(game, move):
    x1, y1, x2, y2 = move
    if x1 == x2:  # vertical
        return ((x1, y1, x1 + 1, y1) in game.lines and
                (x1, y1 + 1, x1 + 1, y1 + 1) in game.lines and
                ((x1 + 1, y1, x1 + 1, y1 + 1) not in game.lines or
                 (x1, y1, x1, y1 + 1) not in game.lines)) or \
               ((x1 - 1, y1, x1, y1) in game.lines and
                (x1 - 1, y1 + 1, x1, y1 + 1) in game.lines and
                ((x1 - 1, y1, x1 - 1, y1 + 1) not in game.lines or
                 (x1, y1, x1, y1 + 1) not in game.lines))
    else:  # horizontal
        return ((x1, y1, x1, y1 + 1) in game.lines and
                (x1 + 1, y1, x1 + 1, y1 + 1) in game.lines and
                ((x1, y1 + 1, x1 + 1, y1 + 1) not in game.lines or
                 (x1, y1, x1 + 1, y1) not in game.lines)) or \
               ((x1, y1 - 1, x1, y1) in game.lines and
                (x1 + 1, y1 - 1, x1 + 1, y1) in game.lines and
                ((x1, y1 - 1, x1 + 1, y1 - 1) not in game.lines or
                 (x1, y1, x1 + 1, y1) not in game.lines))

def count_boxes_completed(game, move):
    x1, y1, x2, y2 = move
    count = 0
    if x1 == x2:  # vertical move
        if ((x1, y1, x1 + 1, y1) in game.lines and
            (x1, y1 + 1, x1 + 1, y1 + 1) in game.lines and
            (x1 + 1, y1, x1 + 1, y1 + 1) in game.lines):
            count += 1
        if ((x1 - 1, y1, x1, y1) in game.lines and
            (x1 - 1, y1 + 1, x1, y1 + 1) in game.lines and
            (x1 - 1, y1, x1 - 1, y1 + 1) in game.lines):
            count += 1
    else:  # horizontal move
        if ((x1, y1, x1, y1 + 1) in game.lines and
            (x1 + 1, y1, x1 + 1, y1 + 1) in game.lines and
            (x1, y1 + 1, x1 + 1, y1 + 1) in game.lines):
            count += 1
        if ((x1, y1 - 1, x1, y1) in game.lines and
            (x1 + 1, y1 - 1, x1 + 1, y1) in game.lines and
            (x1, y1 - 1, x1 + 1, y1 - 1) in game.lines):
            count += 1
    return count

def get_best_move(predictions, available_moves, game):
    best_moves = sorted([move for move in available_moves if move_completes_box(game, move)],
                        key=lambda move: count_boxes_completed(game, move), reverse=True)
    if best_moves:
        return best_moves[0]

    safe_moves = [move for move in available_moves if not move_creates_three_sides(game, move)]
    if safe_moves:
        return random.choice(safe_moves)

    best_value = -float('inf')
    best_move = None
    for move in available_moves:
        x1, y1, x2, y2 = move
        if x1 == x2:
            value = predictions[0, y1, x1]
        else:
            value = predictions[1, y1, x1]
        if value > best_value:
            best_value = value
            best_move = move

    if best_move:
        return best_move

    return random.choice(available_moves)

def check_three_sides_and_complete(game):
    available_moves = game.get_available_moves()
    for move in available_moves:
        if move_completes_box(game, move):
            game.add_line(*move)
            if game.check_boxes():
                game.update_scoreboard()
                game.check_winner()
                return True
    return False

def check_and_complete_all_three_sides(game):
    completed = False
    while check_three_sides_and_complete(game):
        completed = True
    return completed

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = model(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def ai_move(game):
    while not game.player_turn:
        completed_box = False

        if check_and_complete_all_three_sides(game):
            completed_box = True
        else:
            board_state = game.get_board_state()
            board_state = torch.FloatTensor(board_state).view(-1, 2 * 25 * 25)
            with torch.no_grad():
                predictions = model(board_state)
            predictions = predictions.view(2, 25, 25).numpy()

            potential_moves = game.get_available_moves()
            best_move = get_best_move(predictions, potential_moves, game)

            if best_move:
                game.add_line(*best_move)
                if game.check_boxes():
                    completed_box = True
                game.update_scoreboard()
                game.check_winner()

        if not completed_box:
            game.player_turn = not game.player_turn
            break
        if len(game.get_available_moves()) == 0:
            game.player_turn = not game.player_turn
            break
