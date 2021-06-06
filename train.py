from __future__ import print_function
import os, sys, time, datetime, json, random, collections

from solver.maze_generator import Maze, Room
from solver.maze_agent import Agent

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout, Flatten, Conv2D

from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD , Adam, RMSprop
from tensorflow.keras.layers import PReLU, LeakyReLU
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = (11, 11)



def return_dataset(maze:Maze, maze_agent:Agent):
    def escRoot(pr, pc, prv, tree_map):
        for tmp in tree_map[pr][pc]:
            if (-1, -1) == tmp:
                sol.append((pr, pc))
                return 1
            
        for pos in tree_map[pr][pc]:
            if prv == pos:
                continue
            if escRoot(pos[0], pos[1], (pr, pc), tree_map) == 1:
                sol.append((pr, pc))
                return 1

    # with open(map_path, 'r') as rf:
    src = maze.mazeMap
    
            
    mazeSize = int((len(src) - 1)/2)
    tree_map, sol = [], []

    ## maze의 index 단위
    for r in range(1, mazeSize + 1):
        tree_map.append([])
        for c in range(1, mazeSize + 1):
            tree_map[-1].append([])
    
    ## mazeMap의 index 단위
    for  r in range(1, mazeSize*2, 2):
        for c in range(1, mazeSize*2, 2):
            rr, cc = int((r+1)/2)-1, int((c+1)/2)-1
            if src[r-1][c] == "0":
                tree_map[rr][cc].append((rr-1, cc))
            if src[r+1][c] == "0":
                tree_map[rr][cc].append((rr+1, cc))
            if src[r][c-1] == "0":
                tree_map[rr][cc].append((rr, cc-1))
            if src[r][c+1] == "0":
                tree_map[rr][cc].append((rr, cc+1))
                                
    start = mazeSize - 1
    end = mazeSize + 1
    
    for r in range(start, end+1):
        for c in range(start, end+1):
            if src[r][c] == "3":
                tree_map[int((r+1)/2)-1][int((c+1)/2)-1].append((-1, -1))
                
    
    escRoot(0, 0, None, tree_map)

    sol.reverse()


    maze_agent.reset((1, 1))
    current_cell = sol[0]
    current_state = maze_agent.export_env()
    action_state = np.array([0])
    action = -1
    for i, (mrow, mcol) in enumerate(sol):
        if i == 0:
            continue
            
        prev_cell = current_cell
        current_cell = (mrow, mcol)
        
        action_disc = (current_cell[0] - prev_cell[0], 
                    current_cell[1] - prev_cell[1])
        if action_disc == (-1, 0):
            action = UP
        elif action_disc == (0, 1):
            action = RIGHT
        elif action_disc == (1, 0):
            action = DOWN
        elif action_disc == (0, -1):
            action = LEFT
        else:
            print("INVALID")
            break
        action_state = np.r_[action_state, np.array(action)]
        canvas = maze_agent.act(action)
        current_state = np.r_[current_state, maze_agent.export_env()]
    #     print(f"{i} 번째 이동 완료 {action} 실행 {action_disc}")
    return current_state[:-1], action_state[1:]


size = 32
iteration = 800 # iteration 당 하나의 랜덤 미로 세트
# filename = f"maze_{size}_{size}.txt"
maze = Maze(size)
# maze.save(filename)

maze_agent = Agent(maze.mazeMap)
data_x, data_y = return_dataset(maze, maze_agent)

for i in range(iteration):
    maze = Maze(size)
    maze.save(f"maze_{size}_{size}.txt")

    maze_agent = Agent(maze.mazeMap)
    data_x_sub, data_y_sub = return_dataset(maze, maze_agent)
    data_x = np.r_[data_x, data_x_sub]
    data_y = np.r_[data_y, data_y_sub]
    if i % 50 == 0:
        print(f"{i} 번째 축적 완료, shape: {data_x.shape}")
print(f"{i} 번째 축적 완료, shape: {data_x.shape}")

# # # 0 번째 축적 완료, shape: (410, 4225)
# # # 50 번째 축적 완료, shape: (14469, 4225)
# # # 100 번째 축적 완료, shape: (27844, 4225)
# # # 150 번째 축적 완료, shape: (41778, 4225)
# # # 200 번째 축적 완료, shape: (55248, 4225)
# # # 250 번째 축적 완료, shape: (68672, 4225)
# # # 300 번째 축적 완료, shape: (81630, 4225)
# # # 350 번째 축적 완료, shape: (95541, 4225)
# # # 400 번째 축적 완료, shape: (108722, 4225)
# # # 450 번째 축적 완료, shape: (124046, 4225)
# # # 500 번째 축적 완료, shape: (138535, 4225)
# # # 550 번째 축적 완료, shape: (152767, 4225)
# # # 600 번째 축적 완료, shape: (167825, 4225)
# # # 650 번째 축적 완료, shape: (181906, 4225)
# # # 700 번째 축적 완료, shape: (196028, 4225)
# # # 750 번째 축적 완료, shape: (209079, 4225)
# # # 799 번째 축적 완료, shape: (223020, 4225)

np.save(data_path / f"data_x_{size}_{size}_{800}-total.npy", data_x)
np.save(data_path / f"data_y_{size}_{size}_{800}-total.npy", data_y)

def show_maze_x(maze):
    canvas = np.copy(maze)   
    nrows, ncols = canvas.shape

    plt.grid('on')
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    player_row, player_col = map(int, np.where(canvas == information_dict["player"]))
    visited_row, visited_col = np.where(canvas == information_dict["visited"])
    goal_row, goal_col = np.where(canvas == information_dict["goal"])

    for r in range(nrows):
        for c in range(ncols):
            if canvas[r,c] > information_dict["wall"]:
                canvas[r,c] = mark_dict["free"]
            else:
                canvas[r, c] = mark_dict["wall"]

            for i in range(len(visited_row)):
                canvas[int(visited_row[i]), int(visited_col[i])] = mark_dict["visited"]

    
    canvas[player_row, player_col] = mark_dict["player"]
    ## Goal Cell
    for i in range(len(goal_row)):
        canvas[goal_row[i], goal_col[i]] = mark_dict["goal"]
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    return img

# maze_agent.show(x_train[3].reshape((21, 21)))\
idx = 20000
show_maze_x(data_x[idx].reshape((size*2+1, size*2+1)))
print("0: LEFT / 1: UP / 2: RIGHT / 3: DOWN")
print(data_y[idx])

# 라벨 분포 보기

total_num = len(data_y)
for label in set(data_y):
    print(f"{label}: {np.sum(data_y == label) / total_num * 100: .2f}%", end="\n")


# # 0:  22.21%
# # 1:  22.23%
# # 2:  27.76%
# # 3:  27.80%

# Model

## ConvNet C-C-C-F-Drop
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, kernel_size=(7, 7)),
  tf.keras.layers.PReLU(),
  tf.keras.layers.Conv2D(16, kernel_size=(9, 9)),
  tf.keras.layers.PReLU(),
  tf.keras.layers.Conv2D(8, kernel_size=(11, 11)),
  tf.keras.layers.PReLU(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Dense(512),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.PReLU(),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Dense(128),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.PReLU(),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Dense(4, activation='softmax')
])

# model.compile(optimizer=Adam(learning_rate=3e-01), loss='mse')

## Simple Neural Network
# model = tf.keras.models.Sequential([
                                    
# #   tf.keras.layers.Conv2D(64, kernel_size=(7, 7)),
# #   tf.keras.layers.PReLU(),
# #   tf.keras.layers.MaxPool2D((3, 3)),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(units=size**2),
#   tf.keras.layers.BatchNormalization(),
#   tf.keras.layers.PReLU(),
#   tf.keras.layers.Dropout(0.25),
#   tf.keras.layers.Dense(units=size**2),
#   tf.keras.layers.BatchNormalization(),
#   tf.keras.layers.PReLU(),
#   tf.keras.layers.Dropout(0.25),
#   tf.keras.layers.Dense(units=size**3),
#   tf.keras.layers.BatchNormalization(),
#   tf.keras.layers.PReLU(),
#   tf.keras.layers.Dropout(0.25),
#   tf.keras.layers.Dense(units=size),
#   tf.keras.layers.BatchNormalization(),
#   tf.keras.layers.PReLU(),
#   tf.keras.layers.Dropout(0.25),
#   tf.keras.layers.Dense(4, activation='softmax')
# ])


model.compile(optimizer=Adam(learning_rate=5e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(x_train, y_train, batch_size=32, epochs=60,
               validation_data=(x_test, y_test), verbose=1)

res = model.evaluate(x_test, y_test, verbose=0)
print(res)