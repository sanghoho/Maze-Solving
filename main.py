from __future__ import print_function
import os, sys, time, datetime, json, random, collections, copy
from solver.maze_utils import find_maze_solution, read_maze, solve_on_maze
from solver.maze_generator import Maze, Room
from solver.maze_agent import Agent

import matplotlib.pyplot as plt

import argparse


parser = argparse.ArgumentParser(description='Argparse Tutorial')
# argument는 원하는 만큼 추가한다.
parser.add_argument('--maze-path', type=str, 
                    help='path for read maze')

parser.add_argument('--size', type=int, 
                    help='an integer for maze size')

parser.add_argument('--is-visible', type=str, default="false", 
                    help='a str for show maze visible')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    args = parser.parse_args()
    filename = args.maze_path
    is_visible = str2bool(args.is_visible)

    ## 1차 시도
    maze = read_maze(filename)
    maze_agent = Agent(maze)

    sol, search = find_maze_solution(maze, maze_agent.player)
    first_time = solve_on_maze(maze_agent, sol, 0.1, is_visible)
    print(f"1차: {first_time}")


    ## 2차 시도
    player = maze_agent.player
    goal = tuple([(x +1)*2 - 1 for x in sol[-1]])

    # Remove Visual Mark on Target
    for x, y in maze_agent.target:
        maze[x][y] = '0'

    maze[player[0]][player[1]] = '3'
    maze[goal[0]][goal[1]] = '2'

    maze_agent_end = Agent(maze)
    sol_end, _ = find_maze_solution(maze, maze_agent_end.player)
    second_time = solve_on_maze(maze_agent_end, sol_end, 0.1, is_visible)
    print(f"2차 시도 마무리: {second_time}\n")

    print(f"미로 탐색에 소요된 시간: MIN {min([first_time, second_time])}")


