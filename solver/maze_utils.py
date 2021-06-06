from .maze_agent import Agent

import time
import matplotlib.pyplot as plt

from IPython import display

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

arrow_dict = {
    0: '←',
    1: '↑',
    2: '→',
    3: '↓'
}

head_dict = {
    0: '◀',
    1: '▲',
    2:'▶',
    3: '▼'
}

def find_maze_solution(maze:list, player:tuple):
    def escRoot(pr, pc, prv, tree_map, search):
        """미로 탈출 경로 탐색
        * (pr, pc): 현재 위치
        * prv: 이전 위치
        """
        search.append((pr, pc))

        # 종료 분기
        for tmp in tree_map[pr][pc]:
            if (-1, -1) == tmp:
                sol.append((pr, pc))
                return 1

        for pos in tree_map[pr][pc]:
            if len(search) > 1500:
                print("unable to find solution on second")
                break
            if prv == pos:
                continue
            if escRoot(pos[0], pos[1], (pr, pc), tree_map, search) == 1:
                sol.append((pr, pc))
                
                return 1

    # with open(map_path, 'r') as rf:
    src = maze


    mazeSize = int((len(src) - 1)/2)
    tree_map, search, sol = [], [], []

    ## maze의 index 단위: maze 크기 만큼 이중 리스트 생성
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
                
            if src[r][c] == "3":
                tree_map[int((r+1)/2)-1][int((c+1)/2)-1].append((-1, -1))
                

    escRoot(int((player[0]+1)/2 -1), int((player[1]+1)/2 -1), None, tree_map, search)

    sol.reverse()
    
    return sol, search




def read_maze(file):
    maze = []
    with open(file, 'r') as f:
        src = f.readlines()

    mapSize = int((len(src) - 1)/2)
    
    for i, row in enumerate(src):
        for j, item in enumerate(row.split('\n')[0]):
            if j == 0:
                maze.append([item])
            else:
                maze[i].append(item)
    start = mapSize - 1 
    end = mapSize + 1
    maze[start][start] = maze[start][end] = maze[end][start] = maze[end][end] = "3"
                
    return maze
            

    
def solve_on_maze(agent:Agent, solution:list, wait_time:float=1.0, is_visible:bool=True):
    if len(solution) == 0:
        return 10000.0 
    agent.reset(agent.player)
    action_time, rotate_time = 0.3, 0.1


    current_cell = solution[0]
    action = 2

    status = ""
    total_time = 0.0
    for i, (mrow, mcol) in enumerate(solution):
        if i == 0:
            continue

        prev_action = action
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

        if prev_action == action:
            status = "이동"
            total_time += action_time
        else:
            status = "회전 -> 이동"
            total_time += rotate_time + action_time 
            
        if is_visible:
            try:

                plt.clf()

                canvas = agent.act(action)
                img = agent.show()
                display.clear_output(wait=True)

                display.display(plt.gcf())
                if prev_action != action:
                    print(f"봇 헤드 방향: {head_dict[prev_action]} / {'회전중     '}\n현재 진행시간: {total_time - action_time: .2f} sec")
                print(f"봇 헤드 방향: {head_dict[action]} / {status}({arrow_dict[action]}) 중\n현재 진행시간: {total_time: .2f} sec", )
                time.sleep(wait_time)
            except KeyboardInterrupt:
                break
            plt.clf()
        else:
            canvas = agent.act(action)    
    return total_time