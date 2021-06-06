import copy
import numpy as np
import matplotlib.pyplot as plt

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

# For Train
information_dict = {
    "wall": 0.3,
    "free": 0.5,
    "visited": 0.75,
    "player": 0.9,
    "goal": 1.0

}

# For Drawing
mark_dict = {
    "wall": 0.0,
    "free": 1.0,
    "player": 0.3,
    "visited": 0.6,
    "goal": 0.9
}


# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}




num_actions = len(actions_dict)

class Agent:
    def __init__(self, maze, player="2", goal="3"):
        self._convert_dict = {
            "0": information_dict["free"], # 이동가능
            "1": information_dict["wall"], # 벽
            player: information_dict["free"], # 플레이어
            goal: information_dict["free"] # 골인지점
        }
        self.player, self.target = (), []
        self._maze = self.__convert_maze_format(maze, player, goal)
        self.nrows, self.ncols = self._maze.shape
        self.free_cells = [(r, c) for r in range(self.nrows) for c in range(self.ncols) if self._maze[r, c] == information_dict["free"]]
        for tgt in self.target:
            self.free_cells.remove(tgt)
            if self._maze[tgt] == information_dict["wall"]:
                raise Exception("Invalid maze: target cell cannot be blocked!")
        if not self.player in self.free_cells:
            raise Exception("Invalid Player Location: must sit on a free cell")
        self.reset(self.player)            

    def __convert_maze_format(self, mazeMap:list, player, goal):
        maze_copy = copy.deepcopy(mazeMap)

        for i, line in enumerate(mazeMap):
            for j, elem in enumerate(mazeMap):
                current_elem = mazeMap[i][j]
                if current_elem == player:
                    self.player = (i, j)
                elif current_elem == goal:
                    self.target.append((i, j))
                maze_copy[i][j] = self._convert_dict[current_elem]

        return np.array(maze_copy)

    def reset(self, player):
        self.player = player
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        row, col = player
        self.maze[row, col] = information_dict["player"]
        self.state = (row, col, 'start')
        self.visited = set()

    def update_state(self, action):
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = player_row, player_col, mode = self.state

        if self.maze[player_row, player_col] > information_dict["wall"]:
            self.visited.add((player_row, player_col))  # mark visited cell

        valid_actions = self.valid_actions()
                
        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 2
                self.visited.add((player_row, player_col-1))
            elif action == UP:
                nrow -= 2
                self.visited.add((player_row-1, player_col))
            if action == RIGHT:
                ncol += 2
                self.visited.add((player_row, player_col+1))
            elif action == DOWN:
                nrow += 2
                self.visited.add((player_row+1, player_col))
                
        else:                  # invalid action, no change in rat position
            mode = 'invalid'

        # new state
        self.state = (nrow, ncol, nmode)
        
    def act(self, action):
        self.update_state(action)
        envstate = self.observe()
        return envstate

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > information_dict["wall"]:
                    canvas[r,c] = information_dict["free"]
        # draw the rat
        row, col, valid = self.state
        canvas[row, col] = information_dict["player"]
        return canvas

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self.maze.shape
        ## 가장자리 제거
        if row == 1: # UP 제거
            actions.remove(1)
        elif row == nrows-2: # DOWN 제거
            actions.remove(3)

        if col == 1: # LEFT 제거 
            actions.remove(0)
        elif col == ncols-2: # RIGHT 제거
            actions.remove(2)

        if row > 1 and self.maze[row-1,col] == information_dict["wall"]:
            actions.remove(1)
        if row<nrows-2 and self.maze[row+1,col] == information_dict["wall"]:
            actions.remove(3)

        if col>1 and self.maze[row,col-1] == information_dict["wall"]:
            actions.remove(0)
        if col<ncols-2 and self.maze[row,col+1] == information_dict["wall"]:
            actions.remove(2)

        return actions

    def show(self, external_maze=None):
        plt.grid('on')
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, self.nrows, 1))
        ax.set_yticks(np.arange(0.5, self.ncols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if external_maze is not None:
            canvas = np.copy(external_maze)
        else:
            canvas = np.copy(self.maze)
        nrows, ncols = canvas.shape

        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > information_dict["wall"]:
                    canvas[r,c] = mark_dict["free"]
                else:
                    canvas[r, c] = mark_dict["wall"]


        for row, col in self.visited:
            canvas[row,col] = mark_dict["visited"]

        player_row, player_col, _ = self.state
        canvas[player_row, player_col] = mark_dict["player"]
        ## Goal Cell
        for target_row, target_col in self.target:
            canvas[target_row, target_col] = mark_dict["goal"]
        img = plt.imshow(canvas, interpolation='none', cmap='gray')
        return img

    def export_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > information_dict["wall"]:
                    canvas[r,c] = information_dict["free"]
        # draw the rat
        for row, col in self.visited:
            canvas[row,col] = information_dict["visited"]

        player_row, player_col, _ = self.state
        canvas[player_row, player_col] = information_dict["player"]
        
        ## Goal Cell
        for target_row, target_col in self.target:
            canvas[target_row, target_col] = information_dict["goal"]

        return canvas.reshape((1, -1))
