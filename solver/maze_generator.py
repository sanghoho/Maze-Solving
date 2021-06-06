import random 

class Room:
    def __init__(self, r, c):
        self.r, self.c = r, c
        self.visit = 0
        self.prev = None
        self.drct = [(r + 1, c), (r, c + 1),
                     (r - 1, c), (r, c - 1)]
        random.shuffle(self.drct)


class Maze:
    def __init__(self, size:int=32):
        self.size = size
        self.setup()
        
    def setup(self):
        self._maze = [[Room(r, c) for c in range(self.size)] for r in range(self.size)]
        self._mazeMap = [["1" for c in range(self.size * 2 + 1)] for r in range(self.size * 2 + 1)]
        self.__make(None, self._maze[0][0], self._maze)
        

        start = self.size - 1 # ()
        end = self.size + 1 # ()
        for i in range(start, end+1):
            for j in range(start, end+1):
                self._mazeMap[i][j] = "0"
                
        self._mazeMap[start][start] = self._mazeMap[start][end] = self._mazeMap[end][start] = self._mazeMap[end][end] = "3"
        self._mazeMap[1][1] = "2"
        self.maze = self._maze
        self.mazeMap = self._mazeMap
    
    def __make(self, prev, room:Room, maze:list):
        room.prev = prev
        if room.prev is not None:
            r = prev.r - room.r
            c = prev.c - room.c
            self._mazeMap[(room.r + 1) * 2 - 1 + r][(room.c + 1) * 2 - 1 + c] = "0"
            
        
        room.visit = 1
        self._mazeMap[(room.r + 1) * 2 - 1][(room.c + 1) * 2 - 1] = "0"
        while True:
            if len(room.drct) == 0:
                break
            nr, nc = room.drct.pop()
            if nr >= 0 and nr < self.size and nc >= 0 and nc < self.size:
                if not maze[nr][nc].visit == 1:
                    self.__make(room, maze[nr][nc], maze)
                
    def save(self, filename:str='maze.txt'):
        with open(filename, 'w') as f:
            for r in self.mazeMap:
                for c in r:
                    f.write(c)
                f.write('\n')