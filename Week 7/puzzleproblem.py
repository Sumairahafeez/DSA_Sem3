import heapq

class PuzzleState:
    def __init__(self, board, goal, moves=0, previous=None):
        self.board = board
        self.goal = goal
        self.moves = moves
        self.previous = previous

    def __lt__(self, other):
        return self.cost() < other.cost()

    def cost(self):
        return self.moves + self.manhattan_distance()

    def manhattan_distance(self):
        distance = 0
        for i in range(3):
            for j in range(3):
                if self.board[i][j] != 0:
                    x, y = divmod(self.board[i][j] - 1, 3)
                    distance += abs(x - i) + abs(y - j)
        return distance

    def is_goal(self):
        return self.board == self.goal

    def neighbors(self):
        def swap(board, i1, j1, i2, j2):
            new_board = [row[:] for row in board]
            new_board[i1][j1], new_board[i2][j2] = new_board[i2][j2], new_board[i1][j1]
            return new_board

        neighbors = []
        i, j = next((i, j) for i in range(3) for j in range(3) if self.board[i][j] == 0)
        if i > 0: neighbors.append(PuzzleState(swap(self.board, i, j, i - 1, j), self.goal, self.moves + 1, self))
        if i < 2: neighbors.append(PuzzleState(swap(self.board, i, j, i + 1, j), self.goal, self.moves + 1, self))
        if j > 0: neighbors.append(PuzzleState(swap(self.board, i, j, i, j - 1), self.goal, self.moves + 1, self))
        if j < 2: neighbors.append(PuzzleState(swap(self.board, i, j, i, j + 1), self.goal, self.moves + 1, self))
        return neighbors

def solve_puzzle(start, goal):
    open_set = []
    heapq.heappush(open_set, PuzzleState(start, goal))
    closed_set = set()

    while open_set:
        current = heapq.heappop(open_set)
        if current.is_goal():
            return current

        closed_set.add(tuple(map(tuple, current.board)))

        for neighbor in current.neighbors():
            if tuple(map(tuple, neighbor.board)) not in closed_set:
                heapq.heappush(open_set, neighbor)

    return None

def print_solution(solution):
    path = []
    while solution:
        path.append(solution.board)
        solution = solution.previous
    path.reverse()
    for step in path:
        for row in step:
            print(row)
        print()

if __name__ == "__main__":
    start = [
        [1, 2, 3],
        [4, 0, 5],
        [7, 8, 6]
    ]

    goal = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]

    solution = solve_puzzle(start, goal)
    if solution:
        print_solution(solution)
    else:
        print("No solution found")