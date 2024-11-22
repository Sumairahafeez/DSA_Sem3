class MonkeyBananaProblem:
    def __init__(self):
        self.monkey_position = 'A'
        self.banana_position = 'C'
        self.has_banana = False

    def move_monkey(self, position):
        if position in ['A', 'B', 'C']:
            self.monkey_position = position
            print(f"Monkey moved to {position}")
        else:
            print("Invalid position")

    def pick_banana(self):
        if self.monkey_position == self.banana_position:
            self.has_banana = True
            print("Monkey picked the banana")
        else:
            print("Monkey is not at the banana position")

    def solve(self):
        print("Initial State: Monkey at A, Banana at C")
        self.move_monkey('B')
        self.move_monkey('C')
        self.pick_banana()
        if self.has_banana:
            print("Monkey has the banana!")
        else:
            print("Monkey couldn't get the banana")

if __name__ == "__main__":
    problem = MonkeyBananaProblem()
    problem.solve()