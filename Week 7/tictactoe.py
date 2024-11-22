def print_tic_tac_toe(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)
def print_scoreboard(score):
    print("Scoreboard:")
    for player, points in score.items():
        print(f"{player}: {points}")

        def check_winner(board, player):
            win_conditions = [
                [board[0][0], board[0][1], board[0][2]],
                [board[1][0], board[1][1], board[1][2]],
                [board[2][0], board[2][1], board[2][2]],
                [board[0][0], board[1][0], board[2][0]],
                [board[0][1], board[1][1], board[2][1]],
                [board[0][2], board[1][2], board[2][2]],
                [board[0][0], board[1][1], board[2][2]],
                [board[2][0], board[1][1], board[0][2]],
            ]
            return [player, player, player] in win_conditions

        def tic_tac_toe():
            board = [[" " for _ in range(3)] for _ in range(3)]
            players = ["X", "O"]
            score = {players[0]: 0, players[1]: 0}
            current_player = 0

            for _ in range(9):
                print_tic_tac_toe(board)
                row, col = map(int, input(f"Player {players[current_player]}, enter row and column (0, 1, 2): ").split())
                if board[row][col] == " ":
                    board[row][col] = players[current_player]
                    if check_winner(board, players[current_player]):
                        print_tic_tac_toe(board)
                        print(f"Player {players[current_player]} wins!")
                        score[players[current_player]] += 1
                        break
                    current_player = 1 - current_player
                else:
                    print("Cell already taken, try again.")
            else:
                print_tic_tac_toe(board)
                print("It's a draw!")

            print_scoreboard(score)

        if __name__ == "__main__":
            tic_tac_toe()