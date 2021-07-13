import numpy as np


def solve(board):
    find = find_empty(board)
    if not find:
        return True
    else:
        row, col = find

    for num in possible_numbers(board, (row, col)):
        if valid(board, num, (row, col)):
            board[row][col] = num

            if solve(board):
                return True

            board[row][col] = 0
    return False

def possible_numbers(board, pos):
    y, x = pos
    box_x = x // 3
    box_y = y // 3
    poss_nums = set(range(1, 10))
    appeared_num = set()
    appeared_num |= set(board[:, x])
    appeared_num |= set(board[y, :])
    appeared_num |= set(board[box_y*3:box_y*3+3, box_x*3:box_x*3+3].flatten())
    poss_nums -= appeared_num
    return list(poss_nums)

def valid(box, num, pos):
    for i in range(len(box[0])):
        if box[pos[0]][i] == num and pos[1] != i:
            return False

    for i in range(len(box)):
        if box[i][pos[1]] == num and pos[0] != i:
            return False

    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if box[i][j] == num and (i,j) != pos:
                return False

    return True

def find_empty(box):
    for i in range(len(box)):
        for j in range(len(box[0])):
            if box[i][j] == 0:
                return (i, j)

    return None
