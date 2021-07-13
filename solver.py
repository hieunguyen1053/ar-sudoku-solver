def solve(box):
    find = find_empty(box)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1,10):
        if valid(box, i, (row, col)):
            box[row][col] = i

            if solve(box):
                return True

            box[row][col] = 0
    return False

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
