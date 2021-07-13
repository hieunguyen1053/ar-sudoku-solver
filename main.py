import sys

import cv2

from solver import solve
from utils import *

WIDTH_IMG = 576
HEIGHT_IMG = 576
DEBUG = True

if __name__ == '__main__':
    image = cv2.imread(sys.argv[1])
    model = cv2.dnn.readNetFromONNX('models/mnist_cnn.onnx')

    points = find_board(image)
    matrix = create_perspective_matrix(points)
    imatrix = np.linalg.inv(matrix)

    img_wraped = cv2.warpPerspective(image, matrix, (WIDTH_IMG, HEIGHT_IMG))

    if DEBUG:
        cv2.imshow('Board', img_wraped)
        cv2.waitKey(1)

    gray_wraped = cv2.cvtColor(img_wraped, cv2.COLOR_BGR2GRAY)
    gray_wraped = cv2.GaussianBlur(gray_wraped, (5, 5), 1)

    cells = split_cells(gray_wraped)
    board = predict_digits(cells, model)
    mask = np.zeros((9, 9), dtype=np.int0)
    mask[np.nonzero(board)] = 1

    if DEBUG:
        matrix = np.eye(3)
        pts = np.float32([[0, 0], [WIDTH_IMG, 0], [0, HEIGHT_IMG], [WIDTH_IMG, HEIGHT_IMG]])
        temp_wraped = img_wraped.copy()
        draw_grid(temp_wraped, matrix)
        draw_numbers(temp_wraped, board, matrix, mask)
        cv2.imshow('Detected Board', temp_wraped)
        cv2.waitKey(1)

    solve(board)

    if DEBUG:
        matrix = np.eye(3)
        pts = np.float32([[0, 0], [WIDTH_IMG, 0], [0, HEIGHT_IMG], [WIDTH_IMG, HEIGHT_IMG]])
        temp_wraped = img_wraped.copy()
        draw_grid(temp_wraped, matrix)
        draw_numbers(temp_wraped, board, matrix, mask)
        cv2.imshow('Solved Board', temp_wraped)
        cv2.waitKey(1)

    draw_grid(image, imatrix)
    draw_numbers(image, board, imatrix, mask)

    cv2.imshow('Demo', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
