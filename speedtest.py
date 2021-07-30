import sys
import time

import cv2
import torch

from net import Net
from utils import *

WIDTH_IMG = 576
HEIGHT_IMG = 576
DEBUG = True

def cv_predict_digits(cells, model):
    board = []
    for cell in cells:
        digit = extract_digit(cell)
        if digit is not None:
            input_blob = cv2.dnn.blobFromImage(image=digit)
            model.setInput(input_blob)
            pred = model.forward()
            label = np.argmax(pred) + 1
            probabilityValue = pred[0][np.argmax(pred)]
            if probabilityValue > 0.8:
                board.append(label)
            else:
                board.append(0)
        else:
            board.append(0)
    board = np.array(board).reshape(9, 9)
    return board

def pytorch_predict_digits(cells, model):
    board = []
    for cell in cells:
        digit = extract_digit(cell)
        if digit is not None:
            digit = torch.from_numpy(digit).unsqueeze(0).unsqueeze(0).float()
            pred = model.forward(digit)
            label = torch.argmax(pred).item() + 1
            probabilityValue = pred[0][label - 1].item()
            if probabilityValue > 0.8:
                board.append(label)
            else:
                board.append(0)
        else:
            board.append(0)
    board = np.array(board).reshape(9, 9)
    return board

if __name__ == '__main__':
    image = cv2.imread(sys.argv[1])

    #Setup
    model1 = cv2.dnn.readNetFromONNX('models/mnist_cnn.onnx')
    model2 = Net()
    model2.load_state_dict(torch.load('models/mnist_cnn.pt', map_location=torch.device('cpu')))

    points = find_board(image)
    matrix = create_perspective_matrix(points)

    img_wraped = cv2.warpPerspective(image, matrix, (WIDTH_IMG, HEIGHT_IMG))

    gray_wraped = cv2.cvtColor(img_wraped, cv2.COLOR_BGR2GRAY)
    gray_wraped = cv2.GaussianBlur(gray_wraped, (5, 5), 1)

    cells = split_cells(gray_wraped)
    start = time.time()
    board = cv_predict_digits(cells, model1)
    print('CV Prediction time', time.time() - start)
    print(board)
    start = time.time()
    board = pytorch_predict_digits(cells, model2)
    print('Pytorch Prediction time', time.time() - start)
    print(board)
