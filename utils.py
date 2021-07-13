import cv2
import numpy as np

WIDTH_IMG = 576
HEIGHT_IMG = 576


def find_board(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(gray, (5, 5), 1)
    thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = np.array([])
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                points = approx
                max_area = area

    if points.size != 0:
        points = points.reshape((4, 2))
        rect = np.zeros((4, 1, 2), dtype=np.int32)
        add = np.sum(points, axis=1)
        rect[0] = points[np.argmin(add)]
        rect[3] = points[np.argmax(add)]
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[2] = points[np.argmax(diff)]
    return rect


def create_perspective_matrix(points):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [WIDTH_IMG, 0], [0, HEIGHT_IMG], [WIDTH_IMG, HEIGHT_IMG]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return matrix


def split_cells(image):
    rows = np.vsplit(image, 9)
    cells = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for cell in cols:
            cells.append(cell)
    return cells


def extract_digit(cell):
    height, width = cell.shape
    percent = 0.1
    cell = cell[int(height*percent):height-int(height*percent)+1,
                int(width*percent):width-int(width*percent)+1]
    bitmap = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    (h, w) = bitmap.shape
    percentFilled = cv2.countNonZero(bitmap) / float(w * h)
    if percentFilled > 0.2:
        return None
    contours, _ = cv2.findContours(bitmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(bitmap.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    digit = cv2.bitwise_and(bitmap, bitmap, mask=mask)
    digit = cv2.resize(digit, (32, 32))
    return digit


def predict_digits(cells, model):
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


def draw_grid(image, matrix):
    stepX = WIDTH_IMG // 9
    stepY = HEIGHT_IMG // 9
    pts = []
    for y in [0, 9]:
        for x in range(0, 10):
            startX = x * stepX
            startY = y * stepY
            pts.append([[startX, startY]])
    for x in [0, 9]:
        for y in range(0, 10):
            startX = x * stepX
            startY = y * stepY
            pts.append([[startX, startY]])
    pts = np.array(pts, dtype=np.float32)

    pts = cv2.perspectiveTransform(pts, matrix).astype(np.int32)
    pts = pts.reshape(4, 10, 2)
    pts1, pts2 = pts[0], pts[1]
    for pt1, pt2 in zip(pts1, pts2):
        cv2.line(image, tuple(pt1), tuple(pt2), (0, 0, 255), 3)

    pts1, pts2 = pts[2], pts[3]
    for pt1, pt2 in zip(pts1, pts2):
        cv2.line(image, tuple(pt1), tuple(pt2), (0, 0, 255), 3)


def draw_numbers(image, board, matrix, mask):
    stepX = WIDTH_IMG // 9
    stepY = HEIGHT_IMG // 9

    for y in range(0, 9):
        for x in range(0, 9):
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY
            centerX = (startX + endX) // 2
            centerY = (startY + endY) // 2
            pt = np.array([[[centerX, centerY]]], dtype=np.float32)
            pt = cv2.perspectiveTransform(pt, matrix)
            centerX = int(pt[0][0][0])
            centerY = int(pt[0][0][1])
            color = (0, 0, 0) if mask[y][x] else (0, 0, 255)
            cv2.putText(image, str(board[y, x]), (centerX-10, centerY+10),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, color, thickness=2)
