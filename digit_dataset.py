import os

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def make_dataset(path='data', skip0=True):
    images = []
    labels = []

    if skip0:
        for i in range(1, 10):
            pic_paths = os.listdir(os.path.join(path, str(i)))
            for file_name in pic_paths:
                image = cv2.imread(os.path.join(path, str(i), file_name), cv2.IMREAD_GRAYSCALE)
                image = cv2.equalizeHist(image)
                image = 255 - image
                image = cv2.resize(image, (32, 32))
                images.append(np.expand_dims(image, 0).astype(np.float32))
                labels.append(i-1)
    else:
        for i in range(0, 10):
            pic_paths = os.listdir(os.path.join(path, str(i)))
            for file_name in pic_paths:
                image = cv2.imread(os.path.join(path, str(i), file_name), cv2.IMREAD_GRAYSCALE)
                image = cv2.equalizeHist(image)
                image = 255 - image
                image = cv2.resize(image, (32, 32))
                images.append(np.expand_dims(image, 0).astype(np.float32))
                labels.append(i-1)

    print('Total images: ', len(images))
    print('Total labels: ', len(labels))
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=17)
    return DigitDataset(X_train, y_train), DigitDataset(X_test, y_test)

def make_data(path='data', skip0=True):
    train_dataset, test_dataset = make_dataset(path, skip0)
    params = {
        'train': {
            'images': train_dataset.images,
            'labels': train_dataset.labels,
        },
        'test': {
            'images': test_dataset.images,
            'labels': test_dataset.labels
        }
    }
    torch.save(params, 'data/data.pt')

class DigitDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.from_numpy(image)

        label = self.labels[idx]
        return image, label

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    make_data()
