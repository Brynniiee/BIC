import os
import pickle
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from loadmat import MatDataLoader

class GXWData:
    def __init__(self, batch_num=2, class_ranges=[(0, 3), (3, 5)]):
        loader = MatDataLoader(root_dir='GXW_Data/DASdata')
        self.train_data = loader.train_data
        self.train_labels = loader.train_labels
        self.test_data = loader.test_data
        self.test_labels = loader.test_labels
        self.class_to_idx = loader.class_to_idx

        self.batch_num = batch_num
        self.class_ranges = class_ranges
        self.train_groups, self.val_groups, self.test_groups = self.initialize()

    def initialize(self):
        train_groups = [[] for _ in range(self.batch_num)]
        val_groups = [[] for _ in range(self.batch_num)]
        test_groups = [[] for _ in range(self.batch_num)]

        for train_data, label in zip(self.train_data, self.train_labels):

            for i, (low, high) in enumerate(self.class_ranges):
                
                if low <= label < high:
                    train_groups[i].append((train_data, label))
                    #print(f"label: {label}, low: {low}, high: {high}, i: {i}")  # np
        print(train_groups[0][-1])

        for i in range(self.batch_num):
            n = int(len(train_groups[i]) * 0.9)
            #train_groups = random.shuffle(train_groups)  "永远不要用shuffle的输出赋值变量"
            random.shuffle(train_groups[i])
            val_groups[i] = train_groups[i][n:]
            train_groups[i] = train_groups[i][:n]

        for test_data, label in zip(self.test_data, self.test_labels):

            for i, (low, high) in enumerate(self.class_ranges):
                if low <= label < high:
                    test_groups[i].append((test_data, label))

        return train_groups, val_groups, test_groups

    def getNextClasses(self, i):
        if not (0 <= i < len(self.train_groups)):
            raise ValueError(f"Invalid index {i}, should be between 0 and {len(self.train_groups)-1}")
        return self.train_groups[i], self.val_groups[i], self.test_groups[i]

if __name__ == "__main__":
    GXW_data = GXWData()
    print(len(GXW_data.train_groups[0]))

if __name__ == "__main__":
    loader = MatDataLoader("GXW_Data/DASdata")
    print(f"Loaded {len(loader.train_data)} training samples.")
    print(f"Loaded {len(loader.test_data)} testing samples.")
    print(f"Sample shape: {loader.train_data[0].shape}, dtype: {loader.train_data[0].dtype}")