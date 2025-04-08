import os
import pickle
import random
import numpy as np
from PIL import Image
from torchvision import transforms

class GXWData:
    def __init__(self, path='picklewithsplit.pkl', batch_num=2, class_ranges=[(0, 3), (3, 4)]):
        with open('picklewithsplit.pkl', 'rb') as f:
            class_to_idx = pickle.load(f)['class_to_idx']



            self.train_data = []
            self.train_labels = []
            self.test_data = []
            self.test_labels = []



            while True:
                try:
                    section, data, label = pickle.load(f)
                    if section == 'train':
                        self.train_data.append(data)
                        self.train_labels.append(label)
                    elif section == 'test':
                        self.test_data.append(data)
                        self.test_labels.append(label)
                except EOFError:
                    break  


        self.batch_num = batch_num
        self.class_ranges = class_ranges
        self.train_groups, self.val_groups, self.test_groups = self.initialize()

    def initialize(self):
        train_groups = [[] for _ in range(self.batch_num)]
        val_groups = [[] for _ in range(self.batch_num)]
        test_groups = [[] for _ in range(self.batch_num)]

        for train_data, label in zip(self.train_data, self.train_labels):
            
            train_data_r = train_data[:(875*656)].reshape(875, 656)
            train_data_g = train_data[(875*656):2*(875*656)].reshape(875, 656)
            train_data_b = train_data[2*(875*656):].reshape(875, 656)
            train_data = np.dstack((train_data_r, train_data_g, train_data_b))

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
            test_data_r = test_data[:(875*656)].reshape(875, 656)
            test_data_g = test_data[(875*656):2*(875*656)].reshape(875, 656)
            test_data_b = test_data[2*(875*656):].reshape(875, 656)
            test_data = np.dstack((test_data_r, test_data_g, test_data_b))
            for i, (low, high) in enumerate(self.class_ranges):
                if low <= label < high:
                    test_groups[i].append((test_data, label))
        print("训练集唯一标签:", set(self.train_labels))
        print("测试集唯一标签:", set(self.test_labels))
        return train_groups, val_groups, test_groups

    def getNextClasses(self, i):
        if not (0 <= i < len(self.train_groups)):
            raise ValueError(f"Invalid index {i}, should be between 0 and {len(self.train_groups)-1}")
        return self.train_groups[i], self.val_groups[i], self.test_groups[i]

if __name__ == "__main__":
    GXW_data = GXWData()
    print(len(GXW_data.train_groups[0]))