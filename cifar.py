import pickle
import numpy as np
import os

class Cifar100:
    def __init__(self):
        with open('cifar-100-python/train','rb') as f:
            self.train = pickle.load(f, encoding='latin1')
        with open('cifar-100-python/test','rb') as f:
            self.test = pickle.load(f, encoding='latin1')
        self.train_data = self.train['data']
        self.train_labels = self.train['fine_labels']
        self.test_data = self.test['data']
        self.test_labels = self.test['fine_labels']
        self.batch_num = 5   # number of tasks
        self.class_ranges = [(0, 20), (20, 22), (22, 26), (26, 29), (29, 30)]
        self.train_groups, self.val_groups, self.test_groups = self.initialize()
        

    def initialize(self):
        train_groups = [[] for _ in range(self.batch_num)]
        
        for train_data, train_label in zip(self.train_data, self.train_labels):
            # print(train_data.shape)
            train_data_r = train_data[:1024].reshape(32, 32)
            train_data_g = train_data[1024:2048].reshape(32, 32)
            train_data_b = train_data[2048:].reshape(32, 32)
            train_data = np.dstack((train_data_r, train_data_g, train_data_b))
            for i, (low, high) in enumerate(self.class_ranges):
                if low <= train_label < high:
                    train_groups[i].append((train_data, train_label))
            # if train_label < 20:
            #     train_groups[0].append((train_data,train_label))
            # elif 20 <= train_label < 22:
            #     train_groups[1].append((train_data,train_label))
            # elif 22 <= train_label < 26:
            #     train_groups[2].append((train_data,train_label))
            # elif 26 <= train_label < 29:
            #     train_groups[3].append((train_data,train_label))
            # elif 29 <= train_label < 30:
            #     train_groups[4].append((train_data,train_label))
        
        assert len(train_groups[0]) == (20-0) *500 
        assert len(train_groups[1]) == (22-20)*500 
        assert len(train_groups[2]) == (26-22)*500 
        assert len(train_groups[3]) == (29-26)*500 
        assert len(train_groups[4]) == (30-29)*500 

        val_groups = [[] for _ in range(self.batch_num)]
        for i in range(self.batch_num):
            # val_groups.append([])
            train_sample_num = int(0.9*len(train_groups[i]))
            val_groups[i]   = train_groups[i][train_sample_num:]
            train_groups[i] = train_groups[i][:train_sample_num]
        
        assert len(train_groups[0]) == (20-0) *500*0.9, len(train_groups[0])
        assert len(train_groups[1]) == (22-20)*500*0.9, len(train_groups[1])
        assert len(train_groups[2]) == (26-22)*500*0.9, len(train_groups[2])
        assert len(train_groups[3]) == (29-26)*500*0.9, len(train_groups[3])
        assert len(train_groups[4]) == (30-29)*500*0.9, len(train_groups[4])
        assert len(val_groups[0])   == (20-0) *500*0.1, len(val_groups[0])
        assert len(val_groups[1])   == (22-20)*500*0.1, len(val_groups[1])
        assert len(val_groups[2])   == (26-22)*500*0.1, len(val_groups[2])
        assert len(val_groups[3])   == (29-26)*500*0.1, len(val_groups[3])
        assert len(val_groups[4])   == (30-29)*500*0.1, len(val_groups[4])


        test_groups = [[] for _ in range(self.batch_num)]
        for test_data, test_label in zip(self.test_data, self.test_labels):
            test_data_r = test_data[:1024].reshape(32, 32)
            test_data_g = test_data[1024:2048].reshape(32, 32)
            test_data_b = test_data[2048:].reshape(32, 32)
            test_data = np.dstack((test_data_r, test_data_g, test_data_b))
            for i, (low, high) in enumerate(self.class_ranges):
                if low <= test_label < high:
                    test_groups[i].append((test_data, test_label))
            # if test_label < 20:
            #     test_groups[0].append((test_data,test_label))
            # elif 20 <= test_label < 22:
            #     test_groups[1].append((test_data,test_label))
            # elif 22 <= test_label < 26:
            #     test_groups[2].append((test_data,test_label))
            # elif 26 <= test_label < 29:
            #     test_groups[3].append((test_data,test_label))
            # elif 29 <= test_label < 30:
            #     test_groups[4].append((test_data,test_label))
        assert len(test_groups[0]) == 100*(20-0) , len(test_groups[0])
        assert len(test_groups[1]) == 100*(22-20), len(test_groups[1])
        assert len(test_groups[2]) == 100*(26-22), len(test_groups[2])
        assert len(test_groups[3]) == 100*(29-26), len(test_groups[3])
        assert len(test_groups[4]) == 100*(30-29), len(test_groups[4])

        return train_groups, val_groups, test_groups

    def getNextClasses(self, i):
        if not (0 <= i < len(self.train_groups)):
            raise ValueError(f"Invalid index {i}, should be between 0 and {len(self.train_groups)-1}")
        return self.train_groups[i], self.val_groups[i], self.test_groups[i]

if __name__ == "__main__":
    cifar = Cifar100()
    print(len(cifar.train_groups[0]))
