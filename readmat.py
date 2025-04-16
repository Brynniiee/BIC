import os
import pickle
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from loadmat import MatDataLoader
from tqdm import tqdm

class GXWData:
    def __init__(self, batch_num=2, class_ranges=[[0,1,2,3], [4,5]], train_val_test_split=0.8):
        self.rootdir = 'GXW_Data/DASdata'
        loader = MatDataLoader(self.rootdir)   
        # 出现了testdata中几乎没有少数类的情况，需在loadmat中平均化，才能保证后续的数据增强取得稳定效果
        self.train_data = loader.train_data
        self.train_labels = loader.train_labels
        self.test_data = loader.test_data
        self.test_labels = loader.test_labels
        self.class_to_idx = loader.class_to_idx
        self.train_val_test_split = train_val_test_split
        self.batch_num = batch_num
        self.class_ranges = class_ranges
        self.train_groups, self.val_groups, self.test_groups = self.initialize()
        self.class_names = sorted(os.listdir(self.rootdir))

    def initialize(self):
        train_groups = [[] for _ in range(self.batch_num)]
        val_groups = [[] for _ in range(self.batch_num)]
        test_groups = [[] for _ in range(self.batch_num)]

        for train_data, label in zip(self.train_data, self.train_labels):

            for i, class_list in enumerate(self.class_ranges):
                if label in class_list:
                    train_groups[i].append((train_data, label))
                    #print(f"label: {label}, low: {low}, high: {high}, i: {i}")  # np
        print(train_groups[0][-1])

        for i in range(self.batch_num):
            n = int(len(train_groups[i]) * self.train_val_test_split)
            #train_groups = random.shuffle(train_groups)  "永远不要用shuffle的输出赋值变量"
            random.shuffle(train_groups[i])
            # 不可以random shuffle！会导致数据不均匀
            val_groups[i] = train_groups[i][n:]
            train_groups[i] = train_groups[i][:n]

        for test_data, label in zip(self.test_data, self.test_labels):

            for i, class_list in enumerate(self.class_ranges):
                if label in class_list:
                    test_groups[i].append((test_data, label))
        
        # 统计每类样本数量
        train_data_by_label = self.group_data_by_label(train_groups)
        val_data_by_label = self.group_data_by_label(val_groups)
        test_data_by_label = self.group_data_by_label(test_groups)
        print("Train Samples Before augmentation:", {k: len(v) for k, v in train_data_by_label.items()})
        print("Val Samples Before augmentation:", {k: len(v) for k, v in val_data_by_label.items()})
        print("Test Samples Before augmentation:", {k: len(v) for k, v in test_data_by_label.items()})
        # 生成增强样本
        augmented_train = self.balance_with_augmentation(train_data_by_label)
        augmented_val = self.balance_with_augmentation(val_data_by_label)
        augmented_test = self.balance_with_augmentation(test_data_by_label)
        # 拼回原始数据组
        print(f'train size: {len(train_groups)}')
        for augmented_sample in augmented_train:
            for j, class_range in enumerate(self.class_ranges):
                if augmented_sample[1] in class_range:  # augmented_sample[1] 是标签
                    train_groups[j].append(augmented_sample)
                    break  # 一旦找到对应的 class_range，就可以停止查找
        for augmented_sample in augmented_val:
            for j, class_range in enumerate(self.class_ranges):
                if augmented_sample[1] in class_range:
                    val_groups[j].append(augmented_sample)
                    break
        for augmented_sample in augmented_test:
            for j, class_range in enumerate(self.class_ranges):
                if augmented_sample[1] in class_range:
                    test_groups[j].append(augmented_sample)
                    break
        train_data_by_label = self.group_data_by_label(train_groups)
        val_data_by_label = self.group_data_by_label(val_groups)
        test_data_by_label = self.group_data_by_label(test_groups)
        print("Train samples After augmentation:", {k: len(v) for k, v in train_data_by_label.items()})
        print("Val samples After augmentation:", {k: len(v) for k, v in val_data_by_label.items()})
        print("Test samples After augmentation:", {k: len(v) for k, v in test_data_by_label.items()})




        return train_groups, val_groups, test_groups

    def getNextClasses(self, i):
        if not (0 <= i < len(self.train_groups)):
            raise ValueError(f"Invalid index {i}, should be between 0 and {len(self.train_groups)-1}")
        return self.train_groups[i], self.val_groups[i], self.test_groups[i]
    
    def count_class_samples(self, data):
        class_count = {}
        for _, label in data:
            if label not in class_count:
                class_count[label] = 0
            class_count[label] += 1
        return class_count
    
    def group_data_by_label(self, data):
        all_data = sum(data, [])  # 展平数据
        label_groups = {}
        for sample, label in all_data:
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(sample)
        return label_groups
    
    def balance_with_augmentation(self, data_by_label, augmentation_factor=3):
        """输入按类别分组的数据，返回增强后的 (sample, label) 列表"""
        all_labels = data_by_label.keys()
        class_counts = {k: len(v) for k, v in data_by_label.items()}
        max_count = max(class_counts.values())  
        augmented_data = []

        for label, samples in tqdm(data_by_label.items(), desc="Balancing classes", leave=True):
            current_count = len(samples)
            if current_count <= max_count // 2:
                # 如果类别数量过少,按倍数进行增强，使有效样本数大于最大类别数一半
                augmentation_factor = max_count // current_count - 1
                if augmentation_factor > 0:
                    for s in tqdm(samples, desc=f"Augmenting {label}", leave=False):
                        augmented_samples = self.sample_augmentation(s, augmentation_factor)
                        augmented_data.extend([(a, label) for a in augmented_samples])
        return augmented_data


    def sample_augmentation(self, sample, augmentation_factor):
        """样本增强，返回A个增强版本"""
        augmented = []
        for _ in range(augmentation_factor):  # 生成 A 个增强数据
            s = sample.copy()
            # 加入高斯噪声
            noise = np.random.normal(0, 0.01 * np.std(s), s.shape)
            s_noisy = s + noise
            # 缩放（模拟信号幅度变化）
            scale = np.random.uniform(0.9, 1.1)
            s_scaled = s_noisy * scale
            # 微移（时间轴方向）
            target_len = 10240
            shift = np.random.randint(-20, 20)
            if shift >= 0:
                s_shifted = np.pad(s_scaled, ((0, 0), (shift, 0)), mode='constant')[:, :target_len]
            else:
                s_shifted = np.pad(s_scaled, ((0, 0), (0, -shift)), mode='constant')[:, -shift:target_len]
            if s_shifted.shape[1] < target_len:
                pad_len = target_len - s_shifted.shape[1]
                s_shifted = np.pad(s_shifted, ((0, 0), (0, pad_len)), mode='constant')
            elif s_shifted.shape[1] > target_len:
                s_shifted = s_shifted[:, :target_len]
            augmented.append(s_shifted.astype(np.float32))
        return augmented





if __name__ == "__main__":
    GXW_data = GXWData()
    print(len(GXW_data.train_groups[0]))
    print("Class folder names (sorted):", GXW_data.class_names)
    print("class_to_idx mapping:", GXW_data.class_to_idx)

# if __name__ == "__main__":
#     loader = MatDataLoader("GXW_Data/DASdata")
#     print(f"Loaded {len(loader.train_data)} training samples.")
#     print(f"Loaded {len(loader.test_data)} testing samples.")
#     print(f"Sample shape: {loader.train_data[0].shape}, dtype: {loader.train_data[0].dtype}")