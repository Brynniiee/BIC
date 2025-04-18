import os
import pickle
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from loadmat import MatDataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class GXWData:
    def __init__(self, batch_num=2, class_ranges=[[0,1,2,3], [4,5]], val_train_split=0.2, test_train_split=0.2):
        '''
        following steps included:
            1. load data from loadmat
            2. initialize()
        '''
        self.rootdir = 'GXW_Data/DASdata'
        loader = MatDataLoader(self.rootdir)   
        # 出现了testdata中几乎没有少数类的情况，需在loadmat中平均化，才能保证后续的数据增强取得稳定效果
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        self.all_data = loader.all_data
        self.all_labels = loader.all_labels
        self.class_to_idx = loader.class_to_idx
        self.val_train_split = val_train_split
        self.test_train_split = test_train_split
        self.batch_num = batch_num
        self.class_ranges = class_ranges
        self.train_groups, self.val_groups, self.test_groups = self.initialize()
        self.class_names = sorted(os.listdir(self.rootdir))

    def initialize(self):
        '''
        following steps included:
            1. transfer data from labels & data to list; [(label, data), ...]
            3. data augmentation to balance sample number for each class
            4. split data into train, val and test groups
            5. for counting samples in each class, group data by label
        Returns:
            train_groups: type为list,[(data,label),...]
            val_groups
            test_groups
        '''

        groups = [[] for _ in range(self.batch_num)]
        # train_groups = [[] for _ in range(self.batch_num)]
        # val_groups = [[] for _ in range(self.batch_num)]
        # test_groups = [[] for _ in range(self.batch_num)]

        for all_data, all_label in zip(self.all_data, self.all_labels):
            for i, class_list in enumerate(self.class_ranges):
                if all_label in class_list:
                    groups[i].append((all_data, all_label))

        all_data_by_label = self.group_data_by_label(groups)
        print("Samples Before augmentation:", {k: len(v) for k, v in all_data_by_label.items()})
        augmented_set = self.balance_with_augmentation(all_data_by_label)
        for augmented_sample in augmented_set:
            for j, class_range in enumerate(self.class_ranges):
                if augmented_sample[1] in class_range:  # augmented_sample[1] 是标签
                    groups[j].append(augmented_sample)
                    break  # 一旦找到对应的 class_range，就可以停止查找
        all_data_by_label = self.group_data_by_label(groups)
        print(f"Samples after augmentation:", {k: len(v) for k, v in all_data_by_label.items()})
        train_groups, val_groups, test_groups = self.train_val_test_split(data = groups, val_train_split=self.val_train_split, test_train_split = self.test_train_split, batch_num=self.batch_num)
        # 统计每类样本数量
        train_data_by_label = self.group_data_by_label(train_groups)
        val_data_by_label = self.group_data_by_label(val_groups)
        test_data_by_label = self.group_data_by_label(test_groups)
        print("Train Samples after augmentation:", {k: len(v) for k, v in train_data_by_label.items()})
        print("Val Samples after augmentation:", {k: len(v) for k, v in val_data_by_label.items()})
        print("Test Samples after augmentation:", {k: len(v) for k, v in test_data_by_label.items()})
        return train_groups, val_groups, test_groups


    def train_val_test_split(self, data = [], val_train_split = 0.2, test_train_split=0.2, batch_num = 1):
        """ 划分数据集为训练集、验证集和测试集
        Args:
            data: type 为 dict
            val_train_split: 验证集的比例
            test_train_split: 测试集的比例
            train_groups: 待加入样本的空训练集
            val_groups: 待加入样本的空验证集
            test_groups: 待加入样本的空测试集
        Returns:
            train_groups: type为list,[(data,label),...]
            val_groups
            test_groups
        """
        train_groups = [[] for _ in range(batch_num)]
        val_groups = [[] for _ in range(batch_num)]
        test_groups = [[] for _ in range(batch_num)]
        for task_id, data_group in enumerate(data):  # data_group is a list of (data, label)
            # 分离数据和标签
            print(len(data_group))
            X = [x for x, y in data_group]
            y = [y for x, y in data_group]
            # 先划分 train_val 和 test
            X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_train_split, stratify=y, random_state=42)
            # 再划分 train 和 val
            X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_train_split, stratify=y_trainval, random_state=42)

            train_groups[task_id].extend(list(zip(X_train, y_train)))
            val_groups[task_id].extend(list(zip(X_val, y_val)))
            test_groups[task_id].extend(list(zip(X_test, y_test)))

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
        '''
        for counting samples in each class, group data by label
        Args:
            data: type = list,[(data,label),...]
        Returns:
            label_groups: type = dict, {label: [data1, data2, ...]}
        '''
        all_data = sum(data, [])  # 展平数据
        label_groups = {}
        for sample, label in all_data:
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(sample)
        return label_groups
    
    def balance_with_augmentation(self, data_by_label):
        """按照不同类别的样本数目进行样本增强，使得每个类别的样本总数接近原本总样本数最多的类别
        Args:
            data_by_label: 按照标签分组的数据
        Returns:
            augmented_data: 增强后的数据，type为list,[(label,data),...] 
        """
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
        """生成增强样本，对每个样本返回augmentation_factor个增强版本，输出类型为list, [(label, data), ...]
        Args:
            sample: 输入样本
            augmentation_factor: 增强样本数倍数
        Returns:
            augmented: 增强后的样本列表
        """
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