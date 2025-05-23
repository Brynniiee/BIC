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
            1. transfer data from labels & data to list; [(data, label), ...]
            3. data augmentation to balance sample number for each class
            4. split data into train, val and test groups
            5. group data by tasks
            6. for counting samples in each class, group data by label
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
                if augmented_sample[1] in class_range:  
                    groups[j].append(augmented_sample)
                    break  
        all_data_by_label = self.group_data_by_label(groups)
        print(f"Samples after augmentation:", {k: len(v) for k, v in all_data_by_label.items()})
        train_groups, val_groups, test_groups = self.train_val_test_split(data = groups, val_train_split=self.val_train_split, test_train_split = self.test_train_split, batch_num=self.batch_num)
        # counting samples in each class, group data by label
        train_data_by_label = self.group_data_by_label(train_groups)
        val_data_by_label = self.group_data_by_label(val_groups)
        test_data_by_label = self.group_data_by_label(test_groups)
        print("Train Samples after augmentation:", {k: len(v) for k, v in train_data_by_label.items()})
        print("Val Samples after augmentation:", {k: len(v) for k, v in val_data_by_label.items()})
        print("Test Samples after augmentation:", {k: len(v) for k, v in test_data_by_label.items()})
        print("class_to_idx mapping:", self.class_to_idx)
        return train_groups, val_groups, test_groups


    def train_val_test_split(self, data = [], val_train_split = 0.2, test_train_split=0.2, batch_num = 1):
        """ split data first into train+val & test sets and then split train+val set into train & val sets
        Args:
            data: type = list,[(data,label),...]
            val_train_split: val proportion in train+val set
            test_train_split: test proportion in total set 
            train_groups: empty training set awaiting to be filled
            val_groups 
            test_groups
        Returns:
            train_groups: type = list,[(data,label),...]
            val_groups
            test_groups
        """
        train_groups = [[] for _ in range(batch_num)]
        val_groups = [[] for _ in range(batch_num)]
        test_groups = [[] for _ in range(batch_num)]
        for task_id, data_group in enumerate(data):  # data_group is a list of (data, label)
            # split data and label
            print(len(data_group))
            X = [x for x, y in data_group]
            y = [y for x, y in data_group]
            # train_val & test split
            X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_train_split, stratify=y, random_state=42)
            # train & val split
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
        all_data = sum(data, [])  ## 展平数据
        label_groups = {}
        for sample, label in all_data:
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(sample)
        return label_groups
    
    def balance_with_augmentation(self, data_by_label, assumed_max_count=None):
        """
        Augmentation to balance the number of samples in each class,
        so that the number of samples in each class is greater than 1/2 of the assumed max class size.

        Args:
            data_by_label: dict, {label: [data1, data2, ...]}
            assumed_max_count: int, optional. If provided, use as the max class count instead of real data.

        Returns:
            augmented_data: list of (data, label)
        """
        class_counts = {k: len(v) for k, v in data_by_label.items()}
        if assumed_max_count is not None:
            max_count = assumed_max_count
        else:
            max_count = max(class_counts.values())

        augmented_data = []

        for label, samples in tqdm(data_by_label.items(), desc="Balancing classes", leave=True):
            current_count = len(samples)
            if current_count <= max_count // 2:
                augmentation_factor = max_count // current_count - 1
                if augmentation_factor > 0:
                    for s in tqdm(samples, desc=f"Augmenting {label}", leave=False):
                        augmented_samples = self.sample_augmentation(s, augmentation_factor)
                        augmented_data.extend([(a, label) for a in augmented_samples])
        return augmented_data


    def sample_augmentation(self, sample_data, augmentation_factor):
        """generate augmented samples for data of each sample, 
        return #augmentation_factor augmented samples for each original sample
        Args:
            sample: data of a sample, type = np.ndarray, (15, 10240)
            augmentation_factor: mutiplicative factor
        Returns:
            augmented: type = list, [(15, 10240), ...]
        """
        augmented = []
        for _ in range(augmentation_factor):  # generate augmentation_factor samples 
            s = sample_data.copy()
            # introduce gaussian noise
            noise = np.random.normal(0, 0.01 * np.std(s), s.shape)
            s_noisy = s + noise
            # randomly scale
            scale = np.random.uniform(0.9, 1.1)
            s_scaled = s_noisy * scale
            # randomly shift
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

    def extract_small_balanced_set(self, split='train', per_class=50):
        """
        Extract a small balanced subset from the full set,
        where each class is represented by `per_class` samples,
        and further balance the class sample numbers by augmentation.

        Args:
            split (str): One of 'train', 'val', or 'test'.
            per_class (int): Number of samples per class to initially extract.

        Returns:
            balanced_augmented_subset: list of (data, label)
        """
        if split == 'train':
            groups = self.train_groups
            allowed_classes = getattr(self, 'seen_classes', None)  # Only use seen classes in train
        elif split == 'test':
            groups = self.test_groups # Openset Test Data
            allowed_classes = None
        else:
            raise ValueError(f"Invalid split: {split}. Choose from 'train' or 'test'.")

        # flatten all tasks into one list
        all_data = sum(groups, [])

        # group by label
        label_to_samples = {}
        for data, label in all_data:
            if allowed_classes is not None and label not in allowed_classes:
                continue  # skip unseen classes in 'train'
            if label not in label_to_samples:
                label_to_samples[label] = []
            label_to_samples[label].append(data)

        # sample per_class per label
        balanced_subset = []
        new_label_to_samples = {}
        for label, samples in label_to_samples.items():
            if len(samples) < per_class:
                print(f"Warning: Label {label} only has {len(samples)} samples, using all of them.")
                chosen = samples
            else:
                chosen = random.sample(samples, per_class)
            balanced_subset.extend([(x, label) for x in chosen])
            new_label_to_samples[label] = chosen  # keep grouped by label for augmentation

        # augmentation
        augmented_data = self.balance_with_augmentation(new_label_to_samples, assumed_max_count=2 * per_class)

        final_subset = balanced_subset + augmented_data

        return final_subset





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