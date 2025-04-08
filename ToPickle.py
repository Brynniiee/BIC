import os
import pickle
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

class PickleWithSplitConverter:
    def __init__(self, root_dir='GXW_Data/image', output_path='picklewithsplit.pkl', image_size=(875, 656), train_ratio=0.9):
        self.root_dir = root_dir
        self.output_path = output_path
        self.image_size = image_size
        self.train_ratio = train_ratio

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        #preprocessesing: resize and convert to tensor

        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        self.class_to_idx = {}

        self.process()

    def process(self):
        class_folders = sorted([
            subfolders for subfolders in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, subfolders)) 
        ])
        self.class_to_idx = {name: idx for idx, name in enumerate(class_folders)} 
        # util a dict to store projection of class to indices

        for class_name in tqdm(class_folders, desc="Processing classes", leave=True):
            class_path = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            images = []  #storing images(np arrays) in a list ([[r,g,b],[r,g,b],...])
            file_list = [
                file for file in os.listdir(class_path)
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]

            for file in tqdm(file_list, desc=f'  {class_name}', leave=False):
                img_path = os.path.join(class_path, file)
                try:
                    image = Image.open(img_path).convert('RGB') # Using PIL to open the image & convert to RGB format
                    tensor = self.transform(image) # Resize and ToTensor
                    img_array = (tensor.numpy() * 255).astype('uint8') # np array scaling 0-255(formerly 0-1)
                    r = img_array[0].flatten() 
                    g = img_array[1].flatten()
                    b = img_array[2].flatten()
                    # Flatten the 3D array to 1D (original imgarray.shape: 3*Height*Width)
                    flat = np.concatenate([r, g, b])
                    images.append(flat)
                    # print(f'len(images): {len(images[-1])}')
                except Exception as e:
                    print(f"Failed to load {img_path}: {e}")

            random.shuffle(images)  # put images in random order
            split_idx = int(len(images) * self.train_ratio) # spliting train/test set
            if split_idx == len(images) and len(images) > 1: 
                split_idx -= 1

            self.train_data.extend(images[:split_idx])  
            self.train_labels.extend([class_idx] * split_idx)
            self.test_data.extend(images[split_idx:])
            self.test_labels.extend([class_idx] * (len(images) - split_idx))

        self.save()

    def save(self):
        dataset = {
            'train': {
                'data': self.train_data,
                'labels': self.train_labels
            },
            'test': {
                'data': self.test_data,
                'labels': self.test_labels
            },
            'class_to_idx': self.class_to_idx
        }
        with open(self.output_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"\nDataset Saved to {self.output_path}")
        print(f"   Train samples: {len(self.train_data)}, Test samples: {len(self.test_data)}")
        print(f"   Classes: {len(self.class_to_idx)}")

    def save(self):
        dataset = {
            'train': list(zip(self.train_data, self.train_labels)),  # [(path1, label1), ...]
            'test': list(zip(self.test_data, self.test_labels)),     # [(path2, label2), ...]
            'class_to_idx': self.class_to_idx                        # {'class1': 0, ...}
        }
        # with open(self.output_path, 'wb') as f:
            # pickle.dump(dataset, f)
        with open(self.output_path, 'wb') as f:
            pickle.dump({'class_to_idx': self.class_to_idx}, f) 

            for section_name, data_list, label_list in [
                ('train', self.train_data, self.train_labels),
                ('test', self.test_data, self.test_labels)
            ]:
                for data, label in zip(data_list, label_list):
                    # 每次只 dump 一个样本
                    pickle.dump((section_name, data, label), f)

        print(f"\nDataset saved to {self.output_path}")
        print(f"   Train samples: {len(self.train_data)}, Test samples: {len(self.test_data)}")
        print(f"   Classes: {len(self.class_to_idx)}")
    

# execution
PickleWithSplitConverter(root_dir='GXW_Data/image',output_path='GXW_Data/Pickle/picklewithsplit.pkl')
