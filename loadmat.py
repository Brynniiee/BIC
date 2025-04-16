import os
import scipy.io
import numpy as np
from tqdm import tqdm

class MatDataLoader:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.class_to_idx = {}
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []

        self.load_data()

    def load_data(self):
        class_names = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        for cls_name in tqdm(class_names,desc="Loading classes", leave=True):
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            files = sorted(os.listdir(cls_dir))
            all_files = [os.path.join(cls_dir, f) for f in files if f.endswith('.mat')]

            split_idx = int(0.8 * len(all_files))
            #all_files 是不对的，应当分开文件夹然后再取，否则将会导致test集中没有少样本数的类别
            train_files = all_files[:split_idx]
            test_files = all_files[split_idx:]

            for fpath in tqdm(train_files, desc=f'  {cls_name}', leave=False):
                data = self.load_mat_file(fpath)
                if data is not None:
                    self.train_data.append(data)
                    self.train_labels.append(self.class_to_idx[cls_name])

            for fpath in tqdm(test_files, desc=f'  {cls_name}', leave=False):
                data = self.load_mat_file(fpath)
                if data is not None:
                    self.test_data.append(data)
                    self.test_labels.append(self.class_to_idx[cls_name])

    def load_mat_file(self, file_path):
        try:
            mat_contents = scipy.io.loadmat(file_path)
            valid_keys = [k for k in mat_contents.keys() if not k.startswith("__")]
            if not valid_keys:
                print(f"[WARN] No valid data key found in {file_path}")
                return None

            # 尝试在优先顺序中查找变量名
            preferred_keys = ['subset_data', 'data']
            for key in preferred_keys:
                if key in mat_contents:
                    data = mat_contents[key]
                    break
            else:
                data = mat_contents[valid_keys[0]]  # 如果都没有，用第一个非系统字段
            
            valid_data = 0
            # 检查数据是否符合预期尺寸 (15, 10240)
            if isinstance(data, np.ndarray):

                if data.shape == (15, 10240):
                    valid_data +=1
                    return data.astype(np.float32)
                    
                elif data.shape[1] >= 10240 and data.shape[0] >= 15:
                    # print(f"[INFO] Data shape {data.shape} in {file_path} is larger than expected, trimming to (15, 10240).")
                    data = data[:15, :10240]
                    valid_data +=1
                    return data.astype(np.float32)
                
                else:    
                    # print(f"[WARN] Invalid data shape {data.shape} in {file_path}, expected (15, 10240). Skipping this file.")
                    return None
            else:
                # print(f"[WARN] Invalid or object type data in {file_path}")
                return None
        except Exception as e:
            print(f"[ERROR] Failed to load {file_path}: {str(e)}")
            return None
