import os
import numpy as np
import scipy.io
from tqdm import tqdm
import torch

class ShiftDataLoader:
    def __init__(self, root_dir = 'GXW_Data/Dasdata_adding'):
        self.root_dir = root_dir
        self.all_data = []
        self.all_labels = []  # 全部 label 设为 0

        self.load_data()

    def load_data(self):
        class_names = sorted(os.listdir(self.root_dir))
        print(f"[Shift] Found {len(class_names)} classes: {class_names}")

        for cls_name in tqdm(class_names, desc="Loading shift classes"):
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            files = sorted(os.listdir(cls_dir))
            all_files = [os.path.join(cls_dir, f) for f in files if f.endswith('.mat')]

            for fpath in tqdm(all_files, desc=f'  {cls_name}', leave=False):
                data = self.load_mat_file(fpath)
                if data is not None:
                    self.all_data.append(data)
                    self.all_labels.append(5)  # 全部 label 设为 1

        print(f"[Shift] Loaded {len(self.all_data)} samples from {len(class_names)} classes")

    def load_mat_file(self, file_path):
        try:
            mat_contents = scipy.io.loadmat(file_path)
            valid_keys = [k for k in mat_contents.keys() if not k.startswith("__")]
            if not valid_keys:
                return None

            preferred_keys = ['subset_data', 'data']
            for key in preferred_keys:
                if key in mat_contents:
                    data = mat_contents[key]
                    break
            else:
                data = mat_contents[valid_keys[0]]

            if isinstance(data, np.ndarray):
                if data.shape == (15, 10240):
                    return data.astype(np.float32)
                elif data.shape[1] >= 10240 and data.shape[0] >= 15:
                    data = data[:15, :10240]
                    return data.astype(np.float32)
                else:
                    return None
            else:
                return None
        except Exception as e:
            print(f"[ERROR] Failed to load {file_path}: {str(e)}")
            return None

    def get_shift_data(self):
        return list(zip(self.all_data, self.all_labels))
