import os
import scipy.io
import numpy as np
from tqdm import tqdm

def GXW_data_shift_test(root_dir = 'GXW_Data/Dasdata_adding'):
    """
    Load shift test dataset from a directory structure: class folders with .mat files

    Args:
        root_dir: str, path to the shift test dataset root folder

    Returns:
        shift_test_group: list of (data, label)
        class_to_idx: dict, {class_name: class_idx}
    """

    shift_data = []
    shift_labels = []
    class_names = sorted(os.listdir(root_dir))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    for cls_name in tqdm(class_names, desc="Loading shift test classes", leave=True):
        cls_dir = os.path.join(root_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        files = sorted(os.listdir(cls_dir))
        mat_files = [os.path.join(cls_dir, f) for f in files if f.endswith('.mat')]

        for fpath in tqdm(mat_files, desc=f'  {cls_name}', leave=False):
            data = load_mat_file(fpath)
            if data is not None:
                shift_data.append(data)
                # shift_labels.append(class_to_idx[cls_name])
                shift_labels.append(1)  # all labels are 1 for THIS shift test
                

    # Pack into (data, label) pairs
    shift_test_group = list(zip(shift_data, shift_labels))
    print(f"Shift test samples loaded: {len(shift_test_group)} samples across {len(class_names)} classes")

    return shift_test_group, class_to_idx


def load_mat_file(file_path):
    try:
        mat_contents = scipy.io.loadmat(file_path)
        valid_keys = [k for k in mat_contents.keys() if not k.startswith("__")]
        if not valid_keys:
            print(f"[WARN] No valid data key found in {file_path}")
            return None

        preferred_keys = ['subset_data', 'data']
        for key in preferred_keys:
            if key in mat_contents:
                data = mat_contents[key]
                break
        else:
            data = mat_contents[valid_keys[0]]  # fallback

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
