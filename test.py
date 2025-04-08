import scipy.io

mat_path = 'GXW_Data/DASdata/冲击钻/20240731143419.mat'  
mat = scipy.io.loadmat(mat_path)
print("Keys in .mat file:", mat.keys())
for k in mat:
    if not k.startswith("__"):
        print(f"{k}: {type(mat[k])}, shape={mat[k].shape if hasattr(mat[k], 'shape') else 'N/A'}, dtype={mat[k].dtype if hasattr(mat[k], 'dtype') else 'N/A'}")
