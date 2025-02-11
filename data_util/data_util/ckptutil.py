import numpy as np

# 文件路径
file_path = 'data/save_triplane_and_mlp4/1.npy'

# 加载.npy文件
try:
    data = np.load(file_path)
    # 获取并打印shape
    print(f"The shape of the data in '{file_path}' is: {data.shape}")
except FileNotFoundError:
    print(f"File '{file_path}' not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")