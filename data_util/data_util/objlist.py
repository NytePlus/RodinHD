import os

def get_relative_paths(directory):
    """
    获取指定目录下所有文件的相对路径，并去除文件类型后缀
    """
    relative_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 获取文件的相对路径
            relative_path = os.path.relpath(os.path.join(root, file), directory)
            # 去除文件后缀
            file_name_without_extension = os.path.splitext(relative_path)[0]
            relative_paths.append(file_name_without_extension)
    return relative_paths

def save_to_txt(file_list, output_file):
    """
    将文件列表保存到一个txt文件中
    """
    with open(output_file, 'w') as f:
        for path in file_list:
            f.write('/home/wcc/RodinHD/data/triplane_128_4/'+ path + '\n')

# 目标目录
directory = 'data/triplane_128_4'
# 输出文件路径
output_file = 'data/triplane_128_4/fitting_obj_list.txt'

# 获取相对路径列表
relative_paths = get_relative_paths(directory)
# 保存到txt文件
save_to_txt(relative_paths, output_file)

print(f"Relative paths have been saved to {output_file}")