#!/bin/bash

# 设置输出文件名
output_file="fitting_obj_list_300.txt"

# 如果文件已存在，删除它
if [ -f "$output_file" ]; then
    rm "$output_file"
fi

# 获取当前目录的绝对路径
current_dir=$(pwd)

# 遍历当前目录下的所有文件夹，写入文件
for dir in */; do
    # 只处理文件夹
    if [ -d "$dir" ]; then
        # 获取文件夹的绝对路径
        abs_path="$current_dir/$dir"
        echo "$abs_path" >> "$output_file"
    fi
done

echo "文件夹的绝对路径已写入 $output_file"
