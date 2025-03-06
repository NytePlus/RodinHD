#!/bin/bash

# 设置输出文件名
output_file="fitting_obj_list.txt"

# 如果文件已存在，删除它
if [ -f "$output_file" ]; then
    rm "$output_file"
fi

# 获取当前目录的绝对路径
current_dir=$(pwd)

# 获取所有文件夹并提取其中的数字进行排序
dirs=$(for dir in */; do
    if [ -d "$dir" ]; then
        # 提取文件夹名中的数字部分
        nums=$(echo "$dir" | grep -oP '\d+')
        # 将数字部分分割成数组
        num_array=($nums)
        # 获取第一个数字
        num1=${num_array[0]}
        # 获取第二个数字（如果有）
        num2=${num_array[1]:--1}  # 如果没有第二个数字，则设置为-1
        # 输出文件夹名和数字部分
        echo "$num1 $num2 $dir"
    fi
done | sort -n -k1,1 -k2,2 | awk '{print $3}')

# 将排序后的文件夹绝对路径写入文件
for dir in $dirs; do
    abs_path="$current_dir/$dir"
    echo "${abs_path%/}" >> "$output_file"
done

echo "文件夹的绝对路径已写入 $output_file"