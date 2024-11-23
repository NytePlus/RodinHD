#!/bin/bash

# 源目录
SRC_DIR="../data/portrait3d_nobg"
# 目标目录
DST_DIR="../data/portrait3d_data"

# 遍历源目录中的所有图片文件
find "$SRC_DIR" -type f \( -iname "*.jpg" -o -iname "*.png" \) | while read file; do
    # 替换源路径中的 SRC_DIR 为 DST_DIR
    target="${file/$SRC_DIR/$DST_DIR}"

    # 确保目标目录存在
    mkdir -p "$(dirname "$target")"

    # 替换文件
    cp "$file" "$target"
    echo "Replaced $file -> $target"
done
