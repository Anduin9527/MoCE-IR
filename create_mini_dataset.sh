#!/bin/bash

# 脚本功能：从源数据集目录中为每种退化类型（Blur, Haze, Lowlight, Rain, Snow）
# 复制 10 对（GT 和 LQ）图片到目标微型数据集目录，用于快速测试。

# 源数据集路径 (根据你的实际路径修改)
# 注意：这是你提供的 prompt 中的路径，如果在本地运行请修改为本地路径
SRC_DIR="/data/users/gaoyin/datasets/AIO"

# 目标微型数据集路径 (脚本运行当前目录下的 mini_dataset 文件夹)
DEST_DIR="/data/users/gaoyin/datasets/AIO/mini_dataset_AIO"

# 退化类型列表
TYPES=("Blur" "Haze" "Lowlight" "Rain" "Snow")

# 需要复制的数量
COUNT=10

echo "开始创建微型数据集..."
echo "源路径: $SRC_DIR"
echo "目标路径: $DEST_DIR"

# 创建目标根目录
mkdir -p "$DEST_DIR"

for type in "${TYPES[@]}"; do
    echo "正在处理: $type"
    
    # 定义源 GT 和 LQ 路径
    src_gt_dir="$SRC_DIR/$type/GT"
    src_lq_dir="$SRC_DIR/$type/LQ"
    
    # 定义目标 GT 和 LQ 路径
    dest_gt_dir="$DEST_DIR/$type/GT"
    dest_lq_dir="$DEST_DIR/$type/LQ"
    
    # 检查源目录是否存在
    if [ ! -d "$src_gt_dir" ]; then
        echo "  [警告] 源目录不存在，跳过: $src_gt_dir"
        continue
    fi

    # 创建目标子目录
    mkdir -p "$dest_gt_dir"
    mkdir -p "$dest_lq_dir"
    
    # 计数器
    i=0
    
    # 遍历 GT 目录中的文件 (确保排序)
    # 使用 sort 确保顺序一致，虽然 globe 通常按字母序但 sort 更保险
    for gt_path in $(ls "$src_gt_dir" | sort | head -n "$COUNT"); do
        gt_filename=$(basename "$gt_path")
        
        # 复制 GT 文件
        full_gt_path="$src_gt_dir/$gt_filename"
        if [ -f "$full_gt_path" ]; then
            cp "$full_gt_path" "$dest_gt_dir/"
            
            # 尝试复制对应的 LQ 文件 (假设文件名相同)
            full_lq_path="$src_lq_dir/$gt_filename"
            if [ -f "$full_lq_path" ]; then
                cp "$full_lq_path" "$dest_lq_dir/"
            else
                echo "  [警告] 未找到对应的 LQ 文件: $gt_filename"
            fi
            
            ((i++))
        fi
    done
    
    echo "  已复制 $i 对图片到 $type"
done

echo "完成！微型数据集创建在: $DEST_DIR"
echo "你可以通过修改 options settting 中的 data_file_dir 来使用这个数据集。"
