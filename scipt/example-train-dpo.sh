#!/bin/bash

# 示例脚本：使用通用训练脚本进行Full微调训练

# 设置必要的环境变量
export GLOBAL_BATCH_SIZE=128
export MODEL_PATH="/CommonModels/Qwen/Qwen2.5-1.5B-Base"
export DATASET="./data/train/dpo_chatml_depth_1.jsonl"
export SPLIT_DATASET_RATIO=0
export SWANLAB_EXP_NAME="hotpot-graphgen-dpo(subgraph_depth_1)-qwen2.5-1.5b-base"
export OUTPUT_DIR="./training/ckpts/hotpot/$SWANLAB_EXP_NAME"

# 设置训练类型为full（也可以不设置，因为默认就是full）
export TRAIN_TYPE="full"

# 可选：自定义其他参数
export NUMBER_TRAIN_EPOCHS=1
export LEARNING_RATE=1e-5  # Full微调默认学习率
export MAX_LENGTH=256

conda activate swift

# 执行通用训练脚本
bash ./train/swift-dpo-generic.sh
