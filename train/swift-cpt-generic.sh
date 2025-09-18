#!/bin/bash

# 通用训练脚本 - 通过环境变量配置参数

# 检查必要的环境变量
if [ -z "$MODEL_PATH" ]; then
    echo "错误: 必须设置MODEL_PATH环境变量"
    exit 1
fi

if [ -z "$DATASET" ]; then
    echo "错误: 必须设置DATASET环境变量"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "错误: 必须设置OUTPUT_DIR环境变量"
    exit 1
fi

if [ -z "$SWANLAB_EXP_NAME" ]; then
    echo "错误: 必须设置SWANLAB_EXP_NAME环境变量"
    exit 1
fi

# 自动检测GPU数量（如果未指定）
if [ -z "$NPROC_PER_NODE" ]; then
    if command -v nvidia-smi &> /dev/null; then
        NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
        echo "自动检测到 $NPROC_PER_NODE 个GPU"
    else
        NPROC_PER_NODE=1
        echo "未检测到GPU，设置NPROC_PER_NODE=1"
    fi
fi

# 设置默认值
TRAIN_TYPE=${TRAIN_TYPE:-"full"}  # 默认为full，可选lora
NUMBER_TRAIN_EPOCHS=${NUMBER_TRAIN_EPOCHS:-5}
VAL_DATASET=${VAL_DATASET:-"/env/Projects/jalen/github/GraphGen/training/data_v1/race_eval.jsonl"}
MODEL_TYPE=${MODEL_TYPE:-"qwen2_5"}
MAX_LENGTH=${MAX_LENGTH:-1024}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-16}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-$PER_DEVICE_TRAIN_BATCH_SIZE}  # 默认与训练批次大小相同
TORCH_DTYPE=${TORCH_DTYPE:-"bfloat16"}
EVAL_STEPS=${EVAL_STEPS:-50}
SAVE_STEPS=${SAVE_STEPS:-50}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
LOGGING_STEPS=${LOGGING_STEPS:-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-8}
SWANLAB_PROJECT=${SWANLAB_PROJECT:-"data-synthesis-w-kg-v2.1"}
SWANLAB_TOKEN=${SWANLAB_TOKEN:-"ioMli2NR04L7dvCAc51Ew"}
SWANLAB_WORKSPACE=${SWANLAB_WORKSPACE:-"awsome_ai_team_of_yzw"}

# 计算梯度累积步数（如果设置了GLOBAL_BATCH_SIZE）
if [ ! -z "$GLOBAL_BATCH_SIZE" ]; then
    # 计算公式：GRADIENT_ACCUMULATION_STEPS = GLOBAL_BATCH_SIZE / (PER_DEVICE_TRAIN_BATCH_SIZE * NPROC_PER_NODE)
    GRADIENT_ACCUMULATION_STEPS=$(( $GLOBAL_BATCH_SIZE / ($PER_DEVICE_TRAIN_BATCH_SIZE * $NPROC_PER_NODE) ))
    
    # 确保GRADIENT_ACCUMULATION_STEPS至少为1
    if [ $GRADIENT_ACCUMULATION_STEPS -lt 1 ]; then
        GRADIENT_ACCUMULATION_STEPS=1
        echo "警告: 计算的GRADIENT_ACCUMULATION_STEPS小于1，已设置为1"
        echo "实际全局批次大小将为: $(( $PER_DEVICE_TRAIN_BATCH_SIZE * $NPROC_PER_NODE ))"
    else
        echo "根据GLOBAL_BATCH_SIZE计算的GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"
        echo "实际全局批次大小: $(( $PER_DEVICE_TRAIN_BATCH_SIZE * $NPROC_PER_NODE * $GRADIENT_ACCUMULATION_STEPS ))"
    fi
else
    # 如果未设置GLOBAL_BATCH_SIZE，使用默认值或用户指定的值
    GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
    echo "使用指定的GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"
    echo "实际全局批次大小: $(( $PER_DEVICE_TRAIN_BATCH_SIZE * $NPROC_PER_NODE * $GRADIENT_ACCUMULATION_STEPS ))"
fi

# 根据训练类型设置特定参数
if [ "$TRAIN_TYPE" = "lora" ]; then
    # LoRA特定参数
    LEARNING_RATE=${LEARNING_RATE:-1e-4}
    LORA_RANK=${LORA_RANK:-16}
    LORA_ALPHA=${LORA_ALPHA:-64}
    TARGET_MODULES=${TARGET_MODULES:-"all-linear"}
    
    LORA_ARGS="--lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA --target_modules $TARGET_MODULES"
else
    # Full微调特定参数
    LEARNING_RATE=${LEARNING_RATE:-1e-5}
    LORA_ARGS=""
fi

# 创建日志目录
mkdir -p logs

echo "开始训练: $SWANLAB_EXP_NAME"
echo "训练类型: $TRAIN_TYPE"
echo "模型路径: $MODEL_PATH"
echo "数据集: $DATASET"
echo "输出目录: $OUTPUT_DIR"

# compute steps
if [  -z $MAX_STEPS ]; then
SAMPLE_SIZE=$(wc -l $DATASET | awk '{print $1}')
MAX_STEPS=$(( $SAMPLE_SIZE * $NUMBER_TRAIN_EPOCHS / $GLOBAL_BATCH_SIZE ))
echo "samples: $SAMPLE_SIZE"
fi
echo "steps: $MAX_STEPS"

# 分布式训练命令
NPROC_PER_NODE=$NPROC_PER_NODE swift pt \
    --streaming true \
    --save_only_model true \
    --model "$MODEL_PATH" \
    --dataset $DATASET \
    --train_type "$TRAIN_TYPE" \
    --torch_dtype "$TORCH_DTYPE" \
    --model_type "$MODEL_TYPE" \
    --max_steps "$MAX_STEPS" \
    --max_length "$MAX_LENGTH" \
    $LORA_ARGS \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --logging_steps "$LOGGING_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio "$WARMUP_RATIO" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --dataset_num_proc "$DATASET_NUM_PROC" \
    --report_to swanlab \
    --swanlab_project "$SWANLAB_PROJECT" \
    --swanlab_exp_name "$SWANLAB_EXP_NAME" \
    --swanlab_token "$SWANLAB_TOKEN" \
    --swanlab_workspace "$SWANLAB_WORKSPACE" \
    --deepspeed zero2 \
    --attn_impl flash_attn 2>&1 | tee logs/$SWANLAB_EXP_NAME.log