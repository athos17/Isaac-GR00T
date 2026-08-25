# TrainingRTC 使用说明

本文用于在 Wuji 数据集上训练 TrainingRTC。动作空间为相对 EEF
`xyz + rot6d`，手部关节使用绝对值。

TrainingRTC 使用 `H=32` 的动作块、`30 Hz` 的动作频率，并采样 `d=0..6`
的延迟长度。

## 路径配置

如果使用其他模型或数据集，请修改下面的路径：

```bash
export BASE_MODEL_PATH=/root/models/GR00T-N1.7-3B
export BACKBONE_PATH=/root/models/Cosmos-Reason2-2B
export DATASET_PATH=/root/Isaac-GR00T/examples/wuji_rot6d/teleop_chuneng_spray_water_rot6d_merged_filtered
```

如果完全使用本地模型运行：

```bash
export GR00T_BACKBONE_MODEL_NAME="$BACKBONE_PATH"
export HF_HUB_OFFLINE=1
```

请在仓库根目录执行以下命令。命令直接调用项目 Python，避免 uv 自动同步环境。

## 单步试运行

正式训练前，先使用一张 GPU 执行一个训练 step：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port=29540 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE_MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/wuji_rot6d/wuji_eef_hand_rot6d_h32_config.py \
  --num-gpus 1 \
  --output-dir outputs/training_rtc_smoke \
  --experiment-name training_rtc_smoke \
  --max-steps 1 \
  --global-batch-size 1 \
  --dataloader-num-workers 0 \
  --training-rtc-enabled \
  --training-rtc-max-delay 6 \
  --action-step-hz 30 \
  --training-rtc-loss-mode postfix_only
```

## 单卡训练

根据 GPU 显存调整 `--global-batch-size`、`--max-steps` 和
`--dataloader-num-workers`：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port=29541 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE_MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/wuji_rot6d/wuji_eef_hand_rot6d_h32_config.py \
  --num-gpus 1 \
  --output-dir outputs/training_rtc_wuji_rot6d \
  --experiment-name training_rtc_wuji_rot6d \
  --max-steps 20000 \
  --global-batch-size 1 \
  --dataloader-num-workers 2 \
  --training-rtc-enabled \
  --training-rtc-max-delay 6 \
  --action-step-hz 30 \
  --training-rtc-loss-mode postfix_only
```

## 多卡训练

将 `--nproc_per_node` 和 `--num-gpus` 设置为相同的 GPU 数量：

```bash
.venv/bin/python -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port=29540 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE_MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/wuji_rot6d/wuji_eef_hand_rot6d_h32_config.py \
  --num-gpus 8 \
  --output-dir outputs/training_rtc_wuji_rot6d_8gpu \
  --experiment-name training_rtc_wuji_rot6d_8gpu \
  --max-steps 20000 \
  --global-batch-size 512 \
  --dataloader-num-workers 6 \
  --training-rtc-enabled \
  --training-rtc-max-delay 6 \
  --action-step-hz 30 \
  --training-rtc-loss-mode postfix_only
```

如果使用绝对手臂关节和绝对手部关节动作空间，将
`--modality-config-path` 改为
`data_preprocess/wuji_joint_hand_absolute_h32_config.py`，并提供按照该动作契约
生成的数据集。

TrainingRTC 的运行时调度与训练配置分开设置。运行时延迟参数 `s` 不是训练参数。
