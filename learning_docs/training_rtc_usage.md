# TrainingRTC 使用说明

本文用于在本机的 Wuji 相对 EEF rot6d 数据集上训练 TrainingRTC。动作空间为
相对 EEF `xyz + rot6d` 加绝对手部关节，动作块长度为 `H=32`，频率为
`30 Hz`。

根据 `client_model_eef_inference_timing.jsonl` 中 137 次实测结果，端到端延迟对应
`d=7..11`，因此 checkpoint 的最大延迟支持设为 `11`。训练 PMF 在实测分布之外
保留 5% 的 `d=0`，用于维持无延迟条件下的行为：

| d | 0 | 1..6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|
| probability | 0.05 | 0.00 | 0.04 | 0.20 | 0.18 | 0.50 | 0.03 |

## 本机路径

```bash
export REPO_ROOT=/data_all/liyunhao/Isaac-GR00T
export BASE_MODEL_PATH=/data_all/share/models/GR00T-N1.7-3B
export BACKBONE_PATH=/data_all/share/models/Cosmos-Reason2-2B
export DATASET_PATH=/data_all/liyunhao/Isaac-GR00T/robot_data_pipeline/outputs/grasp_anything_eef_rot6d
export MODALITY_CONFIG_PATH=/data_all/liyunhao/Isaac-GR00T/robot_data_pipeline/outputs/grasp_anything_eef_rot6d_pca12/meta/wuji_eef_hand_rot6d_h32_config.py

cd "$REPO_ROOT"
export GR00T_BACKBONE_MODEL_NAME="$BACKBONE_PATH"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

开始训练前检查输入文件：

```bash
test -f "$BASE_MODEL_PATH/config.json"
test -f "$BACKBONE_PATH/config.json"
test -f "$DATASET_PATH/meta/info.json"
test -f "$DATASET_PATH/meta/stats.json"
test -f "$DATASET_PATH/meta/relative_stats.json"
test -f "$MODALITY_CONFIG_PATH"
```

本机 `.venv` 是 Python 3.10 + PyTorch 2.7.1/cu128，以下命令直接使用该环境。

## 入口说明

`examples/finetune.sh` 和下面的直接 `torch.distributed.run` 命令不是两套训练
实现。前者只是读取 `NUM_GPUS`、`MAX_STEPS`、`GLOBAL_BATCH_SIZE` 等环境变量，
最后仍然调用同一个 `gr00t/experiment/launch_finetune.py`。TrainingRTC 参数可以
通过脚本的 `--` 原样传给这个 Python 入口。

## 单卡试运行

先用一张确认空闲的 GPU 跑一个 optimizer step。下面以 GPU 5 为例；启动前应先用
`nvidia-smi` 确认它仍有足够显存。`CUDA_VISIBLE_DEVICES=5` 后，进程内看到的
`cuda:0` 就是物理 GPU 5。

```bash
CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port=29545 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE_MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$MODALITY_CONFIG_PATH" \
  --num-gpus 1 \
  --output-dir outputs/training_rtc_smoke_d11 \
  --experiment-name training_rtc_smoke_d11 \
  --max-steps 1 \
  --global-batch-size 1 \
  --dataloader-num-workers 0 \
  --shard-size 16 \
  --num-shards-per-epoch 1 \
  --training-rtc-enabled \
  --training-rtc-max-delay 11 \
  --training-rtc-delay-pmf.0 0.05 \
  --training-rtc-delay-pmf.1 0.00 \
  --training-rtc-delay-pmf.2 0.00 \
  --training-rtc-delay-pmf.3 0.00 \
  --training-rtc-delay-pmf.4 0.00 \
  --training-rtc-delay-pmf.5 0.00 \
  --training-rtc-delay-pmf.6 0.00 \
  --training-rtc-delay-pmf.7 0.04 \
  --training-rtc-delay-pmf.8 0.20 \
  --training-rtc-delay-pmf.9 0.18 \
  --training-rtc-delay-pmf.10 0.50 \
  --training-rtc-delay-pmf.11 0.03 \
  --action-step-hz 30 \
  --training-rtc-loss-mode postfix_only
```

之前日志中的 OOM 发生在物理 GPU 0，当时该卡只剩约 31 MiB，并不是数据集或
TrainingRTC action contract 错误。切换到真正空闲的卡即可解决该次启动失败。

## GPU 4 到 7 正式训练

下面命令使用物理 GPU `4,5,6,7`，并保持你当前普通微调命令的 batch 和数据加载
参数。数据目录是非 PCA12 的 `grasp_anything_eef_rot6d`；只有 modality 配置文件
位于 PCA12 输出目录。如果出现显存不足，再把 `GLOBAL_BATCH_SIZE` 降到 `128`。

```bash
cd /data_all/liyunhao/Isaac-GR00T
export CUDA_VISIBLE_DEVICES=4,5,6,7
export GR00T_BACKBONE_MODEL_NAME=/data_all/share/models/Cosmos-Reason2-2B
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_PORT=29531 \
NUM_GPUS=4 \
MAX_STEPS=20000 \
GLOBAL_BATCH_SIZE=256 \
SAVE_STEPS=2000 \
DATALOADER_NUM_WORKERS=6 \
EPISODE_SAMPLING_RATE=1.0 \
SHARD_SIZE=1024 \
NUM_SHARDS_PER_EPOCH=1000 \
USE_WANDB=1 \
uv run bash examples/finetune.sh \
  --base-model-path /data_all/share/models/GR00T-N1.7-3B \
  --dataset-path /data_all/liyunhao/Isaac-GR00T/robot_data_pipeline/outputs/grasp_anything_eef_rot6d \
  --modality-config-path /data_all/liyunhao/Isaac-GR00T/robot_data_pipeline/outputs/grasp_anything_eef_rot6d_pca12/meta/wuji_eef_hand_rot6d_h32_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /data_all/liyunhao/Isaac-GR00T/outputs/wuji_astribot_grasp_anything_eef_rot6d_h32_rtc \
  --experiment-name wuji_astribot_grasp_anything_eef_rot6d_h32_rtc \
  --wandb-project finetune-wuji-astribot-rtc \
  --state-dropout-prob 0.05 \
  -- \
  --gradient-accumulation-steps 2 \
  --save-total-limit 10 \
  --training-rtc-enabled \
  --training-rtc-max-delay 11 \
  --training-rtc-delay-pmf.0 0.05 \
  --training-rtc-delay-pmf.1 0.00 \
  --training-rtc-delay-pmf.2 0.00 \
  --training-rtc-delay-pmf.3 0.00 \
  --training-rtc-delay-pmf.4 0.00 \
  --training-rtc-delay-pmf.5 0.00 \
  --training-rtc-delay-pmf.6 0.00 \
  --training-rtc-delay-pmf.7 0.04 \
  --training-rtc-delay-pmf.8 0.20 \
  --training-rtc-delay-pmf.9 0.18 \
  --training-rtc-delay-pmf.10 0.50 \
  --training-rtc-delay-pmf.11 0.03 \
  --action-step-hz 30 \
  --training-rtc-loss-mode postfix_only
```

`joint` 与 `joints` 不是可互换的拼写。它们是否有影响取决于 dataset feature key
和 modality config 是否完全一致。当前数据和配置使用
`left_hand_joint`、`right_hand_joint`，初始化日志已经得到 `semantic_dim=58`，说明
当前契约已正确匹配，不需要改名。

TrainingRTC 的运行时调度与训练配置分开设置。运行时 client 应使用同一个
`H=32`、`30 Hz` 和 checkpoint 支持的 `max_delay=11`。实测同步推理约 270 ms
时可先令执行步数 `s=10`，每次推理后最多执行 10 个动作，再根据线上新测量动态
计算 `d`；不能使用 `s=32`，否则没有剩余 postfix 可供 RTC 修正。
