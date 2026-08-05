# Wuji/Astribot Rosbag 转 GR00T 数据集

本目录用于把 Wuji/Astribot 的 ROS2 bag 转成当前仓库训练使用的 GR00T LeRobot v2 格式。当前脚本面向“手臂 EEF + 灵巧手关节 + 三路 RGB 视觉”的数据。

转换后的字段包括：

- `observation.state`：左右臂当前 EEF + 左右手当前关节角
- `action`：左右臂 desired EEF + 左右手 command 关节角
- `video`：头部 RGB + 左腕 RGB + 右腕 RGB 三路 MP4

时间对齐策略：

- 使用头部 RGB 的时间戳作为 anchor clock。
- 先计算 11 个目标 topic 的时间范围交集，只保留交集内的头部 RGB anchor 帧。
- 最终 episode 帧数为落在公共时间窗内的 `head_rgb` 帧数。
- 对每个 head timestamp，左右腕相机、state topic、action topic 都取最近邻样本。
- `--max-time-skew` 只是诊断阈值，会写入 metadata/log；不会重采样到 30Hz，也不会改变最终帧数。

## Conda 环境搭建

推荐使用纯 Python 后端 `rosbags`。这个流程不需要服务器安装 ROS，不需要
`source setup.bash`，也不需要 colcon 编译自定义消息包。脚本已经内置了默认
topic 所需的 `astribot_msgs/msg/RobotCartesianState` 消息定义。

### 1. 创建环境

```bash
conda create -n gr00t_data python=3.10 -y
conda activate gr00t_data
```

### 2. 安装纯 Python 依赖

```bash
pip install rosbags numpy pandas pyarrow opencv-python pyyaml
```

### 3. 环境检查

每次运行前只需要激活 conda 环境：

```bash
conda activate gr00t_data
cd /data_all/lyh/Isaac-GR00T
python - <<'PY'
import rosbags
print("pure Python rosbags env ok")
PY
```

如果你手动把 topic 改成其它自定义消息类型，且脚本没有内置对应 `.msg` 定义，
才需要补充消息定义或改用 ROS2 后端。

### 可选：ROS2 后端

脚本仍保留 `--bag-backend ros2`，但它需要 `rosbag2_py`、`rclpy` 和自定义消息包，
也就是需要 ROS2 Python 环境或 Robostack/colcon。没有 ROS 的服务器建议使用默认
`--bag-backend rosbags`。

## 转换命令

rotvec EEF 格式：

```bash
cd /data_all/lyh/Isaac-GR00T
python data_preprocess/wuji_rosbag_to_gr00t.py \
  --input-root /data_all/share/datasets/teleop/data_example \
  --output-dir /data_all/lyh/Isaac-GR00T/data_preprocess/output/wuji_rotvec \
  --eef-rotation-format rotvec \
  --task-description "" \
  --bag-backend rosbags \
  --work-dir /tmp/wuji_bag_cache \
  --num-workers 4 \
  --overwrite
```

rot6d EEF 格式：

```bash
cd /data_all/lyh/Isaac-GR00T
python data_preprocess/wuji_rosbag_to_gr00t.py \
  --input-root /data_all/share/datasets/teleop/data_example \
  --output-dir /data_all/lyh/Isaac-GR00T/data_preprocess/output/wuji_rot6d \
  --eef-rotation-format rot6d \
  --task-description "" \
  --bag-backend rosbags \
  --work-dir /tmp/wuji_bag_cache \
  --num-workers 4 \
  --overwrite
```

`--num-workers` 控制 episode 级并行进程数。默认是 `1`，保持原来的顺序处理；
设置为大于 `1` 时，不同 rosbag 会并行读取、对齐和写视频。建议先从 `4` 开始，
如果机器内存、磁盘 IO 和 ffmpeg 编码压力允许，再继续增大。

默认任务文本为空。需要语言指令时，由外部传入：

```bash
--task-description "pick up the object and place it ..."
```

默认使用消息 `header.stamp` 做时间对齐；如果某些 topic 的 header 时间戳不可靠，
可以改用 rosbag 记录时间：

```bash
--timestamp-source rosbag
```

例如手部 `JointState.header.stamp` 与相机时间存在固定偏移时，使用 rosbag 时间戳
可以避免最近邻对齐把手部动作错位到 episode 前段。

默认 MP4/metadata FPS 会从头部相机时间戳估计。需要固定视频容器 FPS 时可以传：

```bash
--output-fps 30
```

这只影响 MP4 容器和 metadata 的 FPS，不改变 head timestamp anchor 对齐策略。

### 可选：指令低通滤波

默认不做滤波。需要在数据处理阶段按一阶低通/EMA 平滑指令时，可以传：

```bash
--enable-low-pass-filter --filter-scale 0.3
```

滤波公式为：

```text
filtered_t = (1 - filter_scale) * filtered_{t-1} + filter_scale * value_t
```

`filter_scale` 取值范围是 `[0, 1]`：越大越跟随最新指令，越小越平滑但滞后更大。
开启后默认只滤四个 action stream：

- `left_eef_action`：`/astribot_arm_left/endpoint_desired_states`
- `right_eef_action`：`/astribot_arm_right/endpoint_desired_states`
- `left_hand_action`：`/left_hand/joint_commands`
- `right_hand_action`：`/right_hand/joint_commands`

如果确实需要改滤波范围，可以重复传 `--low-pass-filter-stream` 覆盖默认值，例如只滤手部指令：

```bash
--enable-low-pass-filter \
  --filter-scale 0.3 \
  --low-pass-filter-stream left_hand_action \
  --low-pass-filter-stream right_hand_action
```

默认保留各相机的原始图像分辨率写入 MP4。需要在转换阶段统一视频尺寸时，显式传入：

```bash
--image-width 384 --image-height 384
```

## 输出格式

输出目录是 GR00T LeRobot v2 风格：

```text
output_dir/
  meta/
    modality.json
    info.json
    tasks.jsonl
    episodes.jsonl
  data/
    chunk-000/
      episode_000000.parquet
  videos/
    chunk-000/
      observation.images.head_view/
        episode_000000.mp4
      observation.images.left_wrist_view/
        episode_000000.mp4
      observation.images.right_wrist_view/
        episode_000000.mp4
```

parquet 中包含：

- `observation.state`：拼接后的 state float32 数组
- `action`：拼接后的 action float32 数组
- `timestamp`：相对 episode 起点的 head camera 时间
- `frame_index`
- `episode_index`
- `index`
- `task_index`
- `annotation.human.action.task_description`

`meta/modality.json` 描述 state/action 的切片，以及三路视频 key。

## Topic 映射

默认 topic 基于 `/data_all/share/datasets/teleop/data_example`：

| 数据字段 | ROS topic |
| --- | --- |
| `state.left_eef` | `/astribot_arm_left/endpoint_current_states` |
| `state.right_eef` | `/astribot_arm_right/endpoint_current_states` |
| `action.left_eef` | `/astribot_arm_left/endpoint_desired_states` |
| `action.right_eef` | `/astribot_arm_right/endpoint_desired_states` |
| `state.left_hand_joints` | `/left_hand/joint_states` |
| `state.right_hand_joints` | `/right_hand/joint_states` |
| `action.left_hand_joints` | `/left_hand/joint_commands` |
| `action.right_hand_joints` | `/right_hand/joint_commands` |
| `video.head_view` | `/astribot_camera/head_rgbd/color_compress/compressed` |
| `video.left_wrist_view` | `/astribot_camera/left_wrist_rgbd/color_compress/compressed` |
| `video.right_wrist_view` | `/astribot_camera/right_wrist_rgbd/color_compress/compressed` |

可以用 `--topic KEY=/topic` 覆盖任意映射，例如：

```bash
--topic left_eef_action=/remote_control/astribot_arm_left/endpoint_desired_states
```

先做 topic 检查，不反序列化消息：

```bash
python data_preprocess/wuji_rosbag_to_gr00t.py \
  --input-root /data_all/share/datasets/teleop/data_example \
  --dry-run
```

## 训练配置

微调时使用对应的 modality config：

```bash
# rotvec 输出
--embodiment-tag NEW_EMBODIMENT \
--modality-config-path data_preprocess/wuji_eef_hand_rotvec_config.py

# rot6d 输出
--embodiment-tag NEW_EMBODIMENT \
--modality-config-path data_preprocess/wuji_eef_hand_rot6d_config.py
```

rot6d 布局遵循本仓库 `EndEffectorPose` 的约定：`[x, y, z]` 后接旋转矩阵前两行 flatten。
