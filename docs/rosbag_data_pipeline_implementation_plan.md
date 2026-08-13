# ROS2 Bag 数据处理 Pipeline 实施计划

## 1. 文档状态

- 状态：本地实现和旧数据 pilot 完成；Manus + Orin 生产阈值标定待数据
- 目标版本：v1
- 输出格式：LeRobot v2
- 目标输出频率：30 Hz
- EEF 旋转表示：rot6d
- Episode 结果：PASS 或 REJECT
- 实现目录：`robot_data_pipeline/`
- 测试目录：`tests/robot_data_pipeline/`

逐项实现证据见 `robot_data_pipeline/IMPLEMENTATION_AUDIT.md`，旧数据 pilot 和阈值尾部
检查见 `robot_data_pipeline/PILOT_REPORT.md`。最终完成门槛仍是至少 50 条明确确认来源的
Manus + Jetson Orin bag；已搜索全部可读的 `/data_all` 并解析 2,127 个 rosbag metadata，
尚未找到可确认的 Manus + Orin 数据。旧 tactile/teleop bag 不会仅凭 topic 形状或粗略频率
被归类为 Manus。

本项目从零实现，不复用或依赖现有 `wuji_pipeline`，也不依赖
`data_preprocess` 中的 Python 模块。现有代码仅用于确认旧数据的 topic、消息字段、
LeRobot v2 输出布局和 rot6d 数值约定。

## 2. 目标

实现一个配置驱动的 ROS2 bag 数据处理 pipeline，能够：

1. 处理不同机器人本体和不同 topic 映射。
2. 使用 ROS message `header.stamp` 作为统一对齐时钟。
3. 检查原始 topic、时间戳、图像和机器人信号质量。
4. 通过 measured state 自动检测并裁剪有效运动区间。
5. 在高频数据上分析 action-state 响应一致性，但不根据相关性修改时间戳。
6. 在降采样前对允许平滑的 state 信号进行低通和 anti-alias 处理。
7. 以 head camera 为 anchor，将所有模态处理到 30 Hz。
8. 输出绝对 joint action 或绝对 EEF + hand joint action。
9. 直接生成包含多个任务的 LeRobot v2 数据集和完整 QA 报告。
10. 保留原始 rosbag，不对输入数据做任何修改、移动或删除。

## 3. 非目标

v1 明确不负责：

- action/state 归一化。
- absolute 与 relative/delta action 之间的转换。
- action horizon 构造。
- joint 与 EEF action 之间的 FK/IK 转换。
- 根据 action-state cross-correlation 平移 action 或 state。
- spike、损坏图像或缺失帧的自动修复。
- 自动拆分存在中间故障的 episode。
- video-state 视觉运动一致性检查。
- 生成 `relative_stats.json`。
- 复用 `wuji_pipeline` 或 `data_preprocess` 的实现。

`stats.json` 属于训练准备阶段。Pipeline 输出与 LeRobot v2 兼容的数据目录，训练前由
`gr00t/data/stats.py` 生成 `stats.json`；只有模型选择 relative action 时，训练准备阶段
才生成 `relative_stats.json`。

## 4. 已确认的数据契约

### 4.1 时间戳

- 所有目标 publisher 和 rosbag recorder 运行在机器人 Orin。
- `header.stamp` 可认为是对应消息的发布时间。
- Pipeline 默认只使用非零 `header.stamp` 对齐。
- rosbag receive time 仅用于 QA，不参与正常对齐。
- 不允许同一 stream 静默混用 header time 和 rosbag time。
- 缺失、为零或异常的 required stream header timestamp 导致 episode REJECT。
- Pipeline 记录 `bag_time - header_time` 的分布和漂移。

发布时间不能代表 camera 的真实曝光时间，因此 v1 不声称修正 camera pipeline latency。
统一 Orin 主要解决跨 PC 时钟域不一致以及 rosbag receive time 中的 DDS/排队延迟。

### 4.2 Action 定义

训练 action 取实际发送给机器人控制器的 command topic：

- Joint action 使用实际送入控制器的绝对关节位置 command。
- EEF action 使用实际送入控制器的绝对 EEF pose command。
- Hand action 使用实际发送给灵巧手控制器的绝对关节位置 command。
- Manus 原始或映射前 topic 不是 required topic，也不作为训练 action。

Action 在对齐时采用 causal ZOH：对 anchor time `t`，只能选择满足
`command_timestamp <= t` 的最后一条 command。禁止使用未来 command。

### 4.3 输出动作空间

v1 支持两个独立输出：

| 名称 | observation.state | action |
| --- | --- | --- |
| `joint_absolute` | 左右臂 joint state + 左右手 joint state | 左右臂 joint command + 左右手 joint command |
| `eef_absolute_hand_absolute` | 左右 EEF current pose + 左右手 joint state | 左右 EEF desired pose + 左右手 joint command |

两种动作空间写入不同的 LeRobot v2 目录，不能混合在同一个数据集内。一次运行可以请求
一个或两个输出，两者可共享 bag 读取和 QA 结果。

### 4.4 EEF rot6d 约定

内部旋转统一使用单位 quaternion，顺序和消息字段显式记录。处理顺序为：

1. 检查 quaternion 是否 finite。
2. 检查 norm 是否在允许范围内。
3. normalize 到单位 quaternion。
4. 相邻 quaternion 点积小于 0 时，将后一个乘以 -1，保证符号连续。
5. 在两个有效 state 之间按 anchor time 做 SLERP。
6. 转为 rotation matrix。
7. 按现有 Wuji 数据约定取旋转矩阵前两行并 flatten，生成 6 维 rot6d。

符号连续化不改变物理旋转，`q` 和 `-q` 产生完全相同的 rot6d。SLERP 是旋转插值，
不是低通滤波。

## 5. 配置设计

采用两份带 schema version 的 YAML。配置加载必须拒绝未知字段、重复 stream、重复 task、
不存在的输入目录、输入输出目录重叠以及不完整的动作空间定义。

### 5.1 Robot profile

Robot profile 描述本体、topic、消息解析、采样语义和对齐方式。建议路径：

```text
robot_data_pipeline/configs/robots/wuji_astribot_manus.yaml
robot_data_pipeline/configs/robots/wuji_astribot_legacy.yaml
```

旧手套和 Manus 即使 topic 名称一致，也使用不同 profile，以分别表达 hand command 的
200 Hz 和 120 Hz 预期频率。

配置草案：

```yaml
schema_version: robot_profile/v1
name: wuji_astribot_manus

clock:
  source: header
  semantics: publish_time
  require_nonzero: true

output_spaces:
  joint_absolute:
    state_groups:
      - state.left_arm_joint
      - state.right_arm_joint
      - state.left_hand_joint
      - state.right_hand_joint
    action_groups:
      - action.left_arm_joint
      - action.right_arm_joint
      - action.left_hand_joint
      - action.right_hand_joint

  eef_absolute_hand_absolute:
    state_groups:
      - state.left_eef
      - state.right_eef
      - state.left_hand_joint
      - state.right_hand_joint
    action_groups:
      - action.left_eef
      - action.right_eef
      - action.left_hand_joint
      - action.right_hand_joint

streams:
  video.head:
    topic: /astribot_camera/head_rgbd/color_compress/compressed
    adapter: sensor_msgs.compressed_image
    semantic: rgb_image
    required: true
    expected_hz: 30
    alignment: anchor

  video.left_wrist:
    topic: /astribot_camera/left_wrist_rgbd/color_compress/compressed
    adapter: sensor_msgs.compressed_image
    semantic: rgb_image
    required: true
    expected_hz: 30
    alignment: nearest
    max_skew_sec: 0.02
    hard_max_skew_sec: 0.04
    max_consecutive_skew_violations: 1
    max_skew_violation_ratio: 0.005

  state.left_hand_joint:
    topic: /left_hand/joint_states
    adapter: sensor_msgs.joint_state_position
    semantic: joint_position
    required: true
    expected_hz: 200
    alignment: linear
    smoothing:
      type: butterworth
      cutoff_hz: 10.0
      order: 4
      zero_phase: true

  action.left_hand_joint:
    topic: /left_hand/joint_commands
    adapter: sensor_msgs.joint_state_position
    semantic: absolute_joint_position_command
    required: true
    expected_hz: 120
    alignment: previous
    smoothing:
      type: none
```

完整 profile 还需要定义：

- 左右 arm joint state 和 command topic。
- 左右 EEF current 和 desired topic。
- 左右 hand joint state 和 command topic。
- 三路 camera topic。
- 每个 joint group 的固定名称、顺序、单位和范围。
- 连续旋转 joint 的 unwrap/wrap 规则。
- EEF quaternion 消息顺序、base frame 和 tool frame。
- 每个 stream 的 expected frequency、gap 和 alignment tolerance。
- 用于 activity detection 的 measured-state groups。

Adapter 名称引用代码中受控的 adapter registry，不允许在 YAML 中执行任意 Python 表达式。

### 5.2 Dataset manifest

Dataset manifest 描述本次处理的目录、语言指令和输出请求。建议路径：

```text
robot_data_pipeline/configs/datasets/teleop_tasks.yaml
```

配置草案：

```yaml
schema_version: dataset_manifest/v1
profile: robot_data_pipeline/configs/robots/wuji_astribot_manus.yaml

processing:
  output_fps: 30
  num_workers: 8
  activity_padding_before_sec: 0.5
  activity_padding_after_sec: 0.5

outputs:
  - action_space: eef_absolute_hand_absolute
    path: /data_all/share/processed/teleop_eef_rot6d
    eef_rotation_format: rot6d

  - action_space: joint_absolute
    path: /data_all/share/processed/teleop_joint_absolute

datasets:
  - task_id: spray_water
    roots:
      - /data_all/share/datasets/teleop_chuneng/teleop_chuneng/spray_water
    instruction: Pick up the spray bottle, pump it, then spray water on the flowers

  - task_id: wipe_cup
    roots:
      - /data_all/share/datasets/teleop_chuneng/teleop_chuneng/wipe_cup
    instruction: Pick up the cup, wipe it with the rag, then place both down
```

v1 每个输入目录对应一个确定的语言指令。Pipeline 按 manifest 顺序、root 顺序和 bag 路径
排序确定 episode 顺序，确保并行处理时输出索引仍然可复现。

## 6. Pipeline 阶段

### Stage 0：配置和输入发现

输入：robot profile、dataset manifest。

处理：

- 校验两个 YAML schema version。
- 校验 output space 引用的 stream 全部存在。
- 校验 activity detection 所需 stream 存在。
- 发现每个 root 下的 rosbag episode。
- 检查重复 bag、重复 task_id 和路径重叠。
- 固定全局处理顺序。
- 计算配置文件 hash 和输入 bag 元数据 hash。

输出：只读的 processing roster。

拒绝条件：配置非法、输入缺失或输入输出路径重叠时，整个 job 失败，不开始转换。

### Stage 1：Rosbag 读取和原始完整性检查

每条消息保留：

- topic name
- message type
- header timestamp
- rosbag receive timestamp
- 原始消息序号
- 解码后的 payload 或延迟解码引用

检查：

- Required topic 是否存在且非空。
- 消息类型是否符合 profile。
- Header timestamp 是否非零、finite、严格单调。
- 重复 timestamp 和 backward timestamp。
- Header 与 bag time 的 offset、p01/p50/p99 和 episode 内漂移。
- 实际频率、median/p95/max interval、large gap 和估计丢帧。
- Payload 是否包含 NaN/Inf。
- Joint 数量、名称和顺序是否稳定。
- Camera 数据能否解码，分辨率和像素格式是否稳定。
- Camera timestamp 重复、编码 payload 重复和连续冻结帧。

必须在任何排序、去重或滤波之前执行 timestamp 检查，避免隐藏原始问题。

硬拒绝：缺 required topic、无法解码、非 finite、header 时钟异常、joint schema 改变、
camera 损坏或 active 区间存在超过硬阈值的 gap。

### Stage 2：Canonicalization

将不同消息转换为内部 typed stream：

- `JointPositionSeries`
- `PositionCommandSeries`
- `PoseSeries`
- `ImageSeries`

处理：

- 根据 joint name 重排，不能假设数组顺序永远固定。
- 检查并记录物理单位。
- 连续旋转 joint 在插值前 unwrap。
- EEF translation 保存为 xyz。
- EEF rotation 保存为单位 quaternion。
- 不在 joint 和 EEF 之间转换。
- 不计算 relative/delta action。

### Stage 3：活动区间检测

Activity detection 只使用 measured state，不使用 action-state absolute error，避免静止时的
controller tracking offset 将整个 episode 判断为活动。

初始信号：

- 左右 EEF translation velocity。
- 左右 hand measured joint velocity。
- Joint-only 本体可改为左右 arm joint velocity 和 hand joint velocity。

初始参考参数：

- EEF velocity threshold：0.05 m/s。
- Hand joint velocity threshold：0.5 rad/s。
- Sliding window：0.5 s。
- 前后 padding：各 0.5 s。
- 裁剪后最少输出帧数：30。

以上参数均进入配置，pilot 阶段根据新数据调整。无有效运动或裁剪后过短时 REJECT。

### Stage 4：高频 action-state 一致性审计

目的：识别 command/state 映射错误、控制链路异常和不稳定响应，不用于修改时间轴。

处理：

- 在活动区间内建立高频 QA 时间网格。
- Action 按 causal ZOH 投影到网格。
- State 线性插值到网格。
- 只分析具有足够 excitation 的 joint/axis。
- 对速度或带限差分做归一化 cross-correlation。
- 搜索物理允许范围内的正 response lag。
- 分左右臂和左右手计算 group consensus。
- 使用滑动窗口检查 lag median、MAD、趋势和突变。
- 计算 command/state 运动方向一致率。
- 将低活动信号标记为 `insufficient_excitation`，不能因此拒绝 episode。

输出指标：

- best response lag，单位秒。
- peak correlation 和次峰差。
- windowed lag median/MAD/range。
- 有效分析时长和 active joint 数量。
- velocity direction agreement。
- tracking error 分布。

v1 pilot 阶段默认只报告这些指标。完成新 Orin 数据标定后，再决定哪些指标成为硬拒绝条件。
无论结果如何，都不根据该阶段平移 action 或 state。

### Stage 5：State 平滑和 anti-alias

默认策略：

| Stream | 默认处理 |
| --- | --- |
| Arm joint measured state | 可配置 Butterworth 低通 |
| Hand joint measured state | 可配置 Butterworth 低通 |
| EEF translation measured state | 可配置 Butterworth 低通 |
| EEF quaternion | normalize、符号连续化，不做逐分量低通 |
| Arm/hand action command | 不滤波 |
| Camera | 不滤波、不补帧 |

初始 state filter 参数：四阶 Butterworth、10 Hz cutoff、zero-phase。输出为 30 Hz 时
Nyquist frequency 为 15 Hz，因此 cutoff 必须低于 15 Hz。

滤波在高频 state 上执行，必须早于 30 Hz 采样。若原始 timestamp 有轻微 jitter，先在
有效范围内重建局部规则高频网格，再执行滤波。滤波边界必须使用 activity padding，最终
输出时移除 padding，避免边界伪影。

Zero-phase filter 不引入额外相位滞后，但属于离线非因果处理。滤波器类型、cutoff、order、
输入频率估计和实现版本必须写入 pipeline manifest。

### Stage 6：30 Hz 同步

Head camera 是 anchor：

1. 每个有效 head frame 产生一个候选输出 frame。
2. 使用该 head frame 的真实 header timestamp 查询其他 stream。
3. Wrist camera 使用 nearest，并检查 signed skew 和 frame reuse。
4. Joint/EEF translation state 使用有界线性插值。
5. EEF quaternion 使用有界 SLERP。
6. Action 使用 causal ZOH，并记录 action age。
7. 禁止跨越超过阈值的 state gap 插值。
8. 禁止复用未来 action。

LeRobot 的 `timestamp` 写为严格的：

```text
timestamp = frame_index / 30.0
```

所有 MP4 固定编码为 30 fps。真实 head header timestamp、各 stream 的 source timestamp、
signed skew、state bracket gap 和 action age 写入 QA sidecar，不进入训练向量。

如果 head camera 存在丢帧，不复制上一帧进行补帧。超过允许 gap 或 invalid frame ratio 时，
整个 episode REJECT。

### Stage 7：对齐后 QA 和 Episode 决策

检查：

- 三路视频帧数是否与 parquet 行数一致。
- 每路 wrist camera 的 skew mean/p95/max、20 ms 软阈值违规比例和最大连续违规数。
- State interpolation bracket gap mean/p95/max。
- Action age mean/p95/max，以及 future-action violation 数量必须为 0。
- 同一 camera frame 被复用的次数和比例。
- 输出 state/action 是否全部 finite。
- Joint 和 EEF 是否仍在配置范围内。
- 输出 timestamp 是否严格等于 `frame_index / 30`。
- 裁剪后长度是否满足最低要求。

Episode 有三种最终状态：

- PASS：通过所有 hard checks，写入输出数据集。
- PASS_WITH_WARNING：仅存在孤立的 wrist camera 20-40 ms 偏差，且违规比例不超过 0.5%，写入输出数据集并记录 warning。
- REJECT：不写入训练数据，写入 reject report；原始 bag 保持不变。

每个 reject reason 必须是稳定、可搜索的机器标识，例如：

```text
missing_required_topic
zero_header_timestamp
non_monotonic_header_timestamp
camera_decode_failure
joint_schema_mismatch
non_finite_payload
raw_gap_exceeded
state_interpolation_gap_exceeded
action_age_exceeded
wrist_camera_skew_exceeded
no_valid_motion
episode_too_short
```

### Stage 8：LeRobot v2 导出

直接将所有 task 的 PASS 和 PASS_WITH_WARNING episode 写入一个数据集，不先生成 per-task
dataset 再 merge。

输出结构：

```text
output_dir/
  meta/
    info.json
    modality.json
    tasks.jsonl
    episodes.jsonl
    pipeline_manifest.json
  data/
    chunk-000/
      episode_000000.parquet
  videos/
    chunk-000/
      observation.images.head_view/
      observation.images.left_wrist_view/
      observation.images.right_wrist_view/
  quality/
    dataset_summary.json
    episode_reports.jsonl
    rejected_episodes.jsonl
```

Parquet 保持现有训练字段：

- `observation.state`
- `action`
- `timestamp`
- `frame_index`
- `episode_index`
- `index`
- `task_index`
- `annotation.human.action.task_description`

Pipeline 不写 `stats.json` 和 `relative_stats.json`。训练准备阶段必须显式生成 stats 后再启动
GR00T loader。

并行 worker 只产生 episode 临时产物和报告。主进程按预先固定的 roster 顺序收集 PASS
episode，分配连续 episode/global index，然后原子写入最终目录，保证不同 worker 数量下结果一致。

## 7. 建议代码结构

```text
robot_data_pipeline/
  __init__.py
  __main__.py
  cli.py
  config.py
  catalog.py
  models.py
  adapters/
    base.py
    registry.py
    sensor_msgs.py
    astribot_msgs.py
  io/
    rosbag2.py
  quality/
    raw.py
    signal.py
    lag.py
    aligned.py
    decisions.py
  processing/
    canonicalize.py
    activity.py
    filters.py
    rotations.py
    synchronize.py
  export/
    lerobot_v2.py
    video.py
    reports.py
  configs/
    robots/
    datasets/
```

核心数据模型不得引用 `data_preprocess`。Rosbag adapter、处理算法、QA policy 和 exporter
通过明确接口连接，避免把本体 topic 逻辑写进同步器。

## 8. CLI 设计

```bash
# 只校验配置、输入目录和 topic roster
python -m robot_data_pipeline validate \
  --manifest robot_data_pipeline/configs/datasets/teleop_tasks.yaml

# 运行原始 QA，不写 LeRobot 数据
python -m robot_data_pipeline audit \
  --manifest robot_data_pipeline/configs/datasets/teleop_tasks.yaml

# 完整处理和导出
python -m robot_data_pipeline convert \
  --manifest robot_data_pipeline/configs/datasets/teleop_tasks.yaml

# 汇总已有 quality report
python -m robot_data_pipeline summarize \
  --quality-dir /path/to/output/quality
```

所有命令支持 `--dry-run`。`convert` 默认拒绝覆盖非空输出目录；只有显式 `--overwrite`
才允许替换 pipeline 自己创建的目标目录。

## 9. 实施阶段与验收标准

### Phase 1：配置、bag reader 和 raw audit

交付：

- 两类 YAML schema 和校验器。
- Deterministic bag catalog。
- Pure Python ROS2 bag reader 和自定义消息 adapter registry。
- Raw timestamp/frequency/payload/camera QA。
- `validate` 和 `audit` CLI。

验收：

- 可以读取指定旧 bag 和一条新 Manus bag。
- 能正确报告 30/120/200/250 Hz stream。
- 能区分 header timestamp 和 bag receive timestamp。
- 缺失 topic、零 timestamp、重复 timestamp 和损坏图片的测试可稳定触发拒绝原因。
- 不导入 `data_preprocess` 和 `wuji_pipeline`。

### Phase 2：Canonicalization、activity 和 lag audit

交付：

- Named joint reorder 和 schema validation。
- Quaternion normalize/sign continuity。
- State-based activity detection。
- 高频 action-state lag 和 direction audit。

验收：

- `q/-q` 产生相同 rot6d。
- 已知旋转轨迹的 SLERP 数值正确。
- 人工构造的固定 response lag 可在允许误差内估计。
- 静止 joint 标记 insufficient excitation，不产生假 reject。
- Activity crop 在不同输入频率下使用相同的秒级参数。

### Phase 3：Filtering 和 synchronization

交付：

- State Butterworth/anti-alias。
- Joint/translation interpolation。
- Quaternion SLERP。
- Action causal ZOH。
- Camera nearest alignment。
- Per-frame alignment diagnostics。

验收：

- 10 Hz 以下测试信号基本保留，15 Hz 以上信号得到预期衰减。
- Zero-phase filter 不改变测试信号的峰值时刻。
- ZOH 永远不选择 anchor 之后的 command。
- 插值不能跨越超限 gap。
- 输出 timestamp 严格为 `frame_index / 30`。

### Phase 4：LeRobot v2 exporter 和 runner

交付：

- Joint absolute exporter。
- EEF rot6d + hand absolute exporter。
- MP4 writer。
- 多任务直接导出。
- PASS/REJECT report 和 deterministic parallel runner。

验收：

- Parquet、视频和 metadata 的 episode/frame 数完全一致。
- 单 worker 与多 worker 输出索引、任务映射和数值一致。
- 分别生成 joint 和 EEF 两个独立数据集。
- 运行训练准备 stats 命令后，GR00T LeRobot loader 可以加载输出数据集。

### Phase 5：Pilot 和阈值标定

数据：

- 至少 20 条旧手套 bag。
- 至少 50 条新 Manus + Orin bag。
- 包含静止、快速手部动作、快速 EEF 动作和已知异常样本。

处理：

- 所有尚未标定的动态指标先以 report-only 运行。
- 对频率、gap、camera skew、action age、state interpolation gap、lag 和滤波前后频谱做分布统计。
- 人工检查分布尾部样本和 reject 候选。
- 冻结 robot profile v1 阈值。

验收：

- 正常数据不会因频率由 200 Hz 变为 120 Hz 被错误拒绝。
- Action ZOH age 与 120 Hz command 周期一致。
- State filter 不明显削弱最快的有效手部动作。
- 所有 hard reject 都能由报告定位到具体 stream、timestamp 和阈值。

## 10. 测试计划

Unit tests：

- YAML schema、未知字段和路径冲突。
- Timestamp monotonicity、duplicate、gap 和 offset drift。
- Joint name reorder、缺 joint 和重复 joint。
- Quaternion normalize、zero norm、sign continuity、SLERP 和 rot6d。
- Activity detection 的秒级参数与输入采样率无关。
- Butterworth frequency response 和边界处理。
- Linear interpolation、bounded gap 和 causal ZOH。
- Camera nearest tie-breaking、skew 和 frame reuse。
- PASS/REJECT reason 稳定性。
- LeRobot metadata ranges、global index 和 task mapping。

Integration tests：

- 合成 rosbag 覆盖 30/120/200/250 Hz。
- 一条真实旧 bag 的 audit 和 convert smoke test。
- 一条新 Manus bag 的 audit 和 convert smoke test。
- 同一输入重复运行的 determinism test。
- Joint 和 EEF 两种输出的 LeRobot loader smoke test。

Golden tests：

- 固定小 bag 对应固定 QA JSON、Parquet 数值和 metadata。
- Golden 更新必须显式审核，不能由普通测试自动覆盖。

## 11. 性能和可靠性要求

- Episode 级并行，单 episode 内保持确定性。
- 不一次性将全部 bag 的解码图像保存在内存。
- Camera payload 尽可能延迟到选中 anchor frame 后再解码。
- 临时输出写到 job 专属临时目录，成功后原子发布。
- 单 episode 失败不终止其他 episode，但配置或 schema 失败终止整个 job。
- 每次运行记录代码版本、配置 hash、依赖版本、命令行和输入 bag roster。
- 输出目录不得位于任何 input root 内。
- Raw bag 永远只读。

## 12. 仍需讨论和冻结的事项

以下事项不阻塞 Phase 1，但必须在 Phase 3 或 pilot 前确认：

1. State filter 的最终参数。当前建议为四阶 Butterworth、10 Hz、zero-phase；需要确认
   arm joint、hand joint 和 EEF translation 是否统一使用 10 Hz，还是分别配置 8/10/12 Hz。
2. Lag audit 的物理搜索范围和窗口长度。建议先从 0 到 300 ms、2 到 4 s 滑窗开始，
   但必须根据实际控制器响应标定。
3. Lag/方向一致性在 v1 是否只报告，还是 pilot 后启用 hard reject。建议先只报告。
4. Raw QA 和 aligned QA 的 hard thresholds，包括 camera gap、wrist skew、state bracket gap、
   action age 和允许的 invalid frame ratio。
5. Activity detection 的最终阈值和前后 padding。当前参考值为 EEF 0.05 m/s、hand
   0.5 rad/s、0.5 s window、前后各 0.5 s padding。
6. Quaternion norm 的 hard reject 范围和 EEF 最大角速度阈值。
7. Joint 软限位、速度和加速度阈值是由 robot profile 静态提供，还是从 URDF/控制器配置导入。
8. Dataset manifest 是否需要支持同一 task 的多个语言 paraphrase。v1 默认每个目录一个指令。
9. 一次 `convert` 同时生成两种 action space，还是生产环境要求每次只生成一种。建议支持
   同时请求，但始终写入两个独立目录。
10. Pipeline 的独立依赖安装方式。建议增加项目 optional extra，例如
    `uv sync --extra data-pipeline`，显式包含 `rosbags`、`PyYAML` 和 `pyarrow`。

建议先确认第 1、3、4、5 项，再进入 filtering、episode policy 和 production profile 的实现。
