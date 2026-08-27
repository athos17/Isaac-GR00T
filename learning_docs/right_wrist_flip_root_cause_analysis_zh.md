# 右腕跨 chunk 翻腕问题：现象、训练来源与根因分析

## 1. 文档目的

本文总结右腕在实际推理后期出现翻腕、掌心逐渐朝上的问题，说明已经确认的事实、训练数据可能引入的来源、推理阶段的放大机制，以及针对 relative EEF pose 训练的修复路线。

本文只讨论问题本身和最终修复方向，不把临时对比实验当作部署方案。

## 2. 问题摘要

右腕翻腕不是一次随机的 90 度姿态跳变，而是一个小的、方向稳定的旋转偏置在跨 action chunk 执行过程中被反复组合，逐渐累积成大角度翻转。

当前最可信的因果链是：

```text
训练标签使用 measured current -> desired command
        |
        v
同一时刻的 current-to-desired 跟踪误差被写入 relative label
        |
        v
右腕存在稳定的同方向旋转偏置，模型将其学成 relative action
        |
        v
动作接近结束后，正常运动项变小，固定偏置成为主要输出
        |
        v
每个新 chunk 都把该偏置重新组合到新的 reference 上
        |
        v
右腕姿态持续漂移，最终出现掌心朝上的翻腕
```

问题的核心不是单独的 Rot6D 解码错误，也不是单独的控制器失稳，而是：

> 训练标签中的稳定 current-to-desired 偏置，与 relative action 的跨 chunk 组合方式共同形成了长期姿态漂移。

## 3. 已确认的训练和数据语义

### 3.1 GR00T 原生 relative 定义

GR00T 的 relative EEF action 使用：

```text
T_relative(i, k) = inverse(T_state(i)) @ T_action(i + k)
```

其中：

```text
T_state  = measured/current EEF pose
T_action = desired EEF command pose
```

关键实现位于：

- [`gr00t/data/stats.py:190-205`](../gr00t/data/stats.py:190)
- [`gr00t/data/state_action/state_action_processor.py:621-646`](../gr00t/data/state_action/state_action_processor.py:621)
- [`gr00t/data/state_action/pose.py:657-675`](../gr00t/data/state_action/pose.py:657)

因此当前训练语义是 `measured/current -> desired/future`，不是 `desired(i) -> desired(i+k)`。

当前 Wuji EEF 配置也明确将左右 EEF 声明为 relative：

- [`data_preprocess/wuji_eef_hand_rot6d_config.py:40-50`](../data_preprocess/wuji_eef_hand_rot6d_config.py:40)

### 3.2 数据源 topic

数据转换使用：

```text
右腕 state : /astribot_arm_right/endpoint_current_states
右腕 action: /astribot_arm_right/endpoint_desired_states
```

对应代码：

- [`data_preprocess/wuji_rosbag_to_gr00t.py:88-92`](../data_preprocess/wuji_rosbag_to_gr00t.py:88)
- [`data_preprocess/wuji_rosbag_to_gr00t.py:999-1030`](../data_preprocess/wuji_rosbag_to_gr00t.py:999)

各数据流按共同的 anchor timestamp 取最近样本后写入同一帧。因此，当前数据是同一观测时刻的 measured-to-desired 差以及未来 desired trajectory。

## 4. 数据中的直接证据

对 `grasp_anything_eef_rot6d` 全部 75 个 episode、63725 帧，使用项目实际的 Rot6D 解码方式统计右腕：

```text
same-frame relative rotation vector:
mean   = [-0.252°, +0.208°, -1.057°]
median = [-0.362°, +0.135°, -1.130°]
```

右腕第三旋转分量约 98% 为负，且 75/75 个 episode 的 episode 平均值都为负。这个方向一致性说明它不是偶发噪声，而是数据链路中的稳定偏置。

同一帧平移差的全数据统计约为：

```text
translation magnitude median = 10.020 mm
dx = -6.929 mm
dy = +5.846 mm
dz = +2.157 mm
```

按 episode 时间位置分段后的右腕旋转差：

| 区间 | relative rotvec z 中位数 | relative 旋转总角度中位数 |
| --- | ---: | ---: |
| 前 10% | -1.114° | 1.316° |
| 中间 10% | -1.198° | 1.358° |
| 后 10% | -1.005° | 1.346° |

偏置并不是只在动作结束时才出现。它在整个 episode 中都存在，只是在动作结束阶段更容易表现为持续漂移。

低速子集（平移速度不高于 `0.01830 m/s`、角速度不高于 `3.292°/s`）仍有 `9.229 mm` 平移误差、`1.349°` 旋转误差，rotvec-z 中位数为 `-0.989°`。因此这里的“静态”是低速/准静态，不应把末段简单等同于完全静止。

## 5. 为什么静止阶段仍可能有 command-state 差异

理想情况下，如果 desired 和 measured 使用同一坐标系、严格同一时间、同一 EEF 定义，且控制器无稳态误差，则静止时应满足：

```text
T_desired = T_measured
T_relative = Identity
```

但真实机器人中，速度为零不等于 command-state 完全为零。可能来源包括：

- 柔顺/阻抗控制、重力和摩擦造成的稳态误差；
- desired EEF frame 与 measured EEF frame 的标定差异；
- command、执行和测量之间的延迟；
- 最近时间戳匹配带来的残余时差；
- state 和 action 的滤波相位或截止频率不同；
- episode 在控制器完全 settle 前结束。

当前数据还有一个明确的预处理不一致：

[`robot_data_pipeline/outputs/grasp_anything_eef_rot6d/meta/pipeline_manifest.json:126-150`](../robot_data_pipeline/outputs/grasp_anything_eef_rot6d/meta/pipeline_manifest.json:126)

```text
state.right_eef 使用 10 Hz Butterworth、zero-phase 滤波
action.right_eef 没有对应的同等滤波项
```

因此训练输入中的 state 是过滤后的 current，而 action 是未进行同等处理的 desired。即使机器人在视觉上已经基本停止，两个数据流之间仍可能保留稳定的相对偏置。

## 6. 为什么翻腕主要出现在动作结束阶段

在动作进行中，模型输出可以看成：

```text
relative output = 正常动作变化 + 固定 current-to-desired 偏置
```

正常动作变化较大时，固定偏置被运动项掩盖，最终 target 可能仍然看起来合理。

动作接近结束后：

1. desired trajectory 变化变小；
2. measured state 也变得相对平稳；
3. 训练标签中的固定偏置没有消失；
4. 模型继续输出一个小的同方向 relative rotation；
5. 每个新 chunk 都把该旋转应用到新的 reference 上。

如果模型在静止状态持续输出近似固定的旋转 `R_bias`，则跨 chunk 的组合近似为：

```text
C_1 = S_1 @ R_bias
C_2 = S_2 @ R_bias ≈ C_1 @ R_bias
C_3 ≈ C_2 @ R_bias
...
```

单个 chunk 只有约 1° 的偏置，经过几十个 chunk 后就可能形成几十度的姿态漂移，最终达到掌心朝上的状态。

因此“末段翻腕”是一个长期累积的表现位置，而不是说明末段突然产生了新的 90° 训练标签。

## 7. 已排除或暂不支持的解释

### 7.1 不是简单的 action/state 索引错一帧

已有数据质量检查显示：

- 右 EEF action age 约 2 ms；
- 右 EEF state bracket gap 约 4 ms；
- 没有 future-action violation；
- 没有跨 episode 拼接或明显时间错位证据。

这不能证明同步完全无误，但目前没有证据支持“简单的 i/i+1 错位”是主因。

### 7.2 不是单纯的控制器先翻腕

实际推理日志中，异常首先出现在策略生成的 target，actual measured pose 随后才跟随。说明控制器主要是在执行一个已经逐渐漂移的 target，而不是先自行产生 90° 旋转。

### 7.3 不是单纯的 percentile clipping

q01/q99 clipping 会放大 Rot6D 解码误差，尤其是输出接近边界时，但原始训练数据在 clipping 之前已经存在稳定的右腕同方向偏置。因此 clipping 更像放大器，不是根源。

## 8. 根因定位

当前证据支持以下优先级：

### 主根因：relative label 把 tracking error 编码进了训练目标

在当前定义中，`k=0` 目标就是：

```text
inverse(T_measured(i)) @ T_desired(i)
```

所以静止或低速阶段，模型目标并不天然是 identity。

### 主要数据来源：右腕 current/action 存在系统性偏置

偏置在 75 个 episode 和整个时间轴上方向一致，符合坐标系、标定、控制器稳态误差或预处理不一致的特征。

### 放大机制：跨 chunk relative composition

策略每次输出的是相对于 reference 的姿态，固定 residual 会随 chunk 反复组合，形成积分效应。

### 次要放大因素：state-only filtering 和 Rot6D clipping

state/action 预处理不对称会增加 residual 稳定性；Rot6D clipping 会增加姿态重建误差，但两者都不能解释全部问题。

## 9. 如果继续使用 relative pose，推荐的修复路线

### 9.1 重新定义为 desired-to-desired relative

重新构造训练目标：

```text
T_relative(i,k) = inverse(T_desired(i)) @ T_desired(i+k)
```

这样保证：

```text
T_relative(i,0) = Identity
```

静止阶段模型学习的是 identity，而不是 measured-to-desired tracking error。

需要完整执行：

1. 重新生成 relative action；
2. 重新生成 `relative_stats.json`；
3. 检查 `k=0` 的旋转向量接近零；
4. 重新训练 checkpoint；
5. 推理阶段保持同一 relative 定义和 reference 语义。

### 9.2 保留 current-to-desired，但先消除数据偏置

如果必须保持 GR00T 原生语义，则需要在训练前解决：

1. state/action 时间对齐；
2. measured 和 desired 的 EEF frame 标定；
3. state/action 的滤波一致性；
4. 低层控制器稳态误差；
5. episode 结束前的 settling 帧数量。

只有当静止窗口中的 current-to-desired 误差接近零时，原生 current-to-desired relative 才不会把固定偏置反复教给模型。

### 9.3 不建议作为根治的临时措施

以下措施可以用于安全保护或验证，但不能替代重新构造训练标签：

- 对右腕固定减去一个 Euler 角；
- 直接修改 Rot6D 某个分量；
- 只依赖输出 clamp；
- 不重新训练而改变 checkpoint 的 action representation。

姿态偏置补偿应使用完整 SE(3) 变换，而不是对某个 Euler 或 Rot6D 分量做独立相减。

## 10. 修复后的验收标准

### 数据层

```text
静止窗口的 relative rotvec median 接近 [0, 0, 0]
右腕第三分量不再在绝大多数 episode 中同方向偏置
state/action 过滤和时间对齐规则一致
```

### 模型输出层

在任务完成、视觉状态保持不变时：

```text
连续多个 chunk 的右腕 relative rotvec 不应持续同方向增长
k=0 或 hold 状态的旋转输出应接近 identity
```

### 实际执行层

```text
target palm normal 不应跨越 z=0
target-measured rotation error 不应单调累积
右腕姿态在任务结束后保持 bounded，而不是持续积分
```

## 11. 最终结论

当前训练方案在 relative 定义上与 GR00T 原生方案一致：都是 measured/current state 到 desired action 的相对变换。

问题并不是 relative pose 本身不能训练，而是当前 relative label 把真实系统中的右腕 current-to-desired 稳定偏差编码成了模型目标。动作结束后，正常运动项减弱，该偏差被持续输出，并在跨 chunk 组合中累积，形成翻腕。

如果必须继续使用 relative pose，最稳妥的根治路线是：

```text
desired-to-desired relative labels
统一 state/action 的时间、坐标系和滤波处理
重新生成 relative statistics
重新训练 checkpoint
对静止/hold 状态进行 identity 验收
```

这比继续调整推理阶段的 reference 方式更接近问题根源。

## 附录：B 模式 command-reference 对比实验记录

该实验的目的，是判断 B 模式 command-reference 的变化会不会改变已经观察到的右腕漂移，以及区分“推理 reference 放大”与“训练标签产生”的影响。它是诊断实验，不是最终部署方案。

### 实验设置

两次实验都使用同一 checkpoint、同一模型输入和同一控制参数：

```text
EEF frame          = chassis
control            = 30 Hz
execute horizon    = 32
EEF filter scale   = 0.35
max EEF step       = 0.03 m
max rotation step  = 10 degrees
```

日志配置：

- [shadow 配置](../logs/replay_b_shadow_20260826_131152/replay_b_shadow_config.json)
- [真实执行配置](../logs/replay_b_real_20260826_133048/replay_b_real_config.json)

### Shadow 实验

Shadow 配置中 `allow_replay_reference_b_send=false`。虽然请求的 reference 模式是 command-reference，但实际发送仍使用原来的 measured-state reference；command-reference 只在日志中计算，没有进入机器人执行闭环。

因此 shadow 实验可以验证两种 target 的差异，但不能验证 command-reference 被真实执行后的跨 chunk 累积效果。它的主要结论是：

```text
shadow 计算没有形成 command-reference 闭环
不能用 shadow 结果判断真实 command-reference 是否会累积翻腕
```

日志目录：

- [`logs/replay_b_shadow_20260826_131152`](../logs/replay_b_shadow_20260826_131152)

### 真实执行实验

真实执行配置中 `allow_replay_reference_b_send=true`，command-reference 真正发送给机器人：

```text
首个 chunk reference = measured_initialization
后续 chunk reference = previous_handoff_command
```

右腕每个 loop 第一个 target 的 palm-normal z 变化如下：

```text
0.999 -> 0.772 -> 0.474 -> 0.252 -> 0.159 -> 0.090
-> 0.020 -> -0.028 -> -0.039 -> -0.024 -> -0.076 -> -0.217 -> -0.299
```

也就是说，目标掌心法向量先从接近朝下逐步接近水平，再进入负 z 区域，随后实际 measured pose 延迟跟随。日志中的 target-measured rotation error 在 target 已经发生明显偏移后才增大，说明实际翻腕首先由策略 target/reference 组合产生，而不是 controller 先自行翻转。

真实执行日志目录：

- [`logs/replay_b_real_20260826_133048`](../logs/replay_b_real_20260826_133048)

### 对根因定位的意义

该对比实验支持三个结论：

1. command-reference 会把模型输出中的固定 relative 偏置直接变成 command-to-command 的累积项，因此能够明显放大姿态漂移。
2. 真实执行中 target 先异常、actual 后跟随，说明低层 controller 不是初始翻腕的主要来源。
3. 该实验只能说明 reference 组合是放大机制，不能把训练数据中的 current-to-desired 偏置归因于推理 reference 本身。根治仍然需要修正 relative label 语义、数据对齐/滤波一致性和训练 checkpoint。
