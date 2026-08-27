你现在位于 **GR00T 训练机**。

这台机器可以访问：

* 原始 ROS bag；
* 当前转换后的 GR00T dataset；
* 数据预处理代码；
* 实际训练使用的 GR00T repository / state-action processor；
* 当前训练配置和 statistics。

当前任务只进行 **Phase-1 数据根因诊断**。

不要修改训练方案，不要重新训练，不要修改部署代码。

---

# 0. 问题背景

当前 GR00T N1.7 真机策略使用 relative EEF pose。

右腕在实际推理后期会跨 action chunk 持续翻腕，表现为一个小的同方向旋转逐步积累。

当前 processed dataset 已观察到：

```text
same-frame current -> desired relative rotation:

mean   = [-0.252°, +0.208°, -1.057°]
median = [-0.362°, +0.135°, -1.130°]
```

其中右腕 rotvec-z：

```text
约 98% 为负
75 / 75 episode 的 episode mean 均为负
```

同时存在约：

```text
dx = -6.85 mm
dy = -3.67 mm
dz = +6.71 mm
```

的 same-frame translation residual。

当前最需要回答的不是“怎么改 representation”，而是：

1. raw ROS bag 中是否本来就存在这个 residual；
2. residual 是否近似固定 SE(3) transform；
3. residual 是否可以由时间 lag / controller delay 解释；
4. preprocessing 是否明显改变它；
5. GR00T 实际最终写入 H=32 relative action chunk 的 label 到底是什么。

---

# 1. 强制约束

本阶段只允许诊断。

不要：

* 修改 dataset；
* 修改 preprocessing；
* 修改 GR00T state/action semantics；
* 改成 desired-to-desired；
* 改成 measured-to-measured；
* 改 inference reference；
* 改 Rot6D / RotVec representation；
* 修改 normalization/clipping；
* 重新训练；
* 运行真机。

所有 rotation analysis 必须通过：

```text
quaternion
→ rotation matrix
→ SO(3)
→ rotvec / log map
```

完成。

不要对 Euler angle 或 Rot6D component 做直接减法作为姿态误差。

---

# Part A — 审计真实数据链路

首先阅读项目实际使用的 ROS bag conversion / preprocessing pipeline。

重点确认右腕：

```text
state:
  /astribot_arm_right/endpoint_current_states

action:
  /astribot_arm_right/endpoint_desired_states
```

但不要仅根据已有文档假设这些是最终实际字段。

从代码确认：

* topic；
* ROS message type；
* pose field；
* timestamp field；
* source/header timestamp vs rosbag receive timestamp；
* parent/reference frame；
* current EEF frame；
* desired EEF frame；
* quaternion convention；
* ROS bag → canonical/frame → dataset 的完整变换；
* state/action 是否使用相同 timestamp matching；
* state/action 是否应用了不同 filtering。

输出一个：

```text
diagnostics/right_wrist_phase1/pipeline_audit.md
```

明确记录真实代码路径和实际语义。

---

# Part B — Raw ROSBag same-frame SE(3) residual

直接从原始 ROS bag 提取右腕：

$$
T_c(t)
$$

和：

$$
T_d(t)
$$

其中：

```text
Tc = measured/current EEF
Td = desired/command EEF
```

必须尽量基于 ROS message 原始 pose。

按当前 pipeline 实际使用的 source-time 对齐规则构造 same-frame pair。

计算：

$$
E_i = T_{c,i}^{-1}T_{d,i}.
$$

将其转换成：

```text
translation x/y/z [mm]
translation magnitude [mm]

rotvec x/y/z [degree]
rotation magnitude [degree]
```

---

# Part C — Raw residual 全量统计

统计全部 episode。

至少输出：

```text
count
mean
median
std

q01
q05
q25
q75
q95
q99
```

针对：

```text
tx
ty
tz
translation magnitude

rotvec_x
rotvec_y
rotvec_z
rotation magnitude
```

同时输出每 episode：

```text
frame count
translation median
rotvec median
rotation magnitude median
```

特别报告：

```text
rotvec_z median < 0
```

的 episode 数量和比例。

再分别统计：

```text
episode first 10%
episode middle 10%
episode final 10%
```

回答：

> 当前约 -1° bias 是整个 episode 都存在，还是主要只存在于动作末尾？

---

# Part D — Constant SE(3) hypothesis

这是本次最关键的诊断之一。

判断：

$$
E_i \approx X
$$

是否成立，其中 \(X\) 是一个固定 SE(3) transform。

使用正确的 SO(3)/SE(3) averaging 或 optimization 方法估计：

$$
X^\star.
$$

不要平均 Euler angles。

保存 \(X^\star\)：

```text
translation
rotation matrix
quaternion
rotvec
```

然后计算去除固定 transform 后的 residual：

$$
E_i^{res}
$$

具体左乘还是右乘必须根据当前 transform convention 严格推导和验证。

不要机械套公式。

报告：

```text
BEFORE:
median translation magnitude
median rotation magnitude
median rotvec xyz

AFTER removing X*:
median translation magnitude
median rotation magnitude
median rotvec xyz
```

并给：

```text
translation residual reduction %
rotation residual reduction %
```

---

# Part E — Pose dependence

测试 residual 是否随着机器人绝对 pose 系统变化。

分析：

```text
E_i vs current EEF xyz
E_i vs current EEF orientation
```

可以使用：

* workspace bin；
* orientation bin；
* correlation；
* scatter；
* 简单 regression 作为辅助。

核心问题：

> 同一个 \(X^\star\) 能否在不同 workspace 和 wrist orientation 下都较好解释 residual？

如果可以，证据更支持：

```text
fixed frame / TCP / calibration / semantic transform
```

如果 residual 强烈 orientation-dependent，则不要轻易归因于固定外参。

---

# Part F — Lag sweep

测试 fixed time lag 是否可以解释 residual。

构造：

$$
E_i(\tau)
=
T_c(t_i)^{-1}T_d(t_i+\tau).
$$

扫描：

$$
\tau \in [-200,+200]\text{ ms}.
$$

先：

```text
10 ms step
```

找到最优区域后再：

```text
1~2 ms local refinement
```

插值必须使用：

```text
translation: linear
rotation: quaternion SLERP / proper SO(3)
```

禁止：

```text
Euler interpolation
Rot6D component interpolation
```

---

# Part G — Lag objective

对每个 lag：

$$
J_R(\tau)
=
median(\|\log(R_E(\tau))\|)
$$

$$
J_t(\tau)
=
median(\|t_E(\tau)\|)
$$

计算：

```text
best_rotation_lag
best_translation_lag

rotation residual @ tau=0
rotation residual @ best tau
rotation improvement %

translation residual @ tau=0
translation residual @ best tau
translation improvement %
```

生成：

```text
lag_vs_rotation_residual.png
lag_vs_translation_residual.png
```

---

# Part H — Velocity-direction test

如果 residual 主要来自固定 latency：

$$
e_R \approx \omega\tau
$$

则 error 应该随运动速度和方向发生变化。

从 measured/current trajectory 计算 angular velocity。

根据 relevant wrist rotation component 将数据分成：

```text
positive angular velocity
negative angular velocity
near-static
```

分别统计：

```text
count

rotvec-z mean
rotvec-z median
rotvec-z negative ratio

rotation magnitude
```

尤其回答：

> 当 wrist angular velocity 方向反转时，current→desired rotvec-z residual 是否也反转？

如果：

```text
positive omega -> negative residual
negative omega -> negative residual
near-static    -> negative residual
```

则明确说明：

> Simple fixed latency is insufficient to explain the persistent directional bias.

---

# Part I — Quasi-static residual

根据数据的 translational/angular velocity distribution 定义一个保守的 quasi-static threshold。

必须报告实际 threshold。

选择：

```text
low translational velocity
AND
low angular velocity
```

样本。

统计：

$$
T_c^{-1}T_d
$$

如果在 quasi-static 情况下仍稳定存在：

```text
~1° rotation
+
several-mm translation
```

明确记录。

这里只做事实诊断，不进一步修改 controller/action semantics。

---

# Part J — Raw vs Processed sanity check

训练机同时拥有 raw ROS bag 和当前训练 dataset，因此直接在同一任务中完成比较。

分别计算：

### RAW

```text
raw measured current
vs
raw desired command
```

### PROCESSED

```text
state pose as actually stored for training
vs
action pose as actually stored for training
```

使用完全相同的 SE(3) residual metric。

生成：

| Metric                | Raw | Processed | Difference |
| --------------------- | --: | --------: | ---------: |
| tx median             |     |           |            |
| ty median             |     |           |            |
| tz median             |     |           |            |
| translation magnitude |     |           |            |
| rotvec-x median       |     |           |            |
| rotvec-y median       |     |           |            |
| rotvec-z median       |     |           |            |
| rotation magnitude    |     |           |            |

当前 preprocessing 已知可能存在：

```text
state: filtered
action: not equivalently filtered
```

但本阶段**不要完整研究 filter 机制**。

当前只回答：

> preprocessing 是否 materially creates or amplifies the right-wrist residual？

---

# Part K — 审计 GR00T 实际 relative semantics

找到本次训练实际使用的：

```text
modality config
ActionConfig
state_action_processor
pose utilities
stats generation
```

确认：

```text
ActionRepresentation
ActionType
ActionFormat
```

以及实际 right EEF key。

特别确认：

$$
A_i^k
=
T_{state}(i)^{-1}T_{action}(i+k)
$$

是否真的是当前 checkpoint 的实际训练语义。

必须从**实际执行代码**确认：

```text
which transform is inverted
multiplication order
parent/child convention

translation reference frame
rotation reference frame

Rot6D row/column convention
horizon indexing
padding behavior
```

不要只看 README 或已有报告。

---

# Part L — 通过真实 GR00T processor 重建 H=32 label

这是本阶段另一项核心任务。

尽可能直接：

* 调用真实 GR00T processor；
* 或 instrument 真实 training preprocessing path。

不要另写一套“数学等价”的 converter 后直接假设它与训练完全一致。

对每一个 observation index \(i\)：

$$
A_i^0,\ldots,A_i^{31}
$$

获得模型实际训练使用的 right EEF relative action chunk。

保留：

```text
pre-normalization physical action
post-normalization action
```

本阶段主要分析 **pre-normalization physical SE(3)**。

将 Rot6D decode 回合法：

```text
rotation matrix
→ rotvec
```

---

# Part M — H=32 horizon-wise statistics

对：

```text
k = 0 ... 31
```

分别计算：

```text
tx median / mean
ty median / mean
tz median / mean

rotvec-x median / mean
rotvec-y median / mean
rotvec-z median / mean

rotation magnitude median / mean
```

生成：

```text
horizon_vs_tx.png
horizon_vs_ty.png
horizon_vs_tz.png

horizon_vs_rotvec_x.png
horizon_vs_rotvec_y.png
horizon_vs_rotvec_z.png
horizon_vs_rotation_angle.png
```

需要明确判断：

> 当前约 -1° right-wrist orientation residual 是否作为 common offset 广泛存在于整个 H=32 action chunk？

区分例如：

```text
Case A:

k0  ≈ -1.1°
k1  ≈ -1.1°
...
k31 ≈ -1.0°
```

与：

```text
Case B:

k0 = -1.1°
之后快速变化/消失
```

---

# Part N — 分离 chunk reference offset 与 desired trajectory motion

对于每个 action chunk，计算：

$$
D_i^k
=
(A_i^k)^{-1}A_i^{k+1},
\qquad k=0...30.
$$

严格验证 transform convention。

在当前定义下理论上：

$$
A_i^k
=
T_m(i)^{-1}T_d(i+k)
$$

则：

$$
(A_i^k)^{-1}A_i^{k+1}
=
T_d(i+k)^{-1}T_d(i+k+1).
$$

因此 \(D_i^k\) 反映 future desired trajectory 本身的相邻运动，并去掉 observation reference 的 common factor。

分析：

```text
A_i^k rotvec
vs
D_i^k rotvec
```

重点关注 right wrist rotvec-z。

如果出现：

```text
A_i^k:
persistent ~-1° common orientation residual

D_i^k:
near-zero median / balanced sign / normal trajectory motion
```

明确写出：

> The approximately -1° orientation bias is primarily a chunk reference offset introduced by current-to-desired semantics, rather than an approximately -1° incremental rotation contained in every desired trajectory step.

---

# Part O — Episode-end behaviour

再对：

```text
episode final 10%
```

以及可靠的 low-motion subset 统计：

$$
A_i^k
$$

和：

$$
D_i^k.
$$

重点检查是否：

```text
A:
retains non-zero common orientation offset

D:
approaches identity as task motion slows
```

如果成立，它将支持：

```text
normal motion gets small
while
current-to-desired common offset remains
```

从而解释为什么翻腕主要在后期显现。

---

# Part P — Pre-normalization check

只确认：

> 主要 orientation residual 在 normalization / percentile clipping 之前是否已经存在？

如果 pre-normalization SE(3) 中已经存在，则明确写：

```text
Normalization/clipping is not required to generate the observed bias.
```

本阶段不要深入修改 stats/clipping。

---

# Output

创建：

```text
diagnostics/right_wrist_phase1/
```

至少包含：

```text
README.md

pipeline_audit.md

raw_residual_summary.json
processed_residual_summary.json
raw_vs_processed.json

constant_se3_fit.json
lag_sweep_summary.json

per_episode_residual.csv
velocity_conditioned_residual.csv
quasi_static_residual.json

gr00t_training_semantics.json
horizon_relative_label_stats.csv
adjacent_trajectory_motion_stats.csv

lag_vs_rotation_residual.png
lag_vs_translation_residual.png

horizon_vs_rotvec_x.png
horizon_vs_rotvec_y.png
horizon_vs_rotvec_z.png
horizon_vs_rotation_angle.png
```

---

# README 最终必须回答的 7 个问题

## Q1

Raw ROS bag 是否已经存在稳定 right-wrist current→desired bias？

```text
YES / NO / PARTIALLY
```

## Q2

它能否被单一 constant SE(3) transform 很好解释？

给出：

```text
X*
before
after
reduction %
```

## Q3

fixed time lag 能解释多少？

给出：

```text
best lag
rotation reduction
translation reduction
velocity-direction evidence
```

## Q4

quasi-static 时 residual 是否仍存在？

## Q5

raw → processed 是否明显创建或放大 residual？

## Q6

GR00T 实际 H=32 relative label 中 residual 如何随 horizon 分布？

## Q7

最终更像：

```text
common chunk reference offset
```

还是：

```text
actual incremental rotation in desired trajectory
```

---

# Final Root-Cause Classification

只根据证据，将结果分类为一个或多个：

### A — Fixed EEF/TCP/frame semantic offset

### B — Temporal/controller lag

### C — Preprocessing-induced/amplified residual

### D — GR00T relative-label/reference semantic effect

### E — Actual desired trajectory contains directional rotational motion

### F — Mixture

### G — Unresolved

给出证据强弱和数字，不要实施修复。

完成诊断后停止。
