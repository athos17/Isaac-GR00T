# TrainingRTC 当前实现原理与使用方式

本文描述当前仓库中已经实现的 TrainingRTC，包括训练、推理和 runtime scheduler。
本文只记录现状；Client/server 远程 RPC 适配不属于当前实现。

## 1. 目标

TrainingRTC 让模型在动作 chunk 的前 `d` 个 action token 已经被旧 chunk 占用时，
仍然能够生成后续 postfix 动作。训练和推理都将动作 chunk 看成：

```text
action = [prefix of length d, postfix of length H-d]
```

其中：

- `H` 是 action-token horizon；
- `d` 是延迟对应的 prefix 长度；
- `d=0` 表示普通同步推理；
- runtime 的执行步数 `s` 不属于模型训练目标，只由 scheduler 使用。

当前默认配置是 `action_step_hz=30`，训练 checkpoint 默认支持 `d=0..6`，
具体概率由 `training_rtc_delay_pmf` 配置。

## 2. 代码入口

主要实现位置：

| 功能 | 文件 |
| --- | --- |
| TrainingRTC 训练采样和 loss | `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` |
| TrainingRTC 推理采样器 | `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` |
| action layout、padding、编码和解码 | `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` |
| checkpoint contract 写入和校验 | `gr00t/model/gr00t_n1d7/setup.py` |
| 训练配置 | `gr00t/configs/model/gr00t_n1d7.py`、`gr00t/configs/finetune_config.py` |
| 异步延迟和 chunk 状态 | `gr00t/policy/training_rtc_runtime.py` |
| policy 层的 TrainingRTC 输入输出接入 | `gr00t/policy/gr00t_policy.py` |

## 3. 动作表示和 layout

TrainingRTC 作用于展平后的 padded action tensor，不区分某个维度是 EEF 还是 joint。
动作处理仍由 modality config 和 `StateActionProcessor` 完成。

当前支持两类典型动作空间：

1. 双臂相对 EEF `xyz + rot6d`，手部 joint 为 absolute；
2. 双臂和手部 joint 全部为 absolute。

处理顺序是：

```text
raw absolute target
    -> 对 RELATIVE group 按当前 state 转为 relative
    -> 使用对应 statistics normalization
    -> 按 modality key 拼接
    -> padding 到 max_action_dim
```

定义：

- `D_sem`：当前 embodiment 的真实 semantic action dimension；
- `D_model`：模型的 `max_action_dim`；
- 模型 tensor 使用 `[B, H, D_model]`；
- 必须满足 `D_sem <= D_model`；
- padding dimension 不计 loss，也不能传给 controller。

`Gr00tN1d7Processor` 会保存 action layout，包括 semantic dimension、model dimension、
horizon 和各 action group 的 offset。TrainingRTC 开启时，layout 会写入模型配置。

## 4. 训练阶段

### 4.1 延迟采样

模型开启 `training_rtc_enabled` 后，每个 batch sample 独立采样一个 `d`：

```python
d ~ training_rtc_delay_pmf
```

约束为：

```text
0 <= d < H
d <= training_rtc_max_delay
```

`d=H` 被拒绝，因为没有 postfix token 可以计算 loss。

如果 PMF 只有 `d=0`，实现不会额外消耗一次随机数，以保证固定 RNG 下的普通 flow-matching
路径和旧实现一致。

### 4.2 prefix/postfix 构造

训练输入为 normalized action `actions`，形状为 `[B, H, D_model]`。实现首先生成：

```python
noise = randn_like(actions)
tau = sample_time(B)
```

每个样本只有一个 postfix timestep `tau[b]`。对第 `k` 个 action token：

```text
token_t[b, k] = 1.0       if k < d[b]
                 tau[b]  otherwise
```

带噪轨迹为：

```python
noisy = token_t[..., None] * actions + (1 - token_t[..., None]) * noise
noisy[prefix_mask] = actions[prefix_mask]
```

因此：

- prefix 使用 clean action；
- prefix 的连续 timestep 精确为 `1.0`；
- postfix 使用同一个样本级 `tau[b]`；
- postfix 仍按 flow matching 的 `velocity = actions - noise` 训练。

### 4.3 loss mask

训练动作 mask 与 prefix mask 相乘：

```python
loss_mask = action_mask * (~prefix_mask[..., None])
```

最终 loss 是有效 postfix token 和有效 semantic dimension 上的 MSE：

```python
loss = ((pred_velocity - velocity) ** 2 * loss_mask).sum() \
       / (loss_mask.sum() + 1e-6)
```

所以 prefix 不产生 loss，padding dimension 也不产生 loss。

### 4.4 timestep 处理

连续 timestep 通过 `_timestep_to_bucket()` 转成模型使用的离散 bucket：

```python
floor(timestep * num_timestep_buckets)
```

并限制在合法 bucket 范围内。训练时：

- ActionEncoder 接收每个 token 自己的 timestep；
- DiT 主体仍使用 batch/sample 级 timestep；
- 当前实现属于 token-wise ActionEncoder 的 minimal baseline；
- 并不是完整的 token-wise AdaLN DiT。

## 5. 推理阶段

TrainingRTC 推理通过独立选项启用：

```python
options = {
    "rtc_mode": "training",
    "d_cond": d,
}
```

同时，`action_input["action"]` 必须包含 normalized、padded 的 committed prefix。

### 5.1 初始化动作

模型先生成纯 noise action：

```text
actions: [B, H, D_model]
```

然后将前 `d` 个 token 替换为 committed prefix：

```python
actions[:, :d] = committed_prefix[:, :d]
```

`d` 必须满足：

```text
0 <= d < action_horizon
d <= config.training_rtc_max_delay
```

### 5.2 每一步 denoising

每个 denoising step 都会：

1. 计算当前全局 DiT timestep；
2. 将 prefix token 的 ActionEncoder timestep 设置为最后一个 clean bucket；
3. 将 postfix token 的 ActionEncoder timestep 设置为当前 denoising bucket；
4. 执行 action decoder；
5. 使用 Euler update 更新动作；
6. 再次 hard overwrite prefix。

关键约束是：

```python
actions = actions + dt * pred_velocity
actions[prefix_mask] = prefix[prefix_mask]
```

因此 prefix 在整个采样过程中数值不变，postfix 才会被模型更新。

### 5.3 与旧 inference-time RTC 的区别

当前仓库同时保留旧的 inference-time RTC。两者不同：

| 项目 | TrainingRTC | 旧 inference-time RTC |
| --- | --- | --- |
| 入口 | `rtc_mode="training"` | `rtc_overlap_steps` 等选项 |
| prefix | 每一步 hard overwrite | overlap/inpainting |
| timestep | prefix clean bucket，postfix 当前 bucket | 普通全局 timestep |
| 速度调节 | 不使用 ramp | 使用 frozen steps 和 ramp rate |
| 训练支持 | 有 postfix-only loss | 无对应训练 loss |

默认推理行为不会因为 TrainingRTC 代码存在而改变。

## 6. Policy 层数据流

`Gr00tPolicy._get_action()` 在 TrainingRTC 模式下的流程是：

```text
observation
    -> processor 处理图像、语言和 state
    -> 检查 rtc_context
    -> 创建 padded action prefix
    -> 注入 action_input["action"]
    -> 注入 valid_dim_mask
    -> model.get_action(..., options={"rtc_mode": "training", "d_cond": ...})
    -> 使用 request context 的 raw_state_snapshot 解码
    -> 返回 action 和 TrainingRTC metadata
```

返回的 info 包含：

- `training_rtc_reference_timestamp`；
- `training_rtc_c_obs`；
- `training_rtc_d_cond`；
- `training_rtc_chunk_version`；
- `training_rtc_stats_version`；
- `training_rtc_absolute_chunk`。

其中 `training_rtc_absolute_chunk` 用于 runtime 建立新的 absolute action cache。

## 7. 相对 EEF 的 reference 规则

相对 EEF 不能直接复制旧 normalized relative action，因为旧 action 使用的是旧 observation
的 reference frame。

正确流程是：

```text
旧 absolute target cache
    -> 按 controller cursor 对齐前 d 个 target
    -> 使用新请求 t_obs 的 EEF pose rebase
    -> rot6d encode
    -> relative statistics normalization
    -> 形成 committed prefix
```

模型生成的完整 relative EEF chunk 也必须使用同一个请求保存的
`raw_state_snapshot@t_obs` 做 relative-to-absolute decode。不能在 `t_ready` 或 handoff 时
使用更新后的 state 重新 decode，否则会引入额外的 frame jump。

absolute joint 路径不做 frame rebase，直接按 action-grid 对齐复制。

## 8. Runtime scheduler

`TrainingRTCScheduler` 不执行模型采样，只负责延迟和 chunk 状态保护。

### 8.1 延迟换算

action grid 的时间步为：

```python
dt_action = 1.0 / action_step_hz
```

延迟换算为：

```python
d_cond = ceil(latency_seconds / dt_action) + jitter_margin
```

### 8.2 请求上下文

`TrainingRTCRequestContext` 保存：

- `reference_timestamp`；
- `raw_state_snapshot`；
- `c_obs`；
- `d_cond`；
- `chunk_version`；
- `stats_version`；
- `committed_prefix`；
- semantic/model dimension；
- `t_launch`、`t_ready`、`t_handoff` 等 telemetry。

### 8.3 AbsoluteTargetChunkCache

缓存以单调递增的 action cursor 为索引，只保存 absolute targets：

```python
AbsoluteTargetChunkCache(
    targets=...,
    start_cursor=...,
    chunk_version=...,
    stats_version=...,
)
```

它负责：

- 计算某个 cursor 后还剩多少 action；
- 按 cursor 切出 committed prefix；
- 检查旧 chunk coverage 是否足够。

### 8.4 handoff 判断

实际切换位置使用：

```text
d_actual = c_handoff - c_obs
```

而不是简单使用 `c_ready - c_obs`。

判断规则：

| 条件 | 结果 |
| --- | --- |
| `d_actual < 0` | `RTC_CURSOR_INVALID` |
| chunk version 不一致 | `RTC_STALE_REQUEST` |
| `d_actual <= d_cond` | 接受并切换新 absolute chunk |
| `d_actual > d_cond` | `RTC_DELAY_EXCEEDED`，丢弃并重算 |

`compute_handoff_cursor()` 对 action boundary 使用明确的半开区间规则，避免 boundary 前后
出现 off-by-one。

## 9. checkpoint contract

TrainingRTC checkpoint 会记录：

- `training_rtc_enabled`；
- `training_rtc_max_delay`；
- `training_rtc_delay_sampling`；
- `training_rtc_delay_pmf`；
- `training_rtc_loss_mode`；
- `action_step_hz`；
- `training_rtc_action_layout`；
- `training_rtc_stats_version`。

`stats_version` 是对 processor statistics 的确定性 hash。加载 checkpoint 或创建 runtime
request 时，如果 action layout、统计版本或 padded dimension 不匹配，应直接失败。

## 10. 当前已实现的保护和测试覆盖

当前测试已覆盖：

- delay 到 action-grid 的换算；
- `d=0` 的 legacy RNG/noise/timestep 一致性；
- prefix clean timestep 和 postfix shared timestep；
- prefix hard overwrite；
- postfix-only loss mask；
- `d_cond` 越界和 checkpoint delay OOD；
- action boundary 的 handoff cursor；
- stale chunk version；
- absolute cache coverage；
- relative EEF prefix rebase；
- 使用 request snapshot 完整 decode；
- stats/layout mismatch。

## 11. 当前边界

当前仓库已经实现模型侧 TrainingRTC 和本地 runtime scheduler，但以下内容不属于已完成的
TrainingRTC 实现：

- 跨机器 TrainingRTC 专用 Client/server RPC；
- 将 `TrainingRTCRequestContext` 自动序列化为网络协议；
- controller 厂商协议和真实机器人安全停机策略；
- 多请求并发的 ZMQ `DEALER/ROUTER` 服务；
- 完整 token-wise AdaLN DiT ablation。

因此，现有 `PolicyServer/PolicyClient` 仍然是普通同步 `get_action` 服务。要进行远程
TrainingRTC 部署，需要额外增加 RPC adapter，但不应改变本文描述的训练和采样语义。

