# GR00T N1.7 Training-Time RTC 适配计划

> 状态：Phase 0-4 已实现（Full token-wise AdaLN 仍仅保留为后续 ablation；本文件同时作为实现与验收记录）
> 目标：在当前仓库上实现论文 *Training-Time Action Conditioning for Efficient Real-Time Chunking*（TrainingRTC），并支持两种动作表示：
>
> 1. 双臂相对 EEF `xyz + rot6d`，灵巧手绝对 joint；
> 2. 双臂绝对 joint（以及保持绝对表示的手部 joint）。

## 1. 范围和当前基线

本计划在独立分支 `plan/training-rtc-adaptation` 上维护；实现合并前应先在该分支完成单元测试、离线 replay 和 scheduler 仿真。TrainingRTC 的作用对象是展平后的动作块 `action: [B, H, D]`，与每个动作维度是 EEF、joint 还是手部关节没有直接关系。`H` 是动作预测 horizon，`D` 是所有 modality 拼接后的维度，`d` 是训练或推理时注入的延迟前缀长度。每次实际执行的步数 `s` 不属于 TrainingRTC 学习目标或 checkpoint 配置，而是 runtime scheduler 的调度参数；只有 scheduler 在部署时才检查 `d_cond <= H-s`。

当前仓库已经有一个**推理时 RTC**，位置在 [`gr00t/model/gr00t_n1d7/gr00t_n1d7.py`](../gr00t/model/gr00t_n1d7/gr00t_n1d7.py) 的 `get_action_with_features()`：它用 `rtc_overlap_steps`、`rtc_frozen_steps` 和 `rtc_ramp_rate` 做 overlap/inpainting/ramp。训练路径（同文件的 `forward()`）仍然对整个动作块采样一个 batch 级 `t`，并对所有有效 token 计算普通 velocity loss。因此，不能只调现有推理参数来得到 TrainingRTC；需要新增 prefix/postfix token-wise timestep 条件接口和独立的推理模式，但 postfix 的随机 timestep 仍是每个样本共享的单一 `tau[b]`。

非目标：不改 VLM/语言视觉 backbone，不新增第二个 RTC 网络，不把两种动作空间混成一个 checkpoint，也不在本计划阶段替换现有推理时 RTC 的默认行为。

## 1.1 本次重点复核结论

| 检查项 | 计划中的强制语义 | 上线前判定 |
| --- | --- | --- |
| ① timestep | 每个样本一个 postfix 标量 `tau[b]`；所有 prefix token 的连续 timestep 精确为 `1.0`；prefix 不计 loss | fixed-RNG 下 `d=0` 与原始 GR00T FM 逐元素一致；非零 `d` 的 postfix timestep 在同一样本内完全相同 |
| ② relative EEF | prefix 来源是 absolute target cache，以发起请求时的新 observation EEF pose rebase，再归一化；禁止复制旧 normalized relative action | absolute→relative→absolute round-trip 和真实 chunk 边界误差通过阈值 |
| ③ asynchronous runtime | 以 `t_obs`/`c_obs` 为 reference；inference 发起前确定 `d_cond` 并登记 committed prefix；handoff 时计算 `d_actual=c_handoff-c_obs`；`d_actual>d_cond` 时丢弃结果并重算，禁止未 conditioning 的 splice | scheduler 日志能重放 `t_obs/c_obs/t_ready/c_ready/t_handoff/c_handoff/d_cond/d_actual/chunk_version`，且无 stale/越界切换 |

### Repo Verification Record

- `examples/wuji_rot6d/wuji_eef_hand_rot6d_h32_config.py` 的文件名和 `delta_indices=range(32)` 表示实际 horizon 为 32；文件内部的历史变量名/注释仍有 `h40`，不作为 runtime horizon 依据。
- 当前仓库没有已实现的 TrainingRTC asynchronous scheduler。`gr00t/policy/server_client.py` 只提供通用 `get_action` RPC，`gr00t/policy/gr00t_policy.py` 的 `options` 当前不驱动异步 chunk 状态机；Phase 4 必须新增 scheduler/context，但不改变 server/client 的通用 RPC 契约。
- 当前 `Gr00tN1d7Processor.process_observation()` 只生成 horizon action mask；训练 `__call__()` 才生成 `[H,D_model]` 的 dimension mask。Phase 0 必须补齐 inference-side `valid_dim_mask`，使训练和 inference 的 padded action contract 一致。

## 2. 两种动作空间的契约

### 2.1 相对 EEF `xyz + rot6d` + 绝对手部 joint

参考 [`examples/wuji_rot6d/wuji_eef_hand_rot6d_h32_config.py`](../examples/wuji_rot6d/wuji_eef_hand_rot6d_h32_config.py)：左右 EEF action 各为 9 维（`xyz` 3 维 + rotation 6D 6 维），左右手部 joint 保持 absolute。总维度应由 processor/statistics 动态求得：

```text
D = 9 (left EEF) + 9 (right EEF) + D_left_hand + D_right_hand
```

如果每只手是 12 DOF，则 `D = 42`；实现不能把 12 写死。prefix mask 是按**时间步**定义的，因此对一个被固定的 timestep，要同时固定该 timestep 的 EEF 和 hand 所有维度。rotation 的 rot6d 必须沿用仓库 `EndEffectorPose` 的 encode/decode 和数值约定。

### 2.2 双臂绝对 joint + 绝对手部 joint

参考 [`data_preprocess/wuji_joint_hand_absolute_h32_config.py`](../data_preprocess/wuji_joint_hand_absolute_h32_config.py)：`left_joint_space`、`right_joint_space`、`left_hand_joints`、`right_hand_joints` 全部为 `ActionRepresentation.ABSOLUTE`。这一路径不做 frame rebase，不需要 `relative_stats`；使用普通 `action` statistics。即使将来只把“机械臂”改成 absolute 而手部仍是 absolute，处理方式也相同。

两条路径在 TrainingRTC 的 noise/prefix/loss 逻辑上完全相同，差异只发生在数据处理和运行时把旧 chunk 变成 prefix 的步骤。

## 3. 数据、参考系与归一化

当前 [`gr00t/data/state_action/state_action_processor.py`](../gr00t/data/state_action/state_action_processor.py) 的 `apply_action()` 先转换表示再归一化：对标记为 `RELATIVE` 的 EEF，以 state 的最后一帧为 reference；然后使用 `relative_action` statistics。absolute joint 只走普通 action statistics。`launch_finetune.py` 设置 `use_relative_action=True` 时，也只会作用于配置中标记为 `RELATIVE` 的 group。

训练数据必须遵守以下顺序：

1. 从原始记录取得当前 state 和未来的 absolute action target；
2. 对相对 EEF，以当前 state 最后一帧计算 relative transform，并编码为 `xyz + rot6d`；absolute joint/hand 不转换；
3. 对每个 group 使用对应 statistics 归一化，再按 modality config 拼成 `[H, D]`；
4. 在归一化后的 action chunk 上构造 TrainingRTC prefix。

不能对已经是相对 EEF 的数据再次做 relative conversion，也不能用 absolute action 的统计量归一化相对 EEF。

### 3.1 相对 EEF 的运行时换参考系（关键）

旧 observation 下的 normalized relative rot6d 不能直接作为新 observation 下的 prefix：它们的 reference frame 不同。运行时应缓存可重建的 absolute target（推荐在 unnormalize 后、发送给控制器前缓存），下一次推理时：

1. 根据控制器执行游标，从上一 chunk 取出与当前控制时刻对齐、将作为新 chunk 前缀的 `d` 个 absolute EEF targets（不能只按数组下标盲取）；
2. 记录新 observation 的当前 EEF pose；
3. 以新 pose 为 reference，重新计算每个 target 的 relative transform；
4. 用仓库同一套 rot6d 编码和 `relative_action` statistics 归一化；
5. 将得到的 semantic `[d, D_sem]` prefix 与手部 absolute joint prefix 按时间对齐，再按 processor 的同一 layout 补零到模型的 `max_action_dim`，并生成对应 prefix mask；送入 TrainingRTC sampler。

**Decode 规则必须更严格：** 当前 [`Gr00tN1d7Processor.decode_action()`/`unapply()`](../gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py) 会把调用方传入的 state 交给 `StateActionProcessor.unapply_action()`；TrainingRTC 输出的整个 relative EEF chunk（prefix 和 postfix）都必须使用该 inference request 保存的 `raw_state_snapshot@t_obs` 做 relative→absolute conversion。不能在 `t_ready`、handoff 或 cursor 更新时把新的 current state 传给这些 API。decode 后立即形成整个 absolute target chunk；之后 cursor、slice、handoff、absolute target cache 都只在 absolute timeline 上操作。absolute-joint 路径也应在同一时间对齐后形成 absolute chunk。request context 至少保存 `reference_timestamp=t_obs`、`raw_state_snapshot`、`c_obs`、`d_cond`、`chunk_version` 和 `stats_version`，并在 decode 前校验 context 未过期。

这样可以保证 chunk 拼接后 decode 回 absolute target 时不产生额外的 reference-frame 跳变。任何 reference state 错误、缺失、stats version 不匹配或 semantic/padded layout 不匹配，都必须丢弃该 request 结果，而不是尝试用当前 state 修复。

## 4. 训练阶段实现

### 4.0 Semantic action dimension 与 N1D7 padding 契约

当前 [`gr00t/model/gr00t_n1d7/gr00t_n1d7.py`](../gr00t/model/gr00t_n1d7/gr00t_n1d7.py) 将 `config.max_action_dim` 保存为 model action dimension，action encoder/decoder 和采样噪声都使用这个 padded dimension。processor 则从 modality statistics 动态计算 semantic action dimension，按 modality key 拼接后补零到 `max_action_dim`；训练 collator 的 `action_mask` 对 semantic dimension 为 1、dimension padding 为 0，同时对有效 horizon 之外的行置 0。TrainingRTC 必须保留这两个层次：

- `D_sem = StateActionProcessor.get_action_dim(embodiment_tag)`，由当前 modality/statistics 决定；
- `D_model = config.max_action_dim`，必须满足 `D_sem <= D_model`，模型张量和 noise 的 shape 是 `[B,H,D_model]`；
- prefix/noise/velocity/loss 都在 `[B,H,D_model]` 上构造，但 prefix mask 必须是 `valid_horizon_mask[..., None] & valid_dim_mask[:, None, :]`；dimension padding 不得产生 loss 或被送入控制器。为保持当前 N1D7 的 `d=0` 行为，padding 维度的 noise 可以仍按完整 `D_model` 采样，但训练和 inference 必须使用相同 shape、相同 mask 和相同 post-update 处理，不能一侧把 padding noise 清零、另一侧保留随机值；
- 当前 `Gr00tN1d7Processor.__call__()` 已生成 `[H,D_model]` 的训练 `action_mask`，但 `process_observation()` 的 inference mask 目前主要是 horizon mask，不能假定它已经覆盖 dimension padding。TrainingRTC 实现必须显式生成/传递 `valid_dim_mask`，并让 inference 的 semantic prefix、模型 padded tensor、noise、action mask 和 decoder slicing 使用同一 layout/version。decode 前只能取 `[..., :D_sem]` 并按 modality group 切分，不能把 padded zeros 当作真实 joint/EEF action；
- 如果 `D_sem > D_model`、任一 group dimension/statistics 与 checkpoint layout 不一致、或训练和 inference 的 mask/layout 不一致，必须在加载/请求时失败。

这项检查不能只依赖 `max_action_dim` 的默认值（当前默认配置为 132），必须把实际 `D_sem`、`D_model`、group offsets、horizon 和 stats version 写入 checkpoint metadata，并加入两种 Wuji action config 的 shape regression。

### 4.1 配置和采样

在模型/训练配置中增加独立字段，建议命名为：

```text
training_rtc_enabled: bool
training_rtc_max_delay: int  # first checkpoint: 6; runtime support guard, not a silent clamp target
training_rtc_delay_sampling: "uniform" | "empirical" | "fixed"
training_rtc_delay_pmf: dict[int, float]  # fixed/empirical 校准后的 d -> probability
training_rtc_loss_mode: "postfix_only"
```

`delay=0` 样本必须保留，以保持普通同步推理能力；非零 delay 应覆盖部署测得的延迟分布。训练采样只校验 `0 <= d < H`（`d=H` 没有 postfix loss，应拒绝）；不读取也不保存 runtime 的 `s`。首版固定 `H=32`、action-token step rate `action_step_hz=30`（`dt_action=1/30 s`）；4090 实测 controller-ready latency 70--120 ms 对应 `ceil(L_ready / dt_action)=3--4` 个 action-grid 步。建议训练 delay 支持 `d=0..6`，其中 `d=2,3,4` 占主要概率，`d=0` 保留约 20%，`d=1,5,6` 作为较低概率尾部，用于覆盖启动、抖动和后续 5090 的延迟变化。具体概率应作为配置写入 checkpoint，而不是隐含在代码中。

可用于第一轮 ablation 的初始 PMF（之后用真实 latency histogram 校准）：

```text
d:  0     1     2     3     4     5     6
p:  .20   .05   .20   .25   .20   .05   .05
```

### 4.2 Prefix/noise/损失构造

沿用当前 flow matching 的约定：`t=0` 是 noise，`t=1` 是 clean action。对每个 batch 采样 `d`，并为每个样本只采样一个 postfix timestep 标量 `tau[b]`；令 `prefix_mask[b, k] = (k < d_b)`，然后：

```python
noise = randn_like(action)
tau = sample_time((B,), device=action.device)       # 每个样本一个标量 tau
token_t = torch.where(prefix_mask, 1.0, tau[:, None])
noisy = token_t[..., None] * action + (1 - token_t[..., None]) * noise
noisy = torch.where(prefix_mask[..., None], action, noisy)
target_velocity = action - noise
loss_mask = action_mask * (~prefix_mask[..., None])
```

prefix token 使用 clean action、`t=1`，loss mask 为 0；所有 postfix token 使用同一个样本级 `tau[b]`，按普通 flow matching 计算 velocity loss。不能为 postfix 的不同 horizon token 分别采样 timestep，也不能把 prefix 的 `t=1` 与 postfix 的 `tau` 混成一个 batch 标量。`action_mask`（数据中无效维度/填充）仍然有效，最终分母使用两个 mask 的乘积。需要明确记录随机种子、delay 和 `tau`，方便复现与 ablation。

这里要区分连续时间和模型的离散 bucket：插值始终使用连续 `tau`，prefix 的连续 timestep 必须是精确 `1.0`；action encoder/DiT 再通过一个统一 helper 把 `[1.0, tau]` 转成模型所需的 bucket。若保留当前 `num_timestep_buckets` 接口，必须明确 clean endpoint 的 bucket（不能因为 `noise_s=0.999` 就把 prefix 当成 `0.999`），并在 action encoder 和 DiT 中使用同一转换。该 endpoint 约定要写入 checkpoint/config，避免训练和推理不一致。

`d=0` 必须是原始 GR00T FM 的严格退化：使用相同的 `sample_time(B)`、相同的 `noise`、相同的连续 `tau` 参与插值、相同的 timestep bucket 离散化、相同的 `action_mask` 和 velocity target；此时 `prefix_mask` 全为 false、`token_t[:, :] = tau[:, None]`、`loss_mask == action_mask`，不能引入额外的 loss 权重或新的随机变量。实现上最好让 `d=0` 直接调用原始 FM 分支，并用独立 TrainingRTC 分支做 fixed-RNG 对照；应提供固定 RNG/固定输入下的逐元素 noisy trajectory、model timestep、loss 与梯度回归测试，而不只是统计意义上的近似一致。

第一版不做 postfix token 间独立采样；`tau[b]` 在该样本的全部 postfix horizon token 上共享。模型仍需支持 token-wise timestep 输入，因为同一序列内 prefix 是 1、postfix 是 `tau[b]`，但这不等于对每个 postfix token 独立随机采样。

## 5. 模型接口改动

当前 action encoder [`gr00t/model/modules/embodiment_conditioned_mlp.py`](../gr00t/model/modules/embodiment_conditioned_mlp.py) 的 `MultiEmbodimentActionEncoder.forward()` 只接受 `[B]` timestep，并复制到全部 `H` 个 token；[`gr00t/model/modules/dit.py`](../gr00t/model/modules/dit.py) 的 `TimestepEncoder`/`AdaLayerNorm` 也按 batch timestep 广播。这里把“action-token timestep”和“DiT 全局 denoising timestep”明确分成两级，并按两个实验配置实现：

### 5.1 Minimal baseline：只让 ActionEncoder token-wise

这是第一版默认方案，也是与现有 GR00T N1.7 改动最小、最容易验证的方案：

1. action encoder 接受 `[B]`（兼容旧 checkpoint）或 `[B,H]`（TrainingRTC）。TrainingRTC 的 action token 输入为 `[B,H]`，其中 prefix 行是 1、同一样本的所有 postfix 行是同一个 `tau[b]`；只有输入为 `[B]` 时才广播。
2. DiT 保持现有 batch-level timestep 和 AdaLN，不改 `TimestepEncoder`/`AdaLayerNorm` 的调制粒度。训练时 DiT 全局 timestep 使用该样本的 `tau[b]`（按现有 batch 接口传入）；推理时使用当前 Euler denoise step。prefix 的 clean 状态由 ActionEncoder 的 `t=1` 条件表达，不能把 prefix 的 1 强行塞进仍为全局标量的 AdaLN。
3. action head 只为 action encoder 构造 `[B,H]` 的 `token_t`；state token 继续沿用当前 GR00T 的全局 timestep conditioning。alternate VL-DiT 分支也保持相同的全局 timestep 接口。
4. output projection/decoder 保证输出 action token 与输入 token 的位置一一对应。
5. [`gr00t/model/modules/flowmatching_modules.py`](../gr00t/model/modules/flowmatching_modules.py) 已有 `[B,T]` timestep 的雏形，可复用其测试和编码方式，但不能假定它已经接入当前 N1D7 action head；这里的 `[B,T]` 仅用于表达 prefix=1/postfix=`tau[b]`，不是逐 token 独立随机 timestep。
6. 保持普通训练/推理调用签名和旧 checkpoint 兼容；新增字段默认关闭，旧的单 timestep 路径回归测试必须通过。

Minimal baseline 的关键假设是：TrainingRTC 所需的“哪个动作 token 已经是 clean prefix”主要由 action token 自身的 timestep embedding 和 clean action 输入决定，不要求每一层 DiT 对 prefix/postfix 使用不同 AdaLN。若该方案无法学到 prefix 保持/后缀生成，再进入 5.2，而不是默认扩大改动面。

### 5.2 Ablation：整个 DiT AdaLN token-wise

作为独立 ablation checkpoint，再让 DiT 的 `TimestepEncoder`、AdaLayerNorm 及其它 timestep 调制层支持与 hidden sequence 对齐的 `[B,L]` timestep 和 `[B,L,C]` scale/shift：state token 使用单独约定（例如当前 denoise time 或 `t=1`），prefix action token 使用 1，postfix action token 使用 `tau[b]`（推理时为当前 denoise step）。需要覆盖所有 AdaLN block、alternate VL-DiT 和 output path，并比较显存、VRAM 峰值、吞吐、收敛、postfix prefix-sensitivity 与控制指标。不要使用 prefix drift 作为 Minimal/Full 的区分指标：sampler 每一步都会 hard overwrite prefix，prefix drift 理论上被强制消除。该方案不是 minimal baseline 的隐式依赖，不能用未改造的旧 checkpoint 直接加载为等价模型。

## 6. TrainingRTC 推理采样器

新增独立的 `rtc_mode="training"`（或等价的 sampler 类），不要复用现有 `rtc_overlap_steps` 的 ramp/inpainting 语义。每次 denoising 迭代都要覆盖 prefix，避免数值积分使 prefix 漂移：

```python
x = torch.randn(B, H, D, device=device)
x = torch.where(prefix_mask[..., None], prefix, x)
for step in denoise_steps:                 # inference denoise time: 0 -> 1
    x = torch.where(prefix_mask[..., None], prefix, x)
    token_t = torch.full((B, H), step.t, device=device)
    token_t = torch.where(prefix_mask, 1.0, token_t)
    velocity = model(
        observation,
        x,
        action_timestep=token_t,   # ActionEncoder token-wise in minimal baseline
        timestep=step.t,           # DiT global timestep; sequence-valued only in 5.2 ablation
    )
    x = x + step.dt * velocity
    x = torch.where(prefix_mask[..., None], prefix, x)
```

实际代码还需处理 batch 中不同 `d`（用 `[B,H]` mask）、action mask、dtype/device、`num_inference_timesteps` 和最后一步 clamp。上式中的 `prefix` 已经是当前 observation rebase 后的 normalized action；不能从旧 normalized relative action 直接复制。TrainingRTC 模式不使用旧实现的 velocity ramp、伪逆或 inpainting 强度；现有 inference-time RTC 保留为可选 baseline。

## 7. 异步控制器和延迟估计

### 7.1 时间轴定义：`t_obs` 是语义 reference

异步流程必须以同步观测包的采样时间 `t_obs` 为 reference，而不是 inference 发起时间 `t_launch`。定义：

```text
t_obs       = image/state/EEF snapshot 的机器人时间戳
t_launch    = 将该 snapshot 提交给模型的时间（仅用于性能 telemetry）
t_ready     = 新 action chunk 完成后，已经可交给 controller 的时间
c_obs       = t_obs 时旧 chunk 的 controller execution cursor
c_ready     = t_ready 时 action chunk 已可交给 controller 的 execution cursor
t_handoff   = controller 真正切换并允许 new chunk 首个 action 生效的时间
c_handoff   = t_handoff 时 new chunk 首个 action 生效的 action-grid cursor
L_ready     = t_ready - t_obs，包含 observation age、预处理、模型、传输和 controller-ready 开销
d_actual   = c_handoff - c_obs (first action-grid index actually applied from the new chunk)
```

同一个 `t_obs` 必须绑定 image、joint/EEF state 和 execution cursor：image/state 时间偏差超过配置阈值时应等待同步或丢弃该 sample，不能用 `t_launch` 的 state 去做 relative EEF rebase。`t_launch - t_obs` 可以单独记录为 observation/preprocess age，但不能改变 prefix 的 reference frame。对相对 EEF，prefix 和该次输出的 postfix 都携带 `reference_timestamp=t_obs`；若需要换到更新 reference，必须丢弃旧结果并重新推理整段 chunk。

### 7.2 `d_cond`、committed prefix 和 `d_actual`

先测量端到端 P50/P95/P99 的 **controller-ready latency** `L_ready`，而不是只测 model forward。当前 action grid 为 30 Hz（`dt_action=1/30 s`），4090 实测 70--120 ms 对应约 3--4 个 action-grid 步；训练的 `d=6` 是为 jitter、启动开销和后续 5090 预留的尾部。运行时流程如下：

这里的 `s` 定义为 scheduler 从当前旧 chunk 提交到 controller、并在后台等待下一次 action 的实际执行步数。它决定旧 chunk 何时切换、需要多少 committed prefix，以及 `d_cond <= H-s` 是否成立；它不参与 delay PMF、prefix/noise 构造、loss mask、ActionEncoder/DiT 参数或 checkpoint 版本。改变 `s` 只需重新做 scheduler/闭环验证，不应触发模型重训。

1. 第一次请求没有旧 chunk，使用 `d_cond=0`，以观测包 `t_obs` 生成 padded `[H,D_model]`，decode 后立即形成 absolute target cache，并将 `reference_timestamp=t_obs` 一起保存。
2. 获取下一组同步观测时记录新的 `t_obs`、`c_obs` 和 image/state age。在发起 inference **之前**，根据预测的 controller-ready latency `L_ready_est(t_obs)` 计算 `d_cond_raw = ceil(L_ready_est / dt_action) + jitter_margin`。只允许在以下 guard 全部通过时发起标准 TrainingRTC 请求：
   - `0 <= d_cond_raw < H`；
   - 旧 absolute chunk 从 `c_obs` 起有足够 committed-prefix coverage；
   - scheduler 选择的 `s` 满足 `d_cond_raw <= H-s`；
   - `d_cond_raw <= checkpoint.training_rtc_max_delay`（首版 checkpoint 上限为 6）；
   - semantic/padded action layout、stats version、image/state sync 和 request context 校验通过。
   任一条件失败都必须进入明确的 `RTC_DELAY_OOD`、`RTC_CHUNK_COVERAGE`、`RTC_SCHEDULER_CONSTRAINT` 或对应数据一致性 fallback；尤其禁止把 `d_cond_raw` 静默 clamp 到 6 后继续运行。`s` 只在此处作为 scheduler 参数，不进入训练采样或 checkpoint 学习目标。
3. 将旧 absolute chunk 从 `c_obs` 开始的前 `d_cond` 个 target 登记为该请求的 committed prefix。相对 EEF 用这些 absolute target 和 **当前 `t_obs` 的 EEF pose** 重新 rebase、rot6d 编码和归一化；absolute joint 直接取同一 action-grid 对齐的 absolute prefix。请求 metadata 固定保存 `reference_timestamp=t_obs`、`raw_state_snapshot`、`c_obs`、`d_cond`、`chunk_version`、`stats_version`、semantic/padded layout 和 prefix reference。
4. 控制器继续执行旧 chunk，后台运行 TrainingRTC sampler。模型返回 padded normalized action 后，必须在 request context 的 `raw_state_snapshot@t_obs` 上对**整个** relative EEF chunk 做 relative→absolute decode；decode 后立即形成新 absolute target chunk，再计算 action-grid 对齐的 `t_ready`、`c_ready` 和可 handoff 状态。禁止在 `t_ready` 或 handoff 时使用新的 current state decode。
5. 定义 `c_handoff` 为 controller 实际能够让 new chunk 的首个 action 生效的 action-grid cursor，并记录 `t_handoff`。`d_actual = c_handoff - c_obs`，不能简单用 `c_ready - c_obs` 代替。若 ready 落在 action boundary 前、恰好 boundary 上或 boundary 后，scheduler 必须按明确的半开区间和 cursor 规则计算 `c_handoff`，见测试章节。
6. 若 `d_actual <= d_cond` 且 chunk/version 未变化，新 absolute chunk 可以切换：丢弃已执行的前 `d_actual` 行，使用从 `d_actual` 开始的结果。`[d_actual:d_cond]` 仍是模型明确 conditioning 的 committed prefix；slice、cursor、handoff 和 cache 全部操作 absolute chunk，不在切换瞬间重新 relative decode。
7. 若 `d_actual > d_cond`，该结果**不可直接切换**：`[d_cond:d_actual]` 是未被 prefix conditioning、但已经在旧 chunk 中执行的区间，不能静默 splice。继续执行旧 absolute chunk，丢弃这次输出，并以新的同步观测 `t_obs'`/cursor 重新发起请求；新请求的 `d_cond'` 根据新的 `L_ready_est(t_obs')` 重新计算，并重新执行全部 support guards。若旧 chunk 无法覆盖重算期间，则进入预定义 hold/减速/安全停止策略。
7. 若返回时 chunk/version、`c_obs` 或 `reference_timestamp` 失配，也按 stale result 丢弃处理；不得复用旧 relative prefix。所有 stale/drop/relaunch 事件要记录原因。

建议 `H=32`、30 Hz 首版训练 delay 为 `0..6`，概率主要集中在 `2..4`。部署时再由 scheduler 独立选择 `s`，根据稳定性、chunk 边界误差和吞吐调参；任何 `s` 改动都不需要重新训练 checkpoint，但必须重新验证 `d_cond <= H-s` 和旧 chunk 覆盖能力。

## 8. 分阶段实施与验收

### Phase 0：锁定数据契约

- 确认两套 modality config、每个 group 的维度、rot6d 顺序、关节限位和统计文件。
- 建立 raw absolute target、processed relative/absolute action、flatten `[H,D]` 三者的可追溯样例。
- 明确训练和部署都只进行一次 relative conversion。

### Phase 1：ActionEncoder token-wise minimal baseline

- 只改 action encoder 的 token-wise timestep；DiT/AdaLN 和 alternate VL-DiT 保持 batch-level timestep。
- 加 shape、dtype、旧 `[B]` timestep 兼容测试。
- 另开实验配置实现全 DiT AdaLN token-wise ablation，不作为主线依赖。

### Phase 2：TrainingRTC 训练路径

- 加 delay sampler、prefix construction、postfix-only loss 和配置开关。
- 验证固定 RNG/输入下 `d=0` 与原始 flow-matching 的 noisy trajectory、timestep、loss 和梯度逐元素一致。

### Phase 3：TrainingRTC sampler

- 实现每一步 prefix overwrite、不同 `d_b` 的 batch mask 和独立 `rtc_mode`。
- 与现有 inference-time RTC 做离线对照，不改变默认模式。

### Phase 4：异步 runtime

- 加旧 absolute chunk 缓存、执行游标、`t_obs/t_ready/t_handoff` 和 `c_obs/c_ready/c_handoff` 延迟测量，以及仅由 runtime scheduler 执行的 `d_cond/s` 合法性检查；其中必须包含 checkpoint `training_rtc_max_delay` guard 和 `RTC_DELAY_OOD` fallback。
- 分别实现 relative EEF `absolute cache -> t_obs rebase -> normalize -> request-context raw snapshot decode -> absolute chunk`，以及 absolute joint 直接复制；handoff 后只操作 absolute timeline。

### Phase 5：训练两个 checkpoint

- checkpoint A：EEF relative rot6d + hand absolute joint；
- checkpoint B：arm absolute joint + hand absolute joint。
- 可先冻结 VLM，仅训练 action encoder/DiT/projector/decoder，稳定后再评估解冻 backbone 的收益。

### Phase 6：对比实验

- synchronous（`d=0`）、当前 inference-time RTC、TrainingRTC；
- 延迟 sweep、不同 scheduler `s`、不同 prefix 长度和动作空间分别报告：postfix prefix-sensitivity、chunk boundary 的 position/rotation/velocity discontinuity、acceleration、jerk、task success、latency-conditioned success、throughput、端到端 controller-ready latency 和峰值 VRAM；不报告 prefix drift 作为主指标，因为 prefix 每一步被 hard overwrite。
- 固定 observation、初始 noise、`tau` 和 denoise steps，仅替换 prefix（GT-prefix、previous-model-prefix、perturbed-prefix），测量 postfix action 的变化、boundary discontinuity 和任务成功率，确认 postfix 是否真正依赖 prefix conditioning，而非只依赖 observation。
- 增加 robustness evaluation：GT-prefix（理想 absolute cache）、previous-model-prefix（上一模型输出）、perturbed-prefix（位置/rotation/joint 加受控扰动），分别统计 task success、latency-conditioned success、postfix sensitivity、position/rotation/velocity discontinuity、acceleration/jerk 和 OOD fallback 率。

## 9. 必须自动化的测试

- 两套 modality config 的 action layout 和动态 `D` 计算；
- rot6d encode/decode 及 absolute→relative→absolute round-trip；
- absolute joint 路径不会触发 relative conversion；
- relative EEF rebase 后 decode 能恢复同一 absolute target（在浮点容差内）；
- relative EEF 整个输出 chunk 只用 request 保存的 `raw_state_snapshot@t_obs` decode；故意传入 `t_ready`/handoff current state 时测试必须失败或证明该参数未被使用，且错误 reference state 会被单元测试检测为 absolute target 不一致；
- request context 缺少或错配 `reference_timestamp/raw_state_snapshot/c_obs/d_cond/chunk_version/stats_version` 中任一项时拒绝 decode/handoff；
- denoising 每一步 prefix 数值完全不变，postfix 正常更新；
- prefix loss 恒为 0，postfix 有效 token loss 非 0；
- postfix 在同一样本内所有 token 使用同一个 `tau[b]`，prefix token 严格为 `t=1`；
- `d=0` 在固定 RNG/输入下与原始 FM 的 noisy trajectory、timestep bucket、loss 和梯度逐元素一致；
- 训练侧 `d >= H`、负 `d`、越界 mask 正确报错；runtime 分别测试 `d_cond >= H`、`d_cond > H-s`、旧 chunk coverage 不足和 `d_cond > checkpoint.training_rtc_max_delay`；最后一种必须返回 `RTC_DELAY_OOD`，不得 clamp 后运行；
- `d_actual <= d_cond` 正确丢弃已执行输出并切换，`d_actual > d_cond` 严格丢弃并重算，不发生未 conditioning 的 splice；
- launch cursor/chunk version 失配时 stale result 被丢弃；
- image/state/EEF/cursor 的 `t_obs` 同步、`reference_timestamp` 传递和 controller-ready latency 计算正确；`t_launch` 只用于 telemetry；delay 使用 `dt_action/action_step_hz`，测试证明改变底层 servo frequency 不会改变 `d`；
- ready 位于 action boundary 前、恰好 boundary 上和 boundary 后的 scheduler 测试，验证 `t_handoff/c_handoff` 的 off-by-one 规则，并确认 `d_actual == c_handoff-c_obs` 而非 `c_ready-c_obs`；
- 两种 action config 都验证 `D_sem <= D_model`、group offsets、zero padding、valid-dimension mask、horizon mask；训练和 inference 的 prefix/noise/padding/action_mask 完全一致，padded dimension 不计 loss、不进入 decode/controller；
- `action_mask` 与 prefix mask 同时生效且分母不为零；`d=0` 的 padded action/noise/mask 仍与原始 FM 逐元素一致；
- 固定 observation/noise/`tau`，仅替换 GT/previous-model/perturbed prefix，验证 postfix 输出有可测的 prefix-sensitivity，并记录该 sensitivity，而不是检查已 hard-overwrite 的 prefix drift；
- minimal baseline 验证 ActionEncoder 的 `[B,H]` timestep，DiT/AdaLN 仍为 `[B]`；ablation 单独验证 state/action token 与 hidden sequence 对齐的 AdaLN timestep；
- CPU 单元测试、最小 forward shape 测试，以及 GPU 小 batch 数值稳定性测试。

## 10. 风险和决策门

- **参考系错误**：若直接复用旧 normalized relative prefix，会造成 EEF 跳变；Phase 0/4 的 absolute target cache 和 round-trip 测试是上线门槛。
- **rotation 表示不一致**：rot6d 的列/行约定、坐标系或左右手镜像不一致会使训练看似收敛但控制失败；必须复用现有 pose 类并用真实样本可视化检查。
- **ActionEncoder timestep 广播残留**：minimal baseline 允许 DiT/AdaLN 保持全局 timestep，但 action encoder 若仍把 `[B,H]` 压成 `[B]`，TrainingRTC 会退化为普通 FM；通过 hook/单元测试检查 prefix=1、postfix=`tau[b]` 的 action embedding。全 DiT AdaLN 的 token-wise 广播检查只适用于 5.2 ablation。
- **prefix 过长**：训练时 `d` 接近 `H` 会使 postfix 梯度和有效预测空间不足，因此拒绝 `d >= H`；部署时由 scheduler 另外检查 `d_cond <= H-s`，不能把这个 runtime 约束写入 checkpoint。
- **数据统计漂移**：relative 和 absolute statistics 必须分开版本化，且部署 processor 与训练 processor 使用同一份配置。

完成 Phase 4 后，才进行真实机械臂闭环测试；在此之前只允许离线 replay 和仿真验证。

## 11. 当前实现记录

- 分支：`plan/training-rtc-adaptation`；未修改 `main`。
- 已实现：ActionEncoder `[B,H]` timestep、postfix-only TrainingRTC loss、`d=0` legacy FM 随机流回归、每步 prefix hard overwrite、`RTC_DELAY_OOD` support guard、semantic/padded layout metadata、absolute target cache、`t_obs` snapshot rebase/decode、`c_handoff`/`d_actual` scheduler 和 `Gr00tPolicy` TrainingRTC options 接线。
- 已验证：action head、runtime scheduler、policy、Wuji dataset contract 共 33 项测试通过；`compileall` 与 `git diff --check` 通过。
- 环境限制：processor 的完整图像增强测试在当前隔离环境中受已有 albumentations `FractionalCenterCrop` API 不兼容影响，相关新增 layout/decode/fixture 测试通过；尚未进行真实机械臂闭环测试。
