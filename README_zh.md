<div align="center">

  <img src="media/header_compress.png" width="800" alt="NVIDIA Isaac GR00T N1.7 Header">

  <p style="font-size: 1.2em;">
    <a href="https://developer.nvidia.com/isaac/gr00t"><strong>官网</strong></a> |
    <a href="https://huggingface.co/collections/nvidia/gr00t-n17"><strong>模型</strong></a> |
    <a href="https://huggingface.co/collections/nvidia/physical-ai"><strong>数据集</strong></a> |
    <a href="https://arxiv.org/abs/2503.14734"><strong>论文</strong></a> |
    <a href="https://developer.nvidia.com/isaac"><strong>NVIDIA Isaac</strong></a> |
    <a href="FAQ.md"><strong>常见问题</strong></a>
  </p>
</div>

## 目录

- NVIDIA Isaac GR00T
- GR00T N1.7 更新内容
- 安装
- 模型检查点与 embodiment tag
- 数据格式
- 推理
- 微调
- 评估
- 贡献与支持
- 许可证
- 引用

---

## NVIDIA Isaac GR00T

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td style="width:33.33%; text-align:center;">
      <img src="media/unitree_g1.gif" style="max-width:100%; height:auto;">
    </td>
    <td style="width:33.33%; text-align:center;">
      <img src="media/agibot_g1.gif" style="max-width:100%; height:auto;">
    </td>
    <td style="width:33.33%; text-align:center;">
      <img src="media/yam.gif" style="max-width:100%; height:auto;">
    </td>
  </tr>
</table>

> 我们刚刚发布了 GR00T N1.7 Early Access，这是 GR00T N1 的最新版本，采用新的 VLM 骨干（Cosmos-Reason2-2B / Qwen3-VL），并带来更好的性能表现。

> **当前版本为 Early Access（EA）预览版。**
> 你可以下载模型、查看代码并开始基于该技术栈进行开发，但在正式 GA（General Availability）发布前，稳定性与支持保障仍然有限。
>
> **当前可用内容：**
> - 预训练 GR00T N1.7 模型权重与参考代码
> - 使用自定义机器人数据或演示数据进行微调与推理
> - 原型验证、实验与研究用途
>
> **GA 版本将提供：**
> - 带商业支持的生产级部署能力
> - 更完整的基准结果与经过充分验证的稳定特性
> - Pull Request 贡献支持
>
> 欢迎通过仓库 issue 提交反馈。

> 旧版本入口：[N1.6](https://github.com/NVIDIA/Isaac-GR00T/releases/tag/n1.6-release) | [N1.5](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release)

NVIDIA Isaac GR00T N1.7 是一个面向通用人形机器人技能的开放式视觉-语言-动作（VLA）模型。它接收语言、图像等多模态输入，在多样环境中执行操作任务。

GR00T N1.7 训练于多样化机器人数据之上，覆盖双臂、半人形与更大规模的人形数据集，并支持通过后训练适配特定 embodiment、任务和环境。

GR00T N1.7 采用 Apache 2.0 开源代码许可，可用于商业场景。它在性能上与 N1.6 相当，同时由于在预训练中加入了 2 万小时 EgoScale 人类视频数据，泛化能力与语言跟随能力更强。

GR00T N1.7 的网络结构由视觉语言基础模型和扩散 Transformer 动作头组成，后者通过连续动作去噪生成控制输出。结构示意如下：

<div align="center">
<img src="media/model-architecture.png" width="800" alt="model-architecture">
</div>

### 工作流概览

1. **准备数据**：采集机器人演示（视频、状态、动作），并转换为 [GR00T LeRobot 格式](#数据格式)。仓库已附带若干示例数据，便于快速测试。
2. **运行推理**：可直接用基础模型在[预训练 embodiment](getting_started/policy.md#--embodiment-tag) 上做 zero-shot 推理，或使用针对基准任务训练好的检查点。
3. **执行微调**：使用 [`launch_finetune.py`](gr00t/experiment/launch_finetune.py) 与自有数据、模态配置来适配你的机器人。
4. **进行评估**：先做[开环评估](#开环评估)，再在[仿真基准](#基准示例)或真实硬件上通过 [Policy API](getting_started/policy.md) 验证。
5. **部署**：将 `Gr00tPolicy` 接入你的机器人控制器，并可选地通过 [TensorRT](scripts/deployment/README.md) 加速。

## GR00T N1.7 更新内容

GR00T N1.7 在 N1.6 的基础上升级了 VLM 骨干，并改进了代码层工作流。

1. **相对 EEF 动作空间**
   N1.7 采用机器人和人类 embodiment 共享的相对末端执行器动作空间。动作表示为相对当前位姿的增量，而不是绝对目标位姿，这显著提升了泛化能力，也是其跨 embodiment 表现的重要原因。为你的机器人配置相对 EEF，可参考 [`getting_started/finetune_new_embodiment.md`](getting_started/finetune_new_embodiment.md)。

2. **人类视频预训练**
   N1.7 在多样机器人演示之外，还使用了 2 万小时 EgoScale 人类视频进行预训练。由于相对 EEF 动作表示在人类和机器人数据之间一致，模型可以把从人类视频中学到的操作先验直接迁移到机器人控制。

### 相比 N1.6 的关键变化

- **新的 VLM 骨干**：采用 Cosmos-Reason2-2B（Qwen3-VL 架构），替代 N1.6 中的 Eagle 骨干。
- 支持灵活分辨率，并以原始宽高比编码图像，无需 padding。
- 数据处理流水线更简化：`processing_gr00t_n1d7.py`。
- 新增从完整流水线导出到 ONNX 与 TensorRT 的能力，并提升推理频率。

---

## 安装

### 硬件要求

**推理：** 需要 1 块显存 16 GB 以上的 GPU，例如 RTX 4090、L40、H100、Jetson AGX Thor/Orin、DGX Spark。

**微调：** 推荐 1 块或多块显存 40 GB 以上 GPU。官方建议使用 H100 或 L40 节点以获得最佳性能。其他硬件（例如 A6000）也可以运行，但训练时间可能更长。更详细配置见 [硬件建议指南](getting_started/hardware_recommendation.md)。

**不同平台的 CUDA / Python：**
- dGPU：CUDA 12.8，Python 3.10
- Jetson Orin：CUDA 12.6，Python 3.10
- Jetson Thor：CUDA 13.0，Python 3.12
- DGX Spark：CUDA 13.0，Python 3.12

对应的安装脚本与 Dockerfile 位于 `scripts/deployment/`。完整平台矩阵见 [部署与推理指南](scripts/deployment/README.md)。

### 克隆仓库

GR00T 依赖若干 git submodule，请在克隆时一并拉取。

**注意：** `/demo_data` 中的 parquet 数据文件依赖 `git-lfs`，请先安装：

```sh
sudo apt install git-lfs && git lfs install
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
```

如果你已经克隆但没有带 submodule：

```sh
git submodule update --init --recursive
```

### 环境配置

GR00T 使用 [uv](https://github.com/astral-sh/uv) 管理依赖，建议先安装 uv：

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### dGPU（x86_64，默认）

安装 FFmpeg（`torchcodec` 默认视频后端所需）：

```sh
sudo apt-get update && sudo apt-get install -y ffmpeg
```

创建环境并安装 GR00T：

```sh
uv sync --python 3.10
```

默认安装会包含 GPU 依赖（如 `flash-attn`、TensorRT）。

验证安装：

```sh
uv run python -c "import gr00t; print('GR00T installed successfully')"
```

> **关于每次 `uv run` 都出现 `Installing flash-attn...`**
> 这是 `uv` 对 URL 固定 wheel 源进行缓存校验的已知行为，不代表在重新从源码构建。wheel 已经缓存在本地，通常只会额外耗时 2 到 3 秒，仅影响 x86_64 平台。

<details>
<summary><strong>可选：使用 pip 安装（不使用 uv）</strong></summary>

如果你更希望使用 pip/conda，请创建 Python 3.10 虚拟环境并安装：

```sh
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e .
```

注意：GPU 依赖（例如 `flash-attn`、TensorRT）可能需要手动安装；`uv` 流程会自动处理这些依赖。
</details>

> **若微调时报错 `CUDA_HOME is unset`**
> 执行一次 `bash scripts/deployment/dgpu/install_deps.sh` 配置 CUDA 路径，或手动：
>
> ```sh
> export CUDA_HOME=/usr/local/cuda
> ```

> **CUDA 13.x 用户（Thor、Spark 及其他 CUDA 13+ 平台）**
> PyTorch 2.7 将 Triton 固定到 3.3.1，而该版本无法识别 CUDA 13 的主版本号，会导致 Triton 的 `ptx_get_version()` 抛出 `RuntimeError`。请执行：
>
> ```sh
> uv run bash scripts/patch_triton_cuda13.sh
> ```

> **GB300（sm_103）用户**
> Triton 3.3.1 不支持 GB300 架构，`torch.compile` 会失败。建议改用 PyTorch eager 模式或 TensorRT 推理。

> **aarch64 视频后端**
> 在 Thor、Orin、Spark 等 aarch64 平台上，`torchcodec` 是唯一受支持的视频后端。若出现 `NotImplementedError`，请确认安装时已成功安装 `torchcodec`。

<details>
<summary><strong>DGX Spark</strong>（已在 DGX Spark GB10 上测试）</summary>

```bash
bash scripts/deployment/spark/install_deps.sh
source .venv/bin/activate
source scripts/activate_spark.sh
```

更多 Docker 与裸机细节见 [Spark 安装指南](scripts/deployment/README.md#dgx-spark-setup)。
</details>

<details>
<summary><strong>Jetson AGX Thor</strong>（已在 JetPack 7.1 上测试）</summary>

> **较老系统上的 `flash-attn` 问题**
> 例如 Ubuntu 20.04 上 `glibc < 2.35` 时，预编译 wheel 可能报：
> `ImportError: glibc_compat.so: cannot open shared object file`
> 可改为本地源码编译：
>
> ```sh
> uv pip install flash-attn==2.7.4.post1 --no-binary flash-attn --no-cache
> ```

```bash
bash scripts/deployment/thor/install_deps.sh
source .venv/bin/activate
source scripts/activate_thor.sh
```

更多细节见 [Thor 安装指南](scripts/deployment/README.md#jetson-thor-setup)。
</details>

<details>
<summary><strong>Jetson Orin</strong>（已在 JetPack 6.2 上测试）</summary>

```bash
bash scripts/deployment/orin/install_deps.sh
source .venv/bin/activate
source scripts/activate_orin.sh
```

更多细节见 [Orin 安装指南](scripts/deployment/README.md#jetson-orin-setup)。
</details>

如果你希望使用容器化环境来避免系统级依赖冲突，可参考 [Docker 安装指南](docker/README.md)。

---

## 模型检查点与 Embodiment Tag

### 检查点

| 检查点 | 类型 | Embodiment Tag | 说明 |
|---|---|---|---|
| [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) | Base | 见 [预训练 tag 列表](getting_started/policy.md#--embodiment-tag) | 基础模型（3B 参数），支持预训练 embodiment 的 zero-shot 推理，也可作为新任务微调起点 |
| [`nvidia/GR00T-N1.7-LIBERO`](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) | Finetuned | `LIBERO_PANDA` | 在 [LIBERO](https://libero-project.github.io/) 基准上微调 |
| [`nvidia/GR00T-N1.7-DROID`](https://huggingface.co/nvidia/GR00T-N1.7-DROID) | Finetuned | `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` | 在 [DROID](https://droid-dataset.github.io/) 数据集上微调 |
| [`nvidia/GR00T-N1.7-SimplerEnv-Bridge`](https://huggingface.co/nvidia/GR00T-N1.7-SimplerEnv-Bridge) | Finetuned | `SIMPLER_ENV_WIDOWX` | 在 SimplerEnv Bridge（WidowX）上微调 |
| [`nvidia/GR00T-N1.7-SimplerEnv-Fractal`](https://huggingface.co/nvidia/GR00T-N1.7-SimplerEnv-Fractal) | Finetuned | `SIMPLER_ENV_GOOGLE` | 在 SimplerEnv Fractal（Google Robot）上微调 |

旧版本入口：[N1.6 checkpoints](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.6-release) | [N1.5 checkpoints](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release)

### Embodiment Tag

每一条推理或微调命令都需要提供 `--embodiment-tag`。它决定模型使用哪套模态配置（状态键、动作键、归一化方式）。tag **不区分大小写**。

完整的预训练与后训练 tag 列表见 [Policy API 指南中的 Embodiment Tags](getting_started/policy.md#--embodiment-tag)。

---

## 数据格式

GR00T 使用基于 [LeRobot v2 数据集格式](https://github.com/huggingface/lerobot) 的变体，并额外增加 `meta/modality.json` 来描述状态、动作和视频结构。一个数据集目录大致如下：

```text
my_dataset/
  meta/
    info.json
    episodes.jsonl
    tasks.jsonl
    modality.json
  data/chunk-000/
  videos/chunk-000/
```

其中 `modality.json` 负责说明拼接后的 state/action 数组如何拆分为命名字段（例如 `x`、`y`、`z`、`gripper`），以及有哪些视频键可用。`embodiment-tag` 正是依赖它来解释数据。

### 内置示例数据集

| 数据集 | 机器人 | Embodiment Tag | 用途 |
|---|---|---|---|
| `demo_data/droid_sample` | DROID（3 条轨迹） | `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` | Zero-shot 或 DROID 微调模型推理 |
| `demo_data/libero_demo` | LIBERO Panda（5 条轨迹） | `LIBERO_PANDA` | 使用 LIBERO 微调检查点推理 |
| `demo_data/simplerenv_bridge_sample` | WidowX | `SIMPLER_ENV_WIDOWX` | 使用 SimplerEnv Bridge 微调检查点推理 |
| `demo_data/simplerenv_fractal_sample` | Google Robot | `SIMPLER_ENV_GOOGLE` | 使用 SimplerEnv Fractal 微调检查点推理 |
| `demo_data/cube_to_bowl_5` | SO100 机械臂（5 条轨迹） | `NEW_EMBODIMENT` | 自定义 embodiment 微调示例 |
| `demo_data/cube_to_bowl_5_with_mask` | 带逐帧 mask 的 SO100 | `NEW_EMBODIMENT` | [基于 mask 的背景抑制](examples/mask-guided-background-suppression/README.md) 示例 |

下载更多 DROID 示例轨迹：

```sh
python scripts/download_droid_sample.py --num-episodes 10
```

### 使用你自己的数据

将演示数据转换为上述格式即可。如果你的原始数据是 LeRobot v3，可使用：

```sh
python scripts/lerobot_conversion/convert_v3_to_v2.py
```

更完整的数据结构说明见 [数据准备指南](getting_started/data_preparation.md)。

---

## 推理

### Zero-Shot 推理（基础模型）

仓库内置的 `demo_data/droid_sample` 可以直接与基础模型一起使用，无需微调，也无需手动下载检查点：

```bash
uv run python scripts/deployment/standalone_inference_script.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --traj-ids 1 2 \
    --inference-mode pytorch \
    --action-horizon 8
```

该命令会在 2 条 DROID 轨迹上执行开环推理，并将预测动作与真实动作进行对比。首次运行时会从 HuggingFace 自动下载基础模型（约 6 GB）。

### 微调模型推理

对于 post-train embodiment，请使用对应的微调检查点。多数微调模型（例如 DROID、SimplerEnv）可直接通过 HuggingFace 模型 ID 使用，无需手动下载：

```bash
uv run python scripts/deployment/standalone_inference_script.py \
    --model-path nvidia/GR00T-N1.7-DROID \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --traj-ids 1 2 \
    --inference-mode pytorch \
    --action-horizon 8
```

部分模型（如 LIBERO）在 HuggingFace 仓库中采用嵌套目录，不能直接把子路径写入 `--model-path`，因此需要先下载：

```bash
uv run hf download nvidia/GR00T-N1.7-LIBERO \
    --include "libero_10/config.json" "libero_10/embodiment_id.json" \
    "libero_10/model-*.safetensors" "libero_10/model.safetensors.index.json" \
    "libero_10/processor_config.json" "libero_10/statistics.json" \
    --local-dir checkpoints/GR00T-N1.7-LIBERO
```

```bash
uv run python scripts/deployment/standalone_inference_script.py \
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
    --dataset-path demo_data/libero_demo \
    --embodiment-tag LIBERO_PANDA \
    --traj-ids 0 1 2 \
    --inference-mode pytorch \
    --action-horizon 8
```

### Server-Client 推理（部署推荐）

实际部署或仿真评估推荐使用 server-client 架构：策略运行在 GPU 服务器上，轻量客户端通过 ZMQ 发送观测并接收动作。

**终端 1：启动策略服务**

```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --device cuda:0
```

**终端 2：作为客户端执行开环评估**

```bash
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --host 127.0.0.1 \
    --port 5555 \
    --traj-ids 1 2 \
    --action-horizon 8
```

> 如果报 `ZMQError: Address already in use`，说明默认端口 `5555` 已被占用，请改用 `--port <other_port>`。

如需接入真实机器人（例如 DROID 硬件），请参考 [examples/DROID/README.md](examples/DROID/README.md)。如需更快推理，可查看 [部署与推理指南](scripts/deployment/README.md) 中的 TensorRT 方案。

完整的输入输出格式、批量推理和常见问题可参考 [Policy API 指南](getting_started/policy.md)。

---

## 微调

### 复现实验基准结果

每个基准都提供了独立 README，包含数据下载、微调与评估步骤：

| 基准 | Embodiment | 指南 |
|---|---|---|
| LIBERO | `LIBERO_PANDA` | [examples/LIBERO/README.md](examples/LIBERO/README.md) |
| SimplerEnv（Fractal） | `SIMPLER_ENV_GOOGLE` | [examples/SimplerEnv/README.md](examples/SimplerEnv/README.md) |
| SimplerEnv（Bridge） | `SIMPLER_ENV_WIDOWX` | [examples/SimplerEnv/README.md](examples/SimplerEnv/README.md) |
| SO100 | `NEW_EMBODIMENT` | [examples/SO100/README.md](examples/SO100/README.md) |

### 人形机器人全身控制（SONIC）

GR00T N1.7 通过 `UNITREE_G1_SONIC` embodiment tag 支持整机人形控制，并与 [GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) 控制器集成。在这个工作流中，VLA 先预测紧凑的 latent action token，再由学习式全身控制器解码为全身关节命令，包括腿、手臂与手部，实现语言条件下的端到端协调操控与移动。

完整的采集、微调与部署流程见 [GR00T-WholeBodyControl 仓库](https://github.com/NVlabs/GR00T-WholeBodyControl)：

- [数据采集](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/data_collection.html)：使用 SONIC + VR 进行遥操作演示采集
- [VLA Workflow](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html)：在采集数据上微调 Isaac-GR00T N1.7 并部署
- [VLA Inference](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_inference.html)：运行 PolicyServer + SONIC decoder 实时控制

> `UNITREE_G1` embodiment tag 与 [decoupled WBC](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/decoupled_wbc) 兼容，但完整的采集-微调-部署端到端流程仅支持 GEAR-SONIC（`UNITREE_G1_SONIC`）。

### 在你自己的机器人上微调（`NEW_EMBODIMENT`）

若要使用自定义机器人数据与配置微调，请参考详细教程 [`getting_started/finetune_new_embodiment.md`](getting_started/finetune_new_embodiment.md)。

请确保输入数据符合 [GR00T LeRobot 格式](#数据格式)，并通过 `--modality-config-path` 提供模态配置。

**单卡：**

```bash
CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/cube_to_bowl_5 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO100/so100_config.py \
    --num-gpus 1 \
    --output-dir /tmp/test_finetune \
    --max-steps 2000 \
    --global-batch-size 32 \
    --dataloader-num-workers 4
```

**多卡（例如 8xH100）：**

```bash
uv run torchrun --nproc_per_node=8 --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/cube_to_bowl_5 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO100/so100_config.py \
    --num-gpus 8 \
    --output-dir /tmp/test_finetune_8gpu \
    --max-steps 2000 \
    --global-batch-size 32 \
    --dataloader-num-workers 4
```

将 `demo_data/cube_to_bowl_5` 和 `examples/SO100/so100_config.py` 替换为你自己的数据集与模态配置即可。完整示例见 [`examples/SO100`](examples/SO100/README.md)。

> 请使用 `uv run torchrun`，不要直接裸跑 `torchrun`，这样才能确保使用正确的虚拟环境。
> 如需启用 Weights & Biases 日志，请加上 `--use-wandb`。若需要更完整训练配置，可改用 `gr00t/experiment/launch_train.py`。

### 训练建议

- 尽可能在硬件允许的范围内增大 batch size，并训练几千步。
- 由于图像增强包含非确定性，重复运行间可能有 5% 到 6% 波动，比较结果时需要考虑这一点。
- `--state_dropout_prob` 用于训练时随机丢弃状态输入，以增强泛化并减少对状态的依赖：
  - 模型默认值：0.8
  - finetune CLI 默认值：0.2
  - 配置位置：`gr00t/configs/finetune_config.py`
  - 官方基准脚本会按任务覆盖默认值：LIBERO 10-Long 为 0.2，SimplerEnv Bridge 为 0.8，SimplerEnv Fractal 为 0.5
  - 如果你的任务高度依赖本体状态，可适当调低这个值

---

## 评估

### 开环评估

将预测动作与数据集中的真实动作进行对比：

```bash
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path <DATASET_PATH> \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path <CHECKPOINT_PATH> \
    --traj-ids 0 \
    --action-horizon 16
```

命令会在 `/tmp/open_loop_eval/traj_{traj_id}.jpeg` 生成可视化图，展示真实动作、预测动作以及 MSE 指标。你也可以通过 `--save-plot-path <dir>` 指定自定义输出目录。

### 闭环评估

通过 server-client 架构在仿真或真实机器人上测试模型：

```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path <CHECKPOINT_PATH> \
    --device cuda:0 \
    --host 0.0.0.0 --port 5555
```

```python
from gr00t.policy.server_client import PolicyClient

policy = PolicyClient(host="localhost", port=5555)
env = YourEnvironment()
obs, info = env.reset()
action, info = policy.get_action(obs)
obs, reward, done, truncated, info = env.step(action)
```

**使用 ReplayPolicy 调试：**
如果你想先验证环境接线是否正确，而不是立刻加载训练模型，可以在启动服务端时提供 `--dataset-path <DATASET_PATH>` 并省略 `--model-path`，让服务端直接回放数据集中的录制动作。

更多输入输出格式、批量推理和排障信息见 [Policy API 指南](getting_started/policy.md)。

### 基准示例

我们支持通过 server-client 架构在公开基准上进行评估。策略服务端复用项目根目录下的 uv 环境；不同仿真客户端则各自有独立安装脚本。

你可以先运行依赖检查脚本确认评估环境是否准备就绪：

```bash
uv run python scripts/eval/check_sim_eval_ready.py
```

**Zero-shot：**

- [DROID](examples/DROID/README.md)：真实 DROID 机器人；也可使用微调检查点 `nvidia/GR00T-N1.7-DROID`

**Finetuned：**

- [DROID](examples/DROID/README.md)：真实 DROID 机器人
- [LIBERO](examples/LIBERO/README.md)：LIBERO 基准（Franka Panda）
- [SimplerEnv](examples/SimplerEnv/README.md)：Google Robot（Fractal）与 WidowX（Bridge）
- [SO100](examples/SO100/README.md)：SO100 自定义 embodiment 工作流

<details>
<summary><strong>新增一个仿真基准</strong></summary>

每个仿真基准都以 gym `env_name` 的形式注册环境，命名格式为 `{prefix}/{task_name}`，例如：
`libero_sim/LIVING_ROOM_SCENE2_put_soup_in_basket`

评估框架会基于前缀，在 [`gr00t/eval/sim/env_utils.py`](gr00t/eval/sim/env_utils.py) 中查找对应的 `EmbodimentTag`。

> **注意：**
> `env_name` 前缀与 `EmbodimentTag` 的字符串值经常并不相同。例如 `libero_sim` 映射到 `EmbodimentTag.LIBERO_PANDA`，不要假设它们名字一致。

新增步骤：

1. 在 `gr00t/eval/sim/env_utils.py` 的 `ENV_PREFIX_TO_EMBODIMENT_TAG` 中添加映射。
2. 如果同一基准有多个前缀（如 `my_benchmark_v1`、`my_benchmark_v2`），它们必须映射到同一个 `EmbodimentTag`。
3. 在 `tests/gr00t/eval/sim/test_env_utils.py` 中增加对应测试，并更新 `test_all_known_prefixes_present`。
</details>

---

## 贡献与支持

### 贡献

在 Early Access 阶段，仓库暂不接受 Pull Request，以便代码库先完成稳定化。如果你遇到问题或有建议，请在仓库中提交 [Issue](https://github.com/NVIDIA/Isaac-GR00T/issues)。

### 支持

Early Access 阶段的支持为 best-effort，后续会继续迭代并朝更稳定的 GA 版本推进。

## 许可证

- **代码：** Apache 2.0，见 [LICENSE](LICENSE)
- **模型权重：** [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)

## 引用

[论文页面](https://research.nvidia.com/labs/lpr/publication/gr00tn1_2025/)

```bibtex
@inproceedings{gr00tn1_2025,
  archivePrefix = {arxiv},
  eprint     = {2503.14734},
  title      = {{GR00T} {N1}: An Open Foundation Model for Generalist Humanoid Robots},
  author     = {NVIDIA and Johan Bjorck and Fernando Castañeda, Nikita Cherniadev and Xingye Da and Runyu Ding and Linxi "Jim" Fan and Yu Fang and Dieter Fox and Fengyuan Hu and Spencer Huang and Joel Jang and Zhenyu Jiang and Jan Kautz and Kaushil Kundalia and Lawrence Lao and Zhiqi Li and Zongyu Lin and Kevin Lin and Guilin Liu and Edith Llontop and Loic Magne and Ajay Mandlekar and Avnish Narayan and Soroush Nasiriany and Scott Reed and You Liang Tan and Guanzhi Wang and Zu Wang and Jing Wang and Qi Wang and Jiannan Xiang and Yuqi Xie and Yinzhen Xu and Zhenjia Xu and Seonghyeon Ye and Zhiding Yu and Ao Zhang and Hao Zhang and Yizhou Zhao and Ruijie Zheng and Yuke Zhu},
  month      = {March},
  year       = {2025},
  booktitle  = {ArXiv Preprint},
}
```
