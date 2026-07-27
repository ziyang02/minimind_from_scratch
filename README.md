# MiniMind from Scratch（NinjaMind）

[![CI](https://github.com/ziyang02/minimind_from_scratch/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/ziyang02/minimind_from_scratch/actions/workflows/ci.yml?query=branch%3Amaster)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

一个面向学习与 MLE 工程实践的、可测试且可复现的 Decoder-only Transformer 项目：从 Byte-level BPE、预训练、SFT/LoRA，到 DPO/PPO/GRPO、KV Cache 推理、DDP 和可复现实验，尽量用小而完整的代码串起 LLM 生命周期。

> **先说明边界：**这是教学与工程复现项目，不是已经训练好的可用大模型，也不声称发明了 Transformer、LoRA、DPO、PPO 或 GRPO。artifact 记录的 tiny 模型只训练了 2 个 optimizer step，生成结果已经退化，不能用于证明回答质量。

代码仓库：[ziyang02/minimind_from_scratch](https://github.com/ziyang02/minimind_from_scratch)

## 当前实现状态

| 能力 | 状态 | 已验证边界 |
|---|---|---|
| Decoder-only Transformer | 已实现 | RMSNorm、RoPE/可选 YaRN、MHA/GQA/MQA、SwiGLU、严格 causal mask、左 padding、Dense/MoE 均有 CPU 测试 |
| Hugging Face 接口 | 已实现 | `PreTrainedModel`、`GenerationMixin`、legacy/Dynamic/Static KV Cache，以及 HF 本地目录保存/回读 logits round-trip 已测试 |
| Tokenizer 与数据集 | 已实现 | Byte-level BPE、ChatML 风格模板、Pretrain/SFT/DPO/RLAIF/AgentRL dataset；AgentRL 目前只有 dataset，没有 trainer |
| Pretrain / SFT / LoRA / DPO | 已实现 | Pretrain/SFT/LoRA 支持去重分组切分、token-weighted validation、曲线 artifact 与单进程 CPU 精确 resume 回归；DPO 训练链路已验证但尚无精确 resume |
| PPO / GRPO | 已实现实验版 | reference policy、KL、generated-token mask、GAE/group advantage 已测试；reward 仍是 toy containment rule |
| torchrun / DDP | 已实现 | 双进程 CPU/Gloo 已实际跑通 Pretrain、DPO、PPO、GRPO；CUDA/NCCL 与双 GPU 性能未验证 |
| 流式推理 CLI | 已实现 | 本地随机模型 smoke、base/SFT/LoRA `.pth`、本地 HF 目录、KV Cache/no-cache 路径 |
| Gradio WebUI | 已实现，本地链路已验证 | CPU 浏览器中完成一次 prompt/streamed response 并保存截图；未做部署、并发或稳定性验收 |
| 测试与 CI | 已实现 | 2026-07-27 当前工作区 `ruff` 通过、`83 passed, 1 CUDA skip`；`master` 的 GitHub Actions 已在远端通过 |
| Attention benchmark | 已实现 | MHA/GQA/MQA + cache/no-cache 的 tiny CPU benchmark 已运行；没有 CUDA 数据 |
| 长训练与模型质量 | 部分完成 | 已有 validation CE/perplexity、best/latest checkpoint、曲线、held-out SFT generation EM 工具及 0.49M 参数 CPU convergence pilot；尚无正式长训练、可信准确率或可用模型 |

## Pipeline

```mermaid
flowchart LR
    RAW["JSONL text / conversations"] --> TOK["Byte-level BPE tokenizer"]
    TOK --> PT["Pretrain<br/>next-token prediction"]
    PT --> SFT["SFT<br/>assistant-only loss mask"]
    SFT --> LORA["LoRA adapters"]
    SFT --> DPO["DPO<br/>chosen vs rejected"]
    SFT --> PPO["PPO<br/>critic + GAE + KL"]
    SFT --> GRPO["GRPO<br/>group advantage + KL"]
    PT --> INF["Streaming inference"]
    SFT --> INF
    LORA --> INF
    DPO --> INF
    PPO --> INF
    GRPO --> INF
    INF --> CLI["CLI"]
    INF --> WEB["Gradio WebUI"]
    UTILS["trainer_utils<br/>AMP · accumulation · DDP · checkpoint"] -.-> PT
    UTILS -.-> SFT
    UTILS -.-> LORA
    UTILS -.-> DPO
    UTILS -.-> PPO
    UTILS -.-> GRPO
```

## 模型结构

NinjaMind 是可配置的 Decoder-only causal LM。一个 block 的主路径为：

```text
token embedding
  -> RMSNorm -> causal self-attention (MHA / GQA / MQA + RoPE) -> residual
  -> RMSNorm -> SwiGLU Dense FFN 或 top-k MoE -> residual
  -> final RMSNorm -> tied LM head
```

- **RMSNorm**：低精度输入时用 FP32 计算归一化统计，再转换回输入 dtype。
- **RoPE / YaRN**：RoPE table 从 config 延迟构建，不写入 checkpoint；可选 YaRN scaling 用于长上下文实验。
- **MHA / GQA / MQA**：`num_attention_heads` 与 `num_key_value_heads` 独立配置，KV head 在 attention 内正确展开。
- **Mask 与左 padding**：严格 causal mask；position ID 从二维 attention mask 推导，使 batched 左填充与逐样本结果一致。
- **KV Cache**：支持 legacy tuple，以及 Transformers 的 Dynamic/Static cache；增量 logits 与全量重算有回归测试。
- **SwiGLU**：Dense FFN 使用 gate/up/down projection。
- **MoE**：支持 top-k expert routing、归一化 route weight 和 router auxiliary loss；top-k token/weight 配对、padding 排除和梯度均有测试。
- **Hugging Face 兼容**：继承 `PreTrainedModel` / `GenerationMixin`，支持 `generate()`、本地 `save_pretrained()` / `from_pretrained()`。

模型规模由 CLI 参数决定。实验 artifact 使用的是 27,376 参数 tiny 模型，不代表默认配置或生产配置。

## 安装与 5 分钟 CPU 快速开始

要求 Python `>=3.10`。推荐使用 [uv](https://docs.astral.sh/uv/)：

```bash
git clone https://github.com/ziyang02/minimind_from_scratch.git
cd minimind_from_scratch
uv sync --frozen --extra cpu --extra dev

# 离线 tiny 随机模型推理，不下载权重
uv run python scripts/run_model.py

# 快速测试
uv run pytest -q

# 真实 trainer 的 1-step CPU pipeline；产物写到 /tmp，不污染仓库
uv run python scripts/smoke_train.py \
  --steps 1 \
  --output-dir /tmp/minimind-smoke \
  --artifact /tmp/minimind-smoke.json
```

`cpu` 与 `cu130` 是互斥 accelerator extras：Linux CI/CPU 快速开始只解析官方 `torch+cpu` wheel；`cu130` 仅供匹配 CUDA 13.0 的未验证 GPU 环境。macOS 下 `cpu` 会回退到 PyPI 的原生 wheel。

首次安装 PyTorch 的下载时间取决于网络；安装完成后，上面的模型/测试/smoke 都可在 CPU 环境执行。没有 uv 时可使用：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python scripts/run_model.py
pytest -q
```

## 用 demo 数据跑完整训练链路

仓库已经包含可直接使用的 tokenizer。若想从 demo 文本重新训练一个 tokenizer，可运行：

```bash
uv run python scripts/train_tokenizer.py \
  --data_path dataset/demo/pretrain_demo.jsonl \
  --vocab_size 6400 \
  --out_dir /tmp/minimind-tokenizer
```

tiny corpus 实际能学到的词表可能小于 `6400`；后续 trainer 需要通过 `--tokenizer_dir /tmp/minimind-tokenizer` 显式使用它。以下命令使用仓库自带 `tokenizer/`，模型配置与已保存的 smoke artifact 一致。

### 1. Pretrain

```bash
uv run python trainer/train_pretrain.py \
  --data_path dataset/demo/pretrain_demo.jsonl \
  --tokenizer_dir tokenizer \
  --out_dir out/demo \
  --hidden_size 32 \
  --num_hidden_layers 1 \
  --num_attention_heads 4 \
  --num_key_value_heads 2 \
  --max_length 96 \
  --batch_size 2 \
  --max_steps 2 \
  --log_interval 1 \
  --device cpu \
  --seed 42
```

输出：latest/resumable `out/demo/pretrain_32.pth`、compact best validation `out/demo/pretrain_32_best.pth`，以及 `out/demo/metrics/pretrain_*` 指标文件。

### 2. SFT

SFT 只在 assistant response token 上计算 loss。

```bash
uv run python trainer/train_sft.py \
  --data_path dataset/demo/sft_demo.jsonl \
  --tokenizer_dir tokenizer \
  --init_from out/demo/pretrain_32.pth \
  --out_dir out/demo \
  --hidden_size 32 \
  --num_hidden_layers 1 \
  --num_attention_heads 4 \
  --num_key_value_heads 2 \
  --max_length 96 \
  --batch_size 2 \
  --max_steps 2 \
  --log_interval 1 \
  --device cpu \
  --seed 42
```

输出：latest/resumable `out/demo/full_sft_32.pth`、compact best validation `out/demo/full_sft_32_best.pth`，以及 `out/demo/metrics/sft_*` 指标文件。

训练后应同时检查 held-out generation，而不只看 teacher-forced CE。下面的命令复用训练时完全相同的数据、validation fraction 和 split seed，只评估每条留出对话最后一个 assistant target：

```bash
uv run python scripts/evaluate_sft.py \
  --checkpoint out/demo/full_sft_32_best.pth \
  --tokenizer-dir tokenizer \
  --data dataset/demo/sft_demo.jsonl \
  --max-length 96 \
  --validation-fraction 0.1 \
  --split-seed 42 \
  --device cpu \
  --max-new-tokens 64 \
  --limit 0 \
  --output out/demo/metrics/sft_generation_evaluation.json
```

评估的 `--data`、`--max-length`、validation fraction、split seed 和 tokenizer 必须与训练一致。它固定使用 greedy decoding，并输出 strict exact match 与保守 normalized exact match；后者只统一 Unicode NFC、换行编码和首尾空白，不放宽大小写、标点、内部空格或数字表达。`--limit 0` 表示评估全部留出样本；`max_new_tokens` 截短 target 会记为不匹配。当前 CLI 接收完整 Pretrain/SFT checkpoint，不直接接收 LoRA-only adapter。

### 3. LoRA

LoRA 默认注入 attention 的 `q_proj/k_proj/v_proj/o_proj`，冻结基座，只训练 adapter。

```bash
uv run python trainer/train_lora.py \
  --data_path dataset/demo/sft_demo.jsonl \
  --tokenizer_dir tokenizer \
  --init_from out/demo/full_sft_32.pth \
  --out_dir out/demo \
  --lora_rank 2 \
  --lora_alpha 4 \
  --hidden_size 32 \
  --num_hidden_layers 1 \
  --num_attention_heads 4 \
  --num_key_value_heads 2 \
  --max_length 96 \
  --batch_size 2 \
  --max_steps 2 \
  --log_interval 1 \
  --device cpu \
  --seed 42
```

输出：推理 adapter `out/demo/lora_32.pth`、best adapter `out/demo/lora_32_best.pth`，以及用于精确续训的完整状态 `out/demo/lora_32_train.pth`。adapter checkpoint 保存 rank、alpha、targets、基座路径和训练参数；代码也支持把 LoRA 合并回基座权重。

### 4. DPO

DPO 使用 trainable policy 与 frozen reference，分别计算 chosen/rejected assistant response 的 masked sequence log-prob：

```text
L_DPO = -log sigmoid(beta * ((log pi(chosen) - log pi(rejected))
                             - (log ref(chosen) - log ref(rejected))))
```

```bash
uv run python trainer/train_dpo.py \
  --data_path dataset/demo/dpo_demo.jsonl \
  --tokenizer_dir tokenizer \
  --init_from out/demo/full_sft_32.pth \
  --out_dir out/demo \
  --beta 0.1 \
  --hidden_size 32 \
  --num_hidden_layers 1 \
  --num_attention_heads 4 \
  --num_key_value_heads 2 \
  --max_length 96 \
  --batch_size 2 \
  --max_steps 2 \
  --log_interval 1 \
  --device cpu \
  --seed 42
```

输出：`out/demo/dpo_32.pth`。默认 reference 是 `--init_from` 的冻结副本，也可通过 `--reference_from` 指定另一 checkpoint。日志包含 preference accuracy、chosen/rejected implicit reward 和 margin。

### 5. PPO

PPO 包含 frozen reference、KL penalty、critic、masked GAE、clipped policy/value objective。当前 reward 只是“生成文本是否包含参考答案”的 toy rule，不是训练好的 reward model。

```bash
uv run python trainer/train_ppo.py \
  --data_path dataset/demo/rl_demo.jsonl \
  --tokenizer_dir tokenizer \
  --init_from out/demo/full_sft_32.pth \
  --out_dir out/demo \
  --hidden_size 32 \
  --num_hidden_layers 1 \
  --num_attention_heads 4 \
  --num_key_value_heads 2 \
  --max_prompt_len 64 \
  --max_new_tokens 2 \
  --batch_size 1 \
  --ppo_epochs 1 \
  --max_steps 1 \
  --log_interval 1 \
  --device cpu \
  --seed 42
```

输出：`out/demo/ppo_32.pth`。

### 6. GRPO

GRPO 不使用 critic；每个 prompt 采样一组 completion，在组内归一化 reward，并用 frozen reference 提供 token-level KL penalty。

```bash
uv run python trainer/train_grpo.py \
  --data_path dataset/demo/rl_demo.jsonl \
  --tokenizer_dir tokenizer \
  --init_from out/demo/full_sft_32.pth \
  --out_dir out/demo \
  --hidden_size 32 \
  --num_hidden_layers 1 \
  --num_attention_heads 4 \
  --num_key_value_heads 2 \
  --max_prompt_len 64 \
  --max_new_tokens 2 \
  --batch_size 1 \
  --group_size 2 \
  --update_epochs 1 \
  --max_steps 1 \
  --log_interval 1 \
  --device cpu \
  --seed 42
```

输出：`out/demo/grpo_32.pth`。

当前 tokenizer 渲染出的 RL demo prompt 均为 59 tokens，因此两个 RL 示例使用 `--max_prompt_len 64`，避免 smoke 时截掉末尾的 assistant generation header。

上述命令是代码路径 smoke，不是推荐超参数，也不能产生有用模型。若下载正式数据，请先检查磁盘空间和数据许可证：

```bash
uv run python scripts/download_dataset.py
uv run python scripts/download_dataset.py --files dpo.jsonl rlaif.jsonl
```

下载源为 [jingyaogong/minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset)。正式数据不会提交到本仓库。该数据集卡同时列出 Apache-2.0 与 CC-BY-NC-2.0，且数据来自多个来源；这不表示每个文件都可任选 Apache-2.0。使用正式数据前应核对具体文件及原始来源，未确认更宽松的文件级授权前按署名、非商业限制保守处理。

## Validation、训练曲线与断点续训

Pretrain/SFT/LoRA 默认执行 `--validation_fraction 0.1`。切分前会按训练实际消费的字段精确去重；SFT 会把相同角色/非 assistant 内容对话骨架下的不同 assistant 内容放在同一侧，避免 prompt 泄漏。切分由 SHA-256 与 seed 决定，不依赖 JSONL 行顺序或进程全局 RNG。fraction 按完整 group 选择最接近的比例，因此实际比例可能与请求值略有差异；正 fraction 至少需要两个 unique group。当前 demo 的实际结果是：Pretrain `300 raw → 185 unique → 167 train / 18 validation`，SFT `100 → 90 / 10`。传 `--validation_fraction 0` 可关闭 validation，但仍会去重。

Validation CE 按所有有效 target token 的 NLL 总和/数量计算，不平均 batch loss；perplexity 为该纯 CE 的指数，不混入 MoE router auxiliary loss。每次运行会原子更新：

```text
OUT_DIR/metrics/pretrain_metrics.json
OUT_DIR/metrics/pretrain_metrics.csv
OUT_DIR/metrics/pretrain_ce.svg
```

SFT/LoRA 使用相同命名规则。JSON 包含 split fingerprint、训练参数、模型配置和 epoch 0/各 epoch 指标；CSV 与 JSON history 对齐，SVG 同时显示 train/validation CE。

单进程 Pretrain/SFT 的 latest checkpoint 与 LoRA 的 `*_train.pth` 是 `format_version=2` 完整训练状态，包含模型、AdamW、AMP scaler、epoch/batch cursor、原始 LR horizon、Python/CPU RNG，以及硬件可用时的 MPS/所有可见 CUDA generator 状态。长任务建议增加 `--save_interval N`，中断后使用**完全相同**的 stage、模型、tokenizer、数据切分、max length、LR/grad clip、device、batch、accumulation、seed、epochs/max_steps 参数，并额外传 `--resume`：

```bash
uv run python trainer/train_pretrain.py \
  --data_path dataset/demo/pretrain_demo.jsonl \
  --out_dir out/resume_demo \
  --epochs 10 \
  --batch_size 4 \
  --save_interval 50 \
  --device cpu

# 中断后；其他训练参数必须保持相同
uv run python trainer/train_pretrain.py \
  --data_path dataset/demo/pretrain_demo.jsonl \
  --out_dir out/resume_demo \
  --epochs 10 \
  --batch_size 4 \
  --save_interval 50 \
  --device cpu \
  --resume out/resume_demo/pretrain_512.pth
```

`--init_from` 表示只加载权重并开始新阶段，`--resume` 表示严格恢复同一训练运行；raw/v1 checkpoint（包括 compact `_best.pth`）用于 `--resume` 会明确报错。关闭 validation 时不会生成 best；若训练后 validation CE 没改善，best 也可能是 epoch 0 baseline。当前精确 resume 实现支持单进程 CPU/CUDA/MPS：CPU 已有连续/中断状态逐项一致回归，MPS RNG 路径有接口回归但尚无可用 MPS 硬件实跑，CUDA RNG 测试因当前机器无 CUDA 而跳过。跨 PyTorch 版本、硬件或算子实现不承诺 bitwise 一致。DDP 与 DPO/PPO/GRPO 仍只支持权重初始化或阶段末 checkpoint，不能声称精确续训。

## 什么时候需要租 GPU

当前 demo、测试、数据切分、指标评估和亚百万参数本地 pilot 都不需要租卡；先用 CPU 把数据质量、留出集 EM 和 checkpoint 路径跑对。只有在换成更大且完成许可审查的数据、准备多 epoch 正式训练时，才建议先短租一张 16–24GB GPU 做 CUDA/AMP smoke，确认显存、吞吐和恢复后再延长租期。GPU 只能缩短训练时间，不能修复小样本重复、标签单一或验证 EM 偏低的问题。

## torchrun / DDP

公共 trainer 支持：process-group 初始化与清理、`DistributedSampler`、每 epoch `set_epoch()`、rank-0 日志/checkpoint、梯度累积时 `no_sync()`、尾 accumulation window 更新，以及单进程 CPU/MPS/CUDA 回退。

下面是已实际验证的双进程 CPU/Gloo 形式；显式 loopback 地址和端口也可避开某些受限环境中 `--standalone` 的 rendezvous 问题：

```bash
uv run torchrun \
  --nproc-per-node=2 \
  --master-addr=127.0.0.1 \
  --master-port=29555 \
  trainer/train_pretrain.py \
  --data_path dataset/demo/pretrain_demo.jsonl \
  --tokenizer_dir tokenizer \
  --out_dir out/ddp \
  --hidden_size 32 \
  --num_hidden_layers 1 \
  --num_attention_heads 4 \
  --num_key_value_heads 2 \
  --max_length 32 \
  --batch_size 2 \
  --max_steps 1 \
  --log_interval 1 \
  --device cpu \
  --dist-backend gloo \
  --seed 42
```

DPO 的 DDP + `no_sync` accumulation 也已按下列形态验证：

```bash
uv run torchrun \
  --nproc-per-node=2 \
  --master-addr=127.0.0.1 \
  --master-port=29556 \
  trainer/train_dpo.py \
  --data_path dataset/demo/dpo_demo.jsonl \
  --tokenizer_dir tokenizer \
  --init_from out/demo/full_sft_32.pth \
  --out_dir out/ddp \
  --hidden_size 32 \
  --num_hidden_layers 1 \
  --num_attention_heads 4 \
  --num_key_value_heads 2 \
  --max_length 64 \
  --batch_size 1 \
  --accumulation_steps 2 \
  --max_steps 1 \
  --device cpu \
  --dist-backend gloo
```

单机双 GPU/NCCL 的命令形式如下，但本项目**尚未实际验证**，不能据此声称加速收益。仓库提供的 `cu130` extra 对应 PyTorch CUDA 13.0 wheel；只应在驱动/平台匹配时使用，其他 CUDA 版本需按 PyTorch 官方说明调整 index：

```bash
uv sync --frozen --extra cu130 --extra dev
uv run torchrun \
  --nproc-per-node=2 \
  --master-addr=127.0.0.1 \
  --master-port=29557 \
  trainer/train_pretrain.py \
  --data_path dataset/demo/pretrain_demo.jsonl \
  --device cuda \
  --dist-backend nccl
```

## 流式推理 CLI

无 checkpoint 时，`scripts/run_model.py` 会创建 tiny 随机模型，只用于检查离线推理链路：

```bash
uv run python scripts/run_model.py
```

加载前面生成的 SFT checkpoint 并逐 token 累积输出：

```bash
uv run python main.py \
  --tokenizer-dir tokenizer \
  --checkpoint out/demo/full_sft_32.pth \
  --prompt 'What is 2 plus 2?' \
  --device cpu \
  --max-new-tokens 32 \
  --temperature 0 \
  --top-k 0
```

在 SFT 基座上加载 LoRA：

```bash
uv run python main.py \
  --tokenizer-dir tokenizer \
  --checkpoint out/demo/full_sft_32.pth \
  --lora-checkpoint out/demo/lora_32.pth \
  --prompt 'What is 3 plus 4?' \
  --device cpu \
  --max-new-tokens 32 \
  --temperature 0
```

- `--temperature 0` 为 greedy decoding；`--top-k 0` 不裁剪词表。
- 默认使用 KV Cache；传入 `--no-cache` 可走全量重算路径。
- 默认应用本地 chat template；`--raw-prompt` 可直接续写原始文本。
- `--checkpoint` 可接收结构化/legacy `.pth`，也可接收本地 Hugging Face model directory。

## WebUI

安装可选依赖并启动本地 Gradio 页面：

```bash
uv sync --frozen --extra cpu --extra web
uv run python webui.py \
  --tokenizer-dir tokenizer \
  --checkpoint out/demo/full_sft_32.pth \
  --device cpu \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --top-k 40 \
  --server-name 127.0.0.1 \
  --server-port 7860
```

LoRA WebUI 同样增加 `--lora-checkpoint out/demo/lora_32.pth`。Gradio import 是 lazy 的，不安装 `web` extra 不影响训练和 CLI。

2026-07-22 已在本地 CPU 浏览器实际提交 `Name three arithmetic words.`，页面流式返回 `two sum plus`：

![NinjaMind local Gradio WebUI](artifacts/webui.png)

截图使用本地、未提交的 `out/full_sft_128.pth`，只证明 Gradio 与本地流式推理链路能够交互；三个词的短输出不代表模型具备算术能力或一般回答质量。

## 测试与 CI

```bash
uv sync --frozen --extra cpu --extra dev
uv run ruff check .
uv run pytest -q
uv run python scripts/run_model.py
uv run python scripts/smoke_train.py \
  --output-dir /tmp/minimind-smoke \
  --artifact /tmp/smoke_train.json
```

2026-07-27 在当前工作区的最新本地结果：

```text
ruff check .                 All checks passed!
pytest -q                    83 passed, 1 skipped
python scripts/run_model.py  smoke inference OK
```

测试覆盖 RMSNorm、causal/left-padding mask、GQA、legacy/Dynamic/Static KV Cache、HF directory round-trip、Dense/MoE 与 padding-aware router loss、SFT response-preserving truncation、DPO shared-prompt truncation/mask、RL 左截断、LoRA 原子保存/加载/合并、截断后训练张量去重与无泄漏 deterministic split、全局 token-weighted 梯度/validation/perplexity、DDP lazy buffer 回归、JSON/CSV/SVG 原子 artifact、held-out SFT greedy generation evaluator、v2 checkpoint、模型/Adam/RNG 精确 resume 对照、DPO loss 与 DDP 全局 metric reduction、PPO GAE/generated mask/reward、GRPO advantage、DDP 参数/单进程回退、尾 batch/尾 accumulation、Unicode token streaming 和推理采样。唯一 skip 是当前机器没有 CUDA，CUDA RNG round-trip 测试未执行。

`.github/workflows/ci.yml` 会在 push/PR 上执行安装、ruff、pytest、CPU inference smoke 和 training pipeline smoke；`master` 的远端 GitHub Actions 已于 2026-07-22 通过。README 顶部 badge 会显示该 workflow 的当前状态。

## 可复现实验与真实结果

### 0.49M CPU convergence pilot

来源：[cpu_pilot.json](artifacts/cpu_pilot.json)

2026-07-27 在 macOS arm64 CPU、PyTorch 2.12.1、seed 42 上运行 hidden size 128、2 layers、4 query heads、2 KV heads、vocab 372 的 490,752 参数模型。Pretrain 使用 50 epochs、batch 32、LR `1e-3`；SFT 从 best pretrain checkpoint 开始，使用 100 epochs、batch 16、LR `5e-4`。训练权重保存在临时目录且未提交。

| Stage/checkpoint | Optimizer steps | Best validation CE | Latest validation CE | Held-out generation EM |
|---|---:|---:|---:|---:|
| Pretrain | 300 | 0.5482（epoch 24） | 0.6208 | — |
| SFT best-by-CE | 600 | 0.3348（epoch 30） | — | 5/10 |
| SFT latest | 600 | — | 0.3508 | 6/10 |

该 pilot 证明完整 CPU 训练、validation、best/latest 选择和 held-out greedy evaluator 能在非 smoke 步数下共同工作，也显示 teacher-forced CE 最优点不一定对应 generation EM 最优点。它不能作为可信泛化或可用模型证据：demo 只有 100 条结构一致的加法 SFT，pretrain 也覆盖同一算术领域，留出集仅 10 条，且 latest 仍有 4 条算错。

### 2-step CPU training smoke

来源：[smoke_train.json](artifacts/smoke_train.json) · [loss curve](artifacts/smoke_loss.svg) · [generation samples](artifacts/generation_samples.json)

环境：Python 3.12.7、PyTorch 2.12.1、macOS 15.7.4 arm64、CPU、seed 42；模型为 hidden size 32、1 layer、4 query heads、2 KV heads、vocab 372，共 27,376 参数。

| Stage | Step 1 loss | Step 2 loss | 实测阶段耗时 |
|---|---:|---:|---:|
| Pretrain | 5.8845 | 5.8919 | 2.8703 s |
| SFT | 5.8934 | 5.8625 | 3.2447 s |
| LoRA | 5.8959 | 5.9065 | 4.3463 s |
| DPO | 0.6931 | 0.6931 | 2.8307 s |

![2-step smoke loss curves](artifacts/smoke_loss.svg)

这些数值只证明真实 trainer、数据、反向传播、optimizer 和 checkpoint 路径可执行。两点 loss 不能说明收敛，也不能跨 stage 比较；DPO 第二步 preference accuracy 为 `0.500`，但 margin 约为 `-0.0000`，在 4 条 preference demo 上没有统计意义。

**生成质量警告：**`generation_samples.json` 中三个 prompt 的 greedy completion 都是连续 12 个换行（可记为 `\n × 12`）。这是仅训练 2 step 的 tiny 模型发生退化的真实结果，不是有效问答样例。smoke checkpoint 未提交，且交接清理后原 `out/smoke_final/` 已删除；需要使用 `scripts/smoke_train.py` 重新生成。

artifact 的 `git_commit` 是 `9c9f940241da73fe5f4eb06f5b9bd54ca3aed519`，同时明确记录 `git_dirty: true`，因此它代表当时的本地工作区快照，而不是不可变的发布版本。

### MHA / GQA / MQA + KV Cache benchmark

复现命令：

```bash
uv sync --frozen --extra cpu --extra benchmark
uv run python scripts/benchmark_attention.py --output-dir /tmp/minimind-benchmark
```

当前保存的 artifact 来源：[benchmark.json](artifacts/benchmark.json) · [benchmark.csv](artifacts/benchmark.csv) · [benchmark.png](artifacts/benchmark.png)

环境：Python 3.12.7、PyTorch 2.12.1、macOS arm64 CPU、float32、1 thread、seed 42；5 repeats、1 warmup、每个 prefill sample 20 iterations、batch 1、prompt length 128、decode 32、hidden size 64、1 layer、4 query heads。表中为 artifact 记录的 mean ± sample std：

| Attention | KV heads | Params | Prefill latency (ms) | Cached decode (tok/s) | No-cache decode (tok/s) | KV payload (B) | Cache speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| MHA | 4 | 82,144 | 0.4745 ± 0.0276 | 4,133.18 ± 176.40 | 1,879.17 ± 88.02 | 81,920 | 2.1995x |
| GQA | 2 | 78,048 | 0.4803 ± 0.0210 | 3,660.56 ± 341.43 | 1,478.85 ± 184.12 | 40,960 | 2.4753x |
| MQA | 1 | 76,000 | 0.5271 ± 0.0264 | 3,713.70 ± 498.50 | 1,360.27 ± 222.44 | 20,480 | 2.7301x |

表格只做显示精度的四舍五入；每次 repeat 的样本和完整浮点值均保留在 `benchmark.json` / `benchmark.csv`。

![MHA GQA MQA benchmark](artifacts/benchmark.png)

在三个配置中，理论 KV bytes 与实际返回 K/V tensor payload 逐项一致；相对 MHA，GQA 为一半、MQA 为四分之一。这个 tiny CPU workload 中 cache 比 no-cache 快约 2.20–2.73 倍，但 GQA/MQA 的 prefill 或 cached decode 并没有稳定快于 MHA。结果不应外推到其他硬件、batch、序列长度或 CUDA kernel。`cuda_peak_allocated_bytes` 为 `null`，因为 CUDA 未运行。

## 数据格式

所有数据均为一行一个 JSON object 的 JSONL。

Pretrain：

```json
{"text": "zero plus zero equals zero."}
```

SFT：

```json
{"conversations": [{"role": "user", "content": "What is 2 plus 2?"}, {"role": "assistant", "content": "2 plus 2 equals 4."}]}
```

DPO：

```json
{"chosen": [{"role": "user", "content": "What is 2 plus 2?"}, {"role": "assistant", "content": "2 plus 2 equals 4."}], "rejected": [{"role": "user", "content": "What is 2 plus 2?"}, {"role": "assistant", "content": "2 plus 2 equals 5."}]}
```

PPO / GRPO 使用的 RLAIF prompt：

```json
{"conversations": [{"role": "user", "content": "What is 2 plus 2?"}], "answer": "4"}
```

AgentRL dataset 还接受 `tools`、可选 `answer` 与 `gold_tool_calls`，但当前没有 AgentRL trainer。

| Demo file | 样本数 | 用途 |
|---|---:|---|
| `dataset/demo/pretrain_demo.jsonl` | 300 | Pretrain |
| `dataset/demo/sft_demo.jsonl` | 100 | SFT / LoRA |
| `dataset/demo/dpo_demo.jsonl` | 4 | DPO |
| `dataset/demo/rl_demo.jsonl` | 100 | PPO / GRPO |

SFT/DPO 的 loss mask 只覆盖 assistant response；过长 SFT 会从左侧保留最新回复，DPO 会对 chosen/rejected 使用同一份截断后的共享 prompt。RL collate 使用左截断/左 padding，并为实际 generated token（包含 EOS、排除 EOS 后 padding）建立 mask。

## Checkpoint 约定

- Full-model checkpoint 名称为 `{stage}_{hidden_size}[_moe].pth`，例如 `pretrain_32.pth`、`full_sft_32.pth`、`dpo_32.pth`。
- 单进程 Pretrain/SFT 的 `format_version=2` checkpoint 包含模型、optimizer、scaler、训练 cursor、planned LR horizon、run identity 与 RNG；恢复时校验 stage、完整模型 config、tokenizer fingerprint、split、max length、LR/grad clip、batch/accumulation、seed 与训练 horizon，模型使用 `strict=True`。
- `--save_interval` 只在 optimizer boundary 保存；latest 使用常规文件名。`_best.pth` 是 compact `format_version=1` 推理/`--init_from` 权重，不能用于 `--resume`；手动 cosine schedule 通过恢复 optimizer step 与原 planned total steps 保持连续。
- `load_weights()` 与推理 loader 兼容旧的裸 `state_dict`；历史 raw checkpoint 无完整 config 时，可能需要显式传入正确的 attention head 参数。
- LoRA 推理 checkpoint 只保存 adapter tensor，并附带 rank/alpha/targets 与基座 metadata；`*_train.pth` 另存完整可续训状态。加载 adapter 推理时仍必须同时提供 base/SFT checkpoint。
- 本地 Hugging Face directory 也可用于推理，RoPE derived buffer 会在首次 forward 重建。
- `out/` 已加入 `.gitignore`。不要提交正式模型权重、optimizer checkpoint、大数据集、API key 或 `.env`。

## 目录结构

```text
minimind_from_scratch/
├── model/
│   ├── model.py                 # Transformer、attention/cache、Dense/MoE
│   └── model_lora.py            # LoRA 注入、保存/加载、merge
├── dataset/
│   ├── lm_dataset.py            # 各阶段 dataset 与 mask/collate
│   └── demo/                    # 可提交的小型 JSONL
├── trainer/
│   ├── trainer_utils.py         # AMP、DDP、累积、checkpoint、GAE
│   ├── checkpointing.py         # 原子 v2 checkpoint 与严格 resume
│   ├── artifacts.py             # JSON/CSV/SVG 训练指标
│   ├── train_pretrain.py
│   ├── train_sft.py
│   ├── train_lora.py
│   ├── train_dpo.py
│   ├── train_ppo.py
│   └── train_grpo.py
├── inference.py                 # 可复用流式推理
├── main.py                      # CLI entry point
├── webui.py                     # 可选 Gradio UI
├── scripts/
│   ├── train_tokenizer.py
│   ├── download_dataset.py
│   ├── run_model.py
│   ├── smoke_train.py
│   ├── evaluate_sft.py           # 留出集 greedy generation exact match
│   └── benchmark_attention.py
├── tests/                       # CPU unit/regression tests
├── artifacts/                   # JSON/CSV/PNG/SVG 真实小实验产物
├── LICENSE                      # Apache License 2.0（仅覆盖本仓库软件）
└── .github/workflows/ci.yml
```

## 已知限制

1. **没有可用模型质量。** Validation、held-out generation EM 和 0.49M 参数 CPU convergence pilot 已经完成，但当前仍无正式长训练、可信正式准确率或可用 checkpoint；pilot 在 10 条留出样本上只有 60% EM，历史 2-step 模型则生成连续换行。
2. **PPO/GRPO reward 是 toy rule。** 它只检查参考答案是否出现在 completion 中，不等同于 learned reward model、AI judge 或生产 RLHF/RLAIF。
3. **GPU 未验证。** 只验证了单进程 CPU 和双进程 CPU/Gloo；CUDA AMP、NCCL、双 GPU correctness/吞吐/显存均待验证。
4. **WebUI 只做了最小本地浏览器验证。** 没有并发、长会话、错误恢复、部署或鉴权测试；截图中的短输出不能作为模型质量证据。
5. **CI 范围仍限 CPU。** 远端 workflow 已通过，但不覆盖 CUDA/NCCL、多 GPU、长训练或模型质量。
6. **Benchmark 很小且依赖机器。** 只能说明当前 CPU workload；不能声称 GQA/MQA 普遍更快或给出 GPU 显存收益。
7. **AgentRL 不完整。** 有 dataset schema，没有对应 trainer、environment 或 tool executor。
8. **Artifact 来自 dirty worktree。** JSON 保留 commit/environment，但不是已发布 tag 的结果；正式报告应在 clean commit 上重跑。
9. **正式数据与训练权重尚未完成商业用途许可审查。** 软件的 Apache-2.0 许可证不自动覆盖外部数据，也不对训练权重的许可状态作出判断。
10. **精确 resume 范围有限。** 当前只支持单进程 CPU/CUDA/MPS 上的 Pretrain/SFT/LoRA，实际逐项一致训练回归只跑过 CPU，MPS 仅验证 RNG 接口路径，CUDA 当前跳过；DDP 需要逐 rank RNG，DPO/PPO/GRPO 还需要稳定 reference，PPO 还必须持久化 critic，均未假装已经支持。

## 参考、归属与致谢

本项目参考 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 的“小型语言模型全流程”方向、训练阶段命名和公开数据格式，并使用其 [MiniMind dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset) 作为正式数据下载源；感谢原项目作者和贡献者。

本仓库是在自身 git 历史上迭代的个人学习/工程实现，没有复制外部仓库覆盖代码。个人实现与本轮工程化范围包括：Transformer/cache/mask/MoE 修复，LoRA/DPO/PPO/GRPO trainer，DDP 与 checkpoint 设施，数据 mask，流式 CLI/WebUI，CPU 测试、smoke pipeline 和 attention benchmark。底层算法均来自公开研究与社区实践，本项目不主张算法原创性，也不隶属于或代表原 MiniMind 项目。

## 许可证与数据使用边界

本仓库的软件源码以 [Apache License 2.0](LICENSE) 发布。上游 [MiniMind](https://github.com/jingyaogong/minimind) 同样采用 [Apache License 2.0](https://github.com/jingyaogong/minimind/blob/master/LICENSE)。

本仓库的软件许可证不会重新许可外部数据、下载得到的 checkpoint 或训练权重。[MiniMind dataset 数据集卡](https://huggingface.co/datasets/jingyaogong/minimind_dataset) 同时标记 Apache-2.0 与 CC-BY-NC-2.0，并说明数据汇集自多个来源；使用者应记录具体文件、版本/commit、原始来源及其许可证。在完成逐来源审查前，不应据此声称正式数据或训练产物可用于商业用途。
