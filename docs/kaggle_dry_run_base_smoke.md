# Kaggle / 服务器 Dry-run 与 Base Model Smoke Test

## 当前仓库能否直接运行（2026-10-14）

**最新 `main` 已包含 SFT 六个文件和交付清单。** 除外部模型权重及 GPU/依赖环境外，
常规 Dry-run 与 Base Smoke 所需的数据文件已在仓库，无需重新准备 SFT。
此前“文件尚未提交”的判断基于落后的本地分支，已更正。

本次核验发现 Git 换行转换会使部分已交付文件不匹配 SHA-256。
修复将文件字节恢复为与原清单精确匹配的 CRLF，并以 `.gitattributes` 保留字节；
不修改 manifest 哈希、任务内容或训练标签。请使用包含此修复的版本。

| 必需文件 | Kaggle 位置 | 是否随 Git 克隆 |
| --- | --- | --- |
| 冻结任务及 manifest | `data/v0.3b_seed42_final/` | 是 |
| SFT 六个文件 | `data/v0.4a_training_readiness/` | 是 |
| 交付清单 | `data/delivery_manifest.json` | 是；初始化时验证已有清单 |
| 完整 Qwen3 权重/config/tokenizer | `/kaggle/input/models/zhuzhu135/qwen3-1-7b/transformers/default/3` | 外部模型挂载 |

SFT 六个文件：`oracle_sft.jsonl`、`teacher_sft.jsonl`、`source_audit.json`、
`token_stats.json`、`mask_examples.json`、`readiness_report.md`。
直接使用现有导出即可。只有主动重建 SFT 时，才需要另行提供 `outputs/` 下的
原始 Oracle/Teacher 可信证据；这些来源不随 Git 克隆，不影响读取现有已验证导出。

## 1. 克隆、配置与依赖

Kaggle 启用 GPU 和 Internet（克隆/安装时需要）。以下代码块直接粘贴到默认
**Python Notebook 单元格**，每个代码块作为一个单元格，按顺序执行。
`%cd` 修改 Notebook 工作目录，`!` 执行终端命令，`%env` 将环境变量保留给后续单元格。
不要在 Python 单元格中直接写 `set -euo pipefail` 或 `source`；`!source` 也不会将变量保留到下一条命令。

首次克隆（已有仓库则跳过此单元格）：

```python
%cd /kaggle/working
!git clone https://github.com/Junheng-Zhu/RecoveryAgent.git RecoveryAgent
```

配置当前 Notebook 内核（重启内核后重新执行）：

```python
%cd /kaggle/working/RecoveryAgent
import os
os.environ.pop("RECOVERY_DATA_ROOT", None)
%env RECOVERY_MODEL_PATH=/kaggle/input/models/zhuzhu135/qwen3-1-7b/transformers/default/3
%env RECOVERY_DATASET_PATH=/kaggle/working/RecoveryAgent/data
%env RECOVERY_SFT_PATH=/kaggle/working/RecoveryAgent/data/v0.4a_training_readiness
%env RECOVERY_FROZEN_PATH=/kaggle/working/RecoveryAgent/data/v0.3b_seed42_final
%env RECOVERY_OUTPUT_PATH=/kaggle/working/RecoveryAgent/outputs
%env RECOVERY_CHECKPOINT_PATH=/kaggle/working/RecoveryAgent/checkpoints
%env RECOVERY_TOKENIZER_PATH=/kaggle/input/models/zhuzhu135/qwen3-1-7b/transformers/default/3
%env RECOVERY_CACHE_PATH=/kaggle/working/RecoveryAgent/cache/huggingface
%env RECOVERY_DELIVERY_MANIFEST=/kaggle/working/RecoveryAgent/data/delivery_manifest.json
%env HF_HOME=/kaggle/working/RecoveryAgent/cache/huggingface
%env HF_HUB_OFFLINE=1
%env TRANSFORMERS_OFFLINE=1
%env USE_TORCH=1
%env USE_TF=0
%env USE_FLAX=0
%env CUDA_VISIBLE_DEVICES=0
```

安装完成后检查输出；安装失败时不要继续下一格：

```python
!python -m pip install -r requirements-gpu.txt
```

```python
!python scripts/setup_gpu_workspace.py
```

上述配置面向 Kaggle。Linux Terminal 使用普通 Bash 命令时，可运行
`source configs/gpu_server.env.example`；Notebook 使用上面的 `%env` 配置。其他 Linux 服务器应另设 `RECOVERY_MODEL_PATH`、
`RECOVERY_TOKENIZER_PATH` 为真实模型目录；数据默认跟随实际仓库位置。
GPU 环境须与固定 Torch/CUDA 依赖兼容。BF16 为默认，T4/P100 等 GPU 不支持时，
明确选择 float32 并重新检查显存；不能把 BF16 预检失败视为成功。

## 2. Base 配置 Dry-run（不加载权重、不执行推理）

```python
%cd /kaggle/working/RecoveryAgent
!python scripts/evaluate_local_model.py --execution dry-run --model-mode base --split train_smoke
```

预期输出 `status: EVALUATION_CONFIG_ONLY`、`tasks: 3`、
`gpu_inference_verified: false`。这只确认路径、冻结任务与评测配置。

## 3. 训练数据 Dry-run（使用仓库已有 SFT）

```python
%cd /kaggle/working/RecoveryAgent
from pathlib import Path
sft = Path(os.environ["RECOVERY_SFT_PATH"])
required = ["oracle_sft.jsonl", "teacher_sft.jsonl", "source_audit.json",
            "token_stats.json", "mask_examples.json", "readiness_report.md"]
missing = [name for name in required
           if not (sft / name).is_file() or (sft / name).stat().st_size == 0]
assert not missing, f"缺少或为空的 SFT 文件：{missing}"
```

文件检查通过后执行清单验证；失败时停止，不执行下一格：

```python
!python scripts/setup_gpu_workspace.py --write-delivery
```

```python
!python scripts/train_qwen3_lora.py --mode dry-run --output outputs/kaggle_oracle_cpu_dryrun
```

预期 `CPU_DATA_DRY_RUN_PASSED`，Oracle 204 条、Teacher 独立校验。
不加载模型、不执行 Forward/Backward；不是训练成功证明。
模型挂载的 tokenizer/config 必须与 SFT 记录的哈希一致。

## 4. GPU 预检与 Base Model Smoke Test

此入口的 `LocalEngine` 同样检查 SFT、tokenizer 与 GPU 路径，因此也需先完成步骤 3。
这是在三个冻结 Train 任务上运行真实 Base 推理与工具链，不更新参数。

```python
%cd /kaggle/working/RecoveryAgent
!python scripts/check_gpu_environment.py --output outputs/kaggle_gpu_preflight.json
```

下一格确认预检通过，再执行真实推理；断言失败时停止：

```python
import json
from pathlib import Path
preflight = json.loads(Path("outputs/kaggle_gpu_preflight.json").read_text())
assert preflight["gpu_verified"] and not preflight["blockers"], preflight["blockers"]
!python scripts/evaluate_local_model.py --execution gpu --model-mode base --split train_smoke --output outputs/kaggle_base_train_smoke
```

预检须 `gpu_verified: true`、`blockers: []`；退出码 2 时停止。
默认要求至少 12 GiB 空闲显存并支持 BF16，实际可运行条件仍需 GPU 验证。
查看输出目录中的 `benchmark_report.json` 和轨迹/审计记录，确认真实模型生成、
工具解析与执行、Verifier 和预算/停止协议。Base 不一定解题成功；任务失败需查看轨迹，
不能以 Mock 运行代替真实推理验收。

重复运行请换新输出目录；脚本拒绝覆盖旧报告。恢复同一次评测可显式 `--resume`。
`train_qwen3_lora.py --mode smoke` 是 LoRA 一步训练测试，与本页 Base Smoke 不同。

## 本地核验边界

更新到最新 `main` 并恢复原始文件字节后，相关测试 **15 passed**；
训练 Dry-run 实测返回 `CPU_DATA_DRY_RUN_PASSED`。Base 配置 Dry-run 选中 3 个任务。
CPU 核验使用仓库 tokenizer；Kaggle 环境示例使用模型挂载中的 tokenizer，
其 config/template 必须匹配数据记录的哈希。真实 GPU Base Smoke 尚未执行，
仍需检查 Kaggle GPU、依赖、显存、BF16 支持及挂载权重的完整性。
