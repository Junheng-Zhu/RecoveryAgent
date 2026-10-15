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

Kaggle 预装的 torchvision 可能与上面固定安装的 PyTorch 不兼容，导致
`RuntimeError: operator torchvision::nms does not exist`。本项目为纯文本 Qwen3，
无需 torchvision；在本 Notebook 环境中移除它，保留 PyTorch：

```python
!python -m pip uninstall -y torchvision
```

检查真实 Qwen3 模型类能否导入（不加载权重、不运行 GPU 推理）：

```python
!python -c "import torch, transformers; from transformers import Qwen3ForCausalLM; print('Qwen3 import OK', torch.__version__, transformers.__version__)"
```

看到 `Qwen3 import OK` 后再继续。`gpu_verified: true` 只代表已有 GPU/路径检查通过，
不保证 Transformers 延迟加载的模型类及其可选依赖都能导入。
上面 `!python` 每次启动新进程，卸载后重跑导入检查及评测即可；若在 Notebook 内核中
直接导入过 torch/transformers，则重启内核并重新执行环境配置。

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

## 5. 完整 LoRA-SFT 训练

完成数据 Dry-run、GPU 预检及 Base Smoke 后，先执行一步训练 Smoke，检查真实
Forward/Backward、梯度和参数更新。它写入独立的 `_gpu_smoke` 目录。
各代码块分别运行，确认上一格成功后继续。

```python
%cd /kaggle/working/RecoveryAgent
!python scripts/train_qwen3_lora.py --mode smoke
```

完整训练默认使用 Oracle 204 条，2 epochs，batch=1、梯度累积=8，预计 52 次优化器更新。
Teacher 不自动混入。默认训练名为 `qwen3_1.7b_oracle_lora`：

```python
!python scripts/train_qwen3_lora.py --mode train
```

完成时检查 `outputs/qwen3_1.7b_oracle_lora/training_report.json`，最终 Adapter 在
`outputs/qwen3_1.7b_oracle_lora/adapter/`，不是完整合并的 Base 权重。
从已有完整 checkpoint 恢复（将示例 step 换成实际最新完整目录；只在需要恢复时执行）：

```python
!python scripts/train_qwen3_lora.py --mode train --resume checkpoints/qwen3_1.7b_oracle_lora/step-000020
```

Resume 需要同一训练配置、数据、Base/Tokenizer 及依赖版本。跨 Kaggle 会话须先恢复
checkpoint 和原运行 `outputs/qwen3_1.7b_oracle_lora/`；不要用一步 Smoke checkpoint 恢复完整训练。
已完成的运行不要重启覆盖；新实验用新的 `--output` 和 `--checkpoint-path`。

## 6. Base 与已训练 SFT 的正式测试：各 31 个任务

`train_smoke` 是 3 个训练集任务；正式测试为 **IID Test 13 + OOD 18 = 31**。
Base 和 SFT 各执行两条命令，共 62 次任务运行。以下四个代码块分别执行，输出目录独立。
模型、数据、精度、seed、预算和停止协议应保持一致，仅 Adapter 不同。
SFT 命令默认加载步骤 5 完整训练的最终 Adapter；若 Adapter 位于其他目录，请替换路径。

Base：IID Test 13 个。

```python
%cd /kaggle/working/RecoveryAgent
!python scripts/evaluate_local_model.py --execution gpu --model-mode base --split iid_test --output outputs/kaggle_base_iid_test
```

Base：OOD 18 个。

```python
!python scripts/evaluate_local_model.py --execution gpu --model-mode base --split ood --output outputs/kaggle_base_ood
```

SFT：IID Test 13 个。

```python
!python scripts/evaluate_local_model.py --execution gpu --model-mode adapter --adapter outputs/qwen3_1.7b_oracle_lora/adapter --split iid_test --output outputs/kaggle_sft_iid_test
```

SFT：OOD 18 个。

```python
!python scripts/evaluate_local_model.py --execution gpu --model-mode adapter --adapter outputs/qwen3_1.7b_oracle_lora/adapter --split ood --output outputs/kaggle_sft_ood
```

若已完成 SFT，只需重新执行第 1 节环境配置，并确保 Base 挂载、数据、Adapter 与预检
就绪，即可运行上述评测，无需重新训练。仅有 Adapter 时仍需同一完整 Base 模型。

另有 **IID Validation 13 个**，用于训练配置/Checkpoint 选择，不计入正式 31 个测试任务。
先在 Validation 确定配置，冻结选择后再评测 IID Test/OOD，勿用正式测试集选择超参数。

```python
!python scripts/evaluate_local_model.py --execution gpu --model-mode base --split iid_validation --output outputs/kaggle_base_iid_validation
```

```python
!python scripts/evaluate_local_model.py --execution gpu --model-mode adapter --adapter outputs/qwen3_1.7b_oracle_lora/adapter --split iid_validation --output outputs/kaggle_sft_iid_validation
```

## 7. Kaggle 输出位置、保存与恢复

所有默认写入都在可写的 `/kaggle/working/RecoveryAgent/`；模型挂载的 `/kaggle/input/` 只读。
代码写入磁盘不等于已持久保存；Git 提交也不会自动收录训练产物。

| 产物 | 默认位置（相对仓库根目录） |
| --- | --- |
| CPU Dry-run 报告 | `outputs/kaggle_oracle_cpu_dryrun/dry_run_report.json` |
| Base Smoke 报告与轨迹 | `outputs/kaggle_base_train_smoke/` |
| 完整训练日志、配置和报告 | `outputs/qwen3_1.7b_oracle_lora/`：`train_metrics.jsonl`、`training_config.json`、`run_contract.json`、`data_manifest.json`、`gpu_preflight.json`、`training_report.json` |
| 最终 LoRA Adapter | `outputs/qwen3_1.7b_oracle_lora/adapter/`：Adapter 权重、配置及归属/哈希元数据 |
| 可恢复训练 checkpoint | `checkpoints/qwen3_1.7b_oracle_lora/step-XXXXXX/`：Adapter、optimizer/scheduler/RNG/cursor 状态和校验元数据 |
| Base 正式测试 | `outputs/kaggle_base_iid_test/`、`outputs/kaggle_base_ood/` |
| SFT 正式测试 | `outputs/kaggle_sft_iid_test/`、`outputs/kaggle_sft_ood/` |

评测目录包含 `benchmark_report.json`（最终指标）、`trajectories.jsonl`（逐任务轨迹）、
`tasks.jsonl`、`session.json`、`report.json` 和 `termination_diagnostics.jsonl`。
运行未完成时可能尚无最终报告；轨迹逐任务写入。相同评测恢复使用原命令加 `--resume`，
否则选择新目录。训练默认每 10 次更新及最后一步保存 checkpoint；意外终止可能丢失
最近一次完整 checkpoint 之后的更新。仅保存最终 Adapter 足以评测，不足以恢复优化器训练。

在 Kaggle 界面使用 **Save Version** 保存版本及其 Output；如选择 **Save & Run All**，
会重新执行 Notebook，请避免再次写入已有实验目录。仅交互运行的文件不应视为已永久保存，
会话结束前请确认版本 Output 中已有产物，或下载备份。也可将版本 Output 作为 Dataset
挂载到后续 Notebook，挂载后位于只读 `/kaggle/input/`；恢复训练时将所需产物复制回可写目录。

训练/评测结束后可打包输出与 checkpoint，生成独立归档（不包含 Base 模型与缓存）：

```python
%cd /kaggle/working/RecoveryAgent
!tar -czf /kaggle/working/recoveryagent_run_bundle.tar.gz -C /kaggle/working/RecoveryAgent outputs checkpoints
```

通过 Kaggle 文件/Output 面板下载归档，或显示下载链接：

```python
%cd /kaggle/working
from IPython.display import FileLink, display
display(FileLink("recoveryagent_run_bundle.tar.gz"))
```

后续会话将归档上传为 Dataset 后，按实际挂载位置替换路径，解压到已克隆仓库再恢复运行：

```python
%cd /kaggle/working/RecoveryAgent
# 将占位路径换成实际归档位置；仅恢复到没有冲突的运行目录。
!tar -xzf /kaggle/input/YOUR_DATASET/recoveryagent_run_bundle.tar.gz -C /kaggle/working/RecoveryAgent
```

checkpoint 及归档可能较大，需留意磁盘余量；确认备份后再清理。无需将 Base 权重复制到
`/kaggle/working`，也不要将大型训练产物直接提交进代码仓库。

## 本地核验边界

更新到最新 `main` 并恢复原始文件字节后，相关测试 **15 passed**；
训练 Dry-run 实测返回 `CPU_DATA_DRY_RUN_PASSED`。Base 配置 Dry-run 选中 3 个任务。
CPU 核验使用仓库 tokenizer；Kaggle 环境示例使用模型挂载中的 tokenizer，
其 config/template 必须匹配数据记录的哈希。真实 GPU Base Smoke 尚未执行，
仍需检查 Kaggle GPU、依赖、显存、BF16 支持及挂载权重的完整性。

## 故障排查：Base Smoke 出现 torchvision::nms

若预检断言已通过，而 `AutoModelForCausalLM.from_pretrained` 报错
`Failed to import ...modeling_qwen3`，最内层为 `operator torchvision::nms does not exist`，
按第 1 节卸载不需要的 torchvision，再执行 Qwen3 模型类导入检查。
这不是路径、JSON 或冻结任务解析错误，无需修改模型文件或关闭数据校验。
导入检查成功后重跑原 Base Smoke 命令。此导入失败发生在 Collector 创建评测输出之前；
若输出目录确实为空可沿用，若已存在其他运行结果则选择新的 `--output`。
本次根据用户 Kaggle traceback 定位；本地无该 Kaggle GPU 环境，未实测硬件推理。
