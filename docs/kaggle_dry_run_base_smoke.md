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

Kaggle 启用 GPU 和 Internet（克隆/安装时需要）；Notebook 通过 `%%bash` 执行下列 Bash。
**每个新 Bash cell 都要重新 `cd` 和 `source`，环境变量不会跨 cell 保留。**
以下克隆命令用于首次启动；已有仓库时跳过克隆，不覆盖现有数据或运行输出。

```bash
set -euo pipefail
cd /kaggle/working
git clone https://github.com/Junheng-Zhu/RecoveryAgent.git RecoveryAgent
cd /kaggle/working/RecoveryAgent
source configs/gpu_server.env.example
python -m pip install -r requirements-gpu.txt
python scripts/setup_gpu_workspace.py
```

该环境示例面向 Kaggle。其他 Linux 服务器应另设 `RECOVERY_MODEL_PATH`、
`RECOVERY_TOKENIZER_PATH` 为真实模型目录；数据默认跟随实际仓库位置。
GPU 环境须与固定 Torch/CUDA 依赖兼容。BF16 为默认，T4/P100 等 GPU 不支持时，
明确选择 float32 并重新检查显存；不能把 BF16 预检失败视为成功。

## 2. Base 配置 Dry-run（不加载权重、不执行推理）

```bash
set -euo pipefail
cd /kaggle/working/RecoveryAgent
source configs/gpu_server.env.example
python scripts/evaluate_local_model.py \
  --execution dry-run --model-mode base --split train_smoke
```

预期输出 `status: EVALUATION_CONFIG_ONLY`、`tasks: 3`、
`gpu_inference_verified: false`。这只确认路径、冻结任务与评测配置。

## 3. 训练数据 Dry-run（使用仓库已有 SFT）

```bash
set -euo pipefail
cd /kaggle/working/RecoveryAgent
source configs/gpu_server.env.example
# 检查克隆得到的文件；缺失或不匹配时立即停止。
for name in oracle_sft.jsonl teacher_sft.jsonl source_audit.json token_stats.json mask_examples.json readiness_report.md; do
  test -s "$RECOVERY_SFT_PATH/$name"
done
python scripts/setup_gpu_workspace.py --write-delivery
python scripts/train_qwen3_lora.py \
  --mode dry-run --output outputs/kaggle_oracle_cpu_dryrun
```

预期 `CPU_DATA_DRY_RUN_PASSED`，Oracle 204 条、Teacher 独立校验。
不加载模型、不执行 Forward/Backward；不是训练成功证明。
模型挂载的 tokenizer/config 必须与 SFT 记录的哈希一致。

## 4. GPU 预检与 Base Model Smoke Test

此入口的 `LocalEngine` 同样检查 SFT、tokenizer 与 GPU 路径，因此也需先完成步骤 3。
这是在三个冻结 Train 任务上运行真实 Base 推理与工具链，不更新参数。

```bash
set -euo pipefail
cd /kaggle/working/RecoveryAgent
source configs/gpu_server.env.example
python scripts/check_gpu_environment.py --output outputs/kaggle_gpu_preflight.json
python scripts/evaluate_local_model.py \
  --execution gpu --model-mode base --split train_smoke \
  --output outputs/kaggle_base_train_smoke
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
