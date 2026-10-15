# Kaggle / 服务器 Dry-run 与 Base Model Smoke Test

## 当前仓库能否直接运行（2026-10-14）

**仅克隆当前 GitHub 仓库，即使另加模型权重，也不能完成训练 Dry-run 或 GPU Base Smoke。**
代码、固定依赖列表、官方 tokenizer、冻结任务已在仓库；Kaggle 路径已经配置。
但 `outputs/` 被 Git 忽略，训练 SFT 导出和交付清单尚未提交；从头准备 SFT 所需的
可信 Oracle/Teacher 原始轨迹与派生参考文件也不在仓库。不要重新生成 Teacher 数据或用 Mock 替代。

本次已恢复冻结 `config.json` 的原始 CRLF 字节，精确匹配原 manifest 的 SHA-256，
并以 `.gitattributes` 防止克隆时换行转换；没有修改清单哈希或任务内容。
Base 的配置 Dry-run 可用冻结任务执行；训练 Dry-run 和真实 Base 推理还需要下列文件。

| 必需文件 | Kaggle 位置 | 当前是否随 Git 克隆 |
| --- | --- | --- |
| 冻结任务及 manifest | `data/v0.3b_seed42_final/` | 是 |
| SFT 六个文件 | `data/v0.4a_training_readiness/` | 否 |
| 交付清单 | `data/delivery_manifest.json` | 否；文件就位后生成 |
| 完整 Qwen3 权重/config/tokenizer | `/kaggle/input/models/zhuzhu135/qwen3-1-7b/transformers/default/3` | 外部模型挂载 |

SFT 六个文件：`oracle_sft.jsonl`、`teacher_sft.jsonl`、`source_audit.json`、
`token_stats.json`、`mask_examples.json`、`readiness_report.md`。
从原开发机取回经过验证的 v0.4-A 导出，放入仓库 `data/v0.4a_training_readiness/` 后，
再生成交付清单。文件必须通过原有哈希、来源、标签、模板校验。
若要从头执行 `prepare_qwen3_sft.py`，必须先补齐其 `source_directories()` 指定的
Oracle/Teacher 证据，包括 `outputs/v0.3b_sft_reference_final/oracle_reference_sft_train.jsonl`
和对应任务/轨迹；当前克隆中的文件不足以重建原可信训练集。

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

## 3. 训练数据 Dry-run（先补齐 SFT 六个文件）

```bash
set -euo pipefail
cd /kaggle/working/RecoveryAgent
source configs/gpu_server.env.example
# 缺文件则立即停止；先从可信导出补齐，不能跳过校验。
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

本轮仅做 CPU 校验和配置 Dry-run；没有 Kaggle 模型挂载或 GPU，未执行真实 Base 推理。
已确认冻结字节完整性。仍缺 SFT 导出与可信来源，训练 Dry-run 保持阻塞，不能承诺直接克隆可完整运行。

本地相关测试结果：12 passed、3 failed。三个失败均为缺少 SFT 导出或交付清单；
Base 配置 Dry-run 实测返回 `EVALUATION_CONFIG_ONLY`，选中 3 个任务。
来源审计实测在缺少 Oracle 派生参考文件处失败。
