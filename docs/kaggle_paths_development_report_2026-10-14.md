# Kaggle 训练与测评路径开发报告

日期：2026-10-14

## 需求与实现

- GitHub 克隆代码位置：`/kaggle/working/RecoveryAgent`。
- 默认模型权重：`/kaggle/input/models/zhuzhu135/qwen3-1-7b/transformers/default/3`。
- 数据直接来自仓库 `data/`；冻结数据在 `data/v0.3b_seed42_final`，SFT 导出在 `data/v0.4a_training_readiness`，交付清单在 `data/delivery_manifest.json`。
- `recovery_agent/paths.py` 统一服务于训练器和本地模型评测入口，保留 CLI/环境变量覆盖和显式启用的旧服务器布局。仓库根目录按实际克隆位置解析，不依赖终端 cwd。
- 初始化跳过 `/kaggle/input` 下模型目录的创建和写入探测；模型读取和完整性预检仍执行。
- `scripts/prepare_qwen3_sft.py` 默认输出与训练器读取目录一致。
- `configs/gpu_server.env.example` 改为 Kaggle 配置，清除旧 `RECOVERY_DATA_ROOT`，将运行输出、缓存和 checkpoint 放在仓库工作目录。显式配置时 tokenizer 来自挂载模型；不配置时 CPU 准备仍可用仓库官方 tokenizer。
- 交付清单的新生成记录使用实际文件位置，去除旧 `/data` 目标。已有清单仍保持校验与禁止覆盖。
- README 原 B.0 路径、手动交付和 Linux 启动说明已修改，训练/恢复/Base/Adapter 命令更新为 Kaggle 位置，并补充首次克隆的 SFT 准备步骤。

## 验证结果

- 路径针对性测试：4 passed，1 deselected（依赖缺失导出数据的交付测试单列阻塞）。新增测试覆盖默认模型位置、仓库数据目录、模型覆盖，以及初始化不触碰只读模型目录。
- Python 编译检查、环境示例 Bash 语法检查、`git diff --check` 通过。
- `setup_gpu_workspace.py --dry-run` 确认模型为指定 Kaggle 路径，其他目录按实际仓库位置解析。
- 三组训练/测评相关测试：6 passed、8 failed。失败原因：当前环境缺少 PyYAML；仓库缺少 SFT 导出和交付清单；既有冻结数据与清单哈希不一致。
- `prepare_qwen3_sft.py --audit-only --output work/sft_audit` 同样在冻结数据哈希校验处失败，尚未生成 SFT 数据。

## 运行前需解决的既有阻塞

安装 README 指定依赖，取得与冻结 manifest 匹配的可信数据，再按 README 生成 SFT 导出和交付清单。不要通过修改 manifest 或绕过哈希校验来掩盖数据差异。本轮未修改冻结数据、训练标签或模型权重；未执行真实 GPU 训练与模型推理。当前环境没有指定 Kaggle 模型挂载，实际 GPU 验收须在 Kaggle 执行。

## 后续运行核验与提交补充

新增 `docs/kaggle_dry_run_base_smoke.md`，明确克隆仓库无法直接完成全部运行，并提供
Base 配置 Dry-run、SFT 文件检查、训练 Dry-run、GPU 预检与 Base Smoke 命令。
冻结 config 的哈希差异已确认为 Git 换行转换：恢复 CRLF 后精确匹配原始清单，
新增 `.gitattributes` 保留其字节。安装 PyYAML 后相关测试为 12 passed、3 failed；
剩余失败均来自未交付的 SFT 文件/清单。真实 Base 推理仍未执行。

## 最新 main 数据核验更正

此前缺少数据的结论基于落后的本地分支。最新 main 已包含六个 SFT 文件及
`data/delivery_manifest.json`。新增恢复五个文件原始 CRLF 字节并固定 Git 属性，
所有字节精确匹配既有交付清单，未改清单哈希。相关测试 15 passed；
训练 Dry-run 返回 CPU_DATA_DRY_RUN_PASSED。现有导出无需重建，只有主动
重建才依赖未提交的原始可信轨迹。Kaggle GPU Base Smoke 仍待实际硬件验证。
