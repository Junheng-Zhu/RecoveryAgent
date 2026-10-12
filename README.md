# RecoveryAgent v0.1 — Local Environment MVP

独立、确定性的工具调用错误恢复实验环境。运行时仅依赖 Python 标准库，
建议 Python 3.11；兼容 Python 3.10。无需 GPU、真实 LLM 或深度学习框架。

## 本地运行（PowerShell，项目根目录）

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-local.txt
python -m pytest -q
python scripts/generate_tasks.py --count 40 --seed 42
python scripts/evaluate.py --policy rule --output outputs/rule --replay
python scripts/evaluate.py --policy oracle --output outputs/oracle --replay
python scripts/evaluate.py --policy retry --output outputs/retry
python scripts/evaluate.py --policy stop --output outputs/stop
python scripts/generate_tasks.py --count 1200 --output data/expanded/tasks.jsonl
```

项目已检测到现有虚拟环境和 pytest，无需安装其他依赖。
输出：每个目录的 trajectories.jsonl 与 report.json。Oracle 报告标记
oracle_certification，禁止作为普通 Policy 评测的模型性能。

## 模块和数据结构

- models.py：ToolCall(name, arguments, call_id)；ToolResult(call_id, tool, ok,
  value, error)；ErrorObservation(code, message, retryable, details)。
- environment.py / tools.py：真实临时目录、内存字典表、严格参数校验、六种工具。
  文件路径限定相对 POSIX 路径；拒绝绝对路径、盘符、反斜杠、.. 和逃逸。
  calculate 只解释数值与 + - * /，不使用 eval。
- faults.py：四类一次性故障，1-based 步骤注入；seed 决定稳定诊断 token。
  参数错误任务的 failed_action 真正包含错误字段，未修正参数时重试仍失败；瞬时错误发生在副作用之前；
  缺失状态删除输入文件、数据库行或旧输出；规划错误要求先获取依赖。
  故障注入自身的状态变化不计入 Policy 写操作账本。
- verifier.py：独立比较完整最终状态和修改账本，检测未授权写入、
  错误中间值及重复写入，写后还原也不能绕过检测。
- policy.py：可替换 Policy.reset / act 接口；rule、retry、stop 三个基线。
- oracle.py：显式参考计划执行器，只用于可恢复性验收或后续生成监督目标。
- runner.py：在线 episode、预算控制、JSONL、指标汇总、确定性重放。
  Reset 重建隔离环境并重置计数器、账本、注入器；退出清理临时目录。

任务 JSON 包含 id、template_family、split、goal、initial_state、
failed_action、fault_type、fault(type/seed/step)、expected_state、
max_recovery_steps、recoverable、allowed_mutations、reference_plan。
goal 是公开结构化任务说明（含自然语言 instruction），不包含最终状态或参考计划。
PolicyInput 仅含 goal、failed_action、observations、remaining_steps；
不传递 task 对象、环境、Verifier、expected_state、reference_plan 或 Oracle。
failed_action 是故障之前拟执行的动作，其参数可能已包含先前工作所得信息；
它不是预先证明正确的恢复答案。

## 任务、切分与指标

10 个模板族 × 4 种故障 = 40 个 Smoke 任务；覆盖复制、文本转换、
记录更新和算术报告。固定模板族映射：upper_label、sum_stock 属于 val（8），record_quota、ratio_summary 属于 test（8），其余 6 族属于 train（24）。训练集包含全部四种操作机制。1200 接口生成每族/故障
30 个参数变体；模板族不跨 split。变体共享结构，不能视为 1200 个不同推理机制。
这是人工合成的环境自检集合，不能用于宣称真实 Agent 泛化能力。
所有 v0.1 任务预先定义为可恢复；预算耗尽及负基线验证失败判定。

- Recovery Success Rate：最终状态精确匹配且无未授权修改/额外副作用的 episode 比例。
- average_recovery_steps：全部 episode 恢复调用数均值，不含起始失败调用。
- average_steps_successful：成功 episode 的调用数均值，无成功时为 null。
- repeated_failure_rate：同一 episode 内工具名、参数和错误码相同的重复失败次数 /
  恢复调用总数；首个失败用于比较，第一次新失败不算重复。
- extra_side_effects：允许资源上的错误中间写入或超出许可次数的写入次数。
- unauthorized_modifications：未允许资源上的写操作事件，另检查直接状态篡改；
  这是检测计数，可能同时捕捉事件与残留差异，不等同唯一受影响资源数。
- by_fault：各类故障分别统计，便于避免汇总掩盖差异。

## Replay 和未来模型接入

Replay 在新隔离环境逐个执行记录中的 call，并逐项比较 result、
final_state、audit、verification；不重新询问 Policy，篡改会报错。
同 seed、同 task、同调用序列的环境行为一致；不保证随机 Policy 自动一致。

未来模型只需实现 Policy 接口，把公开输入序列化为 prompt，再把输出解析为
ToolCall 或 None。运行阶段可独立选择推理后端；环境和 Verifier 不依赖模型。
训练导出应另做一层：仅用 train 的 Oracle 轨迹构建监督目标，去掉 privileged 字段；
val/test 的参考计划始终留在评测进程之外。当前没有 SFT、DPO、RL、偏好排序，
没有 Cortex 接入，也没有模型 API 调用。

隔离保障针对本项目六个模拟工具，不是任意 Python / OS 代码的安全沙箱。
Policy 接口避免正常数据传递泄漏；运行在同一 Python 进程并非恶意代码隔离。

## 指定故障步骤与验收记录

FaultInjector 支持任意正整数 step；Episode 可提供 prefix_actions（成功的前置调用列表）。
例如添加一次 list_files 前置调用，并设置 fault.step=2，失败发生在第二个工具调用。
prefix 的调用与结果也进入轨迹和 Replay；恢复步数不包含前置步骤。

本机已执行验收（2026-10-12）：

| 项目 | 结果 |
| --- | --- |
| pytest | 106 passed |
| Rule Policy | 40/40；平均恢复 3.35 步；重复失败率 0；额外副作用 0 |
| Oracle 可恢复性验收 | 40/40；平均恢复 2.2 步 |
| Retry Policy | 14/40；参数与顺序错误不能直接重试解决 |
| Stop Policy | 0/40 |
| 独立环境重放 | Rule 40/40；Oracle 40/40 |
| 两次独立 Rule 评测 | trajectories.jsonl 逐字节一致 |
| 扩展任务 | 1200 个已生成；pytest 内逐任务验证 Oracle 全部成功 |

详细报告见 outputs/rule/report.json、outputs/oracle/report.json、
outputs/acceptance.json；轨迹见相应目录的 trajectories.jsonl。
这些数值证明合成环境和参考解的可执行性，尚未测量任何模型的恢复能力。
