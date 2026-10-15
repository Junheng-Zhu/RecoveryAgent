> Kaggle Dry-run 与 Base Smoke 的完整命令和文件检查见
> [运行指南](docs/kaggle_dry_run_base_smoke.md)。SFT 导出和交付清单已在 `data/` 中，
> 可直接读取现有文件；真实 Base 推理仍需挂载模型并通过 GPU 预检。

# RecoveryAgent v0.3-A - Stateful Recovery & Trajectory Pipeline

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

## v0.2 — Trajectory & Training Data Pipeline

v0.2 以新增的 recovery_agent/data_pipeline/ 模块实现；v0.1 的工具、
Environment、FaultInjector、Verifier、Oracle、Policy 与 Runner 核心接口保持不变。
仅使用标准库与已有 pytest；不调用模型、不下载权重、不训练 SFT/DPO。

### 一键生成、验证与复现

在项目根目录激活现有虚拟环境，然后执行：

```powershell
python -m pytest -q
python scripts/build_data.py build --output data/v0.2_seed42
python scripts/build_data.py validate --input data/v0.2_seed42
python scripts/build_data.py build --config data/v0.2_seed42/config.json --output data/v0.2_seed42_repro
python scripts/build_data.py compare --first data/v0.2_seed42 --second data/v0.2_seed42_repro
```

生成目录必须为空，已有版本不会被覆盖。可以用新的 --output 或 --seed 生成独立版本。
--config 接受完整配置并优先于 --seed；--preference-decisions 默认 1，表示只在每个
train/val 任务的第一个恢复决策构建偏好原型，避免无必要地放大样本数量。
更大的值允许覆盖前 N 个决策（含终止决策），该参数记录在 manifest 中。

输出结构：

- config.json / manifest.json：数据集版本 0.2.0、seed、任务规模、族与环境配置，
  Pipeline 参数、配置摘要，以及所有生成文件的 SHA-256。不记录时间戳或临时目录路径。
- tasks/train.jsonl、val.jsonl、test.jsonl：800 / 200 / 200 个任务。
- trajectories/train.jsonl、val.jsonl、test.jsonl：特权原始轨迹及逐决策公开投影。
- sft/train.jsonl、val.jsonl：逐恢复动作与停止动作的监督样本。
- preferences/train.jsonl、val.jsonl：共享 Prompt 的候选对及实际执行证据。
- report.json：任务、故障、模板、环境、长度、成功率、样本数和丢弃原因分布。
- quality_report.json：完整重放、重复、因果输入、切分与偏好证据验证结果。
- 不生成 sft/test.jsonl 或 preferences/test.jsonl。

### 保守切分与数据含义

复制任务的 note/receipt 别名归为 copy；两种记录更新归为 record_update；
大小写变换归为 uppercase；加法/除法报告归为 arithmetic_report。
固定切分为 train=copy+record_update（各 400），val=uppercase（200），
test=arithmetic_report（200）。同一恢复机制的别名和所有参数变体不会跨 split。
各 split 使用两套互不重复的真实文件路径布局和数据库命名配置；每个实例还有
独立的文件子目录或记录 ID。各 split 四类故障均衡（200/50/50 个每类）。

这一严格切分用于检验未见机制的泛化：训练恢复目标没有 calculate/uppercase 操作，
尽管 tools Schema 会完整列出六个工具。它不是六类工具均覆盖的 IID 训练集。
未来可另行设计具有更多不同任务组合的新版本，但不能依据 test 结果调整本版本。
1200 个任务仍是少量手工模板的参数化实例，不等于 1200 种独立推理机制。

### 轨迹与统一 Prompt

Collector 使用既有 Oracle 在 Environment 中执行 run_episode，最终由 Verifier
判定，并立即 Replay。逐步保存 goal、visible_history、error_feedback、tools、
prompt、oracle_action、tool_result；任务级保留原始 trace、最终 Verifier、恢复步数、
副作用和 Replay 标记。每个成功任务另导出显式停止决策。

serialization.build_prompt 是训练导出和 ConversationalPolicy 在线适配器共用的入口：
system 指令 → 公开 goal → 已发生的 assistant tool_calls/tool results → 剩余预算。
可见错误详情只允许 path/record_id/table/field/missing/extra，排除 fault_token、
隐藏 seed 及其他注入器元数据。expected_state、reference_plan、Oracle 规则、最终判定、
未来 Tool Result 从不加入模型 Prompt；已经实际读取的备份内容属于合法观察信息。
v0.1 failed_action 中原有的参数信息保留，可能包含失败前已计算出的值；它是可见历史，
不是新增的 Oracle 泄漏。原始 trajectories 与偏好 evidence 是特权审计产物，
不能整体拼入模型输入。

### SFT / DPO 数据约定

SFT 每条含 messages、tools、sample_id、task_id、split、decision_index、prompt_id、
loss_scope。messages 最后一条 assistant 是监督目标；tool_calls 位于 assistant 消息，
function.arguments 是规范化 JSON 字符串，tool 消息包含匹配的 tool_call_id。
tools 为标准 function JSON Schema，字段和类型对应 v0.1 参数校验器。
loss_scope=final_assistant_only：将来训练时只对最后一条 assistant 计算监督损失，
尤其不能把历史失败动作也作为正例学习。当前没有实现 tokenizer 或训练 collator；
接入 Qwen3 时应让同一个 chat template 同时服务训练与在线推理，并正确实现损失掩码。
ConversationalPolicy 接收一个可替换 predictor；当前只有测试替身，无模型实现。
该适配器面向从失败动作开始的恢复 episode，不支持 v0.1 任意 prefix_actions 的
未知历史参数重建；本版本生成任务的故障均在第一步。

Preference 含 prompt={messages,tools}、chosen/rejected（各一条 assistant completion），
统一使用同一动作序列化器；metadata/evidence 不属于模型输入。
两个分支通过重放同一 prefix，校验物理状态、账本、步数、故障状态和前置条件摘要一致。
实际执行候选及相同的参考续程，并保存 result、candidate_verification、最终 Verifier、
完整执行事件、最终状态、账本和预算消耗。每条保留的偏好会再次从新环境独立执行验证。

只有 chosen 完成任务且 rejected 提前停止未满足目标，或候选自身造成不可消除的
账本违规，才保留偏好。若只是固定续程失败、但换一个修复计划仍可能成功，直接丢弃；
如果违规由固定续程的重复写入引起而非候选本身，也丢弃。双方成功不凭动作名字、
步数或直觉分出优劣。候选为停止、重试原调用、未请求写入、只读替代动作；当前是一版
偏好管线自检，负例偏简单，不能声称已具备高质量模型错误分布或 DPO 训练收益。

### 质量检查

validate 会重新读取 JSONL，检查格式与完整数据结构、任务配置的确定性生成、
内容与 ID 去重、族/环境/Prompt 的跨 split 重叠，以及没有 test 训练文件。
成功轨迹逐条 Replay，逐决策按历史前缀重新构造公开输入，检查未来结果/隐藏字段
无法混入 SFT；这使用实际的因果投影比对，不依赖简单的关键词搜索。
所有偏好分支重新执行并比对证据；manifest 的文件摘要检查落盘内容是否被改动。
测试覆盖隐藏 seed/token 与 expected_state 扰动、未来结果注入、相同 Prompt 在线契约、
续程诱导的伪偏好、证据篡改、非法 JSONL、两次生成逐字节一致。

### v0.2 本机验收结果（2026-10-12）

- pytest：161 passed；原有 10 个 v0.1 核心文件 SHA-256 全部保持不变。
- 任务：train 800、val 200、test 200；每个 split 的四类故障均衡。
- Oracle：1200/1200 成功，全部 Verifier 与 Replay 通过，额外副作用和未授权修改为 0。
- SFT：train 2600、val 700 条，其中共有 1000 条显式停止目标。
- Preference：train 1400、val 350 对，1750 对均重新执行验证；选择状态摘要完全相同。
- 4000 个候选中丢弃 2250 个：1150 个偏好不确定，1100 个双方成功；另排除 200 个 test 任务。
- JSONL 重复样本 0、切分泄漏 0、test 训练样本 0、3300 个 SFT 输入因果投影全部匹配。
- 独立 validate 命令通过；根据保存的配置完整再生成一次，14 个产物逐字节一致。
- 本地验收摘要：outputs/v0.2_acceptance.json。原始数据质量报告位于
  data/v0.2_seed42/quality_report.json，分布与丢弃统计位于 report.json。

这些结果验证数据流水线的可执行性与确定性，尚未进行 Qwen3 微调或测量模型收益。

## v0.3-A — Stateful Tools & Real LLM Collection

新增 recovery_agent/stateful/，不改动 v0.1/v0.2 核心模块与冻结数据。
运行依赖为现有 Python 标准库、pytest，以及已安装的 Git；无需 GPU 或模型权重。

### 状态化工具与场景

保留原有六个工具，新增 git_status、git_diff、git_add、git_commit、git_checkout、
execute_select、insert_record、delete_record，共 14 个唯一工具名。
update_record 同时支持真实 SQLite 的 accounts/entries 表；原有内存数据库更新仍保留，
两张 SQLite 表名在扩展环境中为保留名。SQLite 更新不支持 create/upsert，缺失行须显式 insert。

Git 在每个任务的独立临时仓库初始化；固定作者、时间、默认分支，关闭外部配置与 hooks。
只能操作任务指定路径；checkout 只恢复单个文件，source 仅 head/index，并要求提供
与实际内容匹配的 expected_content。无 push、reset、clone、shell 或任意 Git 命令入口；
工具不能读写 .git 元数据。状态摘要保存 HEAD、index、提交树与工作区状态，而非 .git 字节。
Git 的提交冲突是 expected_head 乐观并发冲突，文件冲突是受保护 restore 冲突，
当前没有多分支 merge/rebase 冲突解析。

SQLite 使用每个环境独立的本地 :memory: 连接，启用外键与事务。
工具只拼装固定 SQL 模板，表/列白名单与类型校验，所有值使用参数绑定；
execute_select 不接受任意 SQL 字符串或脚本。事务失败由真实 CHECK 约束触发并 rollback，
当前为单写入事务，不包含跨工具的长期事务或多行转账协议。
所有成功写入进入 Audit；失败尝试保留 ToolResult。Reset 重新建立两种隔离状态。

八个模板、每个默认两个实例，共 16 个任务：

| 模板 | 恢复机制 | 默认 split |
| --- | --- | --- |
| git_stage_missing | status/diff 诊断，暂存指定文件后提交 | train |
| git_revision_conflict | 读取真实 HEAD 后重新执行乐观提交 | train |
| git_no_change | 诊断后停止，不能创建空提交 | train |
| git_checkout_conflict | 读取当前内容后受保护恢复 | val |
| sqlite_primary_key | 查询既有行，用更新替代重复插入 | train |
| sqlite_unique | 查询唯一键占用，使用目标要求的正确值插入 | train |
| sqlite_transaction | 验证失败写入已回滚，执行合法值更新 | train |
| sqlite_dependency | 先建立父记录，再插入外键子记录 | test |

train/val/test 为 12/2/2 个任务；新任务与 v0.2 数据完全独立。
Val/Test 可以评测、认证和 Replay，但训练导出与默认在线采样仅使用 Train。
正式采集 50～100 个 train 实例时可增加 --variants；这仍只是八种机制的参数变体。

### Mock、真实 API 与断点续采

```powershell
python -m pytest -q
python scripts/collect_stateful.py --count 16 --output outputs/v0.3_mock
python scripts/collect_stateful.py --clean --count 16 --output outputs/v0.3_clean
python scripts/collect_stateful.py --count 16 --output outputs/v0.3_mock --resume
```

默认 Mock 是明确标记的 oracle_mock：参考计划脚本用于协议/环境自检，
其成功率不代表任何模型能力。测试另使用异常/非法动作的脚本响应，验证错误处理。

只有加 --online 才读取 .env 并请求真实 API；测试与 Mock 命令不会发请求。
.env 只需 MODEL_NAME、OPENAI_BASE_URL、OPENAI_API_KEY；密钥文件已加入 Git ignore。
base_url 必须为无内嵌凭证的 HTTPS 地址；Authorization 不会写进 Prompt、数据或日志。

```powershell
python scripts/collect_stateful.py --online --count 4 --seed 42 --output outputs/v0.3_online --max-requests 16 --episode-requests 8 --max-tool-calls 8 --max-tokens 4096 --max-total-tokens 160000 --max-cost-cny 10
python scripts/collect_stateful.py --online --count 4 --seed 42 --output outputs/v0.3_online --max-requests 16 --episode-requests 8 --max-tool-calls 8 --max-tokens 4096 --max-total-tokens 160000 --max-cost-cny 10 --resume
```

上面的 online 命令会产生 API 用量；只能在需要真实实验时手动执行。
若需指定温度或超时，使用 --temperature / --timeout；默认不传 temperature，
使 qwen3.8-max 保持服务端默认值 0.6，不人为降到会被服务修正的范围。
API 协议参考：[千问 OpenAI Chat 手册](https://platform.qianwenai.com/docs/api-reference/chat/openai-chat)。

请求与工具次数分别受硬预算控制；多工具调用按返回顺序执行，超出剩余预算时整批拒绝，
不部分执行、不静默修复、不调用 Oracle 替代模型动作。JSON 参数错误、空响应、截断、
调用 ID 重用等作为明确解析反馈保留；普通非空文本表示模型停止，是否完成仍由 Verifier 判定。
HTTP/限流/超时错误保留，传输层不自动重试；episode 最多容忍两次 API 错误。
思考输出触及 max_tokens 而截断时保留失败，可在新的实验版本显式增加预算。

费用控制使用可配置单价估算：默认输入 30 元/M、输出 90 元/M 是保守预算参数，
不是宣称真实账单单价。请求前按 UTF-8 字节及额外 framing 预留输入预算，
按 max_tokens 预留输出；得到 usage 后结算，超时/未知计费保留整笔预留。
请求数、Token 和估算费用写前预留并原子落盘，断点续采不能重置额度。
实际账单仍以平台合同、地区和价格为准；可在 LLMConfig 调整单价。
输出目录只允许一个 Collector 同时写入；异常崩溃遗留 .collection.lock 时，
先确认对应进程已退出，再删除该目录里的锁文件后续采。

### 轨迹、训练候选与指标

每个完成的 episode 立即追加并 fsync trajectories.jsonl；已完成 ID 不再请求 API。
session.json 记录样本与配置摘要，resume 时必须匹配；尾部中断的不完整 JSON 行会被丢弃，
完整记录中的格式错误不会被忽略。单任务异常保留异常类型并继续下一任务。
中断在任务内部时，已使用/预留费用保留，未完成任务会从隔离初始环境重新执行。

每条记录包含公开 Prompt、经密钥脱敏的原始 Response（包括服务返回的思考字段）、
结构化解析、调用与 ToolResult、前后状态摘要、最终状态及 Audit、Verifier、来源、
模型/配置版本、Token Usage、延迟、API 错误和终止原因。
Replay 不请求模型，重新执行全部已记录动作，核对逐步结果、状态摘要、最终状态与 Audit。

training_candidates/ 将 oracle_mock 与 llm 的 SFT 文件分开。
只有 train 中成功、无违规且通过 Replay 的样本可用于 SFT；失败动作、无意义重复调用、
非法动作和未实际执行的超预算调用不会导出正例。保留模型实际动作，监督范围为最后 assistant。
candidate_evidence.jsonl 是首个决策的同状态隔离分支原型：chosen 来自参考计划，
rejected 来自真实记录；重新执行两条分支，只有严格可验证的偏好才保留，
固定参考续程引入的失败/重复写入不能强行当负例。原始证据含特权环境信息，
不能整体作为模型输入。当前不训练 SFT/DPO，也不声称已生成最终高质量困难负例集合。

report.json 指标约定：

- Recovery Success Rate：恢复 episode 最终状态符合目标，且没有未授权/额外写入的比例。
- Clean Task Success Rate：单独 --clean 评测跳过起始失败调用；仍从任务定义的初始资源状态开始。
- Invalid Tool Call Rate：非法调用/解析尝试数除以全部模型工具尝试数，起始失败调用不计。
- Repeated Failure Rate：同工具、参数和错误码重复失败次数除以恢复调用数。
- Unnecessary Tool Call Rate：相同资源/账本状态下重复调用，或重复相同失败的调用比例；
  这是保守操作性定义，不用最短 Oracle 步数判断所有诊断调用是否必要。
- Unsafe Action Rate：拒绝的 UNSAFE_ACTION 或引入新账本违规的恢复调用比例。
- Average Recovery Steps：实际执行的恢复工具调用均值；解析错误不算执行步骤。
- P50/P95 Latency：所有模型请求的记录延迟分位数；另保留每个 episode elapsed_ms。
- Tokens per Successful Recovery：整批恢复请求使用的总 Token / 成功恢复数；
  包含失败任务的成本，无成功时为 null。缺失 usage 不虚构 Token，用预算预留单独反映风险。
- extra_tool_calls 是相对参考计划长度的描述性差值，不直接等同于无意义调用。

Local Git 子进程每次 10 秒超时；API 默认 90 秒。任务 seed 固定的是样本和环境，
不能保证非零温度远端模型重复给出相同动作；确定性验收针对保存轨迹 Replay。

### v0.3-A 本机验收（2026-10-12）

- 全部测试：190 passed。Git/SQLite 场景、约束回滚、故障、Mock、解析错误、预算、
  API 错误、同状态候选证据、Replay、锁与断点续采均有测试。
- Mock 恢复：16/16 成功；Clean-start Mock：16/16 成功；两组各 16 条均通过 Replay。
- 真实 qwen3.8-max：采样 4 个 train 任务，实际 16 次请求、3 个有模型决策的任务全部成功。
  第 4 个任务在请求前被预算阻止，collection_status=budget_not_evaluated；
  整批受预算约束成功率为 3/4（75%），已尝试任务为 3/3，不把未执行记录解释成模型失败。
- 真实记录共 18 次恢复工具执行、相对参考计划多 11 次调用；保守定义下没有无效重复、
  非法调用或不安全副作用。额外调用不自动等同于无意义调用。
- 真实 API Token：30626（输入 28066、输出 2560）；请求 P50/P95 约 5.96/11.40 秒。
  费用账本按输入 30、输出 90 元/M 的保守参数估算约 1.07238 元，非平台实际账单。
- 真实模型训练候选：15 条 llm SFT；Mock 恢复候选：38 条 oracle_mock SFT，来源分离。
  本批没有足够可靠的实际 Chosen/Rejected 对，candidate_evidence 为空，不人为制造偏好。
- 实际执行在线 --resume 后，请求账本仍为 16，Token 与估算费用未增加。
- v0.2 冻结数据与本轮开始时的原有核心文件哈希全部一致；密钥扫描未在代码、日志、
  轨迹或训练数据发现 API Key。原始 .env 不修改、不提交。

本地报告：outputs/v0.3_mock/、outputs/v0.3_clean/、outputs/v0.3_online/；
最终摘要：outputs/v0.3_acceptance.json。此规模仅验证采集链路，不是正式泛化 benchmark。

## v0.3-B — 任务多样性、真实轨迹与数据加固

独立实现位于 `recovery_agent/hardening/`。复用现有 Environment、OnlinePredictor、
JSONL Collector 和 Episode Runtime；原 Collector 只增加默认关闭的依赖扩展参数。
v0.2 / v0.3-A 数据及原测试集保持原样，不把新增模板回填旧版本。

### 新模板与隔离边界

保留原 8 个模板族，新增以下 8 种恢复机制。原 14 个工具保留，增加受限
`git_switch(branch)`，共 15 个。只能切换当前临时任务的已存在白名单分支，
不能访问用户仓库、创建远程连接、执行 reset/rebase/shell 或任意 SQL。

| 新模板 | 初始故障与正确恢复机制 |
| --- | --- |
| git_wrong_branch | 工作树在 feature；检查状态、切回 main、暂存并提交 |
| git_stale_index | index 比工作树旧；比较差异、重新暂存最新内容 |
| git_multifile_revision | 两文件 revision 不一致；读实际 revision、修复依赖文件、共同提交 |
| git_compensating_commit | 错误内容已提交；禁止回退，读取备份并追加补偿提交 |
| sqlite_delete_dependency | 删除父记录触发外键约束；先查并删除子记录再删父记录 |
| sqlite_unique_swap | 邮箱互换触发 UNIQUE；经过允许的临时值完成交换并保留余额 |
| sqlite_relink | 新父记录缺失；查询旧余额、创建新父记录、重挂子记录、删除旧父记录 |
| mixed_db_git_report | 缺少可提交报告；查询 SQLite、计算总额、写文件、暂存并提交 |

每个任务保存初始状态、失败调用、公开目标、最终合同、逐资源写入额度、
允许中间状态、预算和 Oracle 参考计划。Oracle 根据真实查询结果解析动作。
最终 Verifier 同时检查文件、SQL、Git 分支/提交数/index/tree 和完整 Audit。
JSON 输出允许无害的空白及键顺序差异。失败的约束写入不会生成 Mutation。

SQLite 仍使用受限表/列、绑定参数和每次写入的事务回滚；没有新增跨工具长事务
或通用 SQL。Git 版本冲突由真实分支、index、文件、提交历史及显式依赖合同产生；
当前不模拟多人 merge/rebase。临时 Git 快照缓存只在本隔离环境的写工具后失效。

### 冻结数据与评测协议

正式快照：`data/v0.3b_seed42_final/`，seed=42，6 个参数变体，16 个机制族，
共 96 个任务：Train 52、IID Validation 13、IID Test 13、OOD 18。
IID 共享机制但 variant ID/资源命名空间不重叠；OOD 整族保留
git_compensating_commit、sqlite_relink、sqlite_dependency。
这些 OOD 任务不能被用来选择困难负例或训练参数。
每个 variant 只是参数变体，不声称它是新恢复机制。

Manifest 保存配置、任务摘要、工具 Schema Hash 和文件 SHA-256。
`freeze` 拒绝覆盖已有目录；`validate` 对生成结果逐对象重建，并检查文件哈希。
相同任务与环境可以确定性 Replay；远程模型本身不保证逐字确定性。
`compare` 检查两次在线实验任务、初始状态摘要、预算、schema、解码参数一致，
允许模型 ID/Base URL 不同；不能把 Mock 与真实模型当作同一性能实验。

`--clean` 为配对正常工作流：省略初始失败动作和瞬时注入，从原始状态直接执行
合法计划，仍保留真实 Git/SQL 约束。`--clean --clean-kind control` 是另一个
目标已满足、不得新写入的诊断控制。两种 Clean 协议明确记录，不能混报。

所有命令在项目根目录执行，使用本项目 Python 虚拟环境：
```powershell
Set-Location D:\pyproject\RecoveryAgent
.\.venv\Scripts\Activate.ps1
python -m pytest -q

# 新快照：已有目录请使用新名字，不能覆盖冻结版本
python scripts/harden_data.py freeze --output data/v0.3b_reproduced --variants 6 --seed 42

# 默认不读取凭证、不产生 API 费用；包含独立评测分组
python scripts/harden_data.py collect --dataset data/v0.3b_seed42_final --group all --output outputs/v0.3b_mock_new
python scripts/harden_data.py collect --dataset data/v0.3b_seed42_final --group all --output outputs/v0.3b_mock_new --resume
python scripts/harden_data.py collect --dataset data/v0.3b_seed42_final --group all --clean --output outputs/v0.3b_clean_new
python scripts/harden_data.py export --input outputs/v0.3b_mock_new --output outputs/v0.3b_reference_export_new
python scripts/harden_data.py validate --dataset data/v0.3b_seed42_final --input outputs/v0.3b_mock_new --export outputs/v0.3b_reference_export_new
python scripts/harden_data.py validate --dataset data/v0.3b_seed42_final --input outputs/v0.3b_clean_new
```

### 显式在线：单条 Smoke 与用户触发的批量采集

`MODEL_NAME`、`OPENAI_BASE_URL`、`OPENAI_API_KEY` 从进程环境或本地 .env 获取。
默认要求带日期的模型快照 ID；浮动别名必须显式 `--allow-floating-model`，
并记录无法严格固定后端模型版本的限制。
本轮使用 [qwen3.8-max-0902 快照](https://www.qianwenai.com/models/qwen3.8-max-0902)。
不传 temperature，保留服务默认值 0.6；可以显式 `--temperature`，实际服务端
如何调整需以供应商响应为准。响应返回 model ID 与请求 ID 分开保留。

下面一条命令是一个 Episode，可能包含多次决策请求：
```powershell
$env:MODEL_NAME = 'qwen3.8-max-0902'
python scripts/harden_data.py collect --dataset data/v0.3b_seed42_final --online --task-id 0.3-B-sqlite_delete_dependency-000 --output outputs/v0.3b_online_single_new --max-requests 8 --episode-requests 8 --max-tool-calls 12 --max-tokens 4096 --max-total-tokens 80000 --max-cost-cny 3 --timeout 90
python scripts/harden_data.py collect --dataset data/v0.3b_seed42_final --online --task-id 0.3-B-sqlite_delete_dependency-000 --output outputs/v0.3b_online_single_new --max-requests 8 --episode-requests 8 --max-tool-calls 12 --max-tokens 4096 --max-total-tokens 80000 --max-cost-cny 3 --timeout 90 --resume
python scripts/harden_data.py export --input outputs/v0.3b_online_single_new --output outputs/v0.3b_llm_export_new
python scripts/harden_data.py validate --dataset data/v0.3b_seed42_final --input outputs/v0.3b_online_single_new --export outputs/v0.3b_llm_export_new
```

以下收费批量命令 **本轮不执行**，只有用户自行明确触发时才运行。
默认快照 Train 52 足以抽取 50 个独立任务实例；若要 100 个，请先生成 12 variants
新快照（Train 104），保持该次实验数据固定。实例独立不表示机制族独立。
```powershell
# 用户触发的 50 个 Train Episode；预算耗尽会保留记录并停止计费请求
python scripts/harden_data.py collect --dataset data/v0.3b_seed42_final --online --group train --count 50 --seed 42 --output outputs/v0.3b_teacher50 --max-requests 600 --episode-requests 12 --max-tool-calls 12 --max-total-tokens 2000000 --max-cost-cny 100
# 断点续采必须使用完全相同的模型、任务与预算参数，并追加 --resume
python scripts/harden_data.py export --input outputs/v0.3b_teacher50 --output outputs/v0.3b_teacher50_export
python scripts/harden_data.py validate --dataset data/v0.3b_seed42_final --input outputs/v0.3b_teacher50 --export outputs/v0.3b_teacher50_export

# Val/Test/OOD 独立评测目录，不能合并进训练候选
python scripts/harden_data.py collect --dataset data/v0.3b_seed42_final --group iid_validation --output outputs/v0.3b_iid_validation_mock
python scripts/harden_data.py collect --dataset data/v0.3b_seed42_final --group ood --output outputs/v0.3b_ood_mock
python scripts/harden_data.py compare --first outputs/model_a --second outputs/model_b
```

API 费用采用保守的本地预留价（输入 30、输出 90 元/M Token），并非实际账单。
写前预留全局请求/Token/费用，成功按 usage 结算；未知计费维持预留。
不会自动重试 HTTP；同一 Episode 最多两次 API 错误后结束。
断点仅跳过已完成 Episode；进程在 Episode 中途崩溃可能重做该条，
之前已发请求的预算预留仍保留，不能保证供应商端 exactly-once。
运行目录单写者锁避免并发重复采集。不要同时启动多个进程写同一目录。

### 行为证据与候选数据

`call_review.jsonl` 保存逐调用分类及状态、结果、Verifier 依据：
Diagnostic（新增相关信息）、Redundant（相同查询范围与结果）、
Non-progress（相同物理状态重复同一失败）、Invalid（schema/工具拒绝）、
Unsafe（权限拒绝或新账本违规）、Recovery Action（减少未满足条件）；
证据不足为 uncertain，包括必要但不直接降低目标差距的中间操作。
超出参考步数不自动判错。分类是保守规则和人工复核依据，不是 Oracle 全知的
“最优调用数”。Invalid Tool Call Rate 包含未执行的解析/预算/ID 错误尝试；
逐调用标签只描述实际执行动作，未执行尝试另行统计。
报告含任务级、初始错误类型级、终止原因、调用开销、请求 P50/P95、
含失败开销的 Tokens per Successful Recovery，并区分未评估的预算停止。

SFT 只保留 Train 中最终成功且 Replay/Verifier 通过、确实执行成功的决策。
失败、非法、不安全、冗余、无进展和 uncertain 动作不监督；具有独立依赖证据的中间动作可保留。
重建每个时点公开 Prompt，精确核对历史前缀，拒绝 Expected State、隐藏 seed、
Oracle 计划或未来结果。模型已经看到的初始失败调用/过去错误是恢复输入，
不会成为目标 Assistant。JSONL 使用 messages / tool_calls / tools；
`loss_scope=final_assistant_only`，训练端必须将历史 Assistant、System/User/Tool
token 标签设为 -100，仅监督最后 Assistant，不得全会话计算 loss。
API 返回 reasoning_content 保留在审计轨迹，不作为 SFT 目标。

`llm_sft_train.jsonl` 和 `oracle_reference_sft_train.jsonl` 分开；
source_transport 保留 oracle_mock 原始来源，不能将脚本的成功率报告为 LLM 性能。
可通过 `export --legacy-input outputs/v0.3_online` 单独导入旧 v0.3-A 真实轨迹，
使用旧 schema/Replay 验证并写入新的 legacy_v03a 子目录，不改旧文件，
旧版本与新版本样本不混用同一个 Schema Hash。

DPO 预备遍历真实 LLM 所有决策（不仅第一步），在模型精确前缀的隔离环境中
执行可信参考动作与实际模型动作。状态和 Audit 相同才保留候选。
仅双方执行且 chosen 成功、rejected 有明确证据失败（非法副作用、提前停止、
最终预算内未达合同）才保留。仅固定后续计划失败不证明动作更差，
写入 uncertain_candidates 和丢弃原因；离开参考状态或多动作批次暂不强制配对。
成功、失败、提前停止和非法调用的原始 Episode 全保留，即使没有任何 DPO 对。
`validate` 独立重放分支，并核对 rejected 来源、候选序列化和偏好证据。

### 下一阶段 A100 / Qwen3-1.7B 准备清单

- 固定 Teacher 快照/采样预算，用户触发 50–100 Train 任务；记录真实失败及覆盖率。
- 人工审核逐调用 uncertain、数据合同和成功轨迹，冻结可信 SFT 候选 Manifest。
- 在 GPU 环境另行准备 Qwen3-1.7B tokenizer/chat template，核对 tool_calls、
  tools、结束 token、JSON 参数格式和 final-assistant-only label mask；先抽样解码审查。
- 先测未微调 Baseline，固定 IID Validation/IID Test/OOD/Clean/Recovery、初始状态、
  schema、预算和解码；只使用 Train 训练，IID Validation 选参数。
- 再做小规模 LoRA-SFT，并用相同协议对比；记录成功率和副作用/请求/Token 成本。
- DPO 留待有足够真实学生模型错误与可靠同状态证据后进行；不得用 Teacher/Mock
  成功率推断 Student 或微调收益，不承诺现有候选数量足够训练。

本阶段仅完成数据与验证基础设施；没有下载模型、安装 GPU 栈或执行训练。

离线行为复核保留原始采集文件，不发起 API；输出目录不得已存在：
```powershell
python scripts/harden_data.py review --input outputs/v0.3b_online_smoke --output outputs/v0.3b_online_review_new
python scripts/harden_data.py export --input outputs/v0.3b_online_review_new --output outputs/v0.3b_online_review_export_new
```

### v0.3-B 实际验收（2026-10-12）

- `python -m pytest -q`：254 passed；数据质量校验加固后的定向回归另有 3 passed。
- 新增 8 个独立机制族，保留原 8 族；15 个工具（新增受限 git_switch）。
- 正式数据 96 个任务，Train/IID Validation/IID Test/OOD = 52/13/13/18。
  第二份快照逐字节一致。旧版 106 个源代码/数据/输出文件的 SHA-256 保持不变；
  原 Collector 仅增加兼容扩展参数，原有 190 项测试全部通过。
- Mock 恢复 96/96、配对 Clean 96/96；192 条全部完成独立 Replay/Verifier。
  这些是 Oracle 脚本的环境/流程验证结果，不代表真实模型性能。
- 本轮新采集真实 LLM **1 个 Train Episode**，模型 qwen3.8-max-0902：
  8 次 API 请求、10 次恢复工具调用、17,330 Token（输入 15,920 / 输出 1,410），
  最终 Verifier/Replay 通过，无非法调用或未授权副作用。
  终止原因是 request_budget，模型没有主动完成消息；4 个调用标为 uncertain。
  请求 P50/P95 约 6.36/7.94 秒；保守预留价结算约 0.6045 元，非平台账单。
  实际运行 --resume 后请求计数仍为 8，未追加 API 请求。
- 正式候选：208 条独立 Oracle Reference SFT、4 条可信真实 LLM SFT；
  另独立导入旧 v0.3-A 的 15 条 LLM SFT（不计为本轮新采集）。
  真实 DPO 已验证对为 0；保留 1 个不确定执行证据，不强行标偏好。
  全部成功候选通过因果 Prompt/实际目标/来源/Replay 校验，
  重复样本与 Test 训练样本为 0；API Key 扫描未发现泄漏。

正式交付路径（所有路径相对项目根目录）：

| 内容 | 路径 |
| --- | --- |
| 冻结任务/配置/Manifest | data/v0.3b_seed42_final/ |
| 原始真实采集及预算 | outputs/v0.3b_online_smoke/ |
| 真实行为审核与质量报告 | outputs/v0.3b_online_review/ |
| Mock 恢复与完整质量报告 | outputs/v0.3b_mock_verified/ |
| 配对 Clean 与质量报告 | outputs/v0.3b_clean_verified/ |
| 可信真实 SFT / uncertain 偏好证据 | outputs/v0.3b_llm_trusted_export/ |
| 独立 Oracle SFT / 旧版来源导入 | outputs/v0.3b_sft_reference_final/ |
| 全量测试日志 | outputs/v0.3b_pytest_acceptance.log |
| 验收摘要及候选文件 Hash | outputs/v0.3b_acceptance.json |

重新汇总本轮已验证结果（不发起 API）：
```powershell
python scripts/report_hardening.py
```

可以进入 A100 上的 Baseline 与 tokenizer/工具格式烟测准备；
目前不满足正式 LoRA-SFT 数据量和机制覆盖要求。
尚缺用户触发的 50–100 条真实 Train 轨迹、真实 Git/混合场景验证、
足量可验证模型失败，以及训练端 chat template / loss mask 检查。
不存在已测得的微调收益，也没有执行任何训练。

### 中断后的交付状态

v0.3-B 实现与主要验收已完成，以下为本轮已记录的结果：

- 全量 pytest：254 passed；Mock Recovery / Clean 各 96/96，通过 Replay / Verifier。
- 新增 8 个恢复机制，正式快照 96 个任务，旧版冻结数据保留。
- 本轮真实 LLM：1 个 Episode、8 次请求、17,330 Token；最终状态通过验证，但以请求预算结束。
- 可信候选：4 条真实 LLM SFT、208 条独立 Oracle SFT；可靠 DPO 对 0，uncertain 候选 1。
- 未执行收费批量采集、模型下载或 GPU 训练。

验收报告：`outputs/v0.3b_acceptance.json`。
后续按上方命令由用户显式触发 50–100 条 Train 轨迹采集，再检查 tokenizer/chat template 与最后 Assistant 的 loss mask。
可进入 A100 Baseline/格式验证准备，当前真实数据量尚不足以进行正式 LoRA-SFT。

## v0.3-C — Recovery Termination & Tool Efficiency

本阶段只增加停止协议、可选终止检查及离线诊断，不修改工具 Schema、业务执行、
Environment、Verifier、Fault Injector、任务模板或冻结数据。原有指标口径保持不变。
真实 Teacher10 原始轨迹不覆盖、不清洗；新报告写入独立目录。

### 配置与兼容性

- `--stop-protocol legacy` 是默认值，Prompt 和原实验 session 配置保持兼容。
- `--stop-protocol termination-v0.3c-1` 加入固定公开指令：完成用户目标后停止工具
  并返回完成消息；不要重复确认已经由可见结果建立的事实；目标未完成或有必要
  不确定性时仍可诊断。只改 System 指令，不发送隐藏 Verifier、Expected State、
  Oracle Plan、故障 seed 或私有账本信号。
- `--stop-on-verified-success` 默认关闭。开启后每个工具（包括初始失败调用）
  执行结束时检查现有完整 Verifier，成功即以 `runtime_verified_success` 结束。
  一个 Assistant 批次中也会立即停止后续调用；未执行候选仍保留在原始 Response，
  Audit 和已执行事件不回滚、不删改。此策略属于 Runtime-assisted，
  不计入模型自主完成率，不将检查结果追加到 Prompt。
- 新选项在 session / 轨迹的 `termination_protocol` 中独立记录，
  使用不同输出目录；旧版本默认无新增协议配置，不覆盖既有实验。
- v0.3-C 实验轨迹用于独立诊断；原 v0.3-B SFT exporter 的 Prompt 校验仍按 legacy
  工作，不把新协议或 Runtime-assisted 轨迹直接当作旧版训练候选。

### 五项新增指标

成功一律由原 Verifier 同时检查完整最终状态和累计副作用账本。
非法写入后，即使目标对象再次匹配，也不能算完整成功。逐前缀证据保存于
`episodes.jsonl` 的 `verifier_prefixes`；Replay 核对每步结果、状态和最终 Audit。

| 指标 | 口径与分母 |
| --- | --- |
| First Goal Achieved Step | 首次完整成功的已执行恢复工具步数；初始/失败前缀已满足目标为 0；从未成功为 null。平均值只使用曾成功 Episode。 |
| Post-success Tool Calls | 首次完整成功后的实际恢复调用数，包含同一 Assistant 批次中的后续调用；从未成功记 0。总数与曾成功 Episode 的均值分别报告。 |
| Model Stop Accuracy | 曾完整成功的模型自主模式 Episode 中，以有效 assistant_text 主动结束且最终完整成功的比例。允许晚停，其延迟由 Post-success Calls 表达，不把晚停说成及时停止。 |
| Success Before Budget Exhaustion | 预算终止处理前，合法执行前缀中是否曾完整成功；最后一个获准执行的工具动作也算，未要求仍有剩余配额。分母为同一停止策略的全部 Episode。 |
| Autonomous Completion Rate | 模型自主模式下，assistant_text 主动结束且最终完整成功 / 该模式全部 Episode。Runtime-assisted 全部排除。 |

Model Stop Accuracy 不等同于“成功后零调用”；一次合理确认也可能出现在完整成功之后。
Post-success 调用计数本身不证明所有调用都无意义，不替换 v0.3-B 的行为分类指标。
Runtime-assisted 与 Model-autonomous 使用独立分组和分母；Runtime 组的
Model Stop Accuracy / Autonomous Completion Rate 为不可适用的 null。

### 最小本地命令（无 API）

```powershell
Set-Location D:\pyproject\RecoveryAgent

# 只读取原 Teacher10；不能把输出指向原采集目录，已有报告目录不覆盖
python scripts/diagnose_termination.py --input outputs/v0.3b_teacher10 --output outputs/v0.3c_teacher10_diagnostic_new

# 新停止 Prompt 的本地 Mock（保持原工具/任务）
python scripts/harden_data.py collect --task-id 0.3-B-sqlite_primary_key-000 --stop-protocol termination-v0.3c-1 --output outputs/v0.3c_mock_prompt

# 可选 Runtime-assisted Mock，默认配置仍然关闭
python scripts/harden_data.py collect --task-id 0.3-B-sqlite_primary_key-000 --stop-protocol termination-v0.3c-1 --stop-on-verified-success --output outputs/v0.3c_mock_runtime
python scripts/diagnose_termination.py --input outputs/v0.3c_mock_runtime --output outputs/v0.3c_mock_runtime_diagnostic

python -m pytest -q tests/test_termination_v03c.py
python -m pytest -q
```

上述 Mock 收集命令没有 --online，绝不调用真实 API。离线诊断入口根本没有 API 开关。
报告包含 `episodes.jsonl`、`summary.json`、逐任务表 `report.md`，
以及原任务/轨迹/session 的 SHA-256。

### Teacher10 离线结论

正式报告：`outputs/v0.3c_teacher10_diagnostic/`。以下只来自既有轨迹 Replay，
没有重新请求模型；Teacher 模型 ID 为 qwen3.8-max-0902。

| 任务（省略共同前缀 0.3-B-） | 首次成功步 | 成功后调用 | 最终成功 | 终止原因 |
| --- | ---: | ---: | --- | --- |
| git_multifile_revision-002 | 7 | 4 | 是 | assistant_text |
| mixed_db_git_report-002 | 8 | 4 | 是 | tool_budget |
| sqlite_delete_dependency-002 | 6 | 6 | 否 | tool_budget |
| git_checkout_conflict-000 | 4 | 4 | 是 | tool_budget |
| mixed_db_git_report-000 | 8 | 4 | 是 | tool_budget |
| sqlite_primary_key-002 | 2 | 6 | 否 | tool_budget |
| git_stage_missing-000 | 4 | 4 | 是 | tool_budget |
| mixed_db_git_report-003 | 7 | 5 | 是 | request_budget |
| sqlite_delete_dependency-001 | 8 | 4 | 否 | tool_budget |
| git_no_change-003 | 0 | 8 | 是 | tool_budget |

10/10 曾满足完整目标，最终仅 7/10 成功。首次成功平均 5.4 步；
103 次恢复调用中，49 次在首次成功之后（均值 4.9）。
3 个 SQLite Episode 完成操作后，又把总结写入未授权的 report.txt，
违反“保留无关状态”的完整合同；原 Verifier 正确保留失败，不因数据库仍匹配而洗白。

终止：assistant_text 1、tool_budget 8、request_budget 1。
Model Stop Accuracy = 1/10（10%）；Autonomous Completion Rate = 1/10（10%）。
Success Before Budget Exhaustion = 10/10（100%），包含最后合法动作的前缀成功。
唯一主动停止的 Episode 也有 4 次成功后调用，并非成功后立即停下。
git_no_change 初始已满足目标，故首次成功为 0，随后 8 次调用仍需人工判断必要性。
这些是旧协议行为诊断，不能据此声称新 Prompt 或 Runtime 策略已改善真实模型表现。

### 下一步真实 LLM 对照（本轮未执行）

保持 Runtime-assisted 关闭，以原 Teacher10 为 legacy 对照；新协议只改变停止指令。
固定同 10 个 Task ID/初始状态、qwen3.8-max-0902、seed=42、Schema Hash、
temperature=None（相同模型服务默认值）、max_tokens=4096、timeout=90、
每任务请求 12、工具上限 12（仍受任务自身预算限制）、
全局请求 120、Token 400000、保守费用上限 20 元，使用独立输出目录。

只有用户明确决定进行收费实验后，才运行下面的示例；本轮没有执行：
```powershell
$env:MODEL_NAME = 'qwen3.8-max-0902'
python scripts/harden_data.py collect --online --group train --count 10 --seed 42 --stop-protocol termination-v0.3c-1 --output outputs/v0.3c_teacher10_new_stop --max-requests 120 --episode-requests 12 --max-tool-calls 12 --max-tokens 4096 --max-total-tokens 400000 --max-cost-cny 20 --timeout 90

# 检查新旧数据/模型/预算/解码/Schema 一致，只允许 Prompt 版本不同
python scripts/diagnose_termination.py --compare-first outputs/v0.3b_teacher10 --compare-second outputs/v0.3c_teacher10_new_stop
python scripts/diagnose_termination.py --input outputs/v0.3c_teacher10_new_stop --output outputs/v0.3c_teacher10_new_stop_diagnostic
```

比较入口拒绝模型、任务摘要、预算、seed、Schema 或解码配置不一致，
也拒绝把 Runtime-assisted 运行混入 Prompt A/B。先逐任务比较是否主动停止、
成功后调用数和新增副作用，再看汇总均值；小样本结果不等于训练收益。
Runtime-assisted 的部署策略对照需使用另一目录和单独统计，不替代模型自主停止测试。

### v0.3-C 本地验收

- 核心测试：`python -m pytest -q tests/test_termination_v03c.py`，5 passed（0.34 秒）。
- 全量测试：`python -m pytest -q`，259 passed（371.72 秒）；日志：`outputs/v0.3c_pytest.log`。
- 10 条原始轨迹全部通过离线 Replay；原数据及 Teacher10 目录的文件哈希保持不变。
- 本轮没有真实 API 请求、训练、数据重生成或补充边界测试。
- 验收摘要：`outputs/v0.3c_acceptance.json`；任务完成后结束 Desktop 操作。

## v0.3-C.2 — Redundant Mutations & Termination Refinement

仅新增独立离线归因与 System 停止指令，复用原 Replay / 完整 Verifier。
Git/SQLite 工具、Audit 写入规则、allowed_resources、恢复预算、任务模板、
Collector 和 Training Pipeline 均不修改。旧轨迹与旧指标保留原样。

### 双轮 Teacher10 的逐任务证据

离线输入：`outputs/v0.3b_teacher10/` 与 `outputs/v0.3c_teacher10_new_stop/`。
独立输出：`outputs/v0.3c2_paired_teacher10/`：

- `episodes.jsonl`：20 个 Episode 的首次完整成功、成功后调用、完整 Verifier、
  失败原因及逐 Mutation 证据。包含实际 scoped state、新增 Audit、资源次数/契约
  上限、调用前后 Verifier，不凭动作名称判定。
- `pairs.jsonl`：同一 Task 的旧/新行为、调用数差异及标签增减。
- `summary.json`：沿用原五项终止指标，另列新的诊断标签统计及源文件 SHA-256。
- `report.md`：20 行逐任务表、10 对行为差异及重点 Git 任务的原始执行证据。

| 指标 | Legacy | termination-v0.3c-1 |
| --- | ---: | ---: |
| 最终完整成功 | 7/10 | 7/10 |
| 曾完整成功 | 10/10 | 9/10 |
| 首次完整成功平均步数 | 5.4（10 个） | 5.333（9 个） |
| 成功后调用总数 / 平均数 | 49 / 4.9 | 14 / 1.556 |
| Model Stop Accuracy | 1/10 | 7/9 |
| Autonomous Completion Rate | 1/10 | 7/10 |
| assistant_text / tool_budget / request_budget | 1 / 8 / 1 | 9 / 1 / 0 |

上述新旧真实轨迹已通过同任务、模型、Schema、预算、解码、seed 和 Runtime 开关
的可比性检查。旧轮 3 个失败均为完成数据库工作后写入未经允许的 report.txt。
新轮 2 个仍有同类未授权写入；另 1 个为下面的冗余暂存/额度超限。
实际这 20 条没有独立可确认的 premature_stop 或 incomplete_recovery，
不能把 3 个新轮失败统一解释为“Prompt 让模型提前停了”。

### git_multifile_revision-002：不是提前停止

初始 report.txt 的 revision=3，已经正确暂存；metadata.json 的 revision=2。
初始化暂存属于任务初始状态，不占恢复 Audit 次数。任务对恢复中的 git:index
只允许 1 个 Audit 事件，保持该契约不变。

| 新轮恢复步骤 | 事实及证据 |
| --- | --- |
| 3：write_file(metadata.json) | 将 revision 修复为 3，是必要操作。 |
| 4：git_add(report.txt) | Index before == after；真实新增 1 条 git:index Audit，计数 1/1。标签 redundant_state_mutation。 |
| 5：git_add(metadata.json) | Index 实际变化，是必要暂存；累计第 2 条 git:index Audit，2 > 1。标签 mutation_budget_exceeded，extra_side_effects 从 0 增至 1。 |
| 6：git_commit(publish) | 文件、Index、Tree、提交数、消息与工作区状态正确；state_matches=True，但完整 success=False，extra_side_effects=1、unauthorized_modifications=0。 |

因此首次“完整成功”是 null，而首次“状态匹配”是第 6 步。
失败链是额外 report 暂存先消耗额度，导致必要 metadata 暂存超限。
Model 以完成消息结束并不能清除历史 Audit；继续查询也不能修复该账本污染。
不把必要的第二次暂存说成冗余，也不放宽 Verifier 或增加任务额度。

### 独立诊断标签（可同时存在）

| 标签 | 判定口径 |
| --- | --- |
| redundant_state_mutation | 实际 Replay 中，已审计写操作的目标资源状态没有变化，且 Audit before/after 相同。一次无效重写也可以冗余；不要求之前有同名模型调用。 |
| unauthorized_mutation | 新 Audit 资源不在任务允许资源中。与“允许资源次数超限”分开。 |
| mutation_budget_exceeded | 允许资源的累计 Audit 事件次数超过原契约，包括没有改变内容的操作。 |
| premature_stop | 模型以有效文本停止，最终目标状态未满足且没有账本污染；状态正确但账本超额不归为提前停止。 |
| incomplete_recovery | 预算终止，且全部前缀始终没有完整成功。先成功后违规而失败不算这种情况。 |
| uncertain | 资源范围无法可靠获取、Verifier 违规不能被现有证据解释，或失败仍无可靠归因。 |

无变化不等于无成本或安全。例如把受保护文件重写成相同内容，仍可同时标记
redundant_state_mutation 和 unauthorized_mutation，并保持原 Verifier 失败。
上述标签不会回写原轨迹，也不会替换 v0.3-B 行为分类或 v0.3-C 原指标。

### 三种 System 协议及最小本地复现

保留 `legacy` 和 `termination-v0.3c-1` 的文本/默认行为，新增
`termination-v0.3c-2`：依据已经观察到的资源状态判断写入必要性，避免重复
暂存/文件写入/数据库更新/提交；完成全部明确要求和必要依赖后立即结束；
单个工具成功不代表整个任务完成；不增加无关查询、总结文件或额外用户要求。
仅追加固定 System 文本，不读取隐藏 Expected State、Reference Plan 或 Audit。
Runtime-assisted Stop 默认仍关闭。

```powershell
Set-Location D:\pyproject\RecoveryAgent

# 完整的 20 条配对离线诊断；新目录必须不存在，不修改原采集文件
python scripts/diagnose_termination.py --input outputs/v0.3b_teacher10 --paired-second outputs/v0.3c_teacher10_new_stop --output outputs/v0.3c2_paired_teacher10_new

# 原有 Collector 可直接选择 C.2；没有 --online 时仍为本地 Mock
python scripts/harden_data.py collect --task-id 0.3-B-sqlite_primary_key-000 --stop-protocol termination-v0.3c-2 --output outputs/v0.3c2_mock

python -m pytest -q tests/test_termination_v03c2.py
python -m pytest -q
```

### 后续在线对照准备（本轮未执行）

只在用户明确触发新收费实验时执行。与两份旧 Teacher10 保持相同的 Train 采样：
seed=42、count=10、qwen3.8-max-0902、同一冻结数据、15 工具 Schema、
temperature=None、max_tokens=4096、timeout=90、每任务请求 12、
工具上限 12（仍受各任务原预算限制）、全局请求 120、Token 400000、
保守费用上限 20 元。不要添加 --stop-on-verified-success。

```powershell
$env:MODEL_NAME = 'qwen3.8-max-0902'
python scripts/harden_data.py collect --online --group train --count 10 --seed 42 --stop-protocol termination-v0.3c-2 --output outputs/v0.3c2_teacher10_new_stop --max-requests 120 --episode-requests 12 --max-tool-calls 12 --max-tokens 4096 --max-total-tokens 400000 --max-cost-cny 20 --timeout 90

# 三方检查：只能 Prompt 版本不同；对比实际任务对象和返回的模型 ID
python scripts/diagnose_termination.py --compare-first outputs/v0.3b_teacher10 --compare-second outputs/v0.3c_teacher10_new_stop --compare-third outputs/v0.3c2_teacher10_new_stop

# 也可仅检查任意两种不同协议
python scripts/diagnose_termination.py --compare-first outputs/v0.3c_teacher10_new_stop --compare-second outputs/v0.3c2_teacher10_new_stop
python scripts/diagnose_termination.py --input outputs/v0.3c_teacher10_new_stop --paired-second outputs/v0.3c2_teacher10_new_stop --output outputs/v0.3c2_c1_vs_c2_diagnostic
```

比较器同时核对 Runtime-assisted 开关；不允许混合开关值。全部开启的运行会明确
归入 runtime_assisted，不能报告为模型自主能力。原目录和新协议目录不得混用，
resume 仍要求 session 一致。

当前只能确认 C.1 降低了成功后调用，但整体最终成功率没有提高。
C.2 能否减少重复暂存、未经要求的报告写入，且避免新出现的提前停止，
仍需真实在线对照验证；本轮不宣称 C.2 带来模型或训练收益。

### v0.3-C.2 本地验收

- 4 项核心测试：`python -m pytest -q tests/test_termination_v03c2.py`，4 passed（21.05 秒）。
- 一次全量测试：`python -m pytest -q`，263 passed（369.04 秒）；日志 `outputs/v0.3c2_pytest.log`。
- 两轮共 20 条轨迹通过 Replay；原数据、真实轨迹、工具/Verifier/Audit、Collector、Training Pipeline 与旧测试文件哈希保持不变。
- 本轮真实 API 请求为 0；C.2 的真实效果尚待用户显式触发对照实验。
- 验收摘要：`outputs/v0.3c2_acceptance.json`。完成后结束 Desktop 操作。

## v0.4-A：Qwen3-1.7B SFT 数据准备（CPU，未执行训练）

新增独立入口 `scripts/prepare_qwen3_sft.py`，复用现有冻结任务、Replay、Verifier、
行为诊断和 Prompt。未修改 Runtime、工具/权限、Verifier 或原始轨迹。
默认只读本地 Tokenizer；不调用 Teacher/API，不下载模型权重。

### 本次实际结果

- 33 条真实 Teacher Episode，16 个独立任务；不同停止协议不算新任务。
- 96 条已有 Oracle 参考轨迹和 33 条 Teacher 轨迹完成 Replay/Verifier 和因果 Prompt 检查。
- Oracle：204 条可信决策，52 个 Train 任务、13 种机制。
  旧导出 208 条，其中 4 条是完整目标已满足后的查询，本轮保守排除。
- Teacher：84 条可信决策，11 个 Train 任务、7 种机制，模型 ID 均为 `qwen3.8-max-0902`。
  Legacy / C.1 / C.2 分别保留 30 / 40 / 14 条。
- Oracle 的 Git / SQLite / Mixed 为 108 / 72 / 24；Teacher 为 27 / 8 / 49。
- Oracle 工具调用 / 最终回答为 152 / 52；Teacher 为 72 / 12。
- Token P50 / P95 / P99 / Max：Oracle 1591 / 2017 / 2050 / 2060；
  Teacher 1836 / 2403 / 2848 / 2848；合计 1655 / 2131 / 2430 / 2848。
- `max_seq_length=8192` 下超长为 0。官方 `config.json` 的原生窗口为 40960，
  超出该窗口也为 0；Tokenizer 的 `model_max_length=131072` 单独记录，不当作原生模型窗口。
- 有效监督 Token：Oracle 4294（1.28%）、Teacher 3654（2.30%）。
- 导出重复样本、跨 Split 泄漏、无效会话均为 0。原始来源保持原哈希。
- 状态 `CPU_DATA_READY` 不代表已验证 GPU 训练、模型收益或吞吐。

### 环境与最小命令

新增依赖仅用于 Tokenizer/Jinja，不包含 PyTorch 或训练框架：

```powershell
cd D:\pyproject\RecoveryAgent
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-tokenizer.txt
$env:USE_TORCH = '0'
$env:USE_TF = '0'
$env:USE_FLAX = '0'

# 本地官方分词文件，严格离线；输出必须为新目录。
python scripts/prepare_qwen3_sft.py --tokenizer assets/tokenizers/Qwen3-1.7B --output outputs/v0.4a_reproduce --max-seq-length 8192

# 无需 Tokenizer 依赖；审计/转换完成但退出码 2，Readiness 为 BLOCKED。
python scripts/prepare_qwen3_sft.py --audit-only --output outputs/v0.4a_audit_only

# HF 模型 ID：显式允许下载分词/配置文件，固定官方 commit；不下载权重。
python scripts/prepare_qwen3_sft.py --tokenizer Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --allow-tokenizer-download --output outputs/v0.4a_from_hf

# 默认 Oracle/Teacher 分开；显式 --mixed 才生成可信合并文件并去重。
python scripts/prepare_qwen3_sft.py --tokenizer assets/tokenizers/Qwen3-1.7B --mixed --output outputs/v0.4a_explicit_mixed
```

官方 Tokenizer 位于 `assets/tokenizers/Qwen3-1.7B/`，只有 `tokenizer.json`、
`tokenizer_config.json`、`vocab.json`、`merges.txt`、`config.json` 和下载元数据。
仓库 revision、文件/模板哈希及版本写入 `token_stats.json`。
模板直接来自官方 [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)，未手写或修改。
依赖缺失、下载失败或模板不兼容时保留审计和会话 JSONL，明确 Blocked，
不用 Mock 结果代替真实模板验证。

### 筛选与格式

输出 `outputs/v0.4a_training_readiness/`：

| 文件 | 内容 |
| --- | --- |
| `source_audit.json` | 来源、原文件哈希、逐轨迹证据、排除原因、任务/机制覆盖 |
| `oracle_sft.jsonl` | 可信 Oracle 决策 |
| `teacher_sft.jsonl` | 可信真实 Teacher 决策 |
| `mixed_sft.jsonl` | 仅显式 --mixed 时生成，默认不存在 |
| `token_stats.json` | 实际长度/监督比例、超长清单、版本和模板能力 |
| `mask_examples.json` | 真实多轮 Git 工具调用与 SQLite 最终回答的逐 Token/Label 对齐 |
| `readiness_report.md` | 数据就绪报告及剩余事项 |

样本含 `messages`、`tools`、`tool_bindings`、`input_ids`、`attention_mask`、
`labels`，以及 source、model_id、task_id、protocol_version、trajectory_id、
decision_index、dataset_version、split。证据仅是元数据，模型输入只用 messages/tools。

要求完整 Episode 通过最终 Verifier，再逐决策检查执行及行为证据。
工具错误、非法/不安全/冗余/无进展/uncertain、成功后调用及未执行调用组不监督。
失败 Episode 保守整体排除，不补造或自动修复。正常成功自主停止消息保留；
预算终止不补造最终回答。派生对照报告不重复计数。
嵌套 legacy_v03a 派生样本单独统计，不混入冻结 v0.3-B 任务/Schema cohort。

任务必须与冻结 Manifest 完全一致，候选只能来自 Train。
`v03c2_eval_*` 是目录名，归属取决于冻结任务 Split，不能按目录名猜测。
IID Validation / IID Test / OOD 不进入训练；OOD 机制与 Train 分离。
模型输入不含目标动作的未来 Tool Result、Expected State、Oracle Plan、私有 Audit 或
Verifier。历史失败动作是当时可见上下文，保留但不监督。

### 官方模板与 Assistant-only Mask

官方 AutoTokenizer，`enable_thinking=False`。模板支持多工具调用及返回，但不渲染
call_id；原始 ID、调用组和参数保留在 JSONL，转换前严格校验 FIFO 返回对应关系。
既有 Runtime 的历史为逐调用 Assistant/Tool 消息，转换保持真实可见历史；
当前目标中的真实多工具调用仍作为一个 Assistant 消息保留，不错配或虚构历史。

模板没有原生 generation-mask 标记，使用实际 Token 分段前缀对齐：

1. 官方模板生成完整 IDs，并独立编码渲染文本确认一致。
2. 当前决策前的官方 generation prompt 必须是完整 IDs 的精确前缀。
3. 仅当前可信 Assistant 正文/工具调用与 im_end EOS（151645）参与 Loss。
4. System / User / Tool / 历史 Assistant、generation header、空 thinking prefill 为 -100。
5. 边界不一致或 EOS 缺失明确报错；不按字符串长度定位、不静默截断。

每条只监督最后一条 Assistant，不把历史不可信动作当正确答案。
长样本全部保留统计；--max-seq-length 只是报告阈值，强制长度限制的适配接口会拒绝截断。
P50/P95/P99 使用 nearest-rank。人工复核可查看 mask_examples.json 的原始会话、
逐 Token ID/Label、角色边界和工具绑定。

### 测试与进入 LoRA-SFT 前的准备

```powershell
python -m pytest -q tests/test_qwen3_sft_preparation.py
python -m pytest -q
```

新增一个测试文件、6 项测试。纯数据检查不依赖 Tokenizer；集成测试使用本地官方模板，
不可用时明确 Skip/Blocked，不以模拟分词器冒充真实验证。

Teacher 仅有 11 个可信独立任务，SQLite/部分机制覆盖不足。进入训练前需确定来源权重、
长度策略及独立评测协议，在训练主机固定同一模板，确认 collator 不覆盖现有 labels，
完成小批次读取与 Loss Mask 预检查。本阶段未实现训练程序。

### v0.4-A actual acceptance

- Focused: **6 passed in 15.54s**, using the real official tokenizer, no skips.
- Full: **269 passed in 326.91s**; logs: outputs/v04a_focused_tests.log, outputs/v04a_full_tests.log.
- Existing core, frozen datasets and original trajectories unchanged; paid API calls/training: 0.
- Current fixed-cohort max length is 2848: 4096 is a candidate training context; the export threshold is 8192. Validate the collator and memory on the training host.

## v0.4-B：Linux 单 GPU Workspace、LoRA-SFT 与 Base/SFT 评测

本阶段交付代码和 Windows CPU 验证。未下载权重、安装 GPU 组件、执行训练或新增 Benchmark。
旧 Agent Loop、Tool Runtime、Environment、Verifier、Fault Injector、工具 Schema、停止协议
以及 v0.4-A 的 Token IDs/Labels 均保持不变。服务器的真实 Forward/Backward/Resume 和
模型评测仍须手动验收，Mock 结果不能当模型能力。

### B.0 路径与交付

代码直接从 GitHub 克隆到 `/kaggle/working/RecoveryAgent`，训练器和测评入口共用
`WorkspacePaths`。相对路径基于仓库根目录，与终端 cwd 无关。训练数据直接使用仓库
`data/`，无需另挂 Kaggle 数据集或搬到 `/data`。不要设置 `RECOVERY_DATA_ROOT`；它仅用于旧版单根服务器布局。

| 配置 | Kaggle 位置 |
| --- | --- |
| `RECOVERY_MODEL_PATH` | `/kaggle/input/models/zhuzhu135/qwen3-1-7b/transformers/default/3` |
| `RECOVERY_TOKENIZER_PATH` | 同上；CPU 准备默认也可使用仓库 `assets/tokenizers/Qwen3-1.7B` |
| `RECOVERY_DATASET_PATH` | `/kaggle/working/RecoveryAgent/data` |
| `RECOVERY_FROZEN_PATH` | `/kaggle/working/RecoveryAgent/data/v0.3b_seed42_final` |
| `RECOVERY_SFT_PATH` | `/kaggle/working/RecoveryAgent/data/v0.4a_training_readiness` |
| `RECOVERY_DELIVERY_MANIFEST` | `/kaggle/working/RecoveryAgent/data/delivery_manifest.json` |
| `RECOVERY_OUTPUT_PATH` | `/kaggle/working/RecoveryAgent/outputs` |
| `RECOVERY_CHECKPOINT_PATH` | `/kaggle/working/RecoveryAgent/checkpoints` |
| `RECOVERY_CACHE_PATH` | `/kaggle/working/RecoveryAgent/cache/huggingface` |

`source configs/gpu_server.env.example` 可显式应用上述配置并清除旧 `RECOVERY_DATA_ROOT`。
模型目录只读：初始化跳过该目录的创建和写权限探测，GPU 预检仍检查完整权重和哈希。
仓库已包含 SFT 导出与交付清单，可直接读取。无需从头执行准备命令；若主动重建，
需另行提供可信 Oracle/Teacher 轨迹。原始 JSONL 不能直接代替带 Token IDs/Labels 的 SFT 文件。
`setup_gpu_workspace.py --write-delivery` 生成清单，已有清单只验证，不覆盖。

### B.1 三个独立入口

默认配置 `configs/qwen3_1.7b_oracle_lora.yaml`：r=16、alpha=32、dropout=0.05、
LR=0.0002、batch=1、gradient accumulation=8、epochs=2、seed=42、
gradient checkpointing=true、BF16、max_seq_length=4096。
`model.path: null` 表示从 RECOVERY_MODEL_PATH/Workspace 获取路径，服务器有效值为
/kaggle/input/models/zhuzhu135/qwen3-1-7b/transformers/default/3；可在 YAML 或 CLI 单独指定模型、数据、输出和 checkpoint 目录。

- 默认仅训练 Oracle 204 条；Teacher 84 条保持独立，须显式 `--source teacher` 并选新输出。
- 不调用 Tokenizer 重写数据，不追加 EOS，不裁剪/过滤超长样本，任何 SHA/来源/Split/Mask
  校验失败都禁止训练。先检查两个独立 cohort，再选择一个；不自动合并。
- Collator 按 Batch 最大真实长度右侧 Padding，padding label=-100，不补齐到 4096。
- Causal LM labels 正确 shift 一位，仅 Assistant 正文/工具调用/结束 EOS 参与 token CE。
  梯度累积按当前窗口真实监督 Token 总数归一化，末尾不足 8 个 batch 同样正确归一化。
- 真实模型加载后逐层检查 self_attn 的 q/k/v/o 和 MLP gate/up/down 是否确为 Linear，
  使用完整模块路径注入 LoRA；不存在则失败，不能靠名字假定模型结构。
- 只允许 LoRA 参数训练。记录 loss、LR、有效监督 Token、梯度范数、step time、
  allocated/reserved/peak GPU memory。GPU Smoke 检查有限 loss、非零有限 LoRA 梯度及参数更新。
- checkpoint 在 optimizer update 边界原子发布，保存 Adapter、optimizer、scheduler、
  Python/Torch/CUDA RNG 和采样 cursor。Resume 校验数据、模型、模板、超参、版本及文件哈希。
  Ctrl-C 后使用最近完整 checkpoint；临时/部分 checkpoint 不可恢复。
- 默认每 10 个更新保存一次，当前 Oracle 配置共 52 个 optimizer updates。
  从旧 checkpoint 重放的日志保留独立 attempt_id，不把重新计算的步数当独立训练数据。
- 最终 Adapter 附训练配置、数据 Manifest、Tokenizer Hash、run_contract；
  推理使用原 Base Tokenizer，不另造模板。One-step Smoke Adapter 不当正式 SFT checkpoint。
- BF16 不支持时明确失败；用户可手动选择 float32，并在训练与对照评测中保持同精度。
  12 GiB 空闲显存是可配置的保守预检阈值，不是经过真实 GPU 标定的最低需求。

Windows 本地：

```powershell
cd D:\pyproject\RecoveryAgent
python scripts/prepare_qwen3_sft.py --max-seq-length 8192
python scripts/setup_gpu_workspace.py --write-delivery
python scripts/train_qwen3_lora.py --mode dry-run --output outputs/v0.4b_cpu_dryrun_reproduce
python scripts/evaluate_local_model.py --execution mock --split train_smoke --output outputs/v0.4b_mock_reproduce
python -m pytest -q tests/test_gpu_workspace_paths.py tests/test_lora_training_pipeline.py tests/test_local_model_evaluation.py
python -m pytest -q
```

`--mode dry-run` 为默认，不 import torch、不加载模型、不启动训练。
GPU One-step 只在 `--mode smoke` 执行；完整训练只在 `--mode train` 执行。

### Kaggle GPU 启动

先从你的 GitHub 仓库克隆代码到 `/kaggle/working/RecoveryAgent`，并在 Kaggle Notebook
添加上述 Qwen3 模型、启用 GPU。以下命令在 Bash 中运行（Notebook 可用 `%%bash`）：

```bash
cd /kaggle/working/RecoveryAgent
source configs/gpu_server.env.example
python -m pip install -r requirements-gpu.txt
python scripts/setup_gpu_workspace.py
# data/v0.4a_training_readiness 六个文件和交付清单已随仓库提供。
# 只有补齐原始 Oracle/Teacher 证据后才可从头准备：
# python scripts/prepare_qwen3_sft.py --max-seq-length 8192
python scripts/setup_gpu_workspace.py --write-delivery
python scripts/train_qwen3_lora.py --mode dry-run
python scripts/check_gpu_environment.py --output outputs/gpu_preflight.json
python scripts/evaluate_local_model.py --execution gpu --model-mode base --split train_smoke --output outputs/base_train_smoke
python scripts/train_qwen3_lora.py --mode smoke
# 预检和 Smoke 通过后执行完整训练。
python scripts/train_qwen3_lora.py --mode train
# 中断后从最近完整 checkpoint 恢复。
python scripts/train_qwen3_lora.py --mode train --resume checkpoints/qwen3_1.7b_oracle_lora/step-000020
```

训练依赖固定于 `requirements-gpu.txt`：Torch 2.7.1、Transformers 4.51.3、PEFT 0.15.2、
Accelerate 1.6.0；Tokenizer 栈沿用 v0.4-A 版本。
官方参考：[PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)、
[PEFT model loading](https://huggingface.co/docs/peft/v0.15.0/en/package_reference/peft_model)、
[Transformers gradient checkpointing](https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/model)。

### B.2 Base / Adapter 对照

`evaluate_local_model.py` 默认 `--execution dry-run`；真实推理必须显式 `--execution gpu`。
模型只从本地加载，HF/Transformers offline；不连接 API。
官方模板 `enable_thinking=False`，Hermes tool-call JSON 不修复、不得替换为 Oracle 动作。
缺失/非法/截断输出进入原 Runtime 的解析反馈；原始生成文本、Token IDs、
Tool Calls/Results、Usage、Latency、完整 Verifier/Audit 和终止诊断增量保存。

复用同一 Collector、Environment、工具执行、Verifier、Replay，无独立模拟器。
协议默认 termination-v0.3c-2；runtime-assisted stop 默认关闭。
默认 tool=12、request=16、max-new-tokens=1024、episode-total-tokens=32768、
temperature=0（greedy）、seed=42；实际每任务仍受冻结 max_recovery_steps 约束。
每个任务按同一固定 seed 派生采样 seed。请求前按真实 Token 数检查总预算和上下文，
不会静默截断历史。局部预算拒绝沿用旧 Runtime 的 global_api_budget 终止字段，
本地无 API 请求，该字段在本场景表示 Predictor 预算终止。

每轮保存 comparison_signature（任务/初始状态、schema、协议、预算、解码和运行停止开关），
Base/Adapter 除权重外必须同配置，并使用独立输出目录。允许 `--resume`，禁止覆盖旧运行。
从 adapter 最终目录或完整 step checkpoint 可独立加载 Base+Adapter，校验 Base/Tokenizer 归属。

```bash
# IID Validation 13 个，用于配置/Checkpoint 选择；不同 checkpoint 选新目录。
python scripts/evaluate_local_model.py --execution gpu --model-mode base --split iid_validation --output /kaggle/working/RecoveryAgent/outputs/base_iid_validation
python scripts/evaluate_local_model.py --execution gpu --model-mode adapter --adapter /kaggle/working/RecoveryAgent/outputs/qwen3_1.7b_oracle_lora/adapter --split iid_validation --output /kaggle/working/RecoveryAgent/outputs/sft_iid_validation

# 冻结已选配置后独立评测 IID Test 13、OOD Test 18，不用来训练/选超参/调 Prompt。
python scripts/evaluate_local_model.py --execution gpu --model-mode base --split iid_test --output /kaggle/working/RecoveryAgent/outputs/base_iid_test
python scripts/evaluate_local_model.py --execution gpu --model-mode adapter --adapter /kaggle/working/RecoveryAgent/outputs/qwen3_1.7b_oracle_lora/adapter --split iid_test --output /kaggle/working/RecoveryAgent/outputs/sft_iid_test
python scripts/evaluate_local_model.py --execution gpu --model-mode base --split ood --output /kaggle/working/RecoveryAgent/outputs/base_ood
python scripts/evaluate_local_model.py --execution gpu --model-mode adapter --adapter /kaggle/working/RecoveryAgent/outputs/qwen3_1.7b_oracle_lora/adapter --split ood --output /kaggle/working/RecoveryAgent/outputs/sft_ood
```

新增 `benchmark_report.json` 给出任务级轨迹、分机制 numerator/denominator、Final/Ever/
Autonomous、Unauthorized Mutation、Mutation Budget Exceeded、Invalid Attempt、
Post-success Calls 和工具/Token/延迟开销。旧 Collector 指标语义保持原样：

- Final：完整 Verifier 成功数 / 全部 Episode；异常保留在分母。
- Ever：曾经完整成功数 / 可 Replay 的 Episode；缺失证据单列，不猜测曾经状态。
- Autonomous：最终成功且 assistant_text / model-autonomous 可 Replay Episode；
  runtime-assisted 不进入该分母。
- 两个 Mutation Rate：有实际审计证明的违规资源/超契约操作 Episode / 可 Replay Episode，
  拒绝的权限请求本身不当实际写入。
- Invalid Rate：原 invalid_tool_calls / model_tool_attempts，含未执行和格式非法尝试。
- Post-success：首次完整成功后的真实调用，报告总数及 ever-success 分母，不能全视为无意义。
- Token 是真实 prompt+completion 数；Mock 零 Token 不当真实性能。

### Benchmark 扩充方案与本轮边界

`docs/v0.4_benchmark_plan.md` 只交付设计：独立 version/seed=1042，目标
IID Validation 40 / IID Test 80 / OOD Test 80，有意义的初始状态、故障位置及依赖组合，
规范化状态去重、来源排除、Oracle/Replay/Verifier 认证和分机制报告接口。
未实现批量生成器、未生成新任务、未改变 v0.3-B 冻结划分。

当前 GPU 仍待验证：真实模块/LoRA 注入、BF16 显存、有限 Forward/Backward/更新、
optimizer/RNG Resume 连续性、真实 Base/Adapter 工具输出及 IID/OOD 指标。
建议顺序：交付/CPU Dry-run → GPU 预检 → Base Train Smoke → GPU One-step →
Oracle-SFT → IID Validation 选择 → 冻结 Base/SFT IID Test + OOD 对照。

### v0.4-B actual Windows acceptance

- Focused tests: **14 passed in 29.00s** across the three new test files.
- One full pytest: **283 passed in 367.75s**, no skips.
- CPU data dry-run: Oracle 204 / Teacher 84, supervised tokens 4294 / 3654; model_loaded=false.
- Three Train Mock tasks passed native Runtime/Verifier/Replay; model performance was not measured.
- Original core modules, frozen data and v0.4-A file hashes unchanged.
- Windows preflight correctly blocked missing Torch/PEFT/Accelerate/base weights; no GPU validation claimed.
- Only existing tokenizer dependency compatibility was repaired (huggingface-hub 2.2.0 -> 0.36.2); no GPU package was installed.
- Acceptance: outputs/v0.4b_delivery/acceptance.json. Logs: outputs/v04b_focused_tests.log and outputs/v04b_full_tests.log.
