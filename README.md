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
