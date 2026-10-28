# PCS WebUI

PCS WebUI 用于计算大模型前缀缓存的容量、读写带宽、IOPS、计算量和首 token 时间（TTFT）。可以扫描不同参数、回放有限容量缓存、模拟单块 SSD，并按资源约束比较方案。

计算在本机 CPU 上执行。输入来自模型参数和合成工作负载，不需要下载模型权重。

逐项设置：[Explore 输入](#controls-explore) · [恢复与高级参数](#controls-recovery) · [图表和结果](#controls-results) · [Recommend](#controls-recommend) · [Experiments](#controls-experiments) · [KV Size](#controls-kv) · [45 个高级字段](#controls-kv-fields) · [模型默认值](#controls-model-defaults) · [Architecture](#controls-architecture)。操作示例保留在下文，数值的计算方法和适用范围见[建模说明](#modeling)。

## 启动

先安装 Python 3.9 或更高版本，再运行对应脚本：

| 系统 | 首次安装 | 日常启动 |
| --- | --- | --- |
| macOS | 双击 `install.command` | 双击 `start.command` |
| Windows | 双击 `install.bat` | 双击 `start.bat` |
| macOS / Linux 终端 | `sh install.sh` | `sh start.sh` |

启动后访问 <http://127.0.0.1:8001>。使用期间保持终端打开，按 `Ctrl+C` 停止服务。端口、离线安装和环境问题见 [INSTALL.md](INSTALL.md)。

## 页面和基本概念

| 页面 | 用途 |
| --- | --- |
| [Explore](#explore) | 设置模型和参数，运行批量计算，查看曲线、热力图、缓存状态和单个 Case，导出结果 |
| [Recommend](#recommend) | 按工作负载、带宽、容量、IOPS 和 TTFT 约束筛选方案 |
| [Experiments](#experiments) | 保存、搜索、改名、固定、重开、比较和删除实验 |
| [KV Size](#kv-size) | 计算 KV、Indexer、State 容量，查看公式和组件，比较不同配置 |
| [Architecture](#architecture) | 查看架构图、展开节点、播放动画和导出图形 |

首次打开进入 Explore。Theme 可选择 System、Light 或 Dark。浏览器记忆主题、最近页面和 Explore 的 Simple/Expert 模式。

| 名称 | 含义 |
| --- | --- |
| Case | 一组确定的模型、长度、前缀比例和其他参数，以及该组参数的结果 |
| Sweep | 一次批量计算，包含多个 Case |
| Workload | 请求集合，包括请求数量、到达时间、输入长度、生成长度和前缀共享关系 |
| Runtime | 缓存精度、有效计算吞吐和分析默认值的组合 |
| Prefix ratio | 输入 token 中候选可复用前缀的比例；在两种分析模式中的作用见下文 |
| Logical block | 按 token 数划分的 KV 逻辑块 |
| Checkpoint | 为恢复有状态模型而保存的中间状态 |
| Shard | 缓存分片；单 SSD 仿真只接受一个 shard |

## Explore

### 示例一：比较长度和前缀比例

1. 进入 Explore，选择 `Bandwidth Envelope`。
2. 切换到 Expert，展开块大小和 IO 参数。
3. 点击 `Change models`，取消其他模型，只选 Qwen3-8B。
4. 按下表填写输入，其他选项保持初始值。
5. 点击 `Run Sweep`。

| 输入 | 示例值 |
| --- | --- |
| Compute (TFLOPS) | `2000` |
| Context lengths | `1024, 2048` |
| Prefix hit rates | `50%, 75%` |
| Block sizes | `64` |
| Max IO bytes | `131072` |
| Arrival model | 只选 `fixed` |

这组参数生成四个 Case：一个模型 × 两种长度 × 两种前缀比例。完成后：

- 在 Charts 中切换 Bandwidth、Read、Write、TTFT 等指标。
- 将 View 从 Curves 切换到 Heatmap，按长度和前缀比例查看结果。
- 点击图上的点或表格行，再点击 `Open details` 查看单个 Case。
- 点击 `Export JSON` 或 `Export CSV` 下载结果。

### Case 数量

多个输入值会组合计算。例如：

```text
2 个模型 × 3 个长度 × 4 个前缀比例 × 2 个块大小 = 48 个 Case
```

多种到达模式、Max IO 值和 Replay 容量值也会增加 Case 数；Envelope 不使用到达模式，同时勾选 fixed 和 poisson 只会得到两份相同的结果。DeepSeek V4.1 的 Encoder 场景只扩展适用的模型。运行前查看页面显示的 Case 数量。

### 模型选择

`Change models` 支持按名称搜索，也可按 family、formula、evidence 筛选。`Select visible` 选择当前可见模型。改变筛选条件不会取消已经选中但暂时隐藏的模型；运行前检查已选列表。

当前目录有 56 个可在界面选择的模型，另有 3 个旧配置用于历史实验和 API 兼容。运行配置会检查模型与 Runtime 的组合，不兼容的组合无法执行。

### Simple 和 Expert

Simple 显示常用输入。Expert 另外显示块大小、Max IO、到达模式、负载分布、缓存分片、预热以及 State/Sparse/Draft/Decode 选项。两种视图共用输入，切换视图不会重算结果。

### 内置预设

| 预设 | 模型与采样 | 默认 Case 数 |
| --- | --- | ---: |
| Quick Compare | 5 个模型；长度 32,768；前缀比例 70%、95%、99%、99.9% | 20 |
| Full Tail | 5 个模型；长度 65,536、131,072、262,144；55 个前缀比例采样点 | 825 |

Quick Compare 使用 Cohere Command R+、DeepSeek R1、DeepSeek V3.2、Kimi K3、DeepSeek V4 Flash。Full Tail 使用 Kimi K2.5、GLM-5.2、Kimi K3、DeepSeek V4 Flash、DeepSeek V4 Pro。

两个预设均使用 256-token 块、固定到达模式和 131,072-byte Max IO。Full Tail 的前缀比例从 0.5 到 0.9999，靠近 100% 的采样更密。点击预设会重设它管理的模型和采样参数。

## 分析模式

| 项目 | Bandwidth Envelope | Cache Replay |
| --- | --- | --- |
| 输入前缀比例 | 假定指定逻辑前缀已经驻留 | 指定请求可能复用的前缀上限 |
| 缓存容量 | 不模拟容量竞争和驱逐 | 按容量执行准入、保护和驱逐 |
| 请求 | 每个 Case 只分析 1 个请求，不使用负载参数 | 按负载参数生成多个请求，到达顺序决定每次查找时缓存中已有的对象 |
| 写入 | 不计，Write 恒为 0 | 计入 miss 新块和可选的 decode 写入 |
| 主要结果 | 恢复字节、计算量、容量估计、带宽和时间要求 | 实际命中、驱逐、驻留量、恢复范围和 IO 需求 |
| 单 SSD 仿真 | 需要切换到 Replay | 开启后可对单 shard Case 执行 |

先用 Envelope 比较模型和恢复需求；需要分析容量不足、请求共享和驱逐时，使用 Replay。两种模式的具体算法见[建模说明](#modeling-modes)。

### 前缀的四个阶段

```text
Configured → Shareable → Resident → Recoverable
```

| 阶段 | 含义 |
| --- | --- |
| Configured | 请求指定的候选前缀长度 |
| Shareable | 按对象身份和块对齐后，可以共享的部分 |
| Resident | 查找时仍在缓存中的连续前缀 |
| Recoverable | 满足状态和 checkpoint 条件，能够用于恢复计算的部分 |

Resident 有值不代表能直接从该位置继续计算。有状态模型可能只有更早的 checkpoint，需要从那里重算一段 token。

`Recovery boundary ratio` 为请求恢复边界比例的 p50。对 DeepSeek V4.1，它表示 Encoder 重放的起点，不能直接作为可复用 global KV 的占比。

## Cache Replay

### 示例二：扫描缓存容量

继续使用 Qwen3-8B，在 Expert 模式下设置：

| 输入 | 示例值 |
| --- | --- |
| Analysis mode | `Cache Replay` |
| Context lengths | `1024` |
| Prefix hit rates | `50%` |
| Block sizes / Max IO bytes | `64` / `131072` |
| Request count / QPS | `20` / `10` |
| Generation length | `16` |
| Prefix groups / Zipf alpha | `1` / `1.3` |
| Arrival model / Seed | `fixed` / `1` |
| Workload shape | `Shared prefix groups` |
| Persist blocks created by misses | 开启 |
| Capacity axis (GiB) | `0, 0.25, 1` |
| Shard count / Placement seed | `1` / `0` |
| Warmup fraction | `0` |

运行后生成三个容量 Case。打开 Cache，查看：

- `Resident prefix vs capacity`：容量变化后实际驻留前缀如何变化。
- `Cache churn vs capacity`：驱逐、刚准入即驱逐和拒绝准入的字节数。
- Capacity 指标：峰值使用量、结束时使用量和受保护字节数。
- Recovery / Hits 指标：可共享比例、驻留比例、恢复边界及实际块/字节命中率。

### 为什么输入 50%，实际命中率低于 50%

输入的比例限制每个请求的候选前缀。实际块命中率以测量区间内全部可缓存块为分母，还受冷启动、共享关系和驱逐影响。

例如 20 个等长请求共享前一半内容，第一个请求冷缓存未命中，之后 19 个请求都命中这一半。块对齐一致时，总体块命中率为：

```text
19 × 50% ÷ 20 = 47.5%
```

预热可以先建立缓存状态。Warmup fraction 指定预热请求占比，预热部分不计入测量区间。`Prompt lineage bootstrap` 则使用带初始前缀的请求继承关系；比较实验时需记录所选负载形状。

### 缓存指标

| 指标 | 含义 |
| --- | --- |
| Actual block hit rate | 命中的块数 / 测量请求的全部可缓存块数 |
| Actual byte hit rate | 命中字节数 / 测量请求的全部可缓存字节数 |
| Eviction | 为满足容量限制而移出的逻辑缓存对象 |
| Self-eviction | 新准入的对象很快又被驱逐 |
| Rejected admission | 因容量或保护条件未能准入的对象 |
| Peak protected cache | 在途请求仍在使用、暂时不能驱逐的驻留字节 |
| Final cache occupancy | 回放结束时的实际驻留字节 |
| Stranded resident | 已驻留但在当前恢复策略下无法用于恢复的字节 |

逻辑缓存驱逐不等于 SSD 擦除。擦除、垃圾回收和写放大属于 SSD 仿真的设备模型。

### 运行 SSD Simulation

1. 在 Replay 结果中选中一个 Case。
2. 确认 `Shard count = 1`，并选择 Reference storage。
3. 打开 Case 详情，点击 SSD Simulation。
4. 查看设备吞吐、时延、排队、写放大及模拟后的 TTFT。

一次交互式仿真最多处理 200,000 个逻辑操作和 2,000,000 个物理 IO。超过限制时，减少请求数、上下文长度或其他扩大 IO 数量的参数。

多 shard 的缓存分析可以运行；单 SSD 仿真只接收一个 shard。容量规划使用的存储额定值和设备仿真结果分别显示。

## 参数说明

### 模型、计算和 IO

| 输入 | 单位与要求 | 影响 |
| --- | --- | --- |
| Compute (TFLOPS) | 正有限数，默认 2000 | 将 FLOPs 换算为计算时间 |
| Context lengths | 正整数 token 数，可填多个值 | 请求长度、缓存字节和计算量 |
| Prefix hit rates | `50%` 或 `0.5`，范围 0–100% | 候选复用长度；含义随分析模式变化 |
| Block sizes | 每块 token 数，可填多个值 | 块对齐、共享边界、块数与 IO 粒度 |
| Max IO bytes | 每次 IO 最大字节数，可填多个值；`none` 表示不指定 | 分块 IOPS、SSD 仿真工作量 |
| Reference storage | 存储配置 | 设备参考线和交互式 SSD 仿真 |
| Persist blocks created by misses | 开关 | 是否写入 miss 新创建的块，以及后续请求是否有机会复用；仅 Replay 生效 |

模型可能要求特定块对齐。不兼容时，页面或接口会给出原因。

Compute 表示分配给当前执行范围的有效算力。用它计算时间时，FLOPs 和吞吐需对应同一个执行范围。它不会直接改变模型的 KV 字节数或算子 FLOPs。

仅计算运算部分时：

```text
计算时间（秒）= FLOPs ÷（Compute TFLOPS × 10^12）
```

例如 4 × 10^12 FLOPs 按 2000 TFLOPS 换算为 2 ms。Explore 的 TTFT 不计恢复读取时间，随附模型的固定开销和首 token decode 均为 0；计入存储带宽的估计见 Recommend 和 SSD Simulation，详见 [TTFT 的计算方法](#modeling-ttft)。

一个 8 MiB 的逻辑操作，按 128 KiB 的 Max IO 分块，计为 64 次 IO。Max IO 留空时，相关 chunked IOPS 指标留空。

### 工作负载

| 输入 | 含义 | 使用方式 |
| --- | --- | --- |
| Request count | 每个 Case 的请求数 | Replay 中会逐请求执行，数量越大，计算量越大 |
| QPS | 每秒到达请求数 | 控制到达间隔，影响在途重叠、缓存命中和流量压力；Replay 的 TTFT 可随命中变化，但不包含计算排队项 |
| Generation length | 每请求生成 token 数 | 用于生成对象和可选 decode 写入 |
| Prefix groups | 前缀分组数 | 分组较多时，请求可能分散到更多前缀 |
| Zipf alpha | 前缀组访问集中程度 | 越大表示访问越集中 |
| Arrival model | fixed / poisson | 固定到达间隔 / 泊松到达 |
| Seed | 合成负载随机种子 | 对比同一负载时保持不变 |
| Workload shape | Shared prefix groups / Prompt lineage bootstrap | 共享前缀组 / 带初始前缀的请求继承关系 |

初始 chat 负载使用 1000 个请求、100 QPS、256 个生成 token、100 个前缀组、Zipf alpha 1.3 和 seed 1。Explore 的长度和命中比例由所选预设或手工输入设置。这些负载参数只在 Cache Replay 中使用；Bandwidth Envelope 每个 Case 只分析一个请求。

### 缓存

| 输入 | 含义 | 默认值 |
| --- | --- | --- |
| Capacity axis (GiB) | 总缓存容量，可填多个值 | 1024 GiB |
| Shard count | 缓存分片数 | 1 |
| Placement seed | 对象到分片的映射种子 | 0 |
| Warmup fraction | 预热请求占比 | 0 |
| Policy | 最长前缀查找、在途保护、延迟准入和 leaf-LRU | radix_leaf_lru |

容量使用二进制 GiB。容量为 0 的 Case 可作为无驻留缓存的对照。增加容量是否提高命中率，还取决于对象共享和恢复策略。

## 状态恢复和高级选项

### Recovery scenario

| 选项 | 含义 |
| --- | --- |
| Model defaults | 按模型配置使用 checkpoint 和状态保存策略 |
| Periodic checkpoints | 按指定 token 间隔保存 checkpoint |
| Ideal boundary (upper bound) | 假定驻留前缀边界正好有可用状态，用于估计上界 |
| No checkpoint state | 不保存 checkpoint 状态 |

周期策略使用 `Checkpoint interval`，初始值为 2048 tokens。`Retain prompt-end checkpoint` 控制是否额外保留 prompt 结束位置的 checkpoint。

只有 prompt-end 状态时，中途命中的前缀可能没有对应状态。此时页面会提示不能直接恢复，可保留模型默认值、改用周期 checkpoint，或选择理想边界假设。比较不同恢复策略时，需同时查看容量和重算量。

State group overrides 可为一组状态单独指定策略，Model exceptions 可为某个模型单独指定。优先级为：模型级设置 → 状态组设置 → 全局恢复场景 → 模型默认值。没有对应状态的模型不会因为这些设置而新增状态组件。

### Footprint view

| 视图 | 统计范围 |
| --- | --- |
| Calculator reference | 标称容量参考 |
| Active execution | 当前执行场景涉及的状态 |
| Prefix Store | 用于前缀持久化存储的对象 |
| Replay resident | 回放驻留范围；实际使用量还需结合 Cache Replay 结果 |

切换容量范围时，需确认比较的是同一类对象。KV Size 的标称容量、Prefix Store 容量和 Replay 实际驻留量不应混用。

### Sparse、Draft 和 Decode

| 选项 | 含义 |
| --- | --- |
| Full restore / Sparse on demand | 完整恢复 / 按需恢复稀疏访问对象 |
| Sparse locality | Clustered lower、Window/global、Seeded uniform、Dispersed upper 四种访问分布 |
| Request union / Per query, no reuse | 合并请求内查询访问并复用 / 查询之间不复用 |
| Storage layout | 全部 KV heads 放在一起，或按 GQA group 划分 |
| Query chunk | 稀疏查询的处理粒度，单位 tokens |
| Sparse seed | 稀疏随机访问种子 |
| Include draft cache | 将模型支持的 draft 缓存计入容量 |
| Draft execution | Disabled 或 Prefill，控制是否计入 draft prefill |
| Decode KV/state writes | 计入生成过程中的 KV、state 和 checkpoint 写入；仅 Replay 生效 |
| Decode step (µs/token) | 留空时集中在完成时写入；填写时按指定间隔分布写入；仅 Replay 生效 |

选项能否使用取决于模型支持。当前 DSA 的 `sparse_on_demand` 与 `materialized` MLA 计算组合不受支持；可在页面选择 Full restore。其他执行算法需在模型配置中明确指定受支持的组合，页面没有独立的算法下拉框。

Decode 选项计算相关写入，不包含完整的 TPOT、continuous batching 或逐 token kernel 调度。

### DeepSeek V4.1

选择该模型后，会出现两组场景：

- Encoder state：Unavailable 或 Available。
- Encoder output：Recompute when missing 或 Retain and reuse。

每组可选一个或两个值；同时选择两组的两个值，会为该模型生成四种组合。场景会保存在 Case、实验和导出中。Available 假定所需状态已经可用，Retain 假定有效输出快照已经可用。

KV Size 中该模型的结构为只读，global KV、SWA、压缩尾和可选 draft 按各自规则计入，缓冲预留单独列出。

## Recommend

### 示例三：比较两个模型

1. 选择 `Bandwidth Envelope`。
2. Models 只选 Qwen3-8B 和 Qwen3-32B，Runtimes 只选 `bf16_reference`。
3. Storage 选择 `single_nvme_ftl` 对应配置。
4. 设置 block sizes 为 `64`、max IO bytes 为 `131072`。
5. Prompt length 填 `1024, 2048`，Prefix ratio 填 `0.5, 0.75`，QPS 填 `10`，到达模式只选 fixed。
6. Request count 填 `20`，其余负载参数按需要设置。
7. 开启 `Use Storage limits`，保持 Dependency TTFT saving 为 `0 ms`。
8. 选择排名目标，点击 `Recommend a design`。

在其他选项均为单值时，此例计算 2 个模型 × 2 个长度 × 2 个前缀比例，共 8 个 Demand Case。结果是否满足条件由计算决定。

### 工作负载预设

Recommend 提供 Chat、RAG、Agent 对应的负载预设，以及 Custom。选择预设会填入该配置的参数。Custom 从空白工作负载字段开始，需要补齐输入。

Coverage axes 中的多个长度、前缀比例、QPS 和到达模式按组合形成待覆盖场景。完全可行方案必须满足每个采样场景的全部已启用约束。

### 约束

| 界面项 | 判定内容 |
| --- | --- |
| Optimistic GB/s ≤（Envelope）/ Read GB/s ≤（Replay） | 前者限制理想隐藏恢复的带宽要求，后者限制平均读需求 |
| Write GB/s ≤ | 仅 Replay；平均写需求不超过上限 |
| Break-even GB/s ≤（Envelope）/ Aggregate dependency GB/s ≤（Replay） | 前者限制依赖 break-even 带宽，后者限制聚合依赖压力 |
| Dependency IOPS ≤ | 仅 Replay；聚合依赖 IOPS 不超过上限，需设置 Max IO |
| Capacity GiB ≤ | 仅 Replay；Retain-all 容量上界不超过限制 |
| Dependency TTFT saving ≥ | 按额定存储带宽估计的 p50 TTFT 节省，不低于指定值 |

`Use Storage limits` 使用每个 Storage 配置自己的限制。勾选某项 Custom 后，为候选统一指定该项上限。至少需要一个有效约束。

随附存储配置的读带宽为 7 GB/s、写带宽为 5 GB/s、随机读为 1,000,000 IOPS、逻辑容量为 1024 GiB。这些值来自 YAML，需按要分析的设备修改。

TTFT saving 默认下限为 0 ms，表示要求缓存路径不慢于无缓存路径。Replay 的 Capacity 约束检查 retain-all 上界；还应查看实际驻留量和容量限制下的命中表现。

### 排名目标

| Ranking objective | 主要比较内容 |
| --- | --- |
| Maximize worst-case normalized resource headroom | 最差场景中，资源限制还剩多少比例的余量 |
| Maximize workload-envelope coverage | 满足约束的采样场景占比 |
| Minimize average storage pressure | 平均存储压力 |
| Maximize worst-case dependency-aware TTFT margin | 最差场景中，相对 TTFT saving 下限还剩多少余量 |

资源 headroom 按 `1 − 需求 / 上限` 计算。例如需求为 6 GB/s、上限为 8 GB/s，余量为 25%。数值小于 0 表示超过限制。

### 结果

- `Coverage`：通过全部已启用约束的采样场景占比。
- `Fully feasible`：所有采样场景都满足要求。
- `Best partial candidate`：没有完全可行方案时，列出部分可行方案及未满足的条件。
- `Binding resource`：余量最小的资源约束。
- `TTFT margin`：相对 TTFT saving 下限的差值，单独于资源约束统计。
- `Sampled hit-rate intervals`：根据实际命中率采样点整理的可行区间，没有对未计算位置进行可行性推断。

缺少必要计算结果的方案不参与性能排名。切换排名目标可重排已有候选；修改模型、工作负载或约束后，需要重新运行。

点击 `Open in Explore`，会带入所选方案的 Case、分析模式、恢复场景和约束判断。可继续用曲线、热力图和详情查看结果。

## 图表、指标和导出

### Charts、Cache、Cases

| 位置 | 操作 |
| --- | --- |
| Charts | 选择指标、曲线或热力图，切换横轴和图例 |
| Cache | 查看容量对实际命中和缓存变动的影响，仅在 Replay 结果中显示 |
| Cases | 搜索、排序、分页查看 Case；每页最多显示 100 行 |
| Open details | 查看所选 Case 的计算详情 |
| Show in table | 在 Cases 表格中定位所选点 |

Case 详情分为 Overview、Recovery、Components、Timing、Cache、Raw JSON 等视图。Components 展示字节和状态组成；Timing 展示无缓存 TTFT、读取时间为 0 时的缓存路径 TTFT、重算和 suffix 计算时间、固定开销、节省上界及各窗口峰值，运行 SSD 仿真后增加模拟 TTFT 和设备结果；TTFT 的绝对值只在这里显示。Raw JSON 可查看原始字段。

### 指标页签

| 页签 | 主要指标 |
| --- | --- |
| Bandwidth | Optimistic fully-hidden bandwidth p99 |
| Read / Write | 平均读需求、平均写需求；Envelope 中 Read 等于单个请求的恢复字节换算值，Write 为 0 |
| IOPS | 聚合依赖窗口中的分块 IOPS |
| TTFT | Dependency-aware TTFT saving upper bound p50；不计恢复读取时间 |
| Capacity | Retain-all 上界、可复用容量、峰值/最终/受保护驻留量 |
| Recovery / Hits | Shareable、Resident、Recovery boundary、实际块/字节命中率、重算 token、恢复字节 |
| Compute | 净节省 FLOPs、净节省比例、重算和 suffix 计算时间 |
| Peak pressure | 1 ms、10 ms、100 ms、1 s 窗口中的峰值带宽、IOPS 和峰均比；只在 Replay 中有比较意义 |

p50 为中位数，p99 为第 99 百分位。标有这些后缀的指标按该 Case 的请求统计；不要将它们与不同 Case 之间的最大值混淆。Envelope 每个 Case 只有一个请求，各分位数相同。

### 三种带宽/时间结果

- **Offered bandwidth / IOPS**：工作负载提出的平均读写需求，等于总字节 ÷（请求数 ÷ QPS）。Envelope 只有一个请求、按 1 秒计，Read 在数值上等于该请求的恢复字节，Write 为 0。
- **Optimistic fully-hidden bandwidth**：在理想重叠条件下，用于将阻塞恢复 IO 隐藏在计算时间内的带宽要求。
- **Dependency-aware break-even bandwidth**：考虑读依赖的先后顺序，使缓存路径不慢于无缓存路径所需的带宽。

Explore 中的 TTFT saving 按零恢复延迟计算，表示节省时间的上界。需要指定存储带宽下的估计时，查看 Recommend；需要包含排队的设备结果时，运行 SSD Simulation。计算方法和算例见[建模说明](#modeling)。

净节省 FLOPs 为负，表示恢复和重算带来的工作量超过省下的计算。TTFT saving 为负，表示对应模型下的缓存路径更慢。

### 曲线

| X axis | 用途 |
| --- | --- |
| Prefix Ratio | 按前缀比例线性排列 |
| Miss Ratio Log | 展开接近 100% 命中的区间；100% 命中单独显示 |
| Cache Capacity | 比较不同容量，仅用于 Replay；0 容量单独显示 |
| Context Length | 比较上下文长度 |

只有其他输入一致的 Case 才会连接成同一条曲线。`Smooth guide` 只调整线条形状，导出的采样值不变。

图例可以显示/隐藏单条序列，也可选择 Show all、Hide all 或 only。模型颜色、线型和标记用于区分系列。图点支持键盘聚焦和选择。

`Device proxy` 显示所选存储配置的额定参考线。设备仿真结果需另行运行。

### 热力图

Heatmap 用前缀比例和上下文长度构成离散网格。Model、Block、Arrival、Max IO、Cache GiB，以及适用模型的 Encoder 场景决定当前切片。

- Global：使用全部结果的色域，适合不同切片之间比较。
- Local：使用当前切片的色域，便于看清切片内部差异。

对数色域需要正值。零值、缺失值和不可评估状态会按各自状态显示。

### 单位

| 单位 | 换算或含义 |
| --- | --- |
| GB/s | 10^9 bytes/s |
| GiB | 2^30 bytes |
| MiB / KiB | 2^20 / 2^10 bytes |
| ms / µs | 1 ms = 1000 µs；导出字段 `_us` 使用微秒 |
| FLOPs / TFLOPs | 运算量；1 TFLOP = 10^12 次运算 |
| TFLOPS | 每秒 10^12 次运算 |
| IOPS | 每秒 IO 次数 |

### 数值状态

| 显示 | 含义及处理 |
| --- | --- |
| 0 | 该项计算结果为零 |
| 空白 / Missing | 没有可用数值，例如未设置 Max IO 或历史结果没有该字段 |
| Infinity / No finite break-even | 当前模型未给出有限带宽阈值，例如没有正的时间节省，或阻塞读取截止时间不为正；不能按数值 0 处理 |
| No overlap window | 完整评估后，需要阻塞恢复数据，但用于隐藏读取的计算窗口不大于 0；未完成评估的场景先显示 Not evaluated |
| Cached, not resumable | 对象已驻留，但缺少可用的恢复状态 |
| No TTFT gain | 当前场景没有 TTFT 节省 |
| Mixed | 同一 Case 内的请求具有不同结果状态，需查看分项统计 |
| Not evaluated | 相关完整性能路径未计算，或 Case 因不支持的组合而停止 |

部分完整前缀命中场景已知 KV 字节和 suffix 工作量，但未计算 final hidden state、logits 或 output-head 的结束过程。完整 TTFT 及依赖这些时间的指标会标为 Not evaluated。Replay 遇到未知完成时间时停止该 Case，避免把部分结果当作完整回放。

### JSON 和 CSV

JSON 包含规范化请求、各 Case、metrics、evaluation、汇总和警告。CSV 每行对应一个 Case，便于用表格或脚本处理。

CSV 中空值和非有限数值可能都显示为空，应结合 `evaluation_status`、结果状态和 JSON 查看原因。JSON 对非有限值可能使用 `Infinity` 等字符串。

修改输入后，旧结果会标记为过期，导出和 SSD 仿真暂时禁用。重新计算，或恢复上一轮参数后，可继续操作。切换显示指标、图例和图表样式不会重新计算。

<a id="modeling"></a>
## 建模说明

本节说明 Explore 主要数值的计算方法、适合回答的问题和适用范围。公式以 MHA/GQA/MQA/MLA 这类只恢复 KV 的模型为例；有状态模型多一段状态重算，规则相同。

<a id="modeling-modes"></a>
### Envelope 与 Replay 的算法

| 项目 | Bandwidth Envelope | Cache Replay |
| --- | --- | --- |
| 分析的请求 | 每个 Case 只有 1 个请求，t = 0 到达 | 按负载参数生成 Request count 个请求，按 QPS 和 Arrival model 依次到达 |
| 命中长度 | ⌊上下文长度 × Prefix ratio⌋，假定已经驻留 | Prefix ratio 只是上限；实际命中是查找时缓存中最长的连续驻留前缀，只计完整块 |
| 共享、容量和驱逐 | 不模拟 | radix 查找、在途保护、完成后准入、leaf-LRU 驱逐 |
| 写入 | 不计，Write 恒为 0 | Persist 开启时写入 miss 新生成的块；decode 写入开启时另计生成阶段的写入 |
| 平均带宽的时间基准 | 1 秒（1 个请求 ÷ 1 QPS） | 测量请求数 ÷ QPS |
| p50 / p99 | 只有一个请求，各分位数相同 | 按测量请求统计，不含预热请求 |
| 不使用的输入 | Request count、QPS、Generation length、Prefix groups、Zipf alpha、Seed、Workload shape、Arrival model、Persist、Decode 写入和 Decode step | — |

Envelope 中同时勾选 fixed 和 poisson 会让 Case 数翻倍，两份结果完全相同。

Replay 中，每个请求的完成时间 = 到达时间 + 缓存路径计算时间（见下一节），计算不排队。请求完成前，它命中的块受保护、不能驱逐；它新生成的块在完成后才能被后续请求命中。QPS 会改变请求之间的重叠、缓存驻留和命中，进而改变重算量及 TTFT。

<a id="modeling-ttft"></a>
### TTFT 的计算方法

TTFT 按单个请求计算，从请求到达到第一个 token：

```text
无缓存 TTFT   = 整段输入的 prefill 计算时间 + 首 token decode
缓存路径 TTFT = 固定开销 + 各阶段计算时间之和 + 首 token decode
计算时间      = FLOPs ÷ Compute
```

缓存路径按顺序包含以下阶段，不需要的阶段为 0：必要的格式转换 → 状态重算（界面中的 Replay compute）→ 未命中 suffix 的计算 → 完整命中时的收尾 → 第一个 token 的 LM head。DeepSeek V4.1 使用自己的执行阶段，在 Timing 页单独列出。

- FLOPs 由模型 YAML 中的公式计算，包括投影、FFN/MoE（只计激活的专家）和 attention。attention 按因果关系逐对计数，suffix 中的每个 token 仍要与整段前缀做 attention；稀疏模型只计选中的部分。一次乘加计 2 次运算。
- softmax 中的减法、求和和除法计入 `elementwise` FLOPs，并按 Compute 换算时间。`exp`、`max_compare` 等特殊操作单独记录在 `special_operation_counts`，不折算为 FLOPs 或时间；未计入的工作列在 `counting_omissions` 中。
- 声明了词表大小的模型按 1 个 token 计入 LM head，两条路径相同；Cohere、Llama 等未声明词表的配置不计 LM head，并在 `counting_omissions` 中注明 `output_head_geometry_not_declared`。
- 随附模型的固定开销（`scheduler_overhead_us`）和首 token decode（`decode_us_per_token`）均为 0，TTFT 实际只剩 prefill 计算时间。
- 已计入 FLOPs 的运算使用同一个 Compute 值。Runtime 支持按算子类别分别设置吞吐，随附 Runtime 都使用单一值。

Explore 的 **TTFT saving = 无缓存 TTFT − 恢复读取时间为 0 时的缓存路径 TTFT**。它不计任何存储时间，在随附模型中等于 Compute 页的“净节省 FLOPs ÷ Compute”，因此是节省时间的上界。

三个位置给出的 TTFT 依次更接近实际：

| 位置 | 恢复读取的处理 | 仍不包含 |
| --- | --- | --- |
| Explore（TTFT 页、Timing 页） | 读取时间按 0 计 | 存储带宽和排队 |
| Recommend | 按存储额定读带宽读取；每个计算阶段等它需要的数据读完才开始 | 请求之间的排队；每个请求单独占用带宽 |
| SSD Simulation | 当前 Case 全部请求的 IO 在一块模拟 SSD 上排队 | 主机、网络和 GPU 侧的传输 |

SSD Simulation 给出三种 TTFT：Optimistic 假定读取与计算完全重叠；Dependency 要求每个阶段等它的数据读完；No-overlap 先读完全部数据再开始计算。Hidden miss ratio 是读取没能被计算时间掩盖的请求占有恢复请求的比例。

主图只显示节省量；无缓存 TTFT 和缓存路径 TTFT 的绝对值在 Case 详情的 Timing 页查看。

<a id="modeling-bandwidth"></a>
### 两个带宽阈值

两个阈值都按单个请求计算，假定读取从请求到达时开始、以恒定速率进行，并且这个请求独占带宽。阻塞恢复字节是计算必须等待的恢复数据，不含可选的 draft 数据。

```text
Optimistic fully-hidden bandwidth     = 阻塞恢复字节 ÷（缓存路径计算时间 − 固定开销）
Dependency-aware break-even bandwidth = 阻塞恢复字节 ÷ 读取截止时间
读取截止时间 = 无缓存 TTFT − 固定开销 − 首 token decode − 从首次使用这些数据的阶段到最后阶段的计算时间
```

KV 由 suffix 阶段使用，所以 KV 类模型的读取截止时间就是“无缓存 TTFT − 缓存路径计算时间”，即省下的计算时间。有状态模型的 checkpoint 由重算阶段使用，截止时间更早；有多组数据时，按首次使用的顺序累计字节，取各截止时间要求中的最大值。

| 阈值 | 回答的问题 | 随命中率的变化 |
| --- | --- | --- |
| Optimistic fully-hidden | 读取与计算完全重叠时，多大带宽能让读取完全不增加 TTFT | 命中率越高，可用来掩盖读取的计算越少，数值上升很快 |
| Dependency-aware break-even | 每个阶段等数据读完再开始时，多大带宽下使用缓存不比全部重算慢 | 恢复字节和省下的时间同时增加，变化通常较小 |

带宽低于 break-even 时，按依赖顺序执行的缓存路径比不使用缓存更慢；带宽达到 fully-hidden 时，在完全重叠的前提下读取不再增加 TTFT。实际系统的重叠程度通常介于这两种假设之间。默认的 Bandwidth 页显示 fully-hidden 的 p99，数值大并不表示至少需要这么多带宽；判断缓存是否划算，应同时查看 Cases 表的 Break-even 列。

阈值先检查完整评估状态，再分别按 optimistic 和 dependency 的规则判断：

| 显示 | 条件 |
| --- | --- |
| Not evaluated | 完整 first-token 路径尚未建模，或 Case 因不支持的组合而未完成评估；此时不套用下面的数值判断 |
| No overlap window | optimistic：已完整评估、阻塞恢复字节大于 0，但用于隐藏读取的计算窗口不大于 0 |
| No finite break-even | dependency：无限带宽下仍没有正的时间节省（`no_compute_saving`），或阻塞读取截止时间不为正（`no_replay_slack`） |
| optimistic = 0 | 已完整评估且没有阻塞恢复数据（`no_restore_required`） |
| dependency = 0 | 已完整评估、存在正的时间节省，且没有阻塞恢复数据；没有正的节省时仍返回 `null / no_compute_saving` |

例如，当前 Qwen3-8B 在 Envelope 中 100% 命中时，完整结束过程未建模，两个阈值均为 `null`，状态为 `not_evaluated`；不能仅因没有 suffix 就归为 No overlap window。0% 命中时，optimistic 阈值为 0，而 dependency 阈值为 `null / no_compute_saving`。

<a id="modeling-example"></a>
### 算例：Qwen3-8B

设置：Qwen3-8B，Bandwidth Envelope，上下文 32,768 tokens，BF16 reference，Compute 2000 TFLOPS，块大小 256，Max IO 131,072 bytes。7 GB/s 是随附存储配置的额定读带宽。

| 项目 | 50% 命中 | 90% 命中 |
| --- | ---: | ---: |
| 无缓存 TTFT | 387.3 ms | 387.3 ms |
| 缓存路径 TTFT（读取时间为 0） | 233.5 ms | 53.1 ms |
| TTFT saving（Explore 显示） | +153.7 ms | +334.2 ms |
| 阻塞恢复字节 | 2.42 GB | 4.38 GB |
| Optimistic fully-hidden bandwidth | 10.3 GB/s | 82.5 GB/s |
| Dependency-aware break-even bandwidth | 15.7 GB/s | 13.1 GB/s |
| 7 GB/s，每个阶段等数据读完（Recommend 的算法） | 578.6 ms，慢 191.4 ms | 678.6 ms，慢 291.4 ms |
| 7 GB/s，读取与计算完全重叠（手算） | 345.1 ms，快 42.1 ms | 625.6 ms，慢 238.3 ms |

- 阈值可按公式验算：2.42 GB ÷ 233.5 ms ≈ 10.3 GB/s；2.42 GB ÷（387.3 − 233.5）ms ≈ 15.7 GB/s。
- Explore 显示能省 153.7 ms 和 334.2 ms，但 7 GB/s 低于两种命中率的 break-even，按依赖顺序执行时使用缓存都更慢；完全重叠时也只有 50% 命中能快 42.1 ms。只看 TTFT 页不能判断缓存是否划算。
- 50% 命中只省约 40% 的计算：suffix 的 token 仍要与全部前缀做 attention，长上下文下后半段 token 更贵。命中率不等于计算节省比例。
- 90% 命中要恢复更多数据，留给读取的计算时间却更短，所以 fully-hidden 从 10.3 GB/s 升到 82.5 GB/s；break-even 反而略有下降。

<a id="modeling-aggregate"></a>
### 聚合依赖带宽和 IOPS

聚合依赖指标考虑多个请求同时在途：每个请求的阻塞恢复数据在“到达时刻到读取截止时间”之间平均分摊（多组数据按各自截止时间分段），再把同一时刻在途的请求相加，取整个 Case 的峰值。截止时间不为正的部分不计入，数量记在 Raw JSON 的 `invalid_window_count`。

IOPS 页显示其中按 Max IO 切分后的分块 IOPS，未设置 Max IO 时为空；带宽值 `aggregate_dependency_required_GBps` 在 JSON/CSV 导出中，也是 Recommend 的 Aggregate dependency GB/s 约束。Envelope 只有一个请求，这些值就是该请求自身的要求；Replay 中它们反映 QPS 和请求重叠带来的总压力，是 Explore 中唯一按截止时间计算并计入并发的带宽和 IOPS 要求。

<a id="modeling-peak"></a>
### Peak pressure 的计算方法

```text
平均带宽 = 总字节 ÷（请求数 ÷ QPS）
窗口峰值 = 任意长度为 W 的时间窗内落入的最大字节数 ÷ W
峰均比   = 窗口峰值 ÷ 平均带宽
```

峰值按事件时刻统计：恢复读取记在请求到达时刻；miss 写入记在缓存路径计算完成时刻；decode 写入未填 Decode step 时也记在完成时刻，填写时按步长分布。读取并没有在传输时间内摊开，所以 1 ms、10 ms 这样的短窗口主要反映单个请求一次要读多少，而不是设备需要持续提供的速率；按截止时间摊开的总要求见[聚合依赖带宽和 IOPS](#modeling-aggregate)。

Envelope 只有一个请求，使用 1 秒统计基准，读峰值 = 该请求的恢复字节 ÷ W。只有恢复读字节大于 0 时，1 ms、10 ms、100 ms、1 s 的读峰均比才分别为 1000、100、10、1；恢复读字节为 0 时，读峰值为 0，但峰均比因分母为 0 而无定义，返回 `null`。Envelope 不计写入，写峰值为 0，写峰均比同样无定义。分析负载并发带来的峰值压力时使用 Replay。

<a id="modeling-limits"></a>
### 不能回答的问题

| 问题 | 原因 |
| --- | --- |
| 真实服务中的 TTFT 和排队时间 | 每个请求按同一 Compute 独立估算，没有计算排队、batching、continuous batching、chunked prefill 和调度。QPS 不增加计算排队项，但可通过缓存驻留和命中变化使 TTFT 增大或减小；程序也不报告计算资源过载 |
| 某块 GPU 的实际 prefill 速度 | 随附 Runtime 对已计入 FLOPs 的运算和各输入长度使用同一个有效算力；不考虑显存带宽、kernel 效率以及 TP/EP 等通信 |
| KV 从存储到 GPU 的完整时间 | 没有建模主机内存、PCIe、网络和反序列化。Explore 不计读取时间，Recommend 只按额定带宽，SSD Simulation 只模拟一块 SSD |
| decode 阶段的速度 | 首 token decode 为 0；不计算 TPOT、decode 吞吐和逐 token 调度，Decode 选项只计入写入 |
| 实际业务负载的命中率 | 只能使用合成负载（共享前缀组或请求继承关系、Zipf 分布），页面不能导入真实 trace |
| 多分片缓存的设备时延 | 单 SSD 仿真只接受一个 shard |
| 实测时间 | 结果标为 uncalibrated，是解析估算；适合在相同假设下比较方案，绝对值只作量级参考 |

## Experiments

完成的 Sweep 和 Recommendation 自动保存到 SQLite。列表支持搜索、改名、固定、重开和删除，也可以选择多个实验批量删除。

系统保留最近 50 个未固定实验，固定实验不计入限额。需要长期保留的结果可使用 Pin。

### 重开与比较

1. 打开 Experiments，找到要查看的记录。
2. 使用重开操作恢复其输入和结果。
3. 比较时选择两轮实验作为 baseline 和 candidate。
4. 查看匹配 Case 的指标变化，以及 Added/Removed 的 Case。

只有规范化输入匹配的 Case 才计算逐项差值。更换模型、计算吞吐或恢复假设可能形成不同 Case。基准值为零或非有限数时，百分比变化留空。

旧实验缺少新指标时，页面保留缺失状态。需要该指标时重新运行，不会在打开旧实验时自动补算。

### 数据库与备份

默认位置：

```text
dashboard_v2/.data/dashboard.sqlite3
```

首次使用时自动创建空库。指定其他位置：

```bash
PCS_DASHBOARD_DB="/path/to/dashboard.sqlite3" sh start.sh
```

```powershell
$env:PCS_DASHBOARD_DB = "C:\data\dashboard.sqlite3"
.\.venv\Scripts\python.exe start.py
```

备份前停止服务，复制整个 `.data/` 目录。使用自定义路径时，一并复制数据库及同名 WAL/SHM 文件。安装 Python 依赖不会清空实验库。

实验保存失败时，当前计算结果仍可留在页面中，界面会提示保存失败。应先导出需要的结果，再检查目录写权限。

### 浏览器状态和后台任务

浏览器记忆主题、最近视图和部分页面状态；SQLite 保存实验。换浏览器后可以读取同一服务器的实验，但浏览器偏好需要重新设置。

关闭页面不会取消服务器任务。取消时使用 Cancel；停止服务会终止未完成任务。服务重启后，已保存实验仍在，未完成任务需重新提交。

## KV Size

1. 选择模型 family 和模型。
2. 设置 token 数、sequence 数和精度。
3. 按需要设置 Indexer、State、checkpoint 或 draft 选项。
4. 查看总容量、组件分解、公式代入和模型参数。
5. 点击 `Add to compare` 保存当前配置，再修改输入添加其他配置。
6. 点击 Compare 并排比较，使用 `Back to calculator` 返回。

最多保存四组比较配置，允许同一模型使用不同长度、精度或状态策略。配置按有效参数去重。比较列表在切换页面时保留，刷新浏览器后清空。

### 容量计算示例

对于每层结构相同、Key/Value 同精度、没有额外状态或量化开销的 GQA，可按下式理解 KV 容量：

```text
每 token 字节数 = 层数 × KV heads ×（Key head dim + Value head dim）× 每元素字节数
总字节数 = 每 token 字节数 × tokens × sequences
```

例如 32 层、8 个 KV heads、Key/Value head dim 都为 128、每元素 2 bytes：

```text
每 token = 32 × 8 × (128 + 128) × 2 = 131,072 bytes
8192 tokens、1 sequence = 1 GiB
```

MLA、稀疏 Indexer、线性 attention 状态、滑动窗口和压缩缓存使用各自的组件公式。它们可能包含固定大小状态、量化 scale、checkpoint 或对齐开销，容量不一定与 token 数线性增长。

Calculation-layer breakdown 展开的是容量公式中的计算层。Known buffer reservations 单独列出，不加到有效 payload 总量中。

## Architecture

Architecture 在首次打开时加载，支持节点展开、下钻、动画、全屏和图形导出。导出格式以菜单提供的选项为准。

主题跟随顶栏设置。切换页面再返回时，图中的位置和展开状态会保留。

## 配置文件

| 目录 | 内容 |
| --- | --- |
| `configs/models/` | 模型结构、attention 参数、计算公式、状态和来源信息 |
| `configs/runtime/` | 精度、有效计算吞吐及分析默认值 |
| `configs/workloads/` | Chat、RAG、Agent 合成负载预设 |
| `configs/storage/` | SSD 容量、带宽、IOPS、时延和设备参数 |
| `configs/cache/` | 缓存容量、分片、准入和驱逐策略 |

### Runtime

| 文件 | 用途 |
| --- | --- |
| `bf16_reference.yaml` | 默认 BF16 参考配置，默认有效算力 2000 TFLOPS |
| `fp4_2pf.yaml` | FP4 缓存精度和 2000 TFLOPS 计算配置 |
| `v4_fp8_mxfp4.yaml` | DeepSeek V4 的条目字节布局和周期 checkpoint 默认值 |
| `v41_fp4.yaml` | 面向 DeepSeek V4.1 的 FP4 运行配置 |

Explore 初始使用 BF16 reference，可在页面修改 Compute。Recommend 的 Runtimes 列表用于选择候选配置；API 可以显式传入 `runtime_config`。

需要修改配置时：

1. 备份要修改的 YAML。
2. 修改对应字段，保持 YAML 缩进和数据类型。
3. 重启服务，运行 `sh start.sh --check`；Windows 使用 `.\.venv\Scripts\python.exe start.py --check`。
4. 先运行少量 Case，查看模型、精度、参数和结果是否对应。

接口只接受程序 `configs/` 目录内的配置路径。普通计算吞吐、请求数、缓存容量等参数可直接在页面输入。

## API

接口与页面使用同一个地址，例如 `http://127.0.0.1:8001`。请求和响应均使用 JSON。

| 接口 | 功能 |
| --- | --- |
| `GET /api/configs` | 获取模型、Runtime、负载、存储、缓存、默认值和限制 |
| `POST /api/sweep` | 运行同步 Sweep，并保存成功结果 |
| `POST /api/sweep/jobs` | 提交后台 Sweep |
| `GET /api/sweep/jobs/{job_id}` | 查询 Sweep 任务 |
| `DELETE /api/sweep/jobs/{job_id}` | 请求取消 Sweep 任务 |
| `POST /api/case` | 获取 Case 详情；`include_storage` 控制是否运行 SSD 仿真 |
| `POST /api/recommendations/jobs` | 提交推荐任务 |
| `GET /api/recommendations/jobs/{job_id}` | 查询推荐任务 |
| `DELETE /api/recommendations/jobs/{job_id}` | 请求取消推荐任务 |
| `GET /api/experiments` | 列出实验 |
| `GET /api/experiments/{id}` | 读取实验及结果 |
| `PATCH /api/experiments/{id}` | 修改名称或固定状态 |
| `DELETE /api/experiments/{id}` | 删除实验 |
| `POST /api/experiments/compare` | 指定 baseline 和 candidate 比较实验 |

### Python 请求示例

先启动服务，再在 Python 中运行以下代码。它使用 Python 标准库，提交四个 Envelope Case。Envelope 不使用其中的 `workload_overrides` 和 `write_miss_blocks`，它们只在 `cache_replay` 中生效：

```python
import json
from urllib.request import Request, urlopen

base_url = "http://127.0.0.1:8001"
payload = {
    "analysis_mode": "bandwidth_envelope",
    "model_configs": ["configs/models/qwen3_8b.yaml"],
    "runtime_config": "configs/runtime/bf16_reference.yaml",
    "workload_config": "configs/workloads/synthetic_chat.yaml",
    "storage_config": "configs/storage/single_nvme_ftl.yaml",
    "compute_config": {
        "timing_mode": "normalized_reference",
        "effective_tflops": {"default": 2000},
    },
    "axes": {
        "seq_len": [1024, 2048],
        "prefix_ratio": [0.5, 0.75],
        "block_size_tokens": [64],
        "arrival_model": ["fixed"],
        "max_io_bytes": [131072],
    },
    "workload_overrides": {
        "request_count": 20,
        "qps": 10,
        "generation_len": 16,
        "prefix_group_count": 1,
        "prefix_zipf_alpha": 1.3,
        "seed": 1,
    },
    "write_miss_blocks": True,
}

request = Request(
    base_url + "/api/sweep",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request, timeout=60) as response:
    result = json.load(response)

print("Case count:", result["case_count"])
for row in result["rows"]:
    print(row["model"], row["seq_len"], row["prefix_ratio"],
          row["evaluation"]["status"])
```

返回结果中的 `request` 保存规范化输入，`rows` 保存各 Case，`metrics` 保存数值，`evaluation` 保存计算状态，`summary` 保存汇总。部分 Case 为 Not evaluated 时，HTTP 请求仍可能成功，应逐行检查状态。

调用 `/api/case` 时，可从 Sweep 中取出 `rows[i]["case"]`，作为请求的 `case` 字段。运行 SSD 仿真再加上 `"include_storage": true`；被选中的 Case 必须满足 Replay 和单 shard 要求。

### 运行限制

| 项目 | 限制 |
| --- | --- |
| Sweep | 最多 1,200 个 Case |
| 页面同步计算 | 100 个 Case 以内；更大 Sweep 使用后台任务 |
| Recommend | 最多 1,200 个实际 Demand Case |
| 单 SSD 仿真 | 200,000 个逻辑操作、2,000,000 个物理 IO |
| 未固定实验 | 保留最近 50 个 |
| 后台终态任务 | 内存中保留最近 8 个，完成一小时后按访问情况清理 |

Sweep 和 Recommend 共用计算锁。同一时刻已有不同计算任务时，新任务返回 HTTP 409。同类且输入相同的后台请求可能复用正在运行的任务。

任务清理与实验删除是两回事：后台任务记录在内存中，成功保存的实验在 SQLite 中。服务重启会清空内存任务。

## 常见问题

| 情况 | 处理方法 |
| --- | --- |
| 首次运行计算很久 | 先只选一个模型，减少长度、请求数和参数轴；确认没有误选 Full Tail |
| 模型列表筛选后 Case 数仍很大 | 筛选不会取消隐藏的已选模型，检查实际选中数量 |
| 输入前缀比例很高，实际命中仍低 | 查看冷启动、Prefix groups、共享对象、容量、驱逐和 Warmup 设置 |
| Resident 有值，Recovery 很低 | 查看 checkpoint/state 是否能支持该命中边界，以及重算 token 数 |
| 容量增大后命中率不变 | 容量可能已足够，或瓶颈在共享关系、块对齐和恢复状态 |
| 高命中率下所需带宽反而升高 | 恢复字节可能增加，同时剩余计算窗口缩短；查看恢复字节与计算时间 |
| IOPS 为空 | 设置 Max IO bytes 后重算，并查看 Case 是否完整评估 |
| 峰均比为空 | 平均带宽为零时比值无定义；旧实验也可能没有该字段 |
| 100% 命中显示 Not evaluated | 完整 first-token 结束过程可能未建模，查看 evaluation reason |
| 某些模型报 unsupported | 查看模型、Runtime、块对齐、Sparse、Draft 和恢复策略组合 |
| 热力图看不到预期数据 | 检查当前模型、长度、容量和场景切片；必要时切换 Global/Local |
| 图表为空但表格有结果 | 检查图例是否全部隐藏、指标是否需要 Replay、Max IO 或有效计算结果 |
| 导出按钮禁用 | 参数改变使结果过期，重新运行或恢复上一轮输入 |
| SSD Simulation 不可用 | 确认 Replay 已开启、shard 为 1、结果未过期且 IO 数量未超限 |
| 推荐没有完全可行方案 | 查看失败约束，调整候选或工程预算，再重新计算 |
| 比较实验只有 Added/Removed | 两轮 Case 输入不匹配，例如模型、Compute 或恢复假设不同 |
| 保存失败但图表正常 | 先导出结果，再检查数据库路径和写权限 |
| 同一模型在不同页面容量不同 | 对齐精度、长度、sequence 数、state/draft 策略和 Footprint 范围 |
| 重启后后台任务找不到 | 从 Experiments 读取已保存结果；未完成任务重新提交 |

安装、端口和 Python 环境问题见 [INSTALL.md](INSTALL.md)。

## 文件目录

| 路径 | 内容 |
| --- | --- |
| `dashboard_v2/server.py` | HTTP 接口和计算调度 |
| `dashboard_v2/recommendation.py` | 约束评估和推荐排序 |
| `dashboard_v2/parallel_sweep.py` | 并行 Case 执行 |
| `dashboard_v2/experiment_store.py` | SQLite 实验存储 |
| `dashboard_v2/index.html`、`dashboard_v2/styles.css`、`dashboard_v2/js/` | 页面、样式和前端功能 |
| `dashboard_v2/architecture/` | 架构页资源 |
| `ssu_sim/cache/` | 缓存回放 |
| `ssu_sim/demand/` | 需求和时间计算 |
| `ssu_sim/models/` | 模型结构、容量和 FLOPs 计算 |
| `ssu_sim/prefix/` | 前缀块标识 |
| `ssu_sim/storage/` | 单 SSD 仿真 |
| `ssu_sim/workload/` | 合成负载 |
| `ssu_sim/utils/` | 配置读取和单位换算 |
| `configs/` | 模型、Runtime、负载、存储和缓存配置 |
| `install.*`、`start.*` | 安装和启动入口 |
| `requirements.txt` | Python 依赖 |
| `pyproject.toml` | Python 包安装配置 |
| `.venv/` | 安装时生成的 Python 环境 |
| `dashboard_v2/.data/` | 使用时生成的实验数据库 |

<a id="controls"></a>
## 逐项控件说明

本节默认值指首次打开页面、没有浏览器偏好、没有正在恢复的任务，也没有重开历史实验时的状态。Explore 从 Quick Compare 和 BF16 reference 开始。预设、模型切换和历史实验会重新填入部分值，具体列在对应项中。

“需重算”表示当前结果会被标为过期，重新运行后才能导出或做 SSD 仿真。“只改显示”不修改 Case 数值，也不重新计算。KV Size 的修改会立即在浏览器中重新计算容量，不修改模型 YAML，也不改变 Explore 或 Recommend 的设置。

[公共设置](#controls-common) · [Explore 设置](#controls-explore) · [恢复与高级参数](#controls-recovery) · [图表和结果](#controls-results) · [Recommend](#controls-recommend) · [Experiments](#controls-experiments) · [KV Size](#controls-kv) · [KV 高级字段](#controls-kv-fields) · [模型默认值](#controls-model-defaults) · [Architecture](#controls-architecture)

<a id="controls-common"></a>
## 公共设置

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Explore / Recommend / Experiments / KV Size / Architecture | 切换页面 | 五个页面，单选 | Explore | 始终可用 | 保留当前页面输入和结果；不重新计算。浏览器记忆最近页面，后台任务可继续运行 |
| Theme | 选择颜色主题 | System / Light / Dark | System | 所有页面；System 跟随操作系统 | 只改显示；写入浏览器偏好，同源标签页同步，Architecture 跟随主页面 |

<a id="controls-explore"></a>
## Explore：输入、模型和运行

### 模式和预设

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Simple / Expert | 选择显示哪些输入 | Simple / Expert | Simple | Explore | 仅切换控件可见性；高级输入继续保留。Simple 的前缀比例显示百分数，Expert 显示小数；浏览器记忆选择 |
| Quick Compare | 填入快速比较参数 | 点击应用 | 初始已选 | 配置加载完成后 | 设为 5 个默认模型、32768 tokens、0.7/0.95/0.99/0.999、256-token 块、fixed、131072-byte Max IO；横轴改为 Prefix Ratio；需重算 |
| Full Tail | 填入高命中率尾部采样 | 点击应用 | 未选 | 配置加载完成后 | 设为另一组 5 模型、65536/131072/262144 tokens 和 55 个命中率点；块/arrival/Max IO 同 Quick Compare，横轴改为 Miss Ratio Log；需重算 |
| Analysis mode | 选择假设前缀已驻留或实际缓存回放 | Bandwidth Envelope / Cache Replay | Bandwidth Envelope | 始终显示 | Replay 显示 Workload behavior 和 Prefix Cache Replay；Envelope 隐藏这两区但保留输入；模式进入 Case，需重算 |
| Compute (TFLOPS) | 设置有效默认计算吞吐 | 单个有限实数 > 0；不能留空 | 2000 TFLOPS | 两种分析模式 | 改变使用该吞吐换算的计算时间、带宽窗口和 TTFT；不改变模型 KV 字节或算子 FLOPs。历史实验中的单算子吞吐可能另有设置；需重算 |
| Reference storage | 选择设备参考值和 SSD 模型 | 当前为 single_nvme_ftl | single_nvme_ftl | Expert | 写入 Case 的存储配置，用于 Device proxy 和 SSD Simulation；更换后结果过期，需要重算 |

Quick Compare 与 Full Tail 的具体模型见前面的预设表。两者还会将 V4.1 Encoder 两组场景重置为 Unavailable 和 Recompute when missing。预设不重置 Compute、工作负载、缓存容量、恢复策略和 Sparse/Draft/Decode 参数；切换预设后仍需检查这些输入。

Explore 的 Runtime 下拉在页面中始终隐藏，是兼容旧输入的控件；不要寻找可见的 Runtime 选择框。当前 Runtime 可从结果和导出中确认，Recommend 可以选择 Runtime。Policy、Case 数、状态摘要、缓存生命周期预览都是只读输出。

### 模型弹窗

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Change models | 打开模型选择 | 点击 | 弹窗关闭 | 配置已加载 | 打开弹窗并聚焦搜索；本操作不改变参数 |
| 模型复选框 | 选择参与计算的模型 | 当前 56 个 active 模型，见模型目录；可多选，至少一个 | Quick Compare 的 5 个模型 | 不兼容 Runtime 的模型禁用；最大长度/块对齐仍会进一步检查 | 改变 Case 数、容量与计算公式；动态更新状态编辑器；需重算 |
| Search models | 查找模型 | 任意文字；忽略大小写；空白表示不限制 | 空白 | 弹窗内 | 匹配模型名称、显示标签、ID 和 attention 描述；只改变可见列表，不取消隐藏的已选模型 |
| All families / Family | 按模型家族筛选 | All families，或模型目录中的 13 个 family | All families | 弹窗内 | 只过滤列表；与搜索、Formula、Evidence 同时生效，不改已选集合 |
| All formulas / Formula | 按容量/attention 公式类别筛选 | All formulas，或公式表中的 10 类 | All formulas | 弹窗内 | 只过滤列表，不修改模型公式或已有选择 |
| All evidence / Evidence | 按配置证据标签筛选 | All evidence / analytical / assumed / calculator_reference / derived_from_official_code / synthetic_scenario / uncalibrated / unsupported | All evidence | 弹窗内；一个模型可含多个标签 | 按模型是否包含该标签过滤；不是改变精度或算法，也不等于对模型整体做可用性判断 |
| Select visible | 批量勾选当前可见模型 | 点击；若可见可选模型已全选，则取消它们 | 可点击 | 仅作用于未禁用的可见模型 | 不改变隐藏模型的勾选；改变选择时更新 Case 数并使结果过期 |
| Close / Done / 点击弹窗外侧 | 关闭模型弹窗 | 点击 | 弹窗打开后可用 | 模型弹窗 | 勾选时已立即生效；关闭弹窗不会撤销选择，也不会自动运行 |

Evidence 标签：analytical 表示解析建模；assumed 表示显式假设；calculator_reference 表示容量计算参考；derived_from_official_code 表示从公开实现推导；synthetic_scenario 表示人工设定场景；uncalibrated 表示未做实机校准；unsupported 表示配置中存在未支持的证据项。

### Comparison points

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Context lengths | 设置输入上下文长度 | 一个或多个正整数 tokens，英文逗号分隔；不超过所选模型上限 | 32768 | 两种模式；每个模型都必须满足其长度范围 | 增加值会扩展 Case 组合；改变 KV 字节和计算量；需重算 |
| Prefix hit rates | 设置候选前缀占输入长度的比例 | 一个或多个 0–1 的小数或 0%–100%；如 0.5, 75%；不能全空 | 70%, 95%, 99%, 99.9%（Expert 为 0.7, 0.95, 0.99, 0.999） | 两种模式 | Envelope 为假定驻留比例，Replay 为可复用上限；增加点增加 Case，100% 可能出现 Not evaluated；需重算 |
| Block sizes | 设置每个逻辑 KV 块的 token 数 | 一个或多个正整数；同时满足全部所选模型的最小值和倍数要求 | 256 | Expert | 改变块对齐、共享/命中边界和操作数；增加值扩展 Case；需重算 |
| Max IO bytes | 设置逻辑操作的最大 IO 分块字节数 | 正整数列表；空白、None、null 表示不指定；允许与数值混合 | 131072 bytes（128 KiB） | Expert | 改变分块 IOPS 和仿真工作量；未指定时相关 IOPS 缺失；增加值扩展 Case；需重算 |
| Arrival model：fixed / poisson | 设置请求到达间隔 | 分别勾选 fixed、poisson，可同时选，至少一个 | fixed 勾选；poisson 不勾选 | Expert；只在 Replay 中起作用 | fixed 为固定间隔，poisson 为泊松到达；同时选择增加 Case；Replay 中影响时间分布和峰值，Envelope 中两份结果相同；需重算 |

列表输入会去重并排序。Context lengths、Prefix hit rates、Block sizes 不能留空；Max IO 是例外，空白就是不指定。块的 token 数与 Max IO 的 byte 数是不同单位。

### Workload behavior：仅 Replay 显示

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Persist blocks created by misses | 控制 miss 新生成块的持久化写入 | 开 / 关 | 开 | Replay 显示；设置保存在两种模式的请求中，但只在 Replay 中起作用 | 影响 Replay 的写需求和后续可复用对象；Envelope 不计写入；需重算 |
| Request count | 设置每个 Case 的普通请求数 | 单个整数 ≥ 0，不能留空 | 1000 | Replay | 改变回放样本和运行量；0 没有普通测量请求，通常用于特殊场景；需重算 |
| QPS | 设置平均请求到达率 | 单个正有限数，单位 requests/s，不能留空 | 100 | Replay | 改变到达间隔、在途重叠、驻留和命中，进而影响 TTFT、平均及峰值压力；不增加计算排队项；需重算 |
| Generation length | 设置每请求生成长度 | 单个整数 ≥ 0，单位 tokens/request | 256 | Replay | 影响模型所生成对象及启用后的 decode 写入；不提供完整 decode 吞吐预测；需重算 |
| Prefix groups | 设置不同前缀身份的组数 | 单个整数 ≥ 1 | 100 | Replay + Expert | 改变共享对象分布；lineage 模式还影响 bootstrap 请求数；需重算 |
| Zipf alpha | 设置前缀组访问集中程度 | 单个有限数 ≥ 0 | 1.3 | Replay + Expert | 值越大访问越集中，0 接近均匀；改变复用和容量竞争；需重算 |
| Seed | 设置负载随机种子 | 单个整数，可为 0 或负数；不能留空 | 1 | Replay + Expert | 改变随机到达/共享样本；复现实验应保持相同；需重算 |
| Workload shape | 选择请求共享结构 | Shared prefix groups / Prompt lineage bootstrap | Shared prefix groups | Replay + Expert | 前者按共享前缀组生成请求；后者先构造 bootstrap 前缀再生成继承请求；改变生命周期与命中，需重算 |

### Prefix Cache Replay

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Enable | 切换是否启用回放 | 勾选 / 取消 | 初始为关，整区隐藏 | 切到 Cache Replay 后在 Expert 中显示 | 与 Analysis mode 同步；取消后回到 Envelope 并隐藏回放区，重新开启用 Analysis mode；需重算 |
| Cache preset | 加载缓存配置的初始参数 | 当前为 radix_leaf_lru | radix_leaf_lru | Replay + Expert | 更换时重设容量、shard、placement seed、warmup 和只读 Policy；这些覆盖会使结果过期 |
| Capacity axis (GiB) | 扫描总缓存容量 | 有限非负数列表；小数允许；GiB × 2^30 四舍五入成 bytes，结果须在 JavaScript 安全整数范围内；不能全空 | 1024 GiB | Replay | 0 是无驻留容量对照；新增容量扩展 Case，改变准入和驱逐；需重算 |
| Shard count | 设置缓存分片数 | 整数 ≥ 1 | 1 | Replay + Expert | 改变对象分布和容量竞争；大于 1 时禁止交互式单 SSD 仿真；需重算 |
| Placement seed | 设置对象到 shard 的映射种子 | 整数，可为 0 或负数 | 0 | Replay + Expert | 改变分片映射；保持不变才能比较同一放置方案；需重算 |
| Warmup fraction | 指定普通请求中的预热比例 | 有限数 0 ≤ 值 < 1 | 0 | Replay + Expert | 先用预热请求建立驻留，再统计剩余请求；测量样本减少；需重算 |
| Prefix Cache Replay / Modeling policies / Model exceptions 的展开箭头 | 展开或收起控件 | 展开 / 收起 | Replay 区在启用后展开；Modeling policies、Model exceptions 收起 | 对应区存在时 | 只改可见性，收起不会清空已输入值；不重算 |

<a id="controls-recovery"></a>
## Explore：恢复、状态与高级参数

### Recovery scenario 与提示按钮

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Scenario | 选择有状态模型如何得到命中边界的状态 | Model defaults / Periodic checkpoints / Ideal boundary (upper bound) / No checkpoint state | Model defaults | 有 blocking state 的模型；Simple 下纯 KV 模型隐藏整区，No checkpoint state 只在 Expert 提供；仅有 V4.1 且无其他 blocking state 模型时改用 Encoder 场景 | Model defaults 使用模型策略；Periodic 按间隔持久化；Ideal 假定边界状态已可用；None 不持久化 checkpoint。改变容量、重算和可恢复边界，需重算 |
| Checkpoint interval | 设置全局周期 checkpoint 间隔 | 正整数 tokens，不能留空 | 2048 | Scenario 为 Periodic 时显示；隐藏时保留值 | 间隔越小通常保存更多状态、减少恢复重算；只影响适用的状态组；需重算 |
| Retain prompt-end checkpoint | 周期 checkpoint 之外是否保留 prompt 结束状态 | 开 / 关 | 关 | Periodic + Expert | 末尾未落在周期边界时可增加一个状态快照；增加存储并改变完整前缀恢复；需重算 |
| Keep model defaults | 保留模型默认恢复方式 | 点击 | 仅在恢复提示出现时显示 | 部分前缀不能使用默认 prompt-end 状态时 | 设为 Model defaults，提示可能继续存在；不把该状态自动变成可恢复；需重算 |
| Use periodic 2048 | 快速设置周期恢复 | 点击 | 提示出现后可用 | 同上 | 设为 Periodic checkpoints，间隔改为 2048；其他覆盖仍可能优先；需重算 |
| Use ideal boundary | 快速选择理想边界场景 | 点击 | 提示出现后可用 | 同上 | 设为 Ideal boundary；用于上界比较，不代表实际已有 checkpoint；需重算 |
| Add 100% control point | 给现有前缀轴补一个完整命中对照点 | 点击；已有 1.0 时不重复添加 | 提示出现后可用 | 同上 | 只将 1.0 加入前缀列表并排序，不改变其他点的恢复策略；可能增加 Case，完整命中仍可能 Not evaluated；需重算 |
| Encoder state | 设置 V4.1 执行端 Encoder 状态是否就绪 | Unavailable / Available，复选，可同时选，至少一个 | Unavailable | 所选模型包含 DeepSeek V4.1 Flash | Available 假定 SWA 及所需压缩尾状态已就绪；同时选时只扩展 V4.1 的 Case；影响恢复重算与 IO；需重算 |
| Encoder tail output / Encoder output | 设置 Encoder 尾部输出如何获得 | Recompute when missing / Retain and reuse，复选，至少一个 | Recompute when missing | 所选模型包含 DeepSeek V4.1 Flash | Retain 假定有效输出快照已在执行端；双选扩展适用 Case；改变执行阶段和时间；需重算 |

### State group overrides / Model exceptions

每个状态组和每个模型例外都使用下面三种控件。首次没有覆盖：选择器为 Inherit，隐藏的周期值为 2048，prompt end 复选框为关。实际继承的模型策略和状态组名称列在[模型恢复默认值](#controls-state-defaults)。

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| 状态策略选择器 | 为当前状态组或模型覆盖持久化形式 | Inherit / Prompt end / Periodic / None；支持逐 token 历史的组还提供 Token history | Inherit | 选中模型提供该状态组时；Model exceptions 需展开 | Inherit 使用下一优先级；Prompt end 只保留末尾快照；Periodic 周期保存；None 不保存；Token history 保存逐 token 历史。影响容量及恢复边界；需重算 |
| 状态组 checkpoint interval | 设置当前组或模型的周期长度 | 整数 ≥ 1 tokens | 2048 | 该行选择 Periodic 后显示 | 只覆盖这一行的间隔，不等于全局间隔；间隔改变保存数量和恢复重算；需重算 |
| 状态组 prompt end | 设置该行是否额外保留末尾快照 | 开 / 关 | 关 | 该行选择 Periodic 后显示 | 在不对齐周期的 prompt 末尾增加快照；需重算 |

覆盖优先级为：模型级 Model exceptions → 状态组 State group overrides → 该组支持时使用全局 Scenario → 模型默认。默认 Token history 的组不一定受全局周期场景控制，需要在组或模型级明确转换。当前行的容量预览是只读估算。

### Advanced modeling policies

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Footprint view | 选择容量展示范围 | Calculator reference / Active execution / Prefix Store / Replay resident | Prefix Store | Expert，展开 Advanced modeling policies | 分别查看标称参考、执行状态、前缀存储对象、回放驻留范围；该选择进入 Case，需重算。实际缓存使用量仍看 Replay 的 occupancy 指标 |
| Sparse restore | 选择稀疏模型读取哪些持久化对象 | Full restore / Sparse on demand | Full restore | Expert；只有实现该策略的 attention 消费此选项 | 完整恢复全部相关对象，或按查询场景恢复选中对象；影响读字节和依赖。DSA materialized + Sparse on demand 不受支持；需重算 |
| Sparse locality | 设置稀疏访问在块中的分布 | Clustered lower / Window/global / Seeded uniform / Dispersed upper | Window/global | Expert；稀疏模型和对应恢复策略 | 分别为集中下界、窗口加全局、带种子的均匀采样、分散上界；相同元素数可能产生不同触及块数和 IO；需重算 |
| Reuse scope | 设置查询之间是否复用已读对象 | Request union / Per query, no reuse | Request union | Expert；适用稀疏模型 | Request union 对请求内访问求并集；Per query, no reuse 按查询计读，不做跨查询复用，通常增加 IO；需重算 |
| Storage layout | 设置 KV heads 如何放入存储对象 | All KV heads together / Per GQA group | All KV heads together | Expert；适用稀疏模型 | 前者同块放全部 heads，后者按 GQA group 分开；改变读取粒度与 IO 数，不改变模型原始结构；需重算 |
| Query chunk | 设置稀疏查询的处理分组大小 | 整数 ≥ 1 tokens | 64 | Expert；适用稀疏模型 | 改变查询分组和组内访问合并；可影响读次数与复用；需重算 |
| Sparse seed | 设置稀疏随机访问种子 | 整数，可为 0 或负数 | 0 | Expert；随机 locality 场景使用 | 保持其他输入相同也可能产生不同选中块；复现时固定；需重算 |
| Include draft cache | 把 draft/MTP 缓存纳入容量 | 开 / 关 | 关 | Expert；需模型支持 draft 布局 | 增加支持模型的 draft 状态/缓存；不支持的组合会拒绝 Case；需重算 |
| Draft execution | 控制是否计入 draft prefill 执行 | Disabled / Prefill | Disabled | Expert；需模型支持相应执行公式 | 改变执行工作量及时间，和 Include draft cache 是两个独立假设；需重算 |
| Decode KV/state writes | 计入生成过程的 KV/state/checkpoint 写入 | 开 / 关 | 开 | Expert；只在 Replay 中起作用，具体写入组件由模型决定 | 改变 Replay 的逻辑写流量及后续缓存状态；Envelope 不计 decode 写入；不提供完整 TPOT 或调度预测；需重算 |
| Decode step (µs/token) | 设置生成写入的时间间隔 | 空白，或单个有限实数 > 0 µs/token | 空白：burst at completion | Expert；只在 Replay 中起作用，有相应生成写入时才产生流量 | 空白表示完成时集中写；填值表示均匀分步写，且请求会自动开启 decode writes，即使复选框未勾选；主要改变峰值时序；需重算 |

这些选项不会把普通 GQA/MLA 变成稀疏或有状态模型。某个模型不消费某个选项时，数值可能不变；显式不支持的组合会报错。它们仍保存在 Case 配置中，不能因为当前结果相同就认为输入完全相同。

<a id="controls-results"></a>
## Explore：图表、表格、详情和操作

### 结果页签、运行与导出

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Run Sweep | 提交当前输入 | 点击 | 默认参数有效时可点击 | 至少一个模型、到达模式，输入合法且当前无占用任务 | 最多 1200 Case；≤100 同步，较大使用后台任务。完成后更新结果并尝试保存实验 |
| Cancel Sweep | 请求停止当前后台 Sweep | 点击 | 隐藏 | 当前存在可取消后台任务 | 合作式取消，等待当前 Case 完成；不删除此前已保存的实验 |
| Charts / Cache / Cases | 选择结果区域 | Charts / Cache / Cases | Charts | Cache 只在有回放结果时显示 | 仅切换展示区域，保留选中的 Case；不重算 |
| Export JSON | 下载整轮结果 | 点击 | 无结果时禁用 | 存在结果，且结果未过期、没有当前计算任务 | 下载完整当前 Sweep JSON；表格筛选不裁剪导出 |
| Export CSV | 下载 Case 数值表 | 点击 | 无结果时禁用 | 同上 | 每个 Case 一行；包含状态字段，缺失或非有限值可能为空；表格筛选不裁剪导出 |
| Show in table | 定位选中的 Case | 点击 | 未选 Case 时不显示 | 已有选中的曲线点、热力格或表格行 | 切到 Cases 并定位对应分页；不重算 |
| Open details | 查看当前 Case 详情 | 点击 | 未选 Case 时不显示 | 已选中 Case | 切到 Cases 并滚动至详情；详情读取当前 Case，不改变整轮输入 |

### 指标、曲线与图例

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| 指标页签 | 选择图表的纵轴指标或指标组 | Bandwidth / Read / Write / IOPS / TTFT / Capacity / Recovery / Hits / Compute / Peak pressure | Bandwidth | 已有结果时可绘图；无值时给出状态或原因 | 只改显示；各指标含义和组内选项见后表 |
| Metric | 选择当前指标组中的子指标 | Capacity 5 项；Recovery / Hits 8 项；Compute 4 项；Peak pressure 6 种测量，见后表 | 各组分别为 Retain-all capacity upper bound、Recovery boundary ratio p50、Net compute saving p50、Read peak bandwidth | 选择四个扩展指标组后显示；Replay 专用项在 Envelope 下禁用 | 只改显示，组内选择在切换页签时保留 |
| Window | 设置统计峰值的时间窗 | 1 ms / 10 ms / 100 ms / 1 s | 1 ms | Peak pressure | 读取已有对应窗口的峰值；不重新模拟或改变请求 |
| View | 选择曲线或热力图 | Curves / Heatmap | Curves | Charts | 只改显示；两种视图分别记住横轴；Heatmap 隐藏平滑和曲线图例 |
| X axis | 选择横轴 | Curves：Prefix Ratio / Miss Ratio Log / Cache Capacity / Context Length；Heatmap：Prefix Ratio / Miss Ratio Log | Prefix Ratio；Full Tail 预设改为 Miss Ratio Log | Cache Capacity 只用于 Replay 结果；不适用时回退并提示 | 只改显示。切换横轴会清除隐藏系列集合，使图例重新显示；连接其他输入一致的 Case |
| Smooth guide | 为真实采样点绘制平滑引导线 | 开 / 关 | 关 | Curves | 仅改变线条；不增加 Case，不向 JSON/CSV 写入插值值 |
| Device proxy | 显示设备额定值参考线及相关提示 | 开 / 关 | 关 | 当前指标有对应设备阈值时有效 | 只改图表/提示；不执行 SSD 仿真，不改变计算结果 |
| Show all / Hide all | 批量显示或隐藏曲线系列 | 点击 | 初始全部显示 | Curves，有结果 | 只改显示；全部隐藏时出现提示，不清空结果 |
| 单条系列 / only | 选择显示的曲线系列 | 勾选/取消某条系列；only 只看这一条 | 每条系列显示 | Curves | 只改显示；模型/其他轴组合决定系列，未显示的数据仍会导出 |
| 曲线点 / Heatmap cell | 选择一个已计算 Case | 鼠标点击；键盘聚焦后 Enter/Space | 未选择 | 有对应数据点/单元格 | 选中表格行、同步详情和切片；不修改输入参数 |

### 每个指标选项的含义

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Bandwidth：Optimistic fully-hidden bandwidth p99 | 将阻塞恢复 IO 隐藏在 replay+suffix 计算中的理想带宽要求，取请求 p99 | 选择该指标；结果单位 GB/s | 初始纵轴 | 两模式；没有可用窗口、状态或完整结束过程时显示对应状态；数值大不代表至少需要这么多带宽，见[两个带宽阈值](#modeling-bandwidth) | 仅选择图上已有指标，不重算 |
| Read：Offered read bandwidth | 工作负载平均读需求：总读取字节 ÷（请求数 ÷ QPS） | 选择该指标；结果单位 GB/s | 选择 Read 时显示 | Envelope 只有一个请求、按 1 秒计，数值等于该请求的恢复字节，不代表持续带宽 | 仅选择图上已有指标，不重算 |
| Write：Offered write bandwidth | 持久化 miss 等写入产生的平均写需求 | 选择该指标；结果单位 GB/s | 选择 Write 时显示 | 仅 Replay 有值，受 Persist/decode 写设置影响；Envelope 中为 0 | 仅选择图上已有指标，不重算 |
| IOPS：Aggregate dependency IOPS | 依赖窗口内的聚合分块读操作压力 | 选择该指标；结果单位 IOPS | 选择 IOPS 时显示 | 需 Max IO；Envelope 中等于单个请求自身的要求；缺失/未评估时不补零；计算方法见[聚合依赖带宽和 IOPS](#modeling-aggregate) | 仅选择图上已有指标，不重算 |
| TTFT：Dependency-aware TTFT saving upper bound p50 | 无缓存 TTFT 减去零恢复延迟下的缓存路径时间下界 | 选择该指标；结果单位 ms，导出为 µs | 选择 TTFT 时显示 | 是节省上界，不计恢复读取时间；随附模型中等于净节省 FLOPs ÷ Compute；计入带宽的时间见 Recommend/SSD，计算方法见 [TTFT 的计算方法](#modeling-ttft) | 仅选择图上已有指标，不重算 |
| Retain-all capacity upper bound | 保留所有报告对象的序列化容量上界 | 选择该指标；结果单位 GiB | Capacity 默认 | 不是实际受限缓存驻留 | 仅选择图上已有指标，不重算 |
| Reusable prefix capacity | 可复用前缀对象总容量 | 选择该指标；结果单位 GiB | 非默认 | 两模式 | 仅选择图上已有指标，不重算 |
| Peak cache occupancy | 测量回放中最大驻留字节，包含被保护对象 | 选择该指标；结果单位 GiB；原值 bytes | 非默认 | 仅 Replay | 仅选择图上已有指标，不重算 |
| Final cache occupancy | 回放结束时实际驻留量 | 选择该指标；结果单位 GiB；原值 bytes | 非默认 | 仅 Replay | 仅选择图上已有指标，不重算 |
| Peak protected cache | 测量期间不能驱逐的受保护驻留峰值 | 选择该指标；结果单位 GiB；原值 bytes | 非默认 | 仅 Replay | 仅选择图上已有指标，不重算 |
| Object-shareable prefix ratio | 按对象和对齐后可共享的前缀占比 | 选择该指标；结果单位 % | 非默认 | Envelope 为请求 p50；Replay 为测量请求 token 汇总比例 | 仅选择图上已有指标，不重算 |
| Resident prefix ratio | 驻留前缀占比 | 选择该指标；结果单位 % | 非默认 | Envelope 为假设请求 p50；Replay 为测量 token 汇总比例 | 仅选择图上已有指标，不重算 |
| Recovery boundary ratio p50 | 请求可恢复边界占输入长度的 p50 | 选择该指标；结果单位 % | Recovery / Hits 默认 | V4.1 表示 Encoder 重放起点，不是 global KV 可复用占比 | 仅选择图上已有指标，不重算 |
| Actual block hit rate | 命中块 / 测量请求全部可缓存块 | 选择该指标；结果单位 % | 非默认 | 仅 Replay；分母不只是候选前缀 | 仅选择图上已有指标，不重算 |
| Actual byte hit rate | 命中字节 / 测量请求全部可缓存字节 | 选择该指标；结果单位 % | 非默认 | 仅 Replay | 仅选择图上已有指标，不重算 |
| Replay tokens p50 | 为恢复可用状态重算的 token 数中位数 | 选择该指标；结果单位 tokens | 非默认 | 适用恢复场景 | 仅选择图上已有指标，不重算 |
| Planned restore p50 | 单请求计划逻辑恢复字节的 p50 | 选择该指标；结果单位 GiB；原值 bytes | 非默认 | 不是总容量，也不是实际物理 SSD 流量 | 仅选择图上已有指标，不重算 |
| Stranded resident p50 | 驻留却无法参与当前恢复的字节 p50 | 选择该指标；结果单位 GiB；原值 bytes | 非默认 | 取决于 state/checkpoint 策略 | 仅选择图上已有指标，不重算 |
| Net saved FLOPs p50 | 相对无缓存 prefill 净节省的运算量 | 选择该指标；结果单位 TFLOPs | 非默认 | 负值表示额外计算 | 仅选择图上已有指标，不重算 |
| Net compute saving p50 | 每请求净节省 FLOPs / 无缓存 FLOPs，再取 p50 | 选择该指标；结果单位 % | Compute 默认 | 负值表示额外计算；非完整结果可能未评估 | 仅选择图上已有指标，不重算 |
| Replay compute time p50 | 恢复重算的计算时间 p50 | 选择该指标；结果单位 ms；原值 µs | 非默认 | 不含存储排队 | 仅选择图上已有指标，不重算 |
| Suffix compute time p50 | 新 suffix 的计算及必需转换/算法收尾时间 p50 | 选择该指标；结果单位 ms；原值 µs | 非默认 | 不含 state 恢复和 output head；空 suffix 为 0 | 仅选择图上已有指标，不重算 |

### Peak pressure 的六项测量

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Read peak bandwidth | 选定时间窗内最大 offered 读带宽 | GB/s | Peak pressure 默认 | Window 为 1/10/100 ms 或 1 s | 只切换已有指标 |
| Write peak bandwidth | 选定窗口内最大 offered 写带宽 | GB/s | 非默认 | 同上 | 只切换已有指标 |
| Read peak IOPS | 选定窗口内最大分块读 IOPS | IOPS | 非默认 | 需要 Max IO | 只切换已有指标 |
| Write peak IOPS | 选定窗口内最大分块写 IOPS | IOPS | 非默认 | 需要 Max IO | 只切换已有指标 |
| Read bandwidth peak / average | 选定窗口读峰值 / 全 Case 平均读带宽 | 倍数 × | 非默认 | 平均读为 0 时无定义 | 只切换已有指标 |
| Write bandwidth peak / average | 选定窗口写峰值 / 全 Case 平均写带宽 | 倍数 × | 非默认 | 平均写为 0 时无定义 | 只切换已有指标 |

峰值按事件时刻统计：恢复读取记在请求到达时刻，miss 写入记在计算完成时刻，decode 写入按所选步长或完成时刻记录。Envelope 每个 Case 只有一个请求；恢复读字节大于 0 时，四个窗口的读峰均比为 1000、100、10、1；零读字节时，读峰值为 0、读峰均比为 `null`。这些固定比例来自 1 秒统计基准，不能据此判断负载并发或设备持续吞吐。详见 [Peak pressure 的计算方法](#modeling-peak)。

### Heatmap slice

热力图筛选顺序是 Model → Encoder state → Encoder output → Block → Arrival → Max IO → Cache GiB。每一项的候选值由前面选中的结果限定。无结果时列表为空；已有结果后默认保留有效旧值，否则选择候选值按数字/字符串自然排序的第一项。选择 Case 后可同步为该 Case 所在切片。

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Model | 选择热力图模型 | 当前结果中的模型 | 无结果时空；有结果取第一个有效值 | Heatmap | 只筛选结果；会更新后续切片选项 |
| Encoder state | 选择 V4.1 状态场景切片 | 当前模型结果中的 unavailable / available | 按当前结果取第一个有效值 | Heatmap，且当前候选有 Encoder 状态 | 只筛选，不增加 Case；切换到其他模型时移除不适用场景 |
| Encoder output | 选择 V4.1 输出场景切片 | 当前候选结果中的 recompute / retain | 按当前结果取第一个有效值 | Heatmap，且场景适用 | 只筛选 |
| Block | 固定当前切片的块大小 | 已有结果中的 block_size_tokens | 按当前结果取第一个有效值 | Heatmap | 只筛选；横纵轴仍是前缀比例和长度 |
| Arrival | 固定到达模式 | 当前候选的 fixed / poisson | 按当前结果取第一个有效值 | Heatmap | 只筛选 |
| Max IO | 固定 IO 分块配置 | 当前候选的 byte 值或 None | 按当前结果取第一个有效值 | Heatmap | 只筛选 |
| Cache GiB | 固定缓存容量 | 当前候选结果中的容量 | 按当前结果取第一个有效值 | Heatmap + Replay 容量结果 | 只筛选，不修改容量输入轴 |
| Color scale | 选择颜色范围 | Global / Local | Global | Heatmap | Global 用全部结果色域，Local 用当前切片；只改颜色，不改数值 |

### Cases 与详情

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Filter rows | 搜索 Case 表格 | 任意搜索文字；空白显示全部 | 空白 | Cases | 只筛选表格，回到第 1 页；导出仍包含全部 Case |
| 表头排序按钮 | 按列排序 | 点击某列升序，再次点击降序 | Case 升序 | Cases 表格 | 新列默认升序，并回到第 1 页；数值和导出内容不变 |
| Previous / 页码 / Next | 切换表格分页 | 有效页码；每页 100 行 | 第 1 页 | 有结果；首尾按钮按边界禁用 | 只改变当前显示行，不重新计算 |
| Case 行 | 选择详情对象 | 点击；方向键移动焦点，Enter/Space 选择 | 未选择 | Cases 有结果 | 读取该 Case 的详情，联动曲线、热力切片和选中栏 |
| Overview | 查看 Case 概况 | 点击 | 详情默认页 | 选中 Case | 只切换详情 |
| Recovery | 查看前缀和状态恢复过程 | 点击 | 非默认 | 选中 Case | 只切换详情 |
| Components | 查看组件字节、状态和容量范围 | 点击 | 非默认 | 选中 Case | 只切换详情 |
| Timing | 查看无缓存 TTFT、读取时间为 0 时的缓存路径 TTFT、各阶段计算时间、节省上界和峰值；SSD 仿真后显示模拟 TTFT | 点击 | 非默认；SSD 完成后自动切到此页 | 选中 Case | 只切换详情 |
| Cache | 查看 radix、准入和驱逐详情 | 点击 | 无回放详情时隐藏 | Case 有 cache_replay 报告 | 只切换详情 |
| Raw JSON | 查看完整 Case JSON | 点击 | 非默认 | 选中 Case | 只切换详情；JSON 默认折叠 |
| Copy JSON / Download JSON / 展开 Raw JSON | 复制、下载或展开当前详情 | 点击 | Raw JSON 收起 | 选中 Case 并打开 Raw JSON | 作用于当前 Case 的详情，不是整轮 Sweep；复制需要浏览器剪贴板支持 |
| Run SSD Simulation | 为当前 Case 执行设备仿真 | 点击 | 无选中 Case 时禁用 | Replay、单 shard、结果未过期、没有运行任务；服务器还检查 IO 上限和计算状态 | 产生当前 Case 的 storage_report，成功后切到 Timing；不把它当作整轮所有 Case 的仿真 |

Filter rows 不区分大小写，在 Case ID、模型、attention、长度、前缀比例、驻留比例、Cache GiB、块大小、到达模式、Max IO 和 Encoder 场景中匹配文字；不会搜索 Raw JSON 的全部字段。

可排序列：Case、Model、seq_len、Prefix（Replay 中为 Configured）、Block、Arrival、Max IO、Optimistic BW、Break-even、Read、Write、Dependency IOPS、TTFT upper bound、Status / Risk。Replay 另外显示 Shareable、Resident、Recovery、Cache GiB、Evicted GiB；V4.1 结果还会增加适用的 Encoder 场景列。表头排序只调整显示顺序。

<a id="controls-recommend"></a>
## Recommend：逐项设置

Recommend 的输入与 Explore 分开保存。两页含义相同的选项仍分别列出默认值；修改推荐条件后要重新运行推荐。唯一例外是 Ranking objective，它会直接重排已有结果。

### Design space 与模型筛选

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Analysis mode | 选择推荐要评估的模型 | Bandwidth Envelope / Cache Replay | Bandwidth Envelope | Recommend | 切换约束名称、可见项和评估方式；Replay 显示单个缓存容量输入；需重新推荐 |
| Models | 选择允许搜索的模型 | 56 个 active 模型，见目录；多选、至少一个 | 与 Quick Compare 相同的 5 个 | 有相应模型配置 | 增加模型会扩展候选设计；筛选后仍保留隐藏选择；需重新推荐 |
| Search models | 搜索候选模型 | 文字，空白不限 | 空白 | Recommend Models | 只过滤可见列表，不重新计算或取消勾选 |
| All families / Family | 按家族筛选 | All families 或目录中的 13 个 family | All families | Recommend Models | 只过滤列表 |
| All formulas / Formula | 按公式筛选 | All formulas 或 10 类公式 | All formulas | Recommend Models | 只过滤列表 |
| All evidence / Evidence | 按证据标签筛选 | All evidence 或 Explore 列出的 7 类标签 | All evidence | Recommend Models | 只过滤列表；标签不是运行参数 |
| Select visible | 批量修改可见模型勾选 | 点击；可见项全选时再次点击会取消 | 可点击 | Recommend Models | 只改当前可见可选项，可能改变候选数；需重新推荐 |
| Runtimes | 选择候选运行配置 | bf16_reference / fp4_2pf / v41_fp4 / v4_fp8_mxfp4；多选、至少一个 | 仅 bf16_reference | Recommend Design space | 各 Runtime 提供不同精度和执行默认值；不兼容的模型/Runtime 组合被排除，候选数改变；需重新推荐 |
| Storage | 选择候选存储配置 | 当前为 single_nvme_ftl；多选控件，至少一个 | single_nvme_ftl | Recommend Design space | 改变容量、带宽、IOPS 约束及额定时间估计；同一 Demand Case 不因多 Storage 重复计算 |
| block sizes | 设置候选块大小 | 正整数 tokens 列表 | 256 | Recommend | 不兼容对齐被过滤/拒绝；新增数值扩展设计，需重新推荐 |
| max IO bytes | 设置候选 IO 分块 | 正整数 byte 列表，或空白/None/null | 131072 | Recommend | 影响 IO 计数及 IOPS 可评估性；新增数值扩展设计，需重新推荐 |

### Workload envelope

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Chat / RAG / Agent / Custom | 选择工作负载的起始参数 | 四张卡片，单选；方向键可切换焦点/选择 | Chat | Recommend | 预设重填下表中的长度、比例、QPS、请求生成和到达方式；Custom 清空这些字段并取消到达方式，需填写后才能运行；不创建配置文件 |
| Prompt length (tokens) | 设置需要覆盖的上下文长度 | 正整数 tokens 列表 | 8192 | Recommend | 每个值成为覆盖场景；需重新推荐 |
| Prefix ratio | 设置需要覆盖的前缀比例 | 小数列表，页面要求 0 < 比例 ≤ 1；不接受百分号 | 0.7 | Recommend | 当前页面的正数检查会拒绝 0；零命中对照可用 Explore。各值增加覆盖场景；需重新推荐 |
| QPS | 设置需要覆盖的请求到达率 | 正有限数列表，requests/s | 100 | Recommend | 与长度、比例、到达方式做组合；需重新推荐 |
| Request count | 每个场景的请求数量 | 整数 ≥ 0，不能留空 | 1000 | Recommend | Replay 生成对应数量请求；影响计算量与统计；需重新推荐 |
| Generation length | 每请求生成 token 数 | 整数 ≥ 0 | 256 | Recommend | 影响生成对象和 decode 写场景；需重新推荐 |
| Prefix groups | 设置共享前缀组数 | 整数 ≥ 1 | 100 | Recommend | 改变共享分布和 lineage bootstrap；需重新推荐 |
| Zipf alpha | 设置访问集中程度 | 有限数 ≥ 0 | 1.3 | Recommend | 改变前缀热度分布；需重新推荐 |
| Seed | 设置随机负载种子 | 整数，可为 0 或负数 | 1 | Recommend | 改变负载样本；需重新推荐 |
| Workload shape | 选择共享/继承结构 | Shared prefix groups / Prompt lineage bootstrap | Shared prefix groups | Recommend | 改变请求身份与 bootstrap；选择工作负载卡片不会自动重设此项；需重新推荐 |
| Cache capacity GiB | 设置推荐回放使用的总缓存容量 | 单个有限非负数；GiB，可为小数 | 1024 | 仅 Cache Replay 显示 | 不是容量扫描轴；当前页面使用 shard=1、placement seed=0、warmup=0。改变容量后重新推荐 |
| fixed / poisson | 选择覆盖的到达模式 | 分别勾选，至少一个，可同时选 | 仅 fixed | Recommend | 同时选择会增加场景；Custom 初始两者都不选；需重新推荐 |
| write misses | 控制 miss 新对象写入 | 开 / 关 | 开 | Recommend | 对应 Explore 的 Persist blocks created by misses；需重新推荐 |

工作负载预设只覆盖下表字段和到达模式，不重设 Workload shape、write misses、缓存容量、约束或高级建模参数。每个预设的到达方式均为 fixed、Seed 均为 1。

| 预设 | Prompt tokens | Prefix ratio | QPS | Request count | Generation tokens | Prefix groups | Zipf alpha |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_agent | 32768 | 0.8 | 50.0 | 1000 | 1024 | 200 | 1.1 |
| synthetic_chat | 8192 | 0.7 | 100.0 | 1000 | 256 | 100 | 1.3 |
| synthetic_rag | 16384 | 0.5 | 100.0 | 1000 | 512 | 1000 | 0.9 |

### Modeling policies

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Footprint view | 容量范围 | Calculator reference / Active execution / Prefix Store / Replay resident | Prefix Store | 展开 Modeling policies；只有对应模型能力消费该项 | 含义和影响与 Explore 同名选项一致，修改后需重新推荐；具体见[恢复与高级参数](#controls-recovery) |
| Recovery assumption | 全局恢复假设 | Model default / Exact at resident / Periodic checkpoint / No checkpoint state | Model default | 展开 Modeling policies；只有对应模型能力消费该项 | 含义和影响与 Explore 同名选项一致，修改后需重新推荐；具体见[恢复与高级参数](#controls-recovery) |
| Checkpoint interval | 周期长度 | 正整数 tokens | 2048 | 展开 Modeling policies；只有对应模型能力消费该项 | 含义和影响与 Explore 同名选项一致，修改后需重新推荐；具体见[恢复与高级参数](#controls-recovery) |
| retain prompt-end checkpoint | 是否额外保留末尾状态 | 开 / 关 | 关 | 展开 Modeling policies；只有对应模型能力消费该项 | 含义和影响与 Explore 同名选项一致，修改后需重新推荐；具体见[恢复与高级参数](#controls-recovery) |
| Sparse restore | 完整/按需恢复 | Full restore / Sparse on demand | Full restore | 展开 Modeling policies；只有对应模型能力消费该项 | 含义和影响与 Explore 同名选项一致，修改后需重新推荐；具体见[恢复与高级参数](#controls-recovery) |
| Sparse locality | 稀疏访问分布 | Clustered lower / Window/global / Seeded uniform / Dispersed upper | Window/global | 展开 Modeling policies；只有对应模型能力消费该项 | 含义和影响与 Explore 同名选项一致，修改后需重新推荐；具体见[恢复与高级参数](#controls-recovery) |
| Reuse scope | 查询间复用 | Request union / Per query, no reuse | Request union | 展开 Modeling policies；只有对应模型能力消费该项 | 含义和影响与 Explore 同名选项一致，修改后需重新推荐；具体见[恢复与高级参数](#controls-recovery) |
| Storage layout | KV 头的存储分组 | All KV heads together / Per GQA group | All KV heads together | 展开 Modeling policies；只有对应模型能力消费该项 | 含义和影响与 Explore 同名选项一致，修改后需重新推荐；具体见[恢复与高级参数](#controls-recovery) |
| Query chunk | 查询分组粒度 | 整数 ≥ 1 tokens | 64 | 展开 Modeling policies；只有对应模型能力消费该项 | 含义和影响与 Explore 同名选项一致，修改后需重新推荐；具体见[恢复与高级参数](#controls-recovery) |
| Sparse seed | 稀疏随机种子 | 整数 | 0 | 展开 Modeling policies；只有对应模型能力消费该项 | 含义和影响与 Explore 同名选项一致，修改后需重新推荐；具体见[恢复与高级参数](#controls-recovery) |
| Draft execution | 是否计入 draft prefill | Disabled / Prefill | Disabled | 展开 Modeling policies；只有对应模型能力消费该项 | 含义和影响与 Explore 同名选项一致，修改后需重新推荐；具体见[恢复与高级参数](#controls-recovery) |
| Decode step (µs/token) | decode 写入时间分布 | 空白或正有限数 µs/token；0/负数最终会被拒绝 | 空白，completion burst | 展开 Modeling policies；只有对应模型能力消费该项 | 含义和影响与 Explore 同名选项一致，修改后需重新推荐；具体见[恢复与高级参数](#controls-recovery) |
| include draft cache | 是否纳入 draft 缓存 | 开 / 关 | 关 | 展开 Modeling policies；只有对应模型能力消费该项 | 含义和影响与 Explore 同名选项一致，修改后需重新推荐；具体见[恢复与高级参数](#controls-recovery) |
| decode KV/state writes | 是否纳入生成写入 | 开 / 关 | 开 | 展开 Modeling policies；只有对应模型能力消费该项 | 含义和影响与 Explore 同名选项一致，修改后需重新推荐；具体见[恢复与高级参数](#controls-recovery) |

Recommend 的 Checkpoint interval 和 retain prompt-end 即使在非周期场景下仍可见，只有适用的周期策略使用它们。Decode step 填值会在请求中开启 decode writes。状态组、Model exceptions 和 V4.1 的两组场景同 Explore，初始分别为 Inherit、Unavailable、Recompute when missing；需要重新推荐。

### Constraints & objective

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Use Storage limits | 是否采用每个 Storage 的额定限制 | 开 / 关 | 开 | Recommend | 关掉后未勾 Custom 的资源约束停用；TTFT Enabled 单独控制；至少一个有效约束，修改需重新推荐 |
| Optimistic GB/s ≤（Envelope）/ Read GB/s ≤（Replay）及 Custom | 限制理想隐藏带宽或平均读需求 | Custom 开/关；开启后填单个有限非负 GB/s | Custom 关；有效默认 7 GB/s；首次开 Custom 填入 7 | 两种模式，名称随模式变化 | Envelope 约束 optimistic_overlap_GBps_max，Replay 约束 read_avg_GBps；Custom 对所有候选统一覆盖，需重新推荐 |
| Break-even GB/s ≤（Envelope）/ Aggregate dependency GB/s ≤（Replay）及 Custom | 限制依赖恢复带宽 | Custom 开/关；开启后填有限非负 GB/s | Custom 关；有效默认 7 GB/s；首次开 Custom 填入 7 | 两种模式，名称随模式变化 | Envelope 限制 dependency-aware break-even 最大值，Replay 限制聚合依赖压力；需重新推荐 |
| Write GB/s ≤ 及 Custom | 限制平均写需求 | Custom 开/关；开启后填有限非负 GB/s | Custom 关；有效默认 5 GB/s | 仅 Replay 显示并使用 | 改变候选是否满足写带宽预算；需重新推荐 |
| Dependency IOPS ≤ 及 Custom | 限制聚合依赖读 IOPS | Custom 开/关；开启后填有限非负数，单位 IOPS | Custom 关；有效默认 1000000 IOPS | 仅 Replay；要求明确 Max IO | 没有可计算 IOPS 的候选不能按该约束判为可行；需重新推荐 |
| Capacity GiB ≤ 及 Custom | 限制 retain-all 容量上界 | Custom 开/关；开启后填有限非负 GiB | Custom 关；有效默认 1024 GiB | 仅 Replay 显示并使用 | 约束 retain-all 上界，不是把实际缓存容量输入改成该值；需重新推荐 |
| Dependency TTFT saving：Enabled | 是否启用时间节省下限 | 开 / 关 | 开 | 两种模式 | 关掉后数值框禁用且该约束停用；不改变其他资源约束；需重新推荐 |
| Dependency TTFT saving ≥ | 设置额定存储带宽下 p50 节省时间下限 | 单个有限数，单位 ms，可为负；启用时不能空 | 0 ms | Enabled 开启 | 0 要求不慢于无缓存；正数要求至少节省该时间，负数允许一定变慢；提交时换算 µs，需重新推荐 |
| Ranking objective | 选择排序优先级 | 下方四个目标，单选 | Maximize worst-case normalized resource headroom | Recommend | 已有结果时直接重排，不重新运行 Demand；未计算结果不会因此获得有效排名 |

Custom 数值在首次勾选时用所选 Storage 的该项最小值填入；取消后隐藏并停用该覆盖，重新勾选保留已输入值。Storage 配置变化后，非 Custom 的有效限制随之变化。这里列出的 7/5/1000000/1024 来自随附 single_nvme_ftl。

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Maximize worst-case normalized resource headroom | 优先完整覆盖，再比较最差资源余量和时间余量 | 单选 | 默认 | 有可评估候选 | 重排已有候选 |
| Maximize workload-envelope coverage | 优先覆盖率，再看资源余量和平均压力 | 单选 | 非默认 | 有可评估候选 | 重排已有候选 |
| Minimize average storage pressure | 优先完整覆盖和覆盖率，再比较平均压力 | 单选 | 非默认 | 有可评估候选 | 重排已有候选 |
| Maximize worst-case dependency-aware TTFT margin | 优先完整覆盖和覆盖率，再比较最差时间余量 | 单选 | 非默认 | 有可评估候选 | 重排已有候选 |

### 运行与候选结果

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Recommend a design | 执行设计搜索 | 点击 | 初始合法参数可运行 | 输入完整、至少一个有效约束，且未有占用任务 | 创建后台推荐任务，完成后自动尝试保存实验 |
| Cancel | 请求取消当前推荐 | 点击 | 没有任务时隐藏 | 正在运行推荐 | 等待当前 Case 完成后停止；保留之前的历史结果 |
| Top candidates：Select | 查看一个候选的详细判断 | 结果中的候选按钮 | 完成后选中推荐候选；无推荐则按结果展示 | 有推荐结果 | 更新候选详情、覆盖率、资源/时间余量和采样可行区间；不重算 |
| Open in Explore | 将所选候选带到 Explore | 点击 | 无结果时不可用 | 有可打开的所选候选，结果未过期 | 携带实际 Case、分析模式、恢复假设和推荐约束注解；不会丢弃 Replay 模式 |
| Modeling policies / 结果说明折叠区 | 展开更多设置或结果 | 展开 / 收起 | Modeling policies 收起 | 对应区域存在 | 仅显示变化，输入和结果保留 |

<a id="controls-experiments"></a>
## Experiments：筛选、保存与比较

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Refresh | 重新读取实验列表 | 点击 | 可点击 | Experiments | 更新当前过滤后的列表，不重新计算实验 |
| Kind | 按实验类型筛选 | All / Sweep / Recommendation | All | Experiments | 从数据库筛选类型；不删除被隐藏记录 |
| Search | 按实验名称查找 | 文字；空白不限 | 空白 | Experiments | 更新匹配记录；筛选后隐藏项退出当前批量选择集合 |
| Pinned only | 只显示固定实验 | 开 / 关 | 关 | Experiments | 只过滤列表；不改变 Pin 状态 |
| 实验行复选框 | 选择批量操作对象 | 勾选 / 取消 | 均未选 | 列表中存在记录 | 只改变待操作集合，不修改实验数据 |
| Select all | 选择当前列表全部记录 | 勾选 / 取消；部分选中显示中间状态 | 未选；空列表禁用 | 只对当前筛选结果生效 | 不会选择过滤掉的实验 |
| Delete selected | 删除当前已选实验 | 点击并确认 | 未选择时禁用 | 当前筛选列表中有选中记录 | 永久删除数据库记录；不会删除模型配置；取消确认则不删除 |
| Open | 打开实验输入和保存结果 | 点击 | 每条记录可用 | 记录存在 | Sweep 打开到 Explore；Recommendation 恢复推荐并打开所选候选到 Explore；不自动重算 |
| Pin / Unpin | 固定或取消固定记录 | 点击切换 | 新实验未固定 | 记录存在 | 更新数据库；固定记录不计入最近 50 个未固定实验限额 |
| Rename / Experiment name | 修改实验显示名称 | 弹窗文本，去除首尾空白后非空，最多 120 字符 | 弹窗填当前名字 | 记录存在 | 确认后保存；取消不变；不改输入与计算结果 |
| Delete | 删除单条实验 | 点击并确认 | 记录存在时可用 | Experiments | 永久删除该记录；取消确认则保留 |
| Baseline | 选择对比基准 | 当前过滤列表中的实验 | 空库无选项；有至少两条时优先第二条，否则第一条 | 列表中有实验 | 仅选择比较对象；列表刷新后优先保留原有效选择 |
| Candidate | 选择对比目标 | 当前过滤列表中的实验 | 空库无选项；有记录时优先第一条 | 列表中有实验 | 只选择比较对象 |
| Compare | 比较两条已保存实验 | 点击 | 空库没有可比较记录 | Baseline 和 Candidate 都有值 | 读取已有结果，计算完全匹配 Case 的差值；不同 Case 标 Added/Removed，不重算 |

<a id="controls-kv"></a>
## KV Size：基础选项和比较

首次进入选择 DeepSeek V4 Pro，Tokens per sequence 为 1024，Sequences 为 1，KV precision 为 FP8 / INT8，Indexer precision 为 FP4 / INT4。切换 Model family、Model 或 Restore model defaults 会重新填入该模型的全部默认输入，包括长度、精度、state/draft、checkpoint 和高级字段。

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Model family | 选择模型家族 | 目录中的 13 个 family，单选 | DeepSeek | KV Size | 切换到该 family 的第一个模型并重置输入；立即重算容量 |
| Model | 选择容量计算公式和模型参数 | 当前 family 的 active 模型，单选 | DeepSeek V4 Pro | KV Size | 重置为所选模型默认值；不同模型显示不同高级字段；立即重算 |
| Tokens per sequence | 设置每条 sequence 的 token 数 | 单个十进制整数 ≥ 0；不接受逗号、百分号、小数或空白 | 1024；所有随附模型相同 | KV Size | 立即重算；超过模型配置 context 上限会提示，但仍按标称公式计算 |
| Sequences | 设置同时持有的独立 sequence 数 | 单个十进制整数 ≥ 1 | 1 | KV Size | 各 sequence 的缓存/状态累加；立即重算 |
| KV precision | 设置主 KV 元素精度 | 通常为 BF16 / FP16（2 B）、FP8 / INT8（1 B）、FP4 / INT4（0.5 B）；V4.1 只有 BF16 和 Released 布局 | 普通模型 BF16；V4 Flash/Pro 为 FP8；V4.1 为 Released · FP4 global + FP8 SWA + scales | KV Size | 改变主 KV 字节；普通模型不自动改变独立 Indexer 或 state 字节精度。V4.1 同步锁定的 Indexer；立即重算 |
| Indexer precision | 设置 Indexer 的元素精度 | BF16 / FP16、FP8 / INT8、FP4 / INT4；固定模型不能改 | DSA 为 BF16；V4 为 FP4；MiniMax M3 固定 BF16；V4.1 跟随主精度 | DSA、MiniMax M3、V4、V4.1 显示；MiniMax M3 和 V4.1 锁定 | 独立影响 Indexer 字节；没有有效 Indexer 层时不影响总量；立即重算 |
| Include draft / MTP cache | 是否将 draft/MTP 层的缓存纳入容量 | 开 / 关 | 关 | 仅模型支持且未禁止 draft 时显示；具体模型见默认值表 | 开启后使用 draft/MTP 参数及 V4 压缩数组中的尾部 draft 层；立即重算 |
| Include attention state | 是否计入 attention 状态 | 开 / 关 | Kimi K3、Inkling、Qwen3.5/3.6、V4.1 为开；其他模型不显示 | 对应 stateful 公式显示 | 关掉后 state 参数不参与有效计算，checkpoint 控件禁用；输入保留。V4.1 的 state/tail 输出按开关处理；立即重算 |
| Encoder tail output | 设置 V4.1 是否保留 Encoder 输出快照 | Recompute when missing / Retain and reuse | Recompute when missing | 仅 V4.1；Include attention state 开启时参与计算 | Retain 增加对应输出快照容量；本页只计算容量，不预测重算时延 |
| State checkpoint policy | 选择容量中保存多少份 state | Prompt-end snapshot / Fixed interval | Prompt-end snapshot | Kimi K3、Inkling、Qwen3.5/3.6；Include attention state 开启 | Prompt-end 对非空序列计 1 份；Fixed interval 按 ceil(tokens / interval) 计份数；立即重算 |
| Checkpoint interval | 设置固定 checkpoint 间隔 | 单个十进制整数 ≥ 1 tokens | 2048 | 支持 checkpoint 且 Include attention state 开启、policy 为 Fixed interval；其他情况禁用 | 间隔越小快照份数越多。最后不足一个间隔仍计一份；这里不同于 Explore 的周期加可选 prompt-end 策略 |
| Advanced parameters | 展开模型结构输入 | 展开 / 收起 | 收起 | KV Size；V4.1 结构只读 | 只改变显示；编辑内部字段会立即重算，详见 45 个字段说明 |
| Restore model defaults | 恢复当前模型的全部输入 | 点击 | 可点击 | KV Size | 重置 Tokens=1024、Sequences=1、精度和所有高级值，关闭 draft，恢复 state 和 checkpoint 默认；已加入比较的快照不变 |
| Add to compare | 保存当前有效容量配置用于比较 | 点击 | 无输入错误且未满 4 项时可用 | KV Size | 最多 4 项；按有效参数去重，保存当时的值，后续编辑不会修改已保存快照 |
| Compare | 查看配置对照 | 点击 | 少于 2 项时禁用 | 已有 2–4 个快照 | 显示容量堆叠图和参数表，不重新运行 Explore |
| Clear | 清空比较列表 | 点击 | 没有快照时列表区域隐藏 | 有比较快照 | 移除全部比较快照，不改当前计算器输入 |
| 比较卡片 × | 移除一个比较配置 | 点击 | 有快照后出现 | 比较列表 | 只移除该快照；比较中少于两项时返回计算器 |
| Back to calculator | 返回容量输入界面 | 点击 | 比较视图显示时可用 | 正在比较 | 保留比较列表和计算器输入 |
| Model parameters / Calculation-layer breakdown / Model and advanced parameters | 展开只读参数或逐层结果 | 展开 / 收起 | Model parameters 展开；逐层结果和比较高级参数收起 | 有容量结果或比较结果 | 只改变显示；表格中的来源、总量和逐层明细不能直接编辑 |

本页按整数和 BigInt 计算字节。若 0.5 B 精度与维度组合产生半个字节，会提示调整参数，不自动取整。GQA、MLA、DSA 等公式的逐层展开有 4096 层上限，合并 draft 后也可能触及限制；V4 同样检查主干层数。Tokens 为 0 时，V4 的固定窗口项仍可能占用容量。

<a id="controls-kv-fields"></a>
## KV Size：45 个高级字段

每个字段的精确默认值见[按模型列出的默认参数](#controls-model-defaults)。字段只在该模型提供对应结构时显示；“—”表示不显示。以下“生效”指该字段确实被所选计算分支使用；所有修改只影响当前 KV Size 配置，立即重算，不写回文件。

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 主干 attention 层数 | 整数 ≥ 1；展开上限 4096 | 见按模型默认参数表 | GQA、MLA、DSA、MiniMax M3、V4 | 增加参与累计的主干层；V4 还决定压缩数组中哪些属于主干；立即重算容量 |
| Num Key Value Heads (`num_key_value_heads`) | 通用 KV 头数 | 实际使用时整数 ≥ 1 | 见按模型默认参数表 | GQA、混合 GQA、Inkling、Qwen 线性混合、MiniMax M3 | 按头数增加普通 KV；混合模型有专用 global/SWA 头数时，此值可能只是后备；立即重算容量 |
| Head Dim (`head_dim`) | 通用每头维度 | 实际使用时整数 ≥ 1 | 见按模型默认参数表 | GQA、混合 GQA、Inkling、Qwen 线性混合、MiniMax M3、V4 | 普通同维 Key/Value 各使用一次；混合模型专用维度优先；V4 影响窗口与压缩条目；立即重算容量 |
| Num Nextn Predict Layers (`num_nextn_predict_layers`) | draft 预测层数 | 整数 ≥ 0 | 见按模型默认参数表 | MLA、DSA 等提供该字段且 Include draft 开启 | 增加 draft KV 层；DSA 在未提供单独 draft indexer 数时还作为后备；立即重算容量 |
| Num Mtp Modules (`num_mtp_modules`) | MTP 模块数量 | 整数 ≥ 0 | 见按模型默认参数表 | MiniMax M2 系列等 GQA 公式；Include draft 开启 | 与 Mtp Transformer Layers 相乘得到额外层数；0 时不使用后者；立即重算容量 |
| Mtp Transformer Layers (`mtp_transformer_layers`) | 每个 MTP 模块的 Transformer 层数 | 整数 ≥ 0 | 见按模型默认参数表 | num_mtp_modules > 0 且 Include draft 开启 | 额外层数为模块数 × 本值，增加 draft KV；立即重算容量 |
| Kv Lora Rank (`kv_lora_rank`) | MLA 持久化 latent KV 宽度 | 实际使用时整数 ≥ 1 | 见按模型默认参数表 | MLA、DSA、Kimi K3 的 MLA 部分 | 与 Qk Rope Head Dim 相加后按 token 和层数累计 latent KV；立即重算容量 |
| Qk Rope Head Dim (`qk_rope_head_dim`) | MLA 单独保存的 RoPE Key 宽度 | 实际使用时整数 ≥ 1 | 见按模型默认参数表 | MLA、DSA、Kimi K3 | 增加每 token 的独立 RoPE Key 存储；立即重算容量 |
| Indexer Full Layers (`indexer_full_layers`) | 主干 Indexer 的层数 | 整数 ≥ 0 | 见按模型默认参数表 | DSA 公式 | 只改变 Indexer，不改变 MLA 主缓存；0 且没有 draft indexer 时 Indexer 消失；立即重算容量 |
| Draft Indexer Layers (`draft_indexer_layers`) | draft Indexer 层数 | 整数 ≥ 0 | 见按模型默认参数表 | DSA 且 Include draft 开启 | 追加 Indexer 容量，不直接改变主 KV 层数；立即重算容量 |
| Index Head Dim (`index_head_dim`) | 每个 token/压缩条目的 Indexer 宽度 | 有有效 Indexer 时整数 ≥ 1 | 见按模型默认参数表 | DSA、MiniMax M3、V4；V4 只有压缩比为 4 的层使用 | 增加 Indexer 字节；无 Indexer 分支时编辑不影响总量；立即重算容量 |
| Full Attention Layers (`full_attention_layers`) | 全上下文 attention 层数 | 一般整数 ≥ 0；Kimi MLA 部分要求 ≥ 1 | 见按模型默认参数表 | Kimi K3、Inkling、Qwen 线性混合、混合 GQA | 增加随完整 tokens 增长的 KV/MLA 层数；立即重算容量 |
| Sliding Attention Layers (`sliding_attention_layers`) | 滑动窗口 attention 层数 | 整数 ≥ 0 | 见按模型默认参数表 | Inkling、混合 GQA | 增加按 min(tokens, window) 计算的窗口缓存层；Inkling state 开启时也加 SConv 状态；立即重算容量 |
| Num Global Key Value Heads (`num_global_key_value_heads`) | 全局 attention 专用 KV 头数 | 整数 ≥ 0 | 见按模型默认参数表 | 提供该字段的混合 GQA | 优先于通用 Num Key Value Heads，改变 full-attention KV；立即重算容量 |
| Global Head Dim (`global_head_dim`) | 全局 attention 的 Key 维度 | 整数 ≥ 0 | 见按模型默认参数表 | 提供该字段的混合 GQA | 优先于通用 Head Dim；无单独 Value 维度时也用作 Value 后备；立即重算容量 |
| V Head Dim (`v_head_dim`) | 通用 Value 维度 | 整数 ≥ 0 | 见按模型默认参数表 | 提供该字段的混合 GQA | 用于没有更高优先级 Value 维度的 full/sliding 分支；立即重算容量 |
| Swa Num Key Value Heads (`swa_num_key_value_heads`) | 滑动窗口专用 KV 头数 | 混合 GQA 允许非负整数；Inkling 实际使用要求 ≥ 1 | 见按模型默认参数表 | Inkling、提供该字段的混合 GQA | 优先于通用头数，改变 sliding KV；Inkling 状态中相关卷积尺寸也随之变化；立即重算容量 |
| Swa Head Dim (`swa_head_dim`) | 滑动窗口专用 Key 维度 | 混合 GQA 允许非负整数；Inkling 实际使用要求 ≥ 1 | 见按模型默认参数表 | Inkling、提供该字段的混合 GQA | 改变 sliding Key；无专用 Value 维度时也是 Value 后备；立即重算容量 |
| Swa V Head Dim (`swa_v_head_dim`) | 滑动窗口专用 Value 维度 | 整数 ≥ 0 | 见按模型默认参数表 | 提供该字段的混合 GQA | 只改变 sliding Value，优先于通用 Value 维度；立即重算容量 |
| Sliding Window (`sliding_window`) | 滑动窗口的 token 数 | 实际使用时整数 ≥ 1 tokens | 见按模型默认参数表 | 混合 GQA、Inkling、V4 | 混合/Inkling 使用 min(tokens, window)；V4 标称公式按 window 计窗口条目；立即重算容量 |
| Draft Full Attention Layers (`draft_full_attention_layers`) | draft 中的 full-attention 层数 | 整数 ≥ 0 | 见按模型默认参数表 | Inkling 且 Include draft 开启 | 增加 draft 全局 KV；state 开启时增加对应 SConv 状态；立即重算容量 |
| Draft Sliding Attention Layers (`draft_sliding_attention_layers`) | draft 中的滑动窗口层数 | 整数 ≥ 0 | 见按模型默认参数表 | Inkling 且 Include draft 开启 | 增加 draft 窗口 KV 及适用的 SConv 状态；立即重算容量 |
| Hidden Size (`hidden_size`) | 模型隐藏向量宽度 | 实际使用时整数 ≥ 1 | 见按模型默认参数表 | Inkling 且 Include attention state 开启 | 改变 SConv 状态中的 2 × hidden_size 项；不直接修改该页主 KV 维度；立即重算容量 |
| Sconv Kernel Size (`sconv_kernel_size`) | SConv 卷积核长度 | 整数 ≥ 1 | 见按模型默认参数表 | Inkling 且 state 开启 | 保存历史长度为 kernel−1；1 对应该卷积历史项为零；立即重算容量 |
| Sconv State Bytes Per Element (`sconv_state_bytes_per_element`) | 每个 SConv 状态元素字节数 | 整数 ≥ 1 bytes | 见按模型默认参数表 | Inkling 且 state 开启 | 按该字节宽度累计 SConv 状态，独立于 KV precision；立即重算容量 |
| Kda Layers (`kda_layers`) | KDA 层数 | 整数 ≥ 1 | 见按模型默认参数表 | Kimi K3 且 state 开启 | 乘到每份 KDA checkpoint 的容量上；立即重算容量 |
| Kda Conv Kernel Size (`kda_conv_kernel_size`) | KDA 卷积核长度 | 整数 ≥ 1 | 见按模型默认参数表 | Kimi K3 且 state 开启 | 卷积历史长度为 kernel−1，改变卷积状态容量；立即重算容量 |
| Kda Num Heads (`kda_num_heads`) | KDA Query 卷积分支头数 | 整数 ≥ 1 | 见按模型默认参数表 | Kimi K3 且 state 开启 | 改变卷积状态中的 Query 元素数，不替代单独的 Key/Value heads；立即重算容量 |
| Kda Head Dim (`kda_head_dim`) | KDA Query 卷积分支每头宽度 | 整数 ≥ 1 | 见按模型默认参数表 | Kimi K3 且 state 开启 | 与 Kda Num Heads 相乘进入卷积状态；立即重算容量 |
| Kda Num Key Heads (`kda_num_key_heads`) | KDA Key 头数 | 整数 ≥ 1 | 见按模型默认参数表 | Kimi K3 且 state 开启 | 改变 Key 卷积状态元素数；立即重算容量 |
| Kda Key Head Dim (`kda_key_head_dim`) | KDA Key 每头宽度 | 整数 ≥ 1 | 见按模型默认参数表 | Kimi K3 且 state 开启 | 同时影响 Key 卷积状态和 recurrent 矩阵的 Key 维；立即重算容量 |
| Kda Num Value Heads (`kda_num_value_heads`) | KDA Value 头数 | 整数 ≥ 1 | 见按模型默认参数表 | Kimi K3 且 state 开启 | 同时影响 Value 卷积状态和 recurrent 状态；立即重算容量 |
| Kda Value Head Dim (`kda_value_head_dim`) | KDA Value 每头宽度 | 整数 ≥ 1 | 见按模型默认参数表 | Kimi K3 且 state 开启 | 同时影响 Value 卷积和 recurrent 矩阵的 Value 维；立即重算容量 |
| Kda Conv State Bytes Per Element (`kda_conv_state_bytes_per_element`) | KDA 卷积状态每元素字节数 | 整数 ≥ 1 bytes | 见按模型默认参数表 | Kimi K3 且 state 开启 | 只缩放卷积状态，独立于 KV precision；立即重算容量 |
| Kda Recurrent State Bytes Per Element (`kda_recurrent_state_bytes_per_element`) | KDA recurrent 状态每元素字节数 | 整数 ≥ 1 bytes | 见按模型默认参数表 | Kimi K3 且 state 开启 | 只缩放 recurrent 矩阵状态，独立于 KV precision；立即重算容量 |
| Linear Attention Layers (`linear_attention_layers`) | GDN 线性 attention 层数 | 整数 ≥ 0 | 见按模型默认参数表 | Qwen3.5/3.6 且 state 开启 | 乘到每份 GDN checkpoint 容量上；0 不含线性状态；立即重算容量 |
| Linear Conv Kernel Dim (`linear_conv_kernel_dim`) | GDN 卷积状态保存的长度 | 整数 ≥ 1 | 见按模型默认参数表 | Qwen3.5/3.6 且 state 开启 | 本公式直接乘该长度；与 KDA/SConv 的 kernel−1 写法不同；立即重算容量 |
| Linear Num Key Heads (`linear_num_key_heads`) | GDN Key 头数 | 整数 ≥ 1 | 见按模型默认参数表 | Qwen3.5/3.6 且 state 开启 | 进入卷积状态中的 2 × Hkey × Dkey 项；立即重算容量 |
| Linear Key Head Dim (`linear_key_head_dim`) | GDN Key 每头维度 | 整数 ≥ 1 | 见按模型默认参数表 | Qwen3.5/3.6 且 state 开启 | 影响卷积和 recurrent 矩阵的 Key 维；立即重算容量 |
| Linear Num Value Heads (`linear_num_value_heads`) | GDN Value 头数 | 整数 ≥ 1 | 见按模型默认参数表 | Qwen3.5/3.6 且 state 开启 | 影响卷积中的 Value 分支和 recurrent 状态；立即重算容量 |
| Linear Value Head Dim (`linear_value_head_dim`) | GDN Value 每头维度 | 整数 ≥ 1 | 见按模型默认参数表 | Qwen3.5/3.6 且 state 开启 | 影响卷积和 recurrent 矩阵的 Value 维；立即重算容量 |
| Linear Conv State Bytes Per Element (`linear_conv_state_bytes_per_element`) | GDN 卷积状态每元素字节数 | 整数 ≥ 1 bytes | 见按模型默认参数表 | Qwen3.5/3.6 且 state 开启 | 默认 2 B，独立于主 KV 精度；只缩放卷积部分；立即重算容量 |
| Linear Recurrent State Bytes Per Element (`linear_recurrent_state_bytes_per_element`) | GDN recurrent 状态每元素字节数 | 整数 ≥ 1 bytes | 见按模型默认参数表 | Qwen3.5/3.6 且 state 开启 | 默认 4 B，独立于主 KV 精度；只缩放 recurrent 部分；立即重算容量 |
| Sparse Attention Layers (`sparse_attention_layers`) | 包含稀疏 Indexer 的层数 | 整数 ≥ 0 | 见按模型默认参数表 | MiniMax M3 | 改变 Indexer 层数，不改变 Num Hidden Layers 指定的主 KV 层数；立即重算容量 |
| Compress ratios (`compress_ratios`) | V4 每层压缩比序列 | 非空非负整数列表，英文逗号或空白分隔；条目数至少等于 Num Hidden Layers | 见按模型默认参数表 | DeepSeek V4 Flash/Pro | 0 为 window-only；正值使用 floor(tokens/ratio) 个压缩条目；值 4 额外计 Indexer。draft 关闭只取前 N 层，开启使用全部条目；立即重算容量 |

混合 full/sliding 模型优先使用专用字段：global 头数优先于通用头数；global Key 维度优先于 Head Dim；global Value 优先使用独立 Value 维度；SWA 同理。被专用字段覆盖的通用字段可能仍显示，但修改不会影响该分支。关闭 state/draft 后，其字段仍可保留编辑值，却不进入当前容量和比较身份。

<a id="controls-model-defaults"></a>
## 模型目录与具体默认值

以下名称就是模型选择列表中的可选值。Family/Formula 筛选使用表中的值。Explore 和 Recommend 默认勾选 Cohere Command R+、DeepSeek R1、DeepSeek V3.2、Kimi K3、DeepSeek V4 Flash；其他模型初始未选。KV Size 初始模型为 DeepSeek V4 Pro。

| 模型 | Family | Formula | Context 上限（tokens） | 最小块（tokens） | 块大小倍数 |
| --- | --- | --- | --- | --- | --- |
| Cohere Command A 03-2025 | Cohere | mixed_full_sliding_gqa | 131072 | 1 | 1 |
| Cohere Command A Plus 05-2026 | Cohere | mixed_full_sliding_gqa | 200000 | 1 | 1 |
| Cohere Command R7B 12-2024 | Cohere | mixed_full_sliding_gqa | 132096 | 1 | 1 |
| Cohere Command R+ | Cohere | standard_gqa | 131072 | 1 | 1 |
| Cohere Command R v01 | Cohere | standard_gqa | 131072 | 1 | 1 |
| DeepSeek R1 | DeepSeek | mla | 163840 | 1 | 1 |
| DeepSeek V3 | DeepSeek | mla | 163840 | 1 | 1 |
| DeepSeek V3.2 | DeepSeek | dsa_mla | 163840 | 1 | 1 |
| DeepSeek V4.1 Flash | DeepSeek | deepseek_v41_csa2 | 1048576 | 2 | 2 |
| DeepSeek V4 Flash | DeepSeek | deepseek_v4_hybrid | 1048576 | 128 | 128 |
| DeepSeek V4 Pro | DeepSeek | deepseek_v4_hybrid | 1048576 | 128 | 128 |
| Gemma 4 26B-A4B | Gemma | mixed_full_sliding_gqa | 262144 | 1 | 1 |
| Gemma 4 31B | Gemma | mixed_full_sliding_gqa | 262144 | 1 | 1 |
| Gemma 4 E2B | Gemma | mixed_full_sliding_gqa | 131072 | 1 | 1 |
| Gemma 4 E4B | Gemma | mixed_full_sliding_gqa | 131072 | 1 | 1 |
| GLM-5 | GLM | dsa_mla | 202752 | 1 | 1 |
| GLM-5.1 | GLM | dsa_mla | 202752 | 1 | 1 |
| GLM-5.2 | GLM | dsa_mla | 1048576 | 1 | 1 |
| Inkling | Inkling | inkling_hybrid | 1048576 | 1 | 1 |
| Inkling-Small | Inkling | inkling_hybrid | 1048576 | 1 | 1 |
| Kimi K2.5 | Kimi | mla | 262144 | 1 | 1 |
| Kimi K2.6 | Kimi | mla | 262144 | 1 | 1 |
| Kimi K3 | Kimi | kimi_kda_mla_hybrid | 1048576 | 1 | 1 |
| Llama 3.1 70B | Llama | standard_gqa | 131072 | 1 | 1 |
| Llama 3.1 8B | Llama | standard_gqa | 131072 | 1 | 1 |
| Llama 3.3 70B | Llama | standard_gqa | 131072 | 1 | 1 |
| MiMo-V2.5 | MiMo | mixed_full_sliding_gqa | 1048576 | 1 | 1 |
| MiMo-V2.5-Pro | MiMo | mixed_full_sliding_gqa | 1048576 | 1 | 1 |
| MiniMax M2 | MiniMax | standard_gqa | 196608 | 1 | 1 |
| MiniMax M2.1 | MiniMax | standard_gqa | 196608 | 1 | 1 |
| MiniMax M2.5 | MiniMax | standard_gqa | 196608 | 1 | 1 |
| MiniMax M2.7 | MiniMax | standard_gqa | 204800 | 1 | 1 |
| MiniMax M3 | MiniMax | minimax_msa | 1048576 | 1 | 1 |
| Qwen2.5-14B | Qwen2.5 | standard_gqa | 32768 | 1 | 1 |
| Qwen2.5-32B | Qwen2.5 | standard_gqa | 32768 | 1 | 1 |
| Qwen2.5-72B | Qwen2.5 | standard_gqa | 32768 | 1 | 1 |
| Qwen2.5-7B | Qwen2.5 | standard_gqa | 32768 | 1 | 1 |
| Qwen2.5-Coder-32B | Qwen2.5 | standard_gqa | 32768 | 1 | 1 |
| Qwen3-0.6B | Qwen3 | standard_gqa | 40960 | 1 | 1 |
| Qwen3-14B | Qwen3 | standard_gqa | 40960 | 1 | 1 |
| Qwen3-1.7B | Qwen3 | standard_gqa | 40960 | 1 | 1 |
| Qwen3-235B-A22B | Qwen3 | standard_gqa | 40960 | 1 | 1 |
| Qwen3-30B-A3B | Qwen3 | standard_gqa | 40960 | 1 | 1 |
| Qwen3-32B | Qwen3 | standard_gqa | 40960 | 1 | 1 |
| Qwen3-4B | Qwen3 | standard_gqa | 40960 | 1 | 1 |
| Qwen3.5-0.8B | Qwen3.5 | qwen_linear_full_hybrid | 262144 | 1 | 1 |
| Qwen3.5-122B-A10B | Qwen3.5 | qwen_linear_full_hybrid | 262144 | 1 | 1 |
| Qwen3.5-27B | Qwen3.5 | qwen_linear_full_hybrid | 262144 | 1 | 1 |
| Qwen3.5-2B | Qwen3.5 | qwen_linear_full_hybrid | 262144 | 1 | 1 |
| Qwen3.5-35B-A3B | Qwen3.5 | qwen_linear_full_hybrid | 262144 | 1 | 1 |
| Qwen3.5-397B-A17B | Qwen3.5 | qwen_linear_full_hybrid | 262144 | 1 | 1 |
| Qwen3.5-4B | Qwen3.5 | qwen_linear_full_hybrid | 262144 | 1 | 1 |
| Qwen3.5-9B | Qwen3.5 | qwen_linear_full_hybrid | 262144 | 1 | 1 |
| Qwen3.6-27B | Qwen3.6 | qwen_linear_full_hybrid | 262144 | 1 | 1 |
| Qwen3.6-35B-A3B | Qwen3.6 | qwen_linear_full_hybrid | 262144 | 1 | 1 |
| Qwen3-8B | Qwen3 | standard_gqa | 40960 | 1 | 1 |

长度/块限制适用于 Explore/Recommend。KV Size 超出 Context 上限会提示并继续算标称容量。相同 Family 不代表相同公式或参数。

### Formula 的含义

| Formula | 解释 |
| --- | --- |
| standard_gqa | 标准逐层 KV；可包含 MHA/GQA/MQA 头数结构 |
| mla | latent KV 加独立 RoPE Key |
| dsa_mla | MLA 主缓存加稀疏 Indexer |
| kimi_kda_mla_hybrid | MLA 层与 KDA 卷积/recurrent 状态混合 |
| inkling_hybrid | 全局/滑动 KV 与 SConv 状态 |
| qwen_linear_full_hybrid | full-attention KV 与 GDN 状态 |
| mixed_full_sliding_gqa | 全局和滑动窗口 GQA 混合 |
| minimax_msa | 主 KV 与稀疏 Indexer |
| deepseek_v4_hybrid | 窗口、压缩条目和适用 Indexer |
| deepseek_v41_csa2 | 固定的 Encoder/Decoder global、SWA 和压缩尾布局 |

### KV Size 选择模型后的基础默认值

| 模型 | Tokens | Sequences | KV precision | Indexer precision | Draft | Attention state | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cohere Command A 03-2025；Cohere Command A Plus 05-2026；Cohere Command R7B 12-2024；Cohere Command R+；Cohere Command R v01；Gemma 4 26B-A4B；Gemma 4 31B；Gemma 4 E2B；Gemma 4 E4B；Kimi K2.5；Kimi K2.6；Llama 3.1 70B；Llama 3.1 8B；Llama 3.3 70B；MiMo-V2.5；MiMo-V2.5-Pro；Qwen2.5-14B；Qwen2.5-32B；Qwen2.5-72B；Qwen2.5-7B；Qwen2.5-Coder-32B；Qwen3-0.6B；Qwen3-14B；Qwen3-1.7B；Qwen3-235B-A22B；Qwen3-30B-A3B；Qwen3-32B；Qwen3-4B；Qwen3-8B | 1024 | 1 | BF16 / FP16 | 不显示 | 不显示 | 不显示 | 不显示 |
| DeepSeek R1；DeepSeek V3；MiniMax M2；MiniMax M2.1；MiniMax M2.5；MiniMax M2.7 | 1024 | 1 | BF16 / FP16 | 不显示 | 关（可开启） | 不显示 | 不显示 |
| DeepSeek V3.2；GLM-5；GLM-5.1；GLM-5.2 | 1024 | 1 | BF16 / FP16 | BF16 / FP16 | 关（可开启） | 不显示 | 不显示 |
| DeepSeek V4.1 Flash | 1024 | 1 | Released：FP4 global + FP8 SWA + scales | 跟随 KV（锁定） | 关（可开启） | 开 | 不显示 |
| DeepSeek V4 Flash；DeepSeek V4 Pro | 1024 | 1 | FP8 / INT8 | FP4 / INT4 | 关（可开启） | 不显示 | 不显示 |
| Inkling；Inkling-Small | 1024 | 1 | BF16 / FP16 | 不显示 | 关（可开启） | 开 | Prompt-end；间隔 2048（暂禁用） |
| Kimi K3；Qwen3.5-0.8B；Qwen3.5-122B-A10B；Qwen3.5-27B；Qwen3.5-2B；Qwen3.5-35B-A3B；Qwen3.5-397B-A17B；Qwen3.5-4B；Qwen3.5-9B；Qwen3.6-27B；Qwen3.6-35B-A3B | 1024 | 1 | BF16 / FP16 | 不显示 | 不显示 | 开 | Prompt-end；间隔 2048（暂禁用） |
| MiniMax M3 | 1024 | 1 | BF16 / FP16 | BF16 / FP16（锁定） | 不显示 | 不显示 | 不显示 |

V4.1 的 Encoder tail output 初始为 Recompute when missing。KV Size 中没有显示的控件不会参与该模型计算；内部保留的值不是用户可修改的有效参数。

### 各模型的高级参数默认值

列名与 Advanced parameters 一致。下面列出的都是选择模型或点击 Restore model defaults 后的具体值；展开模型名称查看默认值；参数完全相同的模型合并展示。未列出的高级字段在该模型中不显示。压缩比数组保留完整顺序，便于复制。

### deepseek_v41_csa2：固定的 Encoder/Decoder global、SWA 和压缩尾布局

DeepSeek V4.1 Flash 的高级结构只读，没有可编辑字段。可修改项为上面的长度、sequence、精度、state/draft 和 Encoder tail output。

### deepseek_v4_hybrid：窗口、压缩条目和适用 Indexer

<details>
<summary>DeepSeek V4 Flash</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 43 |
| Head Dim (`head_dim`) | 512 |
| Sliding Window (`sliding_window`) | 128 |
| Index Head Dim (`index_head_dim`) | 128 |
| Compress ratios (`compress_ratios`) | 见下方完整数组 |

DeepSeek V4 Flash — Compress ratios（按层顺序）：

```text
0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0
```

</details>

<details>
<summary>DeepSeek V4 Pro</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 61 |
| Head Dim (`head_dim`) | 512 |
| Sliding Window (`sliding_window`) | 128 |
| Index Head Dim (`index_head_dim`) | 128 |
| Compress ratios (`compress_ratios`) | 见下方完整数组 |

DeepSeek V4 Pro — Compress ratios（按层顺序）：

```text
128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0
```

</details>

### dsa_mla：MLA 主缓存加稀疏 Indexer

<details>
<summary>DeepSeek V3.2</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 61 |
| Kv Lora Rank (`kv_lora_rank`) | 512 |
| Qk Rope Head Dim (`qk_rope_head_dim`) | 64 |
| Num Nextn Predict Layers (`num_nextn_predict_layers`) | 1 |
| Index Head Dim (`index_head_dim`) | 128 |

</details>

<details>
<summary>GLM-5；GLM-5.1</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 78 |
| Kv Lora Rank (`kv_lora_rank`) | 512 |
| Qk Rope Head Dim (`qk_rope_head_dim`) | 64 |
| Num Nextn Predict Layers (`num_nextn_predict_layers`) | 1 |
| Index Head Dim (`index_head_dim`) | 128 |

</details>

<details>
<summary>GLM-5.2</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 78 |
| Kv Lora Rank (`kv_lora_rank`) | 512 |
| Qk Rope Head Dim (`qk_rope_head_dim`) | 64 |
| Num Nextn Predict Layers (`num_nextn_predict_layers`) | 1 |
| Index Head Dim (`index_head_dim`) | 128 |
| Indexer Full Layers (`indexer_full_layers`) | 21 |
| Draft Indexer Layers (`draft_indexer_layers`) | 1 |

</details>

### inkling_hybrid：全局/滑动 KV 与 SConv 状态

<details>
<summary>Inkling</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Full Attention Layers (`full_attention_layers`) | 11 |
| Sliding Attention Layers (`sliding_attention_layers`) | 55 |
| Num Key Value Heads (`num_key_value_heads`) | 8 |
| Head Dim (`head_dim`) | 128 |
| Swa Num Key Value Heads (`swa_num_key_value_heads`) | 16 |
| Swa Head Dim (`swa_head_dim`) | 128 |
| Sliding Window (`sliding_window`) | 512 |
| Draft Full Attention Layers (`draft_full_attention_layers`) | 2 |
| Draft Sliding Attention Layers (`draft_sliding_attention_layers`) | 6 |
| Hidden Size (`hidden_size`) | 6144 |
| Sconv Kernel Size (`sconv_kernel_size`) | 4 |
| Sconv State Bytes Per Element (`sconv_state_bytes_per_element`) | 2 |

</details>

<details>
<summary>Inkling-Small</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Full Attention Layers (`full_attention_layers`) | 7 |
| Sliding Attention Layers (`sliding_attention_layers`) | 35 |
| Num Key Value Heads (`num_key_value_heads`) | 8 |
| Head Dim (`head_dim`) | 128 |
| Swa Num Key Value Heads (`swa_num_key_value_heads`) | 8 |
| Swa Head Dim (`swa_head_dim`) | 128 |
| Sliding Window (`sliding_window`) | 512 |
| Draft Full Attention Layers (`draft_full_attention_layers`) | 2 |
| Draft Sliding Attention Layers (`draft_sliding_attention_layers`) | 6 |
| Hidden Size (`hidden_size`) | 4096 |
| Sconv Kernel Size (`sconv_kernel_size`) | 4 |
| Sconv State Bytes Per Element (`sconv_state_bytes_per_element`) | 2 |

</details>

### kimi_kda_mla_hybrid：MLA 层与 KDA 卷积/recurrent 状态混合

<details>
<summary>Kimi K3</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Full Attention Layers (`full_attention_layers`) | 24 |
| Kv Lora Rank (`kv_lora_rank`) | 512 |
| Qk Rope Head Dim (`qk_rope_head_dim`) | 64 |
| Kda Layers (`kda_layers`) | 69 |
| Kda Conv Kernel Size (`kda_conv_kernel_size`) | 4 |
| Kda Num Heads (`kda_num_heads`) | 96 |
| Kda Head Dim (`kda_head_dim`) | 128 |
| Kda Num Key Heads (`kda_num_key_heads`) | 96 |
| Kda Key Head Dim (`kda_key_head_dim`) | 128 |
| Kda Num Value Heads (`kda_num_value_heads`) | 96 |
| Kda Value Head Dim (`kda_value_head_dim`) | 128 |
| Kda Conv State Bytes Per Element (`kda_conv_state_bytes_per_element`) | 2 |
| Kda Recurrent State Bytes Per Element (`kda_recurrent_state_bytes_per_element`) | 4 |

</details>

### minimax_msa：主 KV 与稀疏 Indexer

<details>
<summary>MiniMax M3</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 60 |
| Num Key Value Heads (`num_key_value_heads`) | 4 |
| Head Dim (`head_dim`) | 128 |
| Sparse Attention Layers (`sparse_attention_layers`) | 57 |
| Index Head Dim (`index_head_dim`) | 128 |

</details>

### mixed_full_sliding_gqa：全局和滑动窗口 GQA 混合

<details>
<summary>Cohere Command A 03-2025</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Full Attention Layers (`full_attention_layers`) | 16 |
| Sliding Attention Layers (`sliding_attention_layers`) | 48 |
| Num Key Value Heads (`num_key_value_heads`) | 8 |
| Head Dim (`head_dim`) | 128 |
| Sliding Window (`sliding_window`) | 4096 |

</details>

<details>
<summary>Cohere Command A Plus 05-2026；Cohere Command R7B 12-2024</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Full Attention Layers (`full_attention_layers`) | 8 |
| Sliding Attention Layers (`sliding_attention_layers`) | 24 |
| Num Key Value Heads (`num_key_value_heads`) | 8 |
| Head Dim (`head_dim`) | 128 |
| Sliding Window (`sliding_window`) | 4096 |

</details>

<details>
<summary>Gemma 4 26B-A4B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Full Attention Layers (`full_attention_layers`) | 5 |
| Sliding Attention Layers (`sliding_attention_layers`) | 25 |
| Num Key Value Heads (`num_key_value_heads`) | 8 |
| Head Dim (`head_dim`) | 256 |
| Sliding Window (`sliding_window`) | 1024 |
| Num Global Key Value Heads (`num_global_key_value_heads`) | 2 |
| Global Head Dim (`global_head_dim`) | 512 |

</details>

<details>
<summary>Gemma 4 31B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Full Attention Layers (`full_attention_layers`) | 10 |
| Sliding Attention Layers (`sliding_attention_layers`) | 50 |
| Num Key Value Heads (`num_key_value_heads`) | 16 |
| Head Dim (`head_dim`) | 256 |
| Sliding Window (`sliding_window`) | 1024 |
| Num Global Key Value Heads (`num_global_key_value_heads`) | 4 |
| Global Head Dim (`global_head_dim`) | 512 |

</details>

<details>
<summary>Gemma 4 E2B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Full Attention Layers (`full_attention_layers`) | 3 |
| Sliding Attention Layers (`sliding_attention_layers`) | 12 |
| Num Key Value Heads (`num_key_value_heads`) | 1 |
| Head Dim (`head_dim`) | 256 |
| Sliding Window (`sliding_window`) | 512 |
| Global Head Dim (`global_head_dim`) | 512 |

</details>

<details>
<summary>Gemma 4 E4B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Full Attention Layers (`full_attention_layers`) | 4 |
| Sliding Attention Layers (`sliding_attention_layers`) | 20 |
| Num Key Value Heads (`num_key_value_heads`) | 2 |
| Head Dim (`head_dim`) | 256 |
| Sliding Window (`sliding_window`) | 512 |
| Global Head Dim (`global_head_dim`) | 512 |

</details>

<details>
<summary>MiMo-V2.5</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Full Attention Layers (`full_attention_layers`) | 9 |
| Sliding Attention Layers (`sliding_attention_layers`) | 39 |
| Num Key Value Heads (`num_key_value_heads`) | 4 |
| Head Dim (`head_dim`) | 192 |
| Sliding Window (`sliding_window`) | 128 |
| V Head Dim (`v_head_dim`) | 128 |
| Swa Num Key Value Heads (`swa_num_key_value_heads`) | 8 |
| Swa Head Dim (`swa_head_dim`) | 192 |
| Swa V Head Dim (`swa_v_head_dim`) | 128 |

</details>

<details>
<summary>MiMo-V2.5-Pro</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Full Attention Layers (`full_attention_layers`) | 10 |
| Sliding Attention Layers (`sliding_attention_layers`) | 60 |
| Num Key Value Heads (`num_key_value_heads`) | 8 |
| Head Dim (`head_dim`) | 192 |
| Sliding Window (`sliding_window`) | 128 |
| V Head Dim (`v_head_dim`) | 128 |
| Swa Num Key Value Heads (`swa_num_key_value_heads`) | 8 |
| Swa Head Dim (`swa_head_dim`) | 192 |
| Swa V Head Dim (`swa_v_head_dim`) | 128 |

</details>

### mla：latent KV 加独立 RoPE Key

<details>
<summary>DeepSeek R1；DeepSeek V3</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 61 |
| Kv Lora Rank (`kv_lora_rank`) | 512 |
| Qk Rope Head Dim (`qk_rope_head_dim`) | 64 |
| Num Nextn Predict Layers (`num_nextn_predict_layers`) | 1 |

</details>

<details>
<summary>Kimi K2.5；Kimi K2.6</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 61 |
| Kv Lora Rank (`kv_lora_rank`) | 512 |
| Qk Rope Head Dim (`qk_rope_head_dim`) | 64 |

</details>

### qwen_linear_full_hybrid：full-attention KV 与 GDN 状态

<details>
<summary>Qwen3.5-0.8B；Qwen3.5-2B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Full Attention Layers (`full_attention_layers`) | 6 |
| Num Key Value Heads (`num_key_value_heads`) | 2 |
| Head Dim (`head_dim`) | 256 |
| Linear Attention Layers (`linear_attention_layers`) | 18 |
| Linear Conv Kernel Dim (`linear_conv_kernel_dim`) | 4 |
| Linear Num Key Heads (`linear_num_key_heads`) | 16 |
| Linear Key Head Dim (`linear_key_head_dim`) | 128 |
| Linear Num Value Heads (`linear_num_value_heads`) | 16 |
| Linear Value Head Dim (`linear_value_head_dim`) | 128 |
| Linear Conv State Bytes Per Element (`linear_conv_state_bytes_per_element`) | 2 |
| Linear Recurrent State Bytes Per Element (`linear_recurrent_state_bytes_per_element`) | 4 |

</details>

<details>
<summary>Qwen3.5-122B-A10B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Full Attention Layers (`full_attention_layers`) | 12 |
| Num Key Value Heads (`num_key_value_heads`) | 2 |
| Head Dim (`head_dim`) | 256 |
| Linear Attention Layers (`linear_attention_layers`) | 36 |
| Linear Conv Kernel Dim (`linear_conv_kernel_dim`) | 4 |
| Linear Num Key Heads (`linear_num_key_heads`) | 16 |
| Linear Key Head Dim (`linear_key_head_dim`) | 128 |
| Linear Num Value Heads (`linear_num_value_heads`) | 64 |
| Linear Value Head Dim (`linear_value_head_dim`) | 128 |
| Linear Conv State Bytes Per Element (`linear_conv_state_bytes_per_element`) | 2 |
| Linear Recurrent State Bytes Per Element (`linear_recurrent_state_bytes_per_element`) | 4 |

</details>

<details>
<summary>Qwen3.5-27B；Qwen3.6-27B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Full Attention Layers (`full_attention_layers`) | 16 |
| Num Key Value Heads (`num_key_value_heads`) | 4 |
| Head Dim (`head_dim`) | 256 |
| Linear Attention Layers (`linear_attention_layers`) | 48 |
| Linear Conv Kernel Dim (`linear_conv_kernel_dim`) | 4 |
| Linear Num Key Heads (`linear_num_key_heads`) | 16 |
| Linear Key Head Dim (`linear_key_head_dim`) | 128 |
| Linear Num Value Heads (`linear_num_value_heads`) | 48 |
| Linear Value Head Dim (`linear_value_head_dim`) | 128 |
| Linear Conv State Bytes Per Element (`linear_conv_state_bytes_per_element`) | 2 |
| Linear Recurrent State Bytes Per Element (`linear_recurrent_state_bytes_per_element`) | 4 |

</details>

<details>
<summary>Qwen3.5-35B-A3B；Qwen3.6-35B-A3B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Full Attention Layers (`full_attention_layers`) | 10 |
| Num Key Value Heads (`num_key_value_heads`) | 2 |
| Head Dim (`head_dim`) | 256 |
| Linear Attention Layers (`linear_attention_layers`) | 30 |
| Linear Conv Kernel Dim (`linear_conv_kernel_dim`) | 4 |
| Linear Num Key Heads (`linear_num_key_heads`) | 16 |
| Linear Key Head Dim (`linear_key_head_dim`) | 128 |
| Linear Num Value Heads (`linear_num_value_heads`) | 32 |
| Linear Value Head Dim (`linear_value_head_dim`) | 128 |
| Linear Conv State Bytes Per Element (`linear_conv_state_bytes_per_element`) | 2 |
| Linear Recurrent State Bytes Per Element (`linear_recurrent_state_bytes_per_element`) | 4 |

</details>

<details>
<summary>Qwen3.5-397B-A17B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Full Attention Layers (`full_attention_layers`) | 15 |
| Num Key Value Heads (`num_key_value_heads`) | 2 |
| Head Dim (`head_dim`) | 256 |
| Linear Attention Layers (`linear_attention_layers`) | 45 |
| Linear Conv Kernel Dim (`linear_conv_kernel_dim`) | 4 |
| Linear Num Key Heads (`linear_num_key_heads`) | 16 |
| Linear Key Head Dim (`linear_key_head_dim`) | 128 |
| Linear Num Value Heads (`linear_num_value_heads`) | 64 |
| Linear Value Head Dim (`linear_value_head_dim`) | 128 |
| Linear Conv State Bytes Per Element (`linear_conv_state_bytes_per_element`) | 2 |
| Linear Recurrent State Bytes Per Element (`linear_recurrent_state_bytes_per_element`) | 4 |

</details>

<details>
<summary>Qwen3.5-4B；Qwen3.5-9B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Full Attention Layers (`full_attention_layers`) | 8 |
| Num Key Value Heads (`num_key_value_heads`) | 4 |
| Head Dim (`head_dim`) | 256 |
| Linear Attention Layers (`linear_attention_layers`) | 24 |
| Linear Conv Kernel Dim (`linear_conv_kernel_dim`) | 4 |
| Linear Num Key Heads (`linear_num_key_heads`) | 16 |
| Linear Key Head Dim (`linear_key_head_dim`) | 128 |
| Linear Num Value Heads (`linear_num_value_heads`) | 32 |
| Linear Value Head Dim (`linear_value_head_dim`) | 128 |
| Linear Conv State Bytes Per Element (`linear_conv_state_bytes_per_element`) | 2 |
| Linear Recurrent State Bytes Per Element (`linear_recurrent_state_bytes_per_element`) | 4 |

</details>

### standard_gqa：标准逐层 KV；可包含 MHA/GQA/MQA 头数结构

<details>
<summary>Cohere Command R+；Qwen2.5-32B；Qwen2.5-Coder-32B；Qwen3-32B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 64 |
| Num Key Value Heads (`num_key_value_heads`) | 8 |
| Head Dim (`head_dim`) | 128 |

</details>

<details>
<summary>Cohere Command R v01</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 40 |
| Num Key Value Heads (`num_key_value_heads`) | 64 |
| Head Dim (`head_dim`) | 128 |

</details>

<details>
<summary>Llama 3.1 70B；Llama 3.3 70B；Qwen2.5-72B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 80 |
| Num Key Value Heads (`num_key_value_heads`) | 8 |
| Head Dim (`head_dim`) | 128 |

</details>

<details>
<summary>Llama 3.1 8B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 32 |
| Num Key Value Heads (`num_key_value_heads`) | 8 |
| Head Dim (`head_dim`) | 128 |

</details>

<details>
<summary>MiniMax M2；MiniMax M2.1；MiniMax M2.5；MiniMax M2.7</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 62 |
| Num Key Value Heads (`num_key_value_heads`) | 8 |
| Head Dim (`head_dim`) | 128 |
| Num Mtp Modules (`num_mtp_modules`) | 3 |
| Mtp Transformer Layers (`mtp_transformer_layers`) | 1 |

</details>

<details>
<summary>Qwen2.5-14B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 48 |
| Num Key Value Heads (`num_key_value_heads`) | 8 |
| Head Dim (`head_dim`) | 128 |

</details>

<details>
<summary>Qwen2.5-7B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 28 |
| Num Key Value Heads (`num_key_value_heads`) | 4 |
| Head Dim (`head_dim`) | 128 |

</details>

<details>
<summary>Qwen3-0.6B；Qwen3-1.7B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 28 |
| Num Key Value Heads (`num_key_value_heads`) | 8 |
| Head Dim (`head_dim`) | 128 |

</details>

<details>
<summary>Qwen3-14B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 40 |
| Num Key Value Heads (`num_key_value_heads`) | 8 |
| Head Dim (`head_dim`) | 128 |

</details>

<details>
<summary>Qwen3-235B-A22B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 94 |
| Num Key Value Heads (`num_key_value_heads`) | 4 |
| Head Dim (`head_dim`) | 128 |

</details>

<details>
<summary>Qwen3-30B-A3B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 48 |
| Num Key Value Heads (`num_key_value_heads`) | 4 |
| Head Dim (`head_dim`) | 128 |

</details>

<details>
<summary>Qwen3-4B；Qwen3-8B</summary>

| 高级字段 | 默认值 |
| --- | --- |
| Num Hidden Layers (`num_hidden_layers`) | 36 |
| Num Key Value Heads (`num_key_value_heads`) | 8 |
| Head Dim (`head_dim`) | 128 |

</details>

<a id="controls-state-defaults"></a>
## 模型恢复默认值

本表给出首次 BF16 reference 下 State group overrides / Model exceptions 选择 Inherit 时的模型默认形式。界面每行选择器初始仍显示 Inherit，不会直接选中表里的策略。全局 Scenario 能否替换它由“使用全局场景”列决定。更换 Runtime 或重开历史实验可能携带不同策略。

| 模型 | 状态组 | 状态内容 | 继承的模型默认值 | 使用全局场景 | 该行可选值 |
| --- | --- | --- | --- | --- | --- |
| Cohere Command A 03-2025；Cohere Command A Plus 05-2026；Cohere Command R7B 12-2024；Gemma 4 26B-A4B；Gemma 4 31B；Gemma 4 E2B；Gemma 4 E4B；Inkling；Inkling-Small；MiMo-V2.5；MiMo-V2.5-Pro | sliding_window_state | 滑动窗口 Key/Value 历史 | Token history | 否；需组/模型级显式覆盖 | Inherit / Prompt end / Periodic / None / Token history |
| DeepSeek V4 Flash；DeepSeek V4 Pro | swa | DeepSeek V4 SWA 状态 | Periodic 2048；不保留 prompt end | 是 | Inherit / Prompt end / Periodic / None / Token history |
| Inkling；Inkling-Small | sconv_state | Inkling 卷积状态 | Prompt end | 是 | Inherit / Prompt end / Periodic / None |
| Kimi K3 | kda_state | Kimi KDA 卷积与 recurrent 状态 | Prompt end | 是 | Inherit / Prompt end / Periodic / None |
| Qwen3.5-0.8B；Qwen3.5-122B-A10B；Qwen3.5-27B；Qwen3.5-2B；Qwen3.5-35B-A3B；Qwen3.5-397B-A17B；Qwen3.5-4B；Qwen3.5-9B；Qwen3.6-27B；Qwen3.6-35B-A3B | linear_state | Qwen GDN 卷积与 recurrent 状态 | Prompt end | 是 | Inherit / Prompt end / Periodic / None |

本表未列出的模型没有这类可编辑状态组。DeepSeek V4.1 使用独立 Encoder state/output 场景，不能把其他模型的 checkpoint 控件套用到它。

### Full Tail 的前缀采样值

点击 Full Tail 后，Prefix hit rates 被填入以下 55 个值；Simple 将它们显示为百分比。

```text
0.5, 0.5729580301, 0.6352703119, 0.688490231, 0.7339445092, 0.7727662782, 0.8059233276, 0.834242231, 0.8584289516, 0.8790864412, 0.8967296714, 0.9117984708, 0.9246684905, 0.9356605676, 0.9450487241, 0.9530669977, 0.9599152765, 0.9657642814, 0.9707598226, 0.9750264341, 0.9786704784, 0.9817827982, 0.9844409805, 0.9867112913, 0.9886503273, 0.9903064269, 0.9917208749, 0.9929289322, 0.9939607145, 0.9948419433, 0.9955945866, 0.9962374072, 0.9967864299, 0.9972553414, 0.9976558312, 0.997997883, 0.9982900241, 0.998539537, 0.998752642, 0.9989346516, 0.999090103, 0.9992228716, 0.9993362671, 0.9994331164, 0.9995158338, 0.9995864814, 0.9996468204, 0.999698355, 0.9997423699, 0.9997799622, 0.9998120693, 0.9998394914, 0.9998629122, 0.9998829155, 0.9999
```

<a id="controls-architecture"></a>
## Architecture：图形操作

Architecture 的操作只改变图的视图、选择和导出，不修改模型、计算输入或实验。默认显示总图，动效开启，未选择路线或节点，缩放为 100%。浏览器偏好为减少动态效果时，部分动画按该偏好停用。主页面 Theme 管理颜色，图内独立主题按钮隐藏。

### 总图和工具栏

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| 动态 / 暂停 | 控制图中动效 | 开 / 暂停 | 开启（受系统减少动态效果设置约束） | Architecture | 只改变动画播放；不会重新计算数据 |
| 演示 / 退出演示 | 切换以图为主的演示布局 | 点击切换；F 快捷键 | 关闭 | Architecture | 调整图的可用视口并重置取景，退出恢复原滚动位置；与模块详情里的浏览器全屏操作不同 |
| 总览 | 清除章节视图，查看全部架构 | 点击 | 初始已经总览，按钮禁用 | 已选某个章节/引导视图后可用 | 恢复总图显示，不修改图形内容 |
| 章节卡片 | 按流程突出一组节点 | 主计算链 / 缓存事件循环 / 恢复与 TTFT / 入口与实验管理 / 容量与 IO / SSD 仿真 / 内存池与拓扑 | 未选章节 | 总图上方 | 高亮相应路线并调整视图；可点击另一个章节或总览切换 |
| 图表指南 | 打开操作入口说明 | 点击；? 快捷键 | 关闭 | Architecture | 显示节点查找、路径、地图、透镜、故事播放、演示六个入口 |
| 图表指南 × | 关闭指南 | 点击或 Escape | 指南关闭 | 指南打开后 | 只关闭指南 |
| 指南中的六项操作 | 跳转到对应功能 | 查找任意节点 / 追踪路径 / 查看完整系统 / 比较语义类型 / 播放引导故事 / 进入演示模式 | 未启动 | 指南打开后 | 调用对应工具；P 可切换引导故事播放，播放会跟随节点；不改变计算结果 |
| 缩小 / 阅读百分比（重置）/ 放大 | 控制总图缩放 | 100%–300%，按钮每次 25%；重置回 100% | 100%；缩小禁用 | Architecture 总图 | 只改变视图；手动缩放/平移会打断自动镜头跟随 |
| 拖动画布 | 移动查看位置 | 在可平移画布中拖动 | 未平移 | 放大或相应取景状态 | 只移动视口；使用重置恢复 |

图内快捷键在图区域获得焦点后使用：/ 查找节点，R 路径，M 地图，L 透镜，P 故事播放，F 演示，E 导出，? 指南。输入框内按键优先用于文字输入。旧版引导的上一/下一章按钮、播放按钮和“复制此刻”在当前布局中隐藏，不需要寻找这些可见按钮；章节卡片和指南/快捷键承担相应入口。

### 节点查找、聚焦和模块详情

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| 查找节点 | 打开节点搜索 | 点击或 / | 关闭 | Architecture | 打开搜索面板；路径选择时搜索范围会限制为可选起点/终点 |
| 搜索图表节点 | 按标签、说明、类型或稳定 ID 查找 | 任意文字；空白列出当前允许的节点 | 空白 | 查找面板 | 只筛选候选，不修改图或计算 |
| 节点查找器 × | 关闭搜索 | 点击或 Escape | 面板打开后可用 | 查找面板 | 不修改计算参数 |
| 总图节点 / 搜索结果 | 查看一个模块 | 点击节点；键盘 Enter/Space | 未打开模块 | 非路径选点状态 | 打开模块详情。路径选点状态下，点击改为选起点/终点；不会运行所描述的 CLI 或分析代码 |
| 图内聚焦 | 显示节点可达性和关系 | Alt、Ctrl 或 Command 加点击总图节点 | 未聚焦 | 总图，且未打开模块详情或路径选点 | 保留图内聚焦而不打开模块弹窗，显示上游/下游和复制链接等操作 |
| 上游 / 下游 | 查看按箭头方向可到达的节点 | 点击上游或下游 | 未选择；数量为 0 的方向禁用 | 单节点处于图内聚焦 | 高亮可达节点和连接，保持原数据；上游是能到达该节点的节点，下游是由它能到达的节点 |
| 复制链接（聚焦） | 复制当前节点或关系的定位链接 | 点击 | 未聚焦时隐藏 | 有当前聚焦对象 | 写入剪贴板；链接使用当前服务地址和图状态 |
| 聚焦卡片 × | 清除聚焦和可达性显示 | 点击 | 未聚焦时隐藏 | 聚焦卡片显示 | 恢复一般图显示 |
| 关系预览 / 固定关系 | 查看已编写的节点连接 | 选择图中提供的关系命中区域或关系条目 | 未固定关系 | 单节点聚焦且提供关系入口时 | 突出连接两端和关系说明；不推断未画出的调用。当前布局隐藏了旧“关系”开关 |
| 关闭模块详情 × / PCS 架构 | 回到总图 | 点击 | 模块弹窗关闭 | 模块详情打开后 | 关闭弹窗并返回原位置，不修改其他页 |
| 模块内部节点 | 选择详细说明 | 点击或 Enter/Space | 打开模块时选择其预设节点 | 模块详情 | 更新右侧实现说明、示例和代码节选；收起面板后选节点会重新展开 |
| 展开内部结构 | 进入选中节点的子图 | 点击 | 仅部分节点提供 | 选中节点有子图 | 进入下一层结构，记录返回位置 |
| 收起 | 收起右侧节点详情 | 点击 | 模块打开后详情展开 | 模块详情 | 给内部图更多空间；再次选节点重新展开 |
| 返回上层 / 返回总图 / 面包屑 | 回到上层结构或退出模块 | 点击或 Escape | 根层显示返回总图 | 模块详情 | 优先回子图上层或关联跳转前的位置；没有上层时关闭弹窗 |
| 上一个模块 / 下一个模块 | 切换讲解模块 | 点击或左右方向键 | 按当前模块在顺序中的位置启用；首尾禁用相应方向 | 模块详情 | 自由探索按 N01–N31 顺序；引导路线按其顺序；重置到新模块根图 |
| 关联模块 | 沿显示的关系查看另一模块 | 点击关系标签 | 按当前模块提供 | 模块详情下方 | 打开关联模块并记住返回历史 |
| 复制提纲 | 复制模块及子图说明 | 点击 | 模块打开后可用 | 模块详情 | 复制文字到剪贴板，不创建实验或计算 |
| 复制代码 | 复制当前代码节选 | 点击 | 有代码片段后可用 | 节点详情中提供代码时 | 复制节选本身，不含行号；不会执行代码 |
| 全屏 | 请求浏览器全屏 | 点击或模块内 F | 未全屏 | 模块详情；需浏览器允许 | 切换浏览器全屏；与总图演示布局分开 |

### 路径

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| 路径 | 追踪有向连接 | 点击或 R；再次点击清除 | 关闭 | Architecture | 进入起点选择；选择起点再选择可达终点，显示图上已经存在的路径 |
| 查找起点 / 查找目标 | 用搜索选择路径两端 | 点击后选择搜索结果 | 路径刚启动时为查找起点 | 路径处于 source/target 阶段 | 起点候选须有出边；终点候选是该起点可达的其他节点 |
| 清除（路径） | 退出路径选择/显示 | 点击 | 路径关闭时隐藏 | 路径面板 | 清空起终点、路径和旅程播放 |
| 复制链接（路径） | 复制起终点定位 | 点击 | 没有结果路径时隐藏 | 找到路径后 | 复制当前路径链接；不保存实验 |
| 路径节点条目 | 跳到路径某一步 | 点击具体节点 | 路径总览状态 | 有路径结果 | 暂停自动旅程，突出该步和邻近边 |
| 上一个路径位置 / 下一个路径位置 | 逐步查看路径 | 点击；首尾按钮禁用 | 初始未步进；首步无法再向前 | 有路径结果 | 更新当前步骤和镜头位置 |
| 播放路径旅程 / 暂停 | 沿路径自动逐步查看 | 点击切换 | 停止 | 有路径且浏览器允许相关动效 | 按路径顺序播放；暂停保留位置；不进行任何后端计算 |
| 总览（路径） | 重新查看整条路径 | 点击 | 已经总览时禁用 | 正在逐步查看路径 | 退出当前步的特写，显示整条路径 |

### 地图、透镜与导出

| 界面项目 | 作用 | 可选值或输入范围 | 首次默认值／初始状态 | 适用条件 | 修改或操作后的影响 |
| --- | --- | --- | --- | --- | --- |
| 地图 / 打开雷达 | 打开整体位置图 | 点击或 M | 关闭 | Architecture | 显示全图节点和当前视口，帮助定位；不改计算 |
| 雷达节点 / 视口 | 在整体位置图中定位 | 点击节点或地图位置，按图提供的视口拖动 | 当前视口 | 地图打开后 | 移动主图视口或聚焦对应节点 |
| 地图 × | 关闭地图 | 点击或再次点击地图工具 | 地图打开后可用 | Architecture | 关闭浮层，保留主图取景 |
| 透镜 | 按节点类型突出显示 | 点击或 L | 关闭 | Architecture | 打开类型选择面板 |
| 后端 / 前端 / 数据库 | 选择要看的节点类型 | 三种类型，可选 0–2 种；再次点击取消 | 都未选 | 透镜面板 | 选一种显示该类节点，选两种比较直接连接；已选两种时第三种禁用 |
| 复制链接（透镜） | 复制类型选择链接 | 点击 | 没有类型选择时无有效链接 | 至少选中一种类型 | 复制当前类型组合 |
| 清除（透镜） | 清除类型选择 | 点击 | 都未选 | 透镜面板 | 恢复未筛选状态，保留相应视口 |
| 透镜 × | 关闭类型面板 | 点击 | 面板关闭 | 面板打开后 | 关闭面板；清除选择用 Clear，关闭本身不等于清除 |
| 导出 | 打开导出格式菜单 | 点击或 E | 关闭 | Architecture | 只展开菜单，选择具体格式后才下载/复制 |
| 分享卡片 | 生成用于分享的概览图片 | 1200×630 PNG | 未导出 | 导出菜单 | 下载概览卡片 |
| 路径分享卡片 | 生成路径摘要图 | 1200×630 PNG | 隐藏或未导出 | 菜单中仅在具有可导出的有效路径快照时提供 | 下载路径卡片；没有有效快照时不显示/不可用 |
| 可达范围分享卡片 | 生成上游/下游范围摘要 | 1200×630 PNG | 隐藏或未导出 | 菜单中仅在当前可达范围满足导出条件时提供 | 下载可达范围卡片 |
| 复制分享卡片 | 复制 PNG 卡片 | 点击 | 未复制；不支持图像剪贴板时禁用 | 导出菜单且浏览器支持 ClipboardItem | 写入剪贴板，不创建文件 |
| 复制图表 | 复制图表 PNG | 点击 | 未复制；不支持图像剪贴板时禁用 | 导出菜单 | 写入剪贴板 |
| PNG | 导出无损位图 | PNG | 未导出 | 导出菜单 | 下载图片，保留当前显示主题 |
| JPEG | 导出紧凑位图 | JPEG | 未导出 | 浏览器支持 JPEG 编码 | 下载图片 |
| WebP | 导出 WebP 位图 | WebP | 未导出 | 浏览器支持 WebP 编码，否则禁用 | 下载图片 |
| SVG | 导出可编辑矢量图 | SVG | 未导出 | 导出菜单 | 下载矢量资源 |
| WebM | 导出动效片段 | 6 秒 WebM | 未导出 | 浏览器支持 canvas 录制和 MediaRecorder，否则禁用 | 生成视频下载；不改变模拟参数 |

复制和全屏需要浏览器权限；失败时页面会提示。图里的节点、路线和代码节选用于解释项目结构，点击不会运行节点所描述的 CLI、内存池或研究实验。
