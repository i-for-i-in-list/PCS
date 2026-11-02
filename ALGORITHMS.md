# PCS WebUI 算法说明

本文说明 PCS WebUI 中各项数值的计算方法：每一项的输入、公式或步骤、输出、前提假设，以及对应的代码位置。界面操作见 [README.md](README.md)，安装见 [INSTALL.md](INSTALL.md)。

| 项目 | 内容 |
| --- | --- |
| 对应代码 | 本仓库提交 `93149e3`（2026-10-28） |
| 覆盖范围 | `ssu_sim/`：模型、负载、缓存回放、需求分析、SSD 仿真；`dashboard_v2/`：参数处理、推荐算法、KV Size 页面 |
| 核对方式 | 公式逐条对照上述代码；第 10.7 节算例和第 11.6 节吞吐数字于 2026-11-02 用该版本代码实际运行得到，复算方法见[附录 A](#appendix-a) |
| 以谁为准 | 本文与代码不一致时以代码为准。README“建模说明”一节与本文第 10 节一致，本文更完整；已知差异见[第 13.4 节](#readme-diff) |

## 目录

0. [范围与阅读方式](#scope)
1. [记号与单位](#notation)
2. [总流程](#pipeline)
3. [合成负载](#workload)
4. [前缀块身份](#block-identity)
5. [容量模型](#capacity)
6. [计算量模型](#compute)
7. [有状态模型的恢复](#recovery)
8. [缓存回放](#replay)
9. [IO 需求](#io-demand)
10. [TTFT 与带宽阈值](#ttft)
11. [单 SSD 仿真](#ssd)
12. [推荐算法](#recommend)
13. [验证状态与已知局限](#validation)
- [附录 A：复算本文的数字](#appendix-a)
- [附录 B：代码索引](#appendix-b)

<a id="scope"></a>
## 0. 范围与阅读方式

PCS 是解析模型：它不运行神经网络，也不读写真实 SSD。它根据模型结构参数和合成负载，计算前缀缓存的容量、读写量、计算量和时间预算。

适合回答的问题：

- 在同一组假设下，比较不同模型、上下文长度、命中率、块大小、缓存容量的相对差异；
- 一次命中需要恢复多少字节、省下多少计算，存储带宽达到多少时使用缓存才划算；
- 在有限容量下，合成负载的实际命中、驱逐和驻留情况。

不适合回答的问题（详见[第 13 节](#validation)）：

- 真实服务中的绝对 TTFT。程序没有计算排队、batching 和调度，算力是一个假定的常数；
- 某型号 GPU 或 SSD 的实测性能。所有时间都未经校准；
- 真实业务流量的命中率。程序只能生成合成负载。

第 3–12 节按同一结构编写：**输入 → 算法 → 输出 → 假设与适用范围 → 代码位置**。代码位置写作“文件 · 函数”，行号见[附录 B](#appendix-b)。

界面中的 Architecture 页绘制于 2026-09-24，其中包含本仓库未随附的模块（例如 `ssu_sim/pool/`）。计算方法以本文和代码为准。

<a id="notation"></a>
## 1. 记号与单位

| 记号 | 含义 |
| --- | --- |
| N | 请求的输入长度（tokens） |
| H_cfg | 配置的候选前缀长度：⌊N × Prefix ratio⌋ |
| H | 命中前缀长度。Envelope 中等于 H_cfg 并假定已驻留；Replay 中是查找时缓存里实际驻留的连续前缀 |
| N − H | 未命中的 suffix 长度 |
| B_r | 恢复边界：可以直接从缓存状态继续计算的位置，B_r ≤ H |
| R = H − B_r | 恢复时需要重算（replay）的 token 数 |
| K | 块大小（Block size，tokens/块） |
| L、L_x | 层数；L_x 表示某一类层的层数 |
| D | 隐藏维度 |
| Hq、Hkv | Q 头数、KV 头数 |
| dk、dv | 每个头的 K、V 维度 |
| C | Compute：有效算力（TFLOPS）。随附 Runtime 均为 2000 |
| BW | 存储带宽（GB/s） |

| 单位与约定 | 说明 |
| --- | --- |
| GB、GB/s | 十进制，1 GB = 10^9 bytes |
| GiB | 二进制，1 GiB = 2^30 bytes |
| 时间 | 内部以 µs 计 |
| FLOPs | 一次乘加（MAC）计 2 次运算 |
| 计算时间 | t（µs）= FLOPs ÷（C × 10^6） |
| 读取时间 | t（µs）= 字节 ÷（BW × 1000） |
| 分位数 | 样本排序后取下标为 round(q/100 × (n − 1)) 的元素，不插值。按请求数加权的分位数取累计权重首次达到 q% 时的值 |

<a id="pipeline"></a>
## 2. 总流程

```text
模型 YAML + Runtime YAML ──→ 模型对象：容量组件（§5）、计算公式（§6）、状态策略（§7）

负载参数 ──→ 合成请求（§3）──→ 块身份（§4）──┬─ Bandwidth Envelope：1 个请求，前缀假定已驻留
                                              └─ Cache Replay：radix + leaf-LRU 回放（§8），
                                                 得到每个请求的实际命中 H 和恢复边界 B_r

每个请求 ──→ 读写的块与字节（§9）──→ 计算量与各阶段时间（§6、§10）
         ──→ TTFT、带宽阈值、聚合依赖压力、峰值窗口（§9、§10）

Case 结果 ──┬─ Explore：图表、表格、Case 详情
            ├─ Recommend：约束判断与排名（§12）
            └─ SSD Simulation：把一个 Replay Case 的 IO 放到一块模拟 SSD 上排队（§11）
```

三个前提贯穿全文：

1. **开环。** 请求的完成时间 = 到达时间 + 缓存路径计算时间。计算不排队；存储读写时间不反馈给缓存回放。
2. **统一算力。** 所有计入的运算按同一个 C 换算成时间（默认 2000 TFLOPS）。这是归一化标尺，不是某块 GPU 的实测值。
3. **序列化字节。** 容量和读写量按序列化后的有效负载计，不含分配器预留、页对齐和文件系统开销。SSD 仿真中按 16 KiB 页取整的部分除外。

| 项目 | Bandwidth Envelope | Cache Replay |
| --- | --- | --- |
| 请求 | 1 个，t = 0 到达 | Request count 个，按 QPS 和到达模型生成 |
| 命中长度 H | H_cfg，假定已驻留 | 查找时最长的连续驻留前缀（§8） |
| 容量与驱逐 | 不模拟 | 按容量准入、保护、驱逐 |
| 写入 | 不计，Write 恒为 0 | Persist 开启时计入 miss 新块，可选 decode 写入 |
| 平均速率的时间基准 | 1 s | 测量请求数 ÷ QPS |

<a id="workload"></a>
## 3. 合成负载

### 输入

| 参数 | 含义 | chat | rag | agent |
| --- | --- | ---: | ---: | ---: |
| Request count | 请求数 n | 1000 | 1000 | 1000 |
| QPS | 平均到达率 | 100 | 100 | 50 |
| Context lengths | 输入长度 N | 8192 | 16384 | 32768 |
| Generation length | 每个请求生成的 token 数 | 256 | 512 | 1024 |
| Prefix ratio | 候选前缀比例 | 0.7 | 0.5 | 0.8 |
| Prefix groups | 前缀组数 G | 100 | 1000 | 200 |
| Zipf alpha | 前缀组流行度的 Zipf 指数 α | 1.3 | 0.9 | 1.1 |
| Workload shape | `shared_prefix_groups` 或 `prompt_lineage_bootstrap` | 前者 | 前者 | 前者 |
| Arrival model | `fixed` 或 `poisson` | fixed | fixed | fixed |
| Seed | 随机种子 | 1 | 1 | 1 |

表中三列是随附的负载预设（`configs/workloads/`）。界面中修改的值会覆盖预设。

### 算法

Bandwidth Envelope 只生成 1 个请求：t = 0，长度 N，H = H_cfg = ⌊N × Prefix ratio⌋。上表中除 Context lengths 和 Prefix ratio 外的参数都不使用。

Cache Replay：

```text
H_cfg = ⌊N × Prefix ratio⌋                     所有请求相同
前缀组权重 w_k = 1 / k^α，k = 1..G
对第 i 个请求（i = 0..n−1）：
    若 H_cfg > 0：按权重 w 随机抽取一个前缀组（Python random.choices，种子为 Seed）
    否则：每个请求单独成组，相互之间没有共享
    到达时间：fixed    t_i = i / QPS
             poisson  t_0 = 0，t_i = t_{i−1} + Exp(QPS)（种子为 Seed + 1）
```

两种 Workload shape：

- `shared_prefix_groups`：同组请求共享前 ⌊H_cfg / K⌋ 个完整块（§4）。
- `prompt_lineage_bootstrap`：先为每个前缀组生成一个 bootstrap 请求（长度 H_cfg，不生成 token，到达时间为负）。它们在第一个正式请求之前全部完成并写入缓存，不计入测量。正式请求以本组的 bootstrap 请求为父请求，继承它的输出对象（§4）。

### 输出

请求列表：到达时间、N、生成长度、前缀组、H_cfg、父请求。

### 假设与适用范围

- 所有请求的 N、生成长度和 H_cfg 都相同，没有长度分布，也不模拟多轮对话中上下文逐轮增长。
- 前缀组的流行度服从 Zipf 分布；同组请求的前缀完全相同。
- 不能导入真实 trace。
- tenant 字段按轮转分配，但不参与块身份，不模拟租户隔离。

### 代码位置

`ssu_sim/workload/synthetic.py` · `generate_synthetic_workload`；`ssu_sim/utils/random.py` · `zipf_weights`、`weighted_choice`；`dashboard_v2/server.py` · `_run_demand_case`。

<a id="block-identity"></a>
## 4. 前缀块身份

### 输入

请求列表、块大小 K。

### 算法

块 ID 是合成身份，不是对真实 token 内容做哈希：

```text
block_id(i) = SHA-256("模型 | 命名空间 | 块号 i | 父块 ID 或 root") 的前 24 个十六进制字符
```

父块 ID 参与计算，所以同一块号在不同前缀下是不同对象，一个请求的块序列构成一条路径。

一个请求的输入路径有 ⌈N / K⌉ 个块：

- 前 ⌊H_cfg / K⌋ 个完整块使用前缀组命名空间，同组请求得到相同的 ID，可以共享；
- 其余块（包括最后一个不满的块）使用请求私有命名空间，不与其他请求共享。

| 可共享长度（Shareable）的来源 | 计算 | 说明 |
| --- | --- | --- |
| `complete_shared_blocks`（`shared_prefix_groups`） | ⌊H_cfg / K⌋ × K | 不满一块的前缀尾部不能共享 |
| `parent_lineage`（`prompt_lineage_bootstrap`） | min(H_cfg, 父请求的输入长度 + 生成长度) | 子请求直接复用父请求的输出对象 ID，可以包括封存的不满块 |

不满块的存储策略 `partial_tail_storage`：

- `sealed_padded`（默认）：不满块按整块存储，作为不可变对象。后续请求在其后追加 token 时生成新版本的对象，旧对象对并发读者仍然有效。
- `complete_blocks_only`：只存储完整块。

生成的 token（decode）在输入路径之后追加使用 `generated:<请求>` 命名空间的对象。

### 输出

每个请求的输入路径、查找路径、输出路径，以及 Configured / Shareable 长度。

### 假设与适用范围

- 身份由（模型、前缀组、块号、父块）决定，相当于假设同组请求的前缀 token 完全一致。
- 不模拟哈希冲突，也不按租户隔离对象。

### 代码位置

`ssu_sim/prefix/block_hasher.py` · `make_block_id`、`build_request_block_ids`、`canonicalize_request_paths`、`_append_immutable_objects`。

<a id="capacity"></a>
## 5. 容量模型

### 5.1 三种口径

| 口径 | 用在哪里 | 含义 |
| --- | --- | --- |
| 参考容量（calculator reference） | KV Size 页 | 按模型 YAML 中 `calculator_reference` 的字段，用统一精度计算的容量；方程与外部参考计算器一致 |
| 前缀存储（序列化块） | Explore、Replay、Recommend、SSD 仿真 | 实际要存储、读取、写入的对象字节：按块、按组件，使用真实的混合精度、scale 和 padding 序列化 |
| 活跃状态（active footprint） | Case 详情的 Footprint view | 正在计算的序列需要保有的状态，例如滑动窗口和单份递归状态 |

多数标准模型三种口径的数值相同。在混合精度（例如 DeepSeek V4）、滑动窗口和 checkpoint 状态上，三者有意不同，不能互相替代。

### 5.2 序列化组件（Explore 口径）

每个模型由若干 cache 组件构成，例如 K、V、latent、RoPE key、indexer、压缩条目。组件 c 的大小：

```text
每层每条目字节  e_c = ⌈E_c × bits_c / 8⌉ + ⌈E_c / G_c⌉ × s_c + pad_c    （配置中也可以直接给定 e_c）
每 token 字节       = L_c × e_c / r_c
每块字节            = L_c × (K / r_c) × e_c                              （K 必须是 r_c 的整数倍）
```

E_c 为每层每条目的元素数，bits_c 为精度位数，G_c 为量化组大小，s_c 为每组 scale 字节数，pad_c 为填充字节，L_c 为存储层数，r_c 为压缩比（普通 KV 为 1）。存储层数可以少于计算层数，例如跨层共享 KV 的模型。

读写都按块计。命中 H 个 token 时读取 ⌈H / K⌉ 块（`sealed_padded`）或 ⌊H / K⌋ 块（`complete_blocks_only`），每块按整块字节计，因此不满块也按整块读取。

### 5.3 十个公式族的参考容量（KV Size 页）

记 T 为 token 数，S 为序列数，p 为主缓存每元素字节数，p_i 为 indexer 每元素字节数。括号中是模型 YAML 中 `calculator_reference.fields` 的字段名。

```text
standard_gqa（MHA / GQA / MQA）
    KV = T·S·(L + L_draft)·Hkv·2·d·p
    L = num_hidden_layers，Hkv = num_key_value_heads，d = head_dim
    L_draft（勾选 draft 时）= num_nextn_predict_layers + num_mtp_modules × mtp_transformer_layers

mla
    MLA = T·S·(L + L_nextn)·(r_kv + d_rope)·p
    r_kv = kv_lora_rank，d_rope = qk_rope_head_dim；L_nextn 仅在勾选 draft 时计入

dsa_mla
    MLA     = T·S·(L + L_nextn)·(r_kv + d_rope)·p
    indexer = T·S·L_idx·d_idx·p_i
    L_idx = indexer_full_layers（缺省为 L），勾选 draft 时另加 draft_indexer_layers；d_idx = index_head_dim

kimi_kda_mla_hybrid
    MLA      = T·S·L_mla·(r_kv + d_rope)·p                  L_mla = full_attention_layers
    KDA 状态 = S·n_ckpt·(conv·b_conv + rec·b_rec)            （勾选 state 时）
    conv = L_kda·(k_conv − 1)·(H·d + Hk·dk + Hv·dv)
    rec  = L_kda·Hv·dv·dk
    k_conv = kda_conv_kernel_size；b_conv、b_rec 为状态的固定精度字节数

inkling_hybrid
    global     = T·S·L_full·Hkv·2·d·p
    sliding    = min(T, W)·S·L_swa·Hkv_swa·2·d_swa·p          W = sliding_window
    SConv 状态 = S·n_ckpt·(k − 1)·[L_full·(2·Hkv·d + 2·D) + L_swa·(2·Hkv_swa·d_swa + 2·D)]·b_sconv

qwen_linear_full_hybrid（full attention + Gated DeltaNet）
    full     = T·S·L_full·2·Hkv·d·p
    GDN 状态 = S·(2·conv + 4·rec)                             （勾选 state 时；只计一份，不随 checkpoint 间隔复制）
    conv = L_lin·k·(2·Hk·dk + Hv·dv)，rec = L_lin·Hv·dk·dv    （卷积历史按 k 计，而 KDA、SConv 按 k − 1 计）

mixed_full_sliding_gqa
    full    = T·S·L_full·Hkv_full·(dk_full + dv_full)·p
    sliding = min(T, W)·S·L_swa·Hkv_swa·(dk_swa + dv_swa)·p

minimax_msa
    main    = T·S·L·2·Hkv·d·p                                 所有层都存主 KV
    indexer = T·S·L_sparse·d_idx·p_i                          只有稀疏层存 indexer

deepseek_v4_hybrid（r_l 为第 l 层的压缩比：0 = 纯滑动窗口，4 = CSA，128 = HCA）
    main_kv = S·[L_active·W·d + Σ_{r_l > 0} ⌊T / r_l⌋·d]·p   每个计入的层固定计 W 个窗口槽位
    indexer = S·Σ_{r_l = 4} ⌊T / 4⌋·d_idx·p_i
    L_active 为计入的层数：不勾选 draft 时为主干层数，勾选时加上 draft 层；W = sliding_window

deepseek_v41_csa2（只支持 (p, p_i) = (0.5, 0.5) 或 (2, 2)）
    e_main = d·p + (FP4 时另加 d/16)，e_idx = d_idx·p_i + (FP4 时另加 d_idx/32)
    e_swa  = d + d/32（FP4）或 2·d（BF16）
    Encoder 全局 KV 与 indexer = S·3·⌊T/2⌋·(e_main + e_idx)    3 个 Encoder owner 层按 2:1 压缩
    Decoder 全局 KV 与 indexer = S·T·(e_main + e_idx)          1 个 Decoder owner 层不压缩
    勾选 state：SWA = S·40·min(T, W)·e_swa（W = sliding_window = 128）；压缩尾 = S·3·2·d·4·(T mod 2)；
               Encoder 输出保留（retain）= S·min(T, W)·(hc·D·2 + hc·4)
    勾选 draft：S·3·min(T, W)·e_swa

n_ckpt：T = 0 时为 0；checkpoint 间隔为 ∞（默认，只在 prompt 末尾存一份）时为 1；否则为 ⌈T / I⌉。
```

KV Size 页在浏览器中用 JavaScript 计算（`dashboard_v2/js/kv_size.js` · `calculateKvSize`），Python 参考实现是 `ssu_sim/models/calculator_reference.py` · `calculate_reference_footprint`，两者是同一组方程。逐模型对拍的结果见[第 13.2 节](#checks)。

### 5.4 已核对的数值

| 模型（Runtime） | 口径 | 数值 |
| --- | --- | ---: |
| Llama 3.1 8B（bf16_reference） | 每 token KV | 131,072 B |
| Qwen3-8B（bf16_reference） | 每 token KV | 147,456 B |
| DeepSeek V3（bf16_reference） | 每 token MLA | 70,272 B |
| Kimi K3（bf16_reference） | 每 token MLA / 每份 KDA 状态 | 27,648 B / 449,372,160 B |
| DeepSeek V4 Pro（v4_fp8_mxfp4） | 每 256-token 块 | 1,284,208 B |
| DeepSeek V4 Flash（v4_fp8_mxfp4） | 每 256-token 块 | 896,960 B |
| DeepSeek V4.1 Flash（v41_fp4） | 每 64-token 块 | 56,960 B |

DeepSeek V4 在 Explore 中使用原生的混合精度序列化：主条目 584 B（448 B FP8 NoPE + 64 个 BF16 RoPE 元素 128 B + 7 B scale + 1 B padding），CSA indexer 条目 66 B；CSA 层压缩比 4，HCA 层压缩比 128。KV Size 页按统一精度计算，因此两处数值不同。V4.1 原生主条目 288 B、indexer 68 B、SWA 528 B；全局 KV 只存在于 4 个 owner 层（0-based 第 2、8、14 层按 2:1 压缩，第 20 层不压缩）。

### 5.5 Explore 中的容量指标

- **Reusable prefix bytes**：对每个（模型，前缀组）取出现过的最大 H，按块计容量，再求和。
- **Cumulative unique write bytes**：全部 miss 写入和 decode 写入的字节之和。
- **Retain-all 上界** = 以上两项之和，相当于假设什么都不驱逐。
- Replay 中的实际驻留量来自 §8 的 radix 状态（used / protected / evictable）。

### 代码位置

`ssu_sim/models/attention.py` · `CacheComponent`、`_prefix_block_indices`、各 `*AttentionProfile`；`ssu_sim/models/calculator_reference.py` · `calculate_reference_footprint`；`dashboard_v2/js/kv_size.js` · `calculateKvSize`；`ssu_sim/demand/analyzer.py` · `analyze_workload`（容量汇总）。

<a id="compute"></a>
## 6. 计算量模型

### 6.1 计数约定

- 一次乘加计 2 FLOPs。只计模型 YAML 中声明了几何参数的运算。
- 因果注意力逐对精确计数。位置 i（从 1 开始）的 token 与自身及之前所有 token 计算注意力。已有上下文 s、新计算 q 个 token 时，注意力对数为：

```text
P(s, q)   = Σ_{i=s+1}^{s+q} i = (s+q)(s+q+1)/2 − s(s+1)/2
带上限 W  ：P_W(s, q) = Σ_{i=s+1}^{s+q} min(i, W)              （滑动窗口、Top-k）
压缩条目  ：Σ min(⌊i / r⌋, cap)                                （压缩比 r 的 KV）
```

- 逐元素运算计入 `elementwise` 类，并按同一算力换算时间。它包括 softmax 中的减法、求和与除法，以及 RMSNorm、RoPE、门控乘法和激活函数中的算术部分。
- exp、tanh、rsqrt、max 比较、top-k 选择等特殊操作只记录次数（`special_operation_counts`），不折算为 FLOPs，也不计时间。
- 未建模的工作列在 `counting_omissions` 中，例如 kernel 启动、通信、量化取整。

### 6.2 三个量

```text
F_nocache = F(q = N, s = 0) + F_head
F_cache   = F_replay(q = R, s = B_r) + F_suffix(q = N − H, s = H) + F_conv + F_head × [N − H > 0 或 R > 0]
净节省     = F_nocache − F_cache                                   （可以为负）
```

- **F_replay**：从恢复边界 B_r 重算到 H，只出现在有状态模型中（§7）。
- **F_conv**：必要的格式转换。MLA 以 latent 形式恢复前缀、且 suffix 非空时，要把恢复的前缀 latent 展开为 K/V：2 × B_r × r_kv × Hq × (d_nope + d_v) × L。只恢复 KV 的模型 B_r = H。
- **F_head**：LM head，按 1 个 token 计：2 × D × vocab，再加最后一层 norm。46 个随附模型在配置中声明了词表大小，按此计入；DeepSeek V4 / V4.1 的 3 个模型在各自的分项账本中计入；其余 7 个（Cohere 4 个、Llama 3 个）未声明词表，不计 LM head，并在 `counting_omissions` 中注明 `output_head_geometry_not_declared`。
- **完整命中**：H = N 且无需重算时，缓存路径中没有任何计算能产生最后一个 token 的 hidden state 或 logits。程序不臆造这部分工作，把结果标为 `not_evaluated`，此时 TTFT 和带宽阈值为 null。

### 6.3 各族公式

以下为一段计算（已有上下文 s、新 token 数 q）的 FLOPs，批大小为 1。

```text
Dense attention（GQA / MHA / MQA）
    投影     = 2·q·D·[(Hq·dk·(1 + g) + Hq·dv)·L + Hkv·(dk + dv)·L_kv]     g = 1 表示有 query gate
    注意力   = 2·P(s,q)·Hq·(dk + dv)·L
    缩放与 softmax（逐元素）= 4·E − rows                             E = P(s,q)·Hq·L，rows = q·Hq·L

MLA（默认 materialized 算法）
    投影     = 2·q·[D·(r_q + r_kv + d_rope) + r_q·Hq·(d_nope + d_rope) + Hq·d_v·D + r_kv·Hq·(d_nope + d_v)]·L
    注意力   = 2·P(s,q)·Hq·(d_nope + d_rope + d_v)·L
    absorbed 算法的注意力 = 2·P(s,q)·Hq·(2·r_kv + d_rope)·L

DSA（MLA + 稀疏 indexer）
    主注意力按 P_topk(s,q) 计，topk = index_topk
    indexer 打分 = L_idx·P(s,q)·(2·h_i·d_i + h_i + (h_i − 1))           按完整因果对计
    indexer 投影 = 2·q·L_idx·(q_in·h_i·d_i + k_in·d_i + D·h_i)

FFN
    Dense FFN = 2·q·D·I·(gated 为 3，否则为 2)·L_ffn
    MoE       = 6·q·k·D_e·I_e·L_moe（k 个路由专家）+ 6·q·n_s·D_s·I_s·L_moe（共享专家）+ 2·q·D·E·L_moe（router）
```

MoE 只计被激活的专家。FFN 另加激活函数和专家输出合并的逐元素运算。

其余公式族的注意力部分：

| 公式族 | 计数方式 |
| --- | --- |
| mixed_full_sliding_gqa | full 层按 P(s,q)，sliding 层按 P_W(s,q)，各层使用自己的头数和维度 |
| inkling_hybrid | full 层按 P，sliding 层按 P_W；另加短卷积 SConv：2·q·k·channels |
| minimax_msa | full 层按 P；稀疏层按块稀疏 Top-k 的对数计，当前不满块占用一个 Top-k 名额；indexer 打分按完整因果对计 |
| qwen_linear_full_hybrid | full 层按 P；Gated DeltaNet 线性层按公开的分块参考算法计数，工作量只与 q 有关，与前缀长度无关；另加短卷积 |
| kimi_kda_mla_hybrid | MLA 层同 MLA；KDA 层按 FLA 参考分块算法（逐通道衰减）计数 |
| deepseek_v4_hybrid | 分项核算投影、压缩、indexer、CSA 和 HCA 注意力（按压缩条目数）、SWA（窗口 128）、MoE 与路由、mHC、RoPE |
| deepseek_v41_csa2 | 按 CED 四个阶段（§7.4）分别核算 Encoder CSA2、Decoder indexer 与重索引、Engram、mHC、MoE、LM head 等 |

### 6.4 时间

```text
t = F ÷ (C × 10^6) µs
```

- 四个随附 Runtime 都是 C = 2000 TFLOPS，`timing_mode = normalized_reference`。56 个随附模型的 `timing_calibration` 均为 `uncalibrated`。
- 代码支持 `component_calibrated` 模式：为 11 个运算类别分别设置吞吐，时间按类别相加。类别为 dense_gemm、dense_attention、sparse_attention、indexer、convolution、recurrent、moe、routing、compression、elementwise 和 unclassified。只要缺少任何一类的吞吐，整个请求就退回归一化时间并给出警告。随附配置没有使用这种模式。
- 随附模型的固定开销 `scheduler_overhead_us` 和首 token decode `decode_us_per_token` 都为 0。

算例：Qwen3-8B（D = 4096，L = 36，Hq = 32，Hkv = 8，d = 128，gated FFN I = 12288），N = 32,768：

| 部分 | FLOPs |
| --- | ---: |
| 投影 | 0.990 × 10^14 |
| 注意力 | 3.167 × 10^14 |
| FFN | 3.562 × 10^14 |
| 逐元素与 LM head | 0.026 × 10^14 |
| 合计 F_nocache | 7.745 × 10^14 |

按 2000 TFLOPS 计为 387.3 ms。

### 代码位置

`ssu_sim/models/compute_counts.py` · `causal_entry_sum`、`causal_pairs`、`block_sparse_pairs`、`delta_rule_work`、`softmax_work`；`ssu_sim/models/llm_model.py` · `_OperationFormulaMixin.cache_compute_breakdown`、`_standard_attention_breakdown`、`_mla_attention_breakdown`、`DSAPrefillFormula`、`DenseFFNProfile`、`MoEFFNProfile`、各 `*PrefillFormula`、`ComputeProfile`（`_with_output_head`、`estimate_timing`、`_flops_to_us`）；`ssu_sim/models/deepseek_compute.py`。

<a id="recovery"></a>
## 7. 有状态模型的恢复

### 7.1 前缀的四个阶段

```text
Configured（H_cfg）→ Shareable（§4）→ Resident（H，§8）→ Recoverable（B_r）
```

只恢复 KV 的模型 B_r = H。含有非 token 线性状态的模型（Gated DeltaNet、KDA、SConv、滑动窗口快照、V4 的 SWA 等）可能只能从更早的 checkpoint 继续，需要重算 R = H − B_r 个 token。

### 7.2 状态持久化策略

| storage_form | 含义 |
| --- | --- |
| `token_history` | 按 token 存全部历史（例如滑动窗口 KV 存全部 token），恢复时按块读取 |
| `snapshot` | 在边界处保存完整状态快照：设置间隔 I 时每 I 个 token 一份；不设间隔时只在 prompt 末尾存一份 |
| `none` | 不持久化 |

`retain_prompt_end` 指定 `snapshot` 是否在 prompt 末尾额外存一份。区间 (P, E] 内新产生的 checkpoint 边界：

```text
{I, 2I, 3I, ...} ∩ (P, E]，若 retain_prompt_end 为真且最后一个边界不是 E，再加上 E
（不设间隔时只有 E）
```

同一 checkpoint group 内的多个张量（例如卷积状态和递归状态）作为一个原子对象准入和恢复。

### 7.3 恢复边界与重算

```text
B_r = max( ∩_g（组 g 的可用边界） ∪ {0} )
R   = H − B_r：从 B_r 到 H 做一次整模型前向
```

- Envelope：可用边界是按策略对长度为 H 的前缀会产生的全部边界。
- Replay：可用边界是实际驻留在命中路径节点上、且不超过 H 的 checkpoint（§8）。
- 任何一个必需组的策略为 `none`，或各组没有共同边界时，B_r = 0。例如组 A 有 {64, 128}、组 B 有 {96, 192}，没有共同边界，B_r = 0。

IO 按 B_r 重新规划：

- token 类组件（KV、latent 等）只读取 [0, B_r) 对应的块。[B_r, H) 中已驻留的部分会被重算覆盖，不读取，记为 stranded resident bytes；
- 每个状态组读取 B_r 处的快照（一个连续 extent）。这些读取是阻塞的，必须在重算开始前到达。

### 7.4 执行阶段

普通模型的阶段顺序：

```text
（必要转换，仅当 R > 0 且需要转换）→ replay（重算 R 个 token）→ suffix（N − H 个 token）→（完整命中收尾）→ LM head
```

每个读取组件标注首次使用它的阶段：R > 0 时，token 组件和状态快照都由 replay 阶段使用；R = 0 时由 suffix 阶段使用。可选读取（例如 draft 缓存）标为 non_blocking，不进入关键路径。

DeepSeek V4.1 使用自己的四个阶段：

| 阶段 | 计算范围 |
| --- | --- |
| encoder_restore | 重算命中前缀尾部 R 个 token 的 Encoder（前 20 层） |
| encoder_suffix | 新 token 的 Encoder（前 20 层） |
| decoder_global_projection | 新 token 的全局投影（第 20 层） |
| decoder_tail | 最后至多 128 个 token 的 Decoder（后 20 层） |

满足以下任一条件时 R = min(H, 128)，否则 R = 0：

- Encoder 状态不可用（`encoder_state = unavailable`）且 H > 0；
- Decoder 尾部的 128 个 token 与命中前缀重叠，且不保留 Encoder 输出（`encoder_tail_policy = recompute`）。

### 代码位置

`ssu_sim/models/state.py` · `StatePersistencePolicy`、`checkpoint_boundaries`、`RestoreExecutionPlan`；`ssu_sim/models/attention.py` · `_checkpoint_read_plan`、`DeepSeekV41AttentionProfile.recovery_plan`；`ssu_sim/demand/analyzer.py` · `_apply_actual_checkpoint_restore`；`ssu_sim/cache/replay.py` · `_actual_recovery_boundary`。

<a id="replay"></a>
## 8. 缓存回放（Cache Replay）

### 输入

请求列表（§3）、块路径（§4）、每块字节数（§5）、缓存配置：总容量（字节）、分片数 n、Placement seed、Warmup fraction（默认 0）；Persist（是否写入 miss 块）、decode 写入。

### 数据结构

每个分片一棵块粒度的前缀树。一个节点对应一个块对象，节点大小 = 该块的序列化字节（§5.2）+ 挂在该节点上的 checkpoint 快照字节。

```text
分片容量：总容量按 divmod 分给 n 个分片，余数分给前几个分片，每个分片多 1 字节
路径所在分片 = SHA-256("seed:第一个块 ID") 的前 8 字节 mod n        整条路径在同一分片
```

### 算法

```text
请求按 (到达时间, request_id) 排序；待完成事件放入按 (完成时间, 序号) 排序的堆
for 每个请求：
    先处理所有完成时间 ≤ 该请求到达时间的完成事件
    阶段：bootstrap（全部在第一个正式请求之前处理完）
          warmup（前 ⌊正式请求数 × Warmup fraction⌋ 个正式请求）
          measured（其余请求）
    查找：沿候选路径（前 ⌈Shareable/K⌉ 或 ⌊Shareable/K⌋ 个块）从根逐块匹配，遇到第一个不驻留的块即停止
          对匹配到的节点加锁（受保护，不能驱逐）
          H   = min(匹配块数 × K, Shareable)
          B_r = 命中节点上的 checkpoint 决定（§7.3）
    完成时间 = 到达时间 + 缓存路径计算时间（§6，不含存储读取时间）
    登记完成事件

完成事件（按完成时间顺序处理）：
    Persist 开启：从 min(与候选路径的共同前缀, 命中块数) 处开始插入输入路径中的新块
                  已存在的同一对象计为去重，不重复写
                  为 (B_r, N] 中按策略产生的边界准入快照，挂在拥有该 token 的节点（第 ⌈b/K⌉ − 1 块）上
    decode 写入开启：插入生成 token 的块和相应快照
    释放查找时加的锁
    驱逐：只要分片已用字节 > 分片容量，就删除最久未访问、未加锁的叶子节点
          （访问时间相同时按创建序号）；父节点变成叶子后进入候选
```

规则：

- 生产者完成之前，后续请求不能命中它新写的块，不模拟“等待在途写入”。
- 单个块大于分片容量时，拒绝该块及其后续块。
- 所有叶子都被锁住、仍然超过容量时报错，不会静默超配。
- checkpoint 与 token 块共用分片容量。checkpoint 不单独驱逐，随所在叶子一起删除；容量不足时拒绝准入该快照，而不是删除所在节点。
- 未命中分类：缺失路径曾经被准入过，计为容量缺失（capacity miss）；否则计为冷缺失（compulsory miss）。

### 输出

| 指标 | 定义 |
| --- | --- |
| Configured / Shareable / Resident / Recovery boundary ratio | 各阶段的 token 总数 ÷ 测量请求的输入 token 总数 |
| Block / Byte hit rate | 命中块数（字节）÷ 可缓存块数（字节） |
| Inserted / Deduplicated / Rejected / Evicted / Self-evicted | 准入、去重、拒绝、驱逐的块；Self-evicted 指驱逐了本次刚写入的块 |
| used / protected / evictable | 某一时刻的快照，同一时刻 protected + evictable = used；三者的峰值分别统计，不能相加 |

只有测量阶段的请求进入后续的 IO 需求和时间计算（§9、§10）；bootstrap 和 warmup 请求只影响缓存状态。

### 假设与适用范围

- 开环：完成时间不含存储读写时间，SSD 仿真的结果不会改变回放过程。
- LRU 的“时间”是逻辑访问序号，不是物理时间。
- 容量按序列化字节计，不含分配器和文件系统开销。

### 代码位置

`ssu_sim/cache/config.py` · `CacheReplayConfig`；`ssu_sim/cache/radix.py` · `RadixLeafLRUCache`（`match_path`、`insert_path`、`attach_checkpoint`、`enforce_capacity`）、`stable_cache_shard`；`ssu_sim/cache/replay.py` · `replay_prefix_cache`。

<a id="io-demand"></a>
## 9. IO 需求

### 输入

每个请求的 H、B_r、N、生成长度、到达和完成时间；块大小 K；Max IO；Persist 与 decode 写入选项。

### 9.1 读取

每个请求在**到达时刻**产生读取，字节 = 每个组件读取的块数 × 每块字节：

- 普通 KV：⌈B_r / K⌉ 块（`complete_blocks_only` 时为 ⌊B_r / K⌋），每块按整块计（§5.2）。只恢复 KV 的模型 B_r = H。
- 状态快照：B_r 处一个 extent（§7.3）。
- 稀疏模型（DSA、MiniMax MSA）默认 `full_restore`，读取完整前缀。选择 `sparse_on_demand` 时，主 KV 只读选中的块（按随机访问计，SSD 仿真中受随机 IOPS 限制），indexer 仍完整读取一次。选中哪些块由局部性场景决定（`window_global`、`clustered_lower`、`seeded_uniform`、`dispersed_upper`），不是模型运行时真实的 Top-k。
- **阻塞恢复字节**：关键路径上必须先到达的读取，不含可选的 draft 读取。

### 9.2 写入（Replay，Persist 开启）

在**完成时刻**写入新块：

```text
写入块号区间 = [⌊P / K⌋, ⌈(P + M) / K⌉)          P 为已有前缀长度，M 为新 token 数
```

suffix 从一个不满块中间开始时，会重写这个块（`sealed_padded`）。Replay 中只写实际新准入的块（一段连续区间）；其他请求已写入的同一对象计为去重。

decode 写入：生成的 token 按同样规则写入，时刻为“到达 + 缓存路径计算时间 + 生成长度 × decode_us_per_token”。随附模型 decode_us_per_token = 0，即完成时刻。

### 9.3 切分与 IOPS

```text
每块分片数    = ⌈块字节 / Max IO⌉                未设置 Max IO 时不计算 IOPS
chunked IOPS = Σ(块数 × 每块分片数) ÷ 时间基准
```

### 9.4 平均速率与峰值窗口

```text
时间基准 T = 测量请求数 ÷ QPS            Envelope 为 1 s；poisson 到达也使用这个名义时长
平均带宽   = 总字节 ÷ T
窗口峰值   = max_t（时刻落在 (t − W, t] 内的事件字节之和）÷ W      W ∈ {1 ms, 10 ms, 100 ms, 1 s}
峰均比     = 窗口峰值 ÷ 平均带宽                                    平均带宽为 0 时无定义（null）
```

事件时刻：读取记在请求到达时刻，miss 写入和 decode 写入记在完成时刻。事件是瞬时的，读取没有在传输时间内摊开，所以 1 ms、10 ms 这样的短窗口主要反映单个请求一次要读多少，而不是设备需要持续提供的速率。按截止时间摊开的总要求见 §10.5。

### 代码位置

`ssu_sim/demand/analyzer.py` · `analyze_bandwidth_envelope`、`analyze_workload`、`_chunk_count`；`ssu_sim/models/attention.py` · `_write_block_indices`、各 profile 的 `map_hit_to_reads` / `map_miss_to_writes`；`ssu_sim/demand/report.py` · `_operation_report`、`_peak_window`。

<a id="ttft"></a>
## 10. TTFT 与带宽阈值

### 输入

每个请求的阶段列表：阶段 j 的计算时间 c_j（§6）以及它首次使用的读取组件和字节（§9）；固定开销 O；首 token decode 时间 D。

O = scheduler_overhead_us + latent 物化时间，只在请求有恢复读取时计入；D = decode_us_per_token。随附配置中两者均为 0。

### 10.1 三个 TTFT

```text
无缓存 TTFT                  T_0 = t(F_nocache) + D
缓存路径 TTFT（读取时间为 0） T_c = O + Σ c_j + D
TTFT saving（Explore 显示）   = T_0 − T_c
```

TTFT saving 不计任何存储时间，是节省时间的上界。

### 10.2 读取截止时间

首次在阶段 j 使用的组件，必须在以下时刻（相对请求到达）之前读完，缓存路径才不比无缓存路径慢：

```text
d_j = T_0 − O − D − Σ_{i ≥ j} c_i
```

截止时间相同的组件合并为一组；各组按首次使用的顺序排列。

### 10.3 两个带宽阈值

```text
Optimistic fully-hidden bandwidth     = 阻塞恢复字节 ÷ (Σ c_j − O)
Dependency-aware break-even bandwidth = max_j（截止时间不晚于 d_j 的各组累计字节 ÷ d_j）
```

只恢复 KV 的模型只有一组，d = T_0 − T_c，即省下的计算时间。有状态模型的 checkpoint 由 replay 阶段使用，截止时间更早。

| 显示 | 条件 |
| --- | --- |
| Not evaluated | 完整命中且收尾未建模（§6.2），或 Case 因不支持的组合未完成评估 |
| No overlap window | optimistic：有阻塞恢复字节，但 Σ c_j − O ≤ 0 |
| No finite break-even | dependency：无限带宽下仍没有正的时间节省（`no_compute_saving`），或某组截止时间 ≤ 0（`no_replay_slack`） |
| optimistic = 0 | 已完整评估且没有阻塞恢复数据（`no_restore_required`） |
| dependency = 0 | 有正的时间节省且没有阻塞恢复数据 |

| 阈值 | 回答的问题 |
| --- | --- |
| Optimistic fully-hidden | 读取与计算完全重叠时，多大带宽能让读取完全不增加 TTFT |
| Dependency-aware break-even | 每个阶段都等它的数据读完再开始时，多大带宽下使用缓存不比全部重算慢 |

带宽低于 break-even 时，按依赖顺序执行的缓存路径比不使用缓存更慢。实际系统的重叠程度通常介于两种假设之间。

### 10.4 额定带宽下的依赖 TTFT（Recommend 使用）

```text
各组按首次使用的顺序在一条读通道上依次读取：ready_g = 累计字节 ÷ (BW × 1000) µs
e = 0；对每个阶段 j：e = max(e, 该阶段所需各组的 ready) + c_j
TTFT_dep = O + e + D，节省 = T_0 − TTFT_dep
```

每个请求独占这条读通道，请求之间不排队。BW 取存储配置的额定顺序读带宽；Case 的结果取按请求数加权的 p50。

### 10.5 聚合依赖带宽与 IOPS

```text
对每个请求：把每组字节在 [上一组的截止时间, 本组的截止时间] 内均匀摊开（第一组从到达时刻开始）
把所有请求的速率按时间叠加，取整个 Case 的最大值 → aggregate_dependency_required_GBps
IOPS 用分片数代替字节，同样计算 → aggregate_dependency_chunked_iops（需设置 Max IO）
截止时间 ≤ 0 的请求不计入，数量记为 invalid_window_count
```

Envelope 只有一个请求，这些值就是该请求自身的要求。Replay 中它们反映 QPS 和请求重叠带来的总压力，是 Explore 中唯一按截止时间计算、并计入并发的带宽和 IOPS 要求。

### 10.6 三处 TTFT 的差别

| 位置 | 恢复读取的处理 | 仍不包含 |
| --- | --- | --- |
| Explore（TTFT 页、Timing 页） | 读取时间按 0 计 | 存储带宽和排队 |
| Recommend | 按额定读带宽读取；每个阶段等它的数据读完才开始（§10.4） | 请求之间的排队；每个请求独占带宽 |
| SSD Simulation | 当前 Case 全部请求的 IO 在一块模拟 SSD 上排队（§11） | 主机、网络和 GPU 侧的传输 |

### 10.7 算例：Qwen3-8B

设置：Qwen3-8B，Bandwidth Envelope，N = 32,768，bf16_reference，C = 2000 TFLOPS，块大小 256，Max IO 131,072 bytes。7 GB/s 是随附存储配置的额定读带宽。

| 项目 | 50% 命中 | 90% 命中 |
| --- | ---: | ---: |
| 无缓存 TTFT | 387.3 ms | 387.3 ms |
| 缓存路径 TTFT（读取时间为 0） | 233.5 ms | 53.1 ms |
| TTFT saving（Explore 显示） | +153.7 ms | +334.2 ms |
| 净节省的计算比例 | 39.7% | 86.3% |
| 阻塞恢复字节 | 2.42 GB | 4.38 GB |
| Optimistic fully-hidden bandwidth | 10.3 GB/s | 82.5 GB/s |
| Dependency-aware break-even bandwidth | 15.7 GB/s | 13.1 GB/s |
| 7 GB/s，每个阶段等数据读完（§10.4） | 578.6 ms，慢 191.4 ms | 678.6 ms，慢 291.4 ms |
| 7 GB/s，读取与计算完全重叠 | 345.1 ms，快 42.1 ms | 625.6 ms，慢 238.3 ms |

- 验算：2.42 GB ÷ 233.5 ms ≈ 10.3 GB/s；2.42 GB ÷（387.3 − 233.5）ms ≈ 15.7 GB/s。
- 50% 命中只省约 40% 的计算：suffix 中每个 token 仍要与全部前缀做注意力，长上下文下后半段的 token 更贵。命中率不等于计算节省比例。
- 7 GB/s 低于两种命中率的 break-even，按依赖顺序执行时使用缓存都更慢。只看 TTFT saving 不能判断缓存是否划算。

### 代码位置

`ssu_sim/models/state.py` · `shared_read_ready_times`、`phase_chain_time_us`、`phase_read_deadlines`、`phase_bandwidth_threshold`；`ssu_sim/demand/analyzer.py` · `analyze_workload`（每个请求的阶段与读取组）、`_component_read_deadlines`；`ssu_sim/demand/report.py` · `_ttft_report`、`_bandwidth_threshold_report`、`_aggregate_dependency_pressure`、`rated_phase_timing_report`。

<a id="ssd"></a>
## 11. 单 SSD 仿真

### 输入

一个 Replay Case（分片数必须为 1）的逻辑 IO 轨迹，以及存储配置。交互运行的上限为 20 万个逻辑操作、200 万个物理 IO。

随附存储配置 `configs/storage/single_nvme_ftl.yaml`：

| 参数 | 值 |
| --- | --- |
| 页 / 擦除块 | 16 KiB / 256 页 |
| 通道 × die × plane | 8 × 2 × 1 = 16 个并行单元 |
| 主机队列深度 | 64 |
| 读 / 编程 / 擦除时延 | 85 / 450 / 3500 µs |
| 额定顺序读 / 写带宽 | 7 / 5 GB/s |
| 随机读 / 写 IOPS | 1,000,000 / 250,000 |
| 逻辑容量 | 1 TiB |
| 预留空间（OP） | 20% |
| GC 触发的空闲页比例 | 5% |
| 默认 IO 单位 | 128 KiB |

这些是研究用的输入参数，不是某型号 SSD 的规格。

### 11.1 轨迹

- 读：请求到达时刻，每个块一个逻辑操作。截止时间 = 到达 + 该组件的读取截止时间（§10.2）。
- 写：完成时刻。设置 Decode step 时，decode 写入按“到达 + 计算时间 + 已生成 token 数 × step”分布。

### 11.2 LBA 分配与切分

```text
所有对象按 SHA-256("seed:对象 ID") 排序，依次连续分配 ⌈对象字节 / 页大小⌉ 个页      不重叠，不回收
分片大小 = min(Max IO, IO 单位)，向下对齐到页，至少 1 页
每个对象按分片大小切成若干物理 IO，最后一片可以较小
```

### 11.3 FTL

- 页映射（L2P）。物理页数 = max(逻辑页数, ⌈逻辑页数 / (1 − OP)⌉)，再按擦除块取整。
- 擦除块 b 属于并行单元 b mod 16。逻辑页优先分配到单元 (LBA mod 16)，该单元满了再依次尝试其他单元。
- 读取从未写过的页时，视为分析开始前已经存在：现场分配物理页，计入 preloaded_pages，不产生写入。
- 写入为异地写，旧页标为无效。
- 空闲页比例 ≤ 5% 时触发 GC：选无效页最多（其次有效页最少）的非打开块，把有效页搬到其他空闲页后擦除。GC 惩罚 = 擦除时延 + 搬迁页数 × 编程时延。
- 写放大 WA = NAND 写入页数 ÷ 主机写入页数。配置中的 write_amplification 参数另按比例增加编程页数和后台编程时间。

### 11.4 排队与时延

按 (到达时间, request_id, LBA) 的顺序逐个处理物理 IO：

```text
1. 主机槽位：取最早空闲的槽位，submit = max(到达, 槽位空闲时刻)
2. 随机 IO 还受 1 / IOPS 的最小间隔限制
3. 并行单元 u = 该 IO 第一个页所在擦除块的单元；start = max(submit, u 的空闲时刻)
4. 时延下限 floor = 读 85 µs / 写 450 µs（随机 IO 取与 1 / IOPS 的较大值）+ GC 惩罚 + 额外编程时间
5. 带宽：读、写各有一条时间线；bw_start = max(start, 时间线当前位置)，bw_end = bw_start + 字节 ÷ (BW × 1000)
6. completion = max(start + floor, bw_end)；更新槽位、单元和带宽时间线
```

### 11.5 输出

| 指标 | 定义 |
| --- | --- |
| Achieved throughput | 完成的字节 ÷（最晚完成时刻 − 最早到达时刻） |
| IOPS | 完成的 IO 数 ÷ 同一时长 |
| Latency | 完成时刻 − 到达时刻（含排队），p50 / p95 / p99 / max |
| Deadline miss | 完成晚于截止时间的读 IO 占比 |
| WA、GC 次数、擦除次数 | 见 §11.3 |

用模拟的读完成时刻重建 TTFT：

```text
ready_c = 组件 c 最后一个读 IO 的完成时刻 − 请求到达时刻
A       = max_c ready_c
Optimistic  TTFT = max(Σ c_j, A + O) + D            读取与计算完全重叠
Dependency  TTFT = §10.4 的阶段链，ready 改用 ready_c   每个阶段等它的数据
No-overlap  TTFT = A + O + Σ c_j + D                先读完全部数据再计算
Hidden miss ratio = 满足 A > Σ c_j − O 的请求数 ÷ 有恢复读取的请求数
```

### 11.6 已知局限（重要）

**1. 有效吞吐远低于额定值。** 以下数字于 2026-11-02 用本版本代码和随附配置测得，复算方法见附录 A.2：

| 读写 | 对象大小 | 分片 | 用到的并行单元 | 吞吐 |
| --- | ---: | ---: | ---: | ---: |
| 读 | 9,437,184 B（Qwen3-8B 的 64-token 块） | 128 KiB | 2 / 16 | 3.08 GB/s |
| 读 | 9,437,184 B | 16 KiB | 16 / 16 | 3.08 GB/s |
| 读 | 114,688 B（7 页，单个 IO） | 128 KiB | 16 / 16 | 7.00 GB/s |
| 写 | 9,437,184 B | 128 KiB | 2 / 16 | 0.58 GB/s |

原因有三：

- 一个 IO 的所有页都按首页所在的那一个并行单元计时，单元的占用时间至少是一次完整的读（85 µs）或编程（450 µs），与 IO 大小无关；
- 连续 128 KiB 分片的首页相隔 8 页，在 16 个单元中只落到 2 个；
- 带宽时间线按处理顺序单调前进，后处理的 IO 不能使用之前留下的空闲带宽，所以同一对象的分片基本只能串行使用这 2 个单元。

KV 对象通常是 8 页的整数倍，因此随附配置下读约 3.1 GB/s、写约 0.58 GB/s，而不是 7 / 5 GB/s，排队时延和截止时间未命中会相应偏大。

**2. 满盘 GC。** 搬迁时找不到空闲页，程序会中断搬迁但仍擦除源块，映射表随后指向已擦除的页。这个问题只在 OP = 0 的极小配置中复现过，尚未确认默认配置下是否会触发。不要用满盘或高占用场景的仿真结果下结论。

**3. 范围。** 不含主机内存、PCIe、网络和反序列化，不模拟控制器固件、读重试等行为。

**4. 开环。** 模拟的完成时刻只用于重建 TTFT，不会反馈给缓存回放中的锁释放和准入。

使用建议：只在同一设备配置下做相对比较（例如不同 QPS、不同缓存容量），不要把这里的绝对吞吐和时延当作设备预测。

### 代码位置

`ssu_sim/storage/profile.py` · `SSDProfile`；`ssu_sim/storage/trace.py` · `build_extent_allocation_table`、`lower_demand_ops`、`_chunk_bytes`；`ssu_sim/storage/simulator.py` · `_FTLState`、`_garbage_collect`、`simulate_ssd`、`_latency_floor_us`、`_ttft_report`；`dashboard_v2/server.py` · `run_case_endpoint`。

<a id="recommend"></a>
## 12. 推荐算法（Recommend）

### 输入

设计空间（模型、Runtime、块大小、Max IO、存储配置）、工作负载范围（上下文长度、Prefix ratio、到达模型、QPS；V4.1 另有 Encoder 场景）、建模选项、约束和排名目标。

### 12.1 候选与场景

```text
计算设计 = 模型 × Runtime × 块大小 × Max IO        剔除不兼容组合：超过最大上下文、块大小不满足倍数或下限
候选     = 计算设计 × 存储配置
场景     = 上下文长度 × Prefix ratio × 到达模型 × QPS（× V4.1 的 Encoder 场景）
```

每个（计算设计，场景）只计算一次，总数不超过 1200，结果复用到每个存储配置。存储配置只影响约束上限和额定带宽下的 TTFT。

### 12.2 约束

| 界面项 | 使用的指标 | 勾选 Use Storage limits 时的上限来源 | 模式 |
| --- | --- | --- | --- |
| Optimistic GB/s ≤ | 各请求 fully-hidden 带宽的最大值 | 额定顺序读带宽 | Envelope |
| Break-even GB/s ≤ | 各请求 break-even 带宽的最大值 | 额定顺序读带宽 | Envelope |
| Read GB/s ≤ | 平均读带宽（§9.4） | 额定顺序读带宽 | Replay |
| Write GB/s ≤ | 平均写带宽 | 额定顺序写带宽 | Replay |
| Aggregate dependency GB/s ≤ | 聚合依赖带宽（§10.5） | 额定顺序读带宽 | Replay |
| Dependency IOPS ≤ | 聚合依赖 IOPS（需设置 Max IO） | 随机读 IOPS | Replay |
| Capacity GiB ≤ | Retain-all 容量上界（§5.5） | 逻辑容量 | Replay |
| Dependency TTFT saving ≥ | 额定读带宽下的依赖 TTFT 节省 p50（§10.4） | 默认 0 ms | 两种 |

界面中填写的上限优先于存储配置的值。

### 12.3 单个场景的判定

```text
上限类约束：利用率 u = 值 ÷ 上限，余量 headroom = 1 − u
           值缺失、非有限或为负时 u = ∞；上限为 0 且值为 0 时 u = 0
下限类约束：margin = 值 − 下限
场景可行 = 所有已启用的约束都满足
```

未完成评估（`not_evaluated`）的场景计为不可行，该候选不参与排名。

### 12.4 候选指标

```text
coverage              = 可行场景数 ÷ 场景数
worst headroom        = 所有场景、所有上限类约束中的最小余量
mean storage pressure = 各场景最大利用率（1 − 该场景的最小余量）的平均值
min TTFT margin       = 所有场景中最小的 margin
binding resource      = 利用率最高的约束
fully feasible        = 所有场景都可行
```

### 12.5 排名

按下表的排序键从左到右比较；缺失的值排在最后，最后按候选 ID 排序，保证结果稳定。

| 排名目标 | 排序键 |
| --- | --- |
| Maximize worst-case normalized resource headroom（默认） | 完全可行优先 → coverage 高 → worst headroom 高 → min TTFT margin 高 |
| Maximize workload-envelope coverage | coverage 高 → 完全可行优先 → worst headroom 高 → mean pressure 低 |
| Minimize average storage pressure | 完全可行优先 → coverage 高 → mean pressure 低 → min TTFT margin 高 |
| Maximize worst-case dependency-aware TTFT margin | 完全可行优先 → coverage 高 → min TTFT margin 高 → worst headroom 高 |

### 12.6 命中率可行区间

对每个（上下文长度，QPS，到达模型），把采样的 Prefix ratio 排序，并标记每个采样点是否可行（V4.1 要求所有 Encoder 场景都可行）。相邻的可行点合并为一个区间；出现多个区间时标为 `non_monotonic`。程序只描述采样点，不在采样点之间插值。

### 代码位置

`dashboard_v2/recommendation.py` · `run_recommendation`、`_compatible_compute_designs`、`_scenarios`、`_evaluate_candidate`、`_add_rated_storage_ttft`、`_effective_constraints`、`_ranking_key`、`_sampled_hit_boundaries`。

<a id="validation"></a>
## 13. 验证状态与已知局限

### 13.1 校准状态

- 56 个随附模型的 `timing_calibration` 均为 `uncalibrated`；4 个随附 Runtime 都以 2000 TFLOPS 作为归一化标尺。没有任何时间结果经过 GPU 或 SSD 实测校准。
- 每个模型配置分别记录各类结论的证据（`ModelEvidence`），类别为 calculator_capacity、active_state_layout、prefix_serialization、compute_formula、sparse_routing 和 timing_calibration。状态取值包括 official、derived_from_official_code、calculator_reference、analytical、synthetic_scenario、scenario_estimate、assumed、uncalibrated、unsupported、upper_bound 和 lower_bound。FLOPs 使用了假定几何（assumed）的模型会在结果中给出警告。

<a id="checks"></a>
### 13.2 已做的一致性检查

以下检查于 2026-11-02 在与 `93149e3` 相同的代码上运行。测试位于开发仓库，未随本仓库提供。

| 检查 | 结果 |
| --- | --- |
| KV Size 页（JavaScript）与 Python 参考实现逐模型对拍，含 GDN checkpoint 和比较页 | 4 项浏览器测试通过 |
| 参考计算器、外部计算器固定快照、Explore 序列化口径三者对拍；FLOPs 分类；恢复与 checkpoint | 40 项测试通过 |
| 缓存回放、radix 保护与驱逐、需求报告、SSD 仿真 | 64 项测试通过 |
| §10.7 的 Qwen3-8B 算例、§11.6 的 SSD 吞吐 | 用本版本代码复算一致（附录 A） |

这些检查说明程序内部的公式和接口自洽，不说明结果与真实硬件一致。

### 13.3 已知局限

| 方面 | 局限 | 对结果的影响 |
| --- | --- | --- |
| 计算时间 | FLOPs ÷ 2000 TFLOPS；不考虑显存带宽、kernel 效率和 TP/EP 通信 | 绝对 TTFT 可能偏高或偏低；不同长度之间的比例也可能与实测不同 |
| 排队与调度 | 没有计算排队、batching、continuous batching、chunked prefill；QPS 只通过缓存驻留和命中影响结果；不报告计算资源过载 | 负载较高时，真实 TTFT 会明显高于本工具的结果 |
| 首 token 与 decode | 首 token decode 为 0；不计算 TPOT 和 decode 吞吐，Decode 选项只计入写入 | — |
| 存储路径 | Explore 的读取时间按 0 计；Recommend 按额定带宽且每个请求独占带宽；SSD 仿真只有一块盘 | Explore 的 TTFT saving 是上界 |
| 主机与网络 | 不含主机内存、PCIe、网络、反序列化和推理引擎连接器的开销 | 真实系统的恢复更慢 |
| SSD 仿真 | 见 §11.6 | 吞吐偏低，排队偏大 |
| 负载 | 合成负载；请求等长；没有多轮增长；不能导入 trace | 命中率可能与真实业务差异很大 |
| 稀疏读取 | 选中的块由场景决定，不是真实的 Top-k | 稀疏读取量是场景估计 |
| 多分片 | 缓存回放支持多分片；SSD 仿真只接受一个分片 | — |
| 完整命中 | 未建模最后一个 token 的 hidden 或 logits 时，结果为 not_evaluated | 相关指标为 null |

<a id="readme-diff"></a>
### 13.4 与 README 的差异

- README“Peak pressure 的计算方法”一节说，填写 Decode step 时 decode 写入按步长分布。实际代码中，Decode step 只改变送入 SSD 仿真的写 IO 时刻（§11.1）；Explore 的峰值窗口仍把 decode 写入记在完成时刻（§9.4）。

<a id="appendix-a"></a>
## 附录 A：复算本文的数字

以下脚本只读取本仓库的代码和配置，不写任何文件。在程序目录下用安装脚本创建的 Python 环境运行；Windows 可把代码保存为文件，再用 `.\.venv\Scripts\python.exe -B 文件名` 运行。

### A.1 第 10.7 节的 Qwen3-8B 算例

```bash
.venv/bin/python -B - <<'EOF'
import sys
sys.path.insert(0, ".")
from ssu_sim.utils.config import load_config
from ssu_sim.models.model_registry import model_from_dict
from ssu_sim.demand.analyzer import analyze_bandwidth_envelope
from ssu_sim.demand.report import phase_timing_samples, rated_phase_timing_report

runtime = load_config("configs/runtime/bf16_reference.yaml")
model = model_from_dict(load_config("configs/models/qwen3_8b.yaml"), runtime)
for ratio in (0.5, 0.9):
    result = analyze_bandwidth_envelope(model, seq_len=32768, prefix_ratio=ratio,
                                        block_size_tokens=256, max_io_bytes=131072)
    job, ttft = result.ttft_jobs[0], result.report["ttft"]
    rated = rated_phase_timing_report(phase_timing_samples(result.ttft_jobs), 7.0)
    print(f"命中 {ratio:.0%}")
    print(f"  无缓存 TTFT       {job.no_cache_ttft_us / 1000:.1f} ms")
    print(f"  缓存路径 TTFT     {job.ideal_cache_ttft_us / 1000:.1f} ms")
    print(f"  阻塞恢复字节      {job.restore_payload_bytes / 1e9:.2f} GB")
    print(f"  fully-hidden      {ttft['optimistic_overlap_GBps']['p50']:.1f} GB/s")
    print(f"  break-even        {ttft['dependency_aware_break_even_GBps']['p50']:.1f} GB/s")
    print(f"  7 GB/s 依赖 TTFT  {rated['ttft_us']['p50'] / 1000:.1f} ms")
EOF
```

预期输出：50% 命中依次为 387.3 ms、233.5 ms、2.42 GB、10.3 GB/s、15.7 GB/s、578.6 ms；90% 命中依次为 387.3 ms、53.1 ms、4.38 GB、82.5 GB/s、13.1 GB/s、678.6 ms。

### A.2 第 11.6 节的 SSD 吞吐

```bash
.venv/bin/python -B - <<'EOF'
import sys
sys.path.insert(0, ".")
from ssu_sim.utils.config import load_config
from ssu_sim.storage import simulate_ssd, ssd_profile_from_dict
from ssu_sim.demand.analyzer import DemandOp

profile = ssd_profile_from_dict(load_config("configs/storage/single_nvme_ftl.yaml"))
cases = (("read", 9_437_184, 131_072, 40), ("read", 9_437_184, 16_384, 40),
         ("read", 114_688, 131_072, 3000), ("write", 9_437_184, 131_072, 40))
for op_type, size, max_io, count in cases:
    ops = [DemandOp(time_us=0.0, request_id="r", model_id="m", op_type=op_type,
                    block_id=f"b{i}", payload_bytes=size, component="kv",
                    operation_weight=1.0, access_pattern="sequential") for i in range(count)]
    result = simulate_ssd(ops, profile, max_io_bytes=max_io)
    units = len({io.parallel_unit for io in result.completed_ios})
    print(f"{op_type:5} 对象 {size:>9,} B  分片 {max_io:>7,} B  "
          f"{result.report['throughput']['achieved_GBps']:.2f} GB/s  使用 {units} 个并行单元")
EOF
```

预期输出：读 3.08 GB/s（2 个单元）、读 3.08 GB/s（16 个单元）、读 7.00 GB/s（16 个单元）、写 0.58 GB/s（2 个单元）。

<a id="appendix-b"></a>
## 附录 B：代码索引

行号对应提交 `93149e3`。

| 主题 | 文件 | 函数或类（行号） |
| --- | --- | --- |
| 合成负载 | `ssu_sim/workload/synthetic.py` | `generate_synthetic_workload`（35） |
| Zipf 与抽样 | `ssu_sim/utils/random.py` | `weighted_choice`、`zipf_weights` |
| 块身份 | `ssu_sim/prefix/block_hasher.py` | `make_block_id`（13）、`build_request_block_ids`（34）、`canonicalize_request_paths`（156）、`_append_immutable_objects`（313） |
| 缓存策略 | `ssu_sim/models/attention.py` | `PrefixCachePolicy`（42） |
| 序列化组件 | `ssu_sim/models/attention.py` | `CacheComponent`（106）、`_ComponentProfileMixin`（376） |
| 各族容量与读写映射 | `ssu_sim/models/attention.py` | `DenseKVAttentionProfile`（513）、`MLAAttentionProfile`（642）、`DSAAttentionProfile`（759）、`DeepSeekV41AttentionProfile`（1066）、`DeepSeekV4HybridAttentionProfile`（1300）、`QwenLinearFullHybridAttentionProfile`（2426）、`KDAHybridAttentionProfile`（2519）、`MiniMaxMSAAttentionProfile`（2620）、`MixedFullSlidingGQAAttentionProfile`（2946）、`InklingHybridAttentionProfile`（3314） |
| 读写块号 | `ssu_sim/models/attention.py` | `_prefix_block_indices`（4104）、`_write_block_indices`（4119）、`_checkpoint_read_plan`（4152） |
| V4.1 恢复阶段 | `ssu_sim/models/attention.py` | `DeepSeekV41AttentionProfile.recovery_plan`（1168） |
| 参考容量 | `ssu_sim/models/calculator_reference.py` | `calculate_reference_footprint`（18） |
| KV Size 页 | `dashboard_v2/js/kv_size.js` | `calculateKvSize`（436） |
| 计数原语 | `ssu_sim/models/compute_counts.py` | `causal_entry_sum`（16）、`causal_pairs`（32）、`block_sparse_pairs`（37）、`delta_rule_work`（57）、`softmax_work`（109） |
| FFN 与 MoE | `ssu_sim/models/llm_model.py` | `DenseFFNProfile`（207）、`MoEFFNProfile`（255） |
| 缓存路径 FLOPs | `ssu_sim/models/llm_model.py` | `_OperationFormulaMixin.cache_compute_breakdown`（601） |
| 各族 FLOPs | `ssu_sim/models/llm_model.py` | `GQAPrefillFormula`（807）、`MLAPrefillFormula`（834）、`DSAPrefillFormula`（885）、`DeepSeekV41PrefillFormula`（966）、`DeepSeekV4PrefillFormula`（1135）、`_standard_attention_breakdown`（1259）、`_mla_attention_breakdown`（1288）、`MixedFullSlidingPrefillFormula`（1405）、`MiniMaxMSAPrefillFormula`（1501）、`QwenLinearHybridPrefillFormula`（1607）、`KDAHybridPrefillFormula`（1720）、`InklingHybridPrefillFormula`（1852） |
| 时间换算与 LM head | `ssu_sim/models/llm_model.py` | `ComputeProfile`（1958）：`estimate_timing`（2043）、`_with_full_hit_status`（2202）、`_with_output_head`（2216）、`_flops_to_us`（2270） |
| DeepSeek 分项账本 | `ssu_sim/models/deepseek_compute.py` | `v4_segment`（118）、`v41_segment`（170）、`v4_head`（249） |
| 状态策略与阶段时间 | `ssu_sim/models/state.py` | `PrefixGranularityPolicy`（48）、`StatePersistencePolicy`（90）、`shared_read_ready_times`（360）、`phase_chain_time_us`（374）、`phase_read_deadlines`（401）、`phase_bandwidth_threshold`（431）、`RestoreExecutionPlan`（459）、`checkpoint_boundaries`（604） |
| 回放配置 | `ssu_sim/cache/config.py` | `CacheReplayConfig`（10） |
| radix 与 LRU | `ssu_sim/cache/radix.py` | `RadixLeafLRUCache`（143）、`match_path`（175）、`insert_path`（209）、`release`（301）、`attach_checkpoint`（314）、`enforce_capacity`（420）、`stable_cache_shard`（583） |
| 回放事件循环 | `ssu_sim/cache/replay.py` | `replay_prefix_cache`（468）、`_actual_recovery_boundary`（1127）、`_admit_checkpoint_interval`（1164）、`_actual_prefill_time_us`（1249） |
| IO 需求 | `ssu_sim/demand/analyzer.py` | `analyze_bandwidth_envelope`（184）、`analyze_workload`（224）、`_apply_actual_checkpoint_restore`（1082）、`_component_read_deadlines`（1215）、`_chunk_count`（1233） |
| 报告与阈值 | `ssu_sim/demand/report.py` | `build_demand_report`（45）、`_operation_report`（250）、`_peak_window`（308）、`phase_timing_samples`（387）、`rated_phase_timing_report`（410）、`_ttft_report`（448）、`_bandwidth_threshold_report`（700）、`_aggregate_dependency_pressure`（814）、`_percentile`（1137） |
| SSD 配置 | `ssu_sim/storage/profile.py` | `SSDProfile`（11）、`ssd_profile_from_dict`（106） |
| LBA 与切分 | `ssu_sim/storage/trace.py` | `build_extent_allocation_table`（41）、`lower_demand_ops`（108）、`_chunk_bytes`（185） |
| SSD 仿真 | `ssu_sim/storage/simulator.py` | `_FTLState`（62）、`_garbage_collect`（223）、`simulate_ssd`（273）、`_latency_floor_us`（355）、`_ttft_report`（465） |
| 推荐 | `dashboard_v2/recommendation.py` | `run_recommendation`（285）、`_compatible_compute_designs`（459）、`_scenarios`（515）、`_evaluate_candidate`（527）、`_add_rated_storage_ttft`（697）、`_effective_constraints`（734）、`_sampled_hit_boundaries`（787）、`_ranking_key`（839） |
| 服务端入口 | `dashboard_v2/server.py` | `run_case_endpoint`（1677）、`_run_demand_case`（2338） |
