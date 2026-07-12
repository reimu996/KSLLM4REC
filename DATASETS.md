# KSLLM4REC 数据集说明

本文记录当前工作区中四个 LLM4Rec 数据目录的定位、规模、来源关系和使用边界。目的不是给出最终训练配比，而是先回答三个问题：每个目录处于数据流水线的哪一层、哪些目录存在派生或重叠关系、哪些数据可以直接作为 SFT 样本使用。

最后核对日期：2026-07-12。

## 1. 术语与统计口径

- **上游结构化数据**：仍保留 `pid`、行为序列、时间戳、caption、SID 映射等字段的 Parquet 表，需要经过 join 和模板转换才能成为 SFT 数据。
- **SFT 样本**：一行一个 JSON list，list 中固定包含一个对象，对象字段为 `system`、`prompt`、`response`。
- **pid**：哈希后的内容、商品、主播或广告 ID，只用于上游表 join。
- **SID**：模型使用的语义内容标识，例如 `<|prod_begin|><s_a_168><s_b_4799><s_c_5345>`。
- **样本数**：JSONL 合法行数或 Parquet 元数据中的行数，不等于独立自然用户数，也不等于真实 item catalog 大小。
- **精确重复**：`system`、`prompt`、`response` 三个字段完全相同。

## 2. 总览

| 路径 | 数据阶段 | 主要格式 | 主数据规模 | 当前定位 |
|---|---|---|---:|---|
| `/home/lyc/Data/dataset` | 官方 SFT 成品 | 12 个 JSONL | 32,480 条，约 436 MiB | 万擎平台下载的官方 SFT 数据 |
| `/home/lyc/Data/Explorer_LLM_Rec_Competition` | 上游结构化数据 | 533 个 Parquet | 5 张不同用途的数据表，约 17 GiB | 用于自行构造扩充 SFT |
| `/home/lyc/Data/HF_kuaishou-llmrec-sft-baseline-0.91` | 社区加工的 SFT 成品 | 1 个 JSONL | 32,705 条，约 232 MiB | 声称取得 0.9107 的社区 baseline |
| `/home/lyc/Data/KSLLM4REC_Explorer_SFT` | 本项目派生的扩充 SFT | 72 个上传 JSONL 分片 | 40,353,206 条，约 33 GiB | 从 Explorer 上游表全量生成的自造数据 |

这四个目录不能视为四份彼此独立、可以直接等权拼接的数据。它们属于两条不同的数据来源链路：

```text
比赛官方 comp_sft
├── 万擎平台拆分下载
│   └── /home/lyc/Data/dataset
└── 社区清洗、改写并加入 CEval
    └── /home/lyc/Data/HF_kuaishou-llmrec-sft-baseline-0.91

Explorer 上游 Parquet
└── KSLLM4REC 生成脚本
    └── /home/lyc/Data/KSLLM4REC_Explorer_SFT
```

当前没有证据证明本地 `/home/lyc/Data/dataset` 是由本地 Explorer Parquet 直接生成的，因此两条链路不能合并描述。

## 3. 官方 SFT：`/home/lyc/Data/dataset`

### 3.1 定位

这是从万擎平台下载的比赛 SFT 数据，已经从上游行为和内容字段转换成自然语言 prompt、SID 和标准 response，可以直接注册为 SFT 数据集。

每行格式如下：

```json
[{"system":"...","prompt":"...","response":"..."}]
```

### 3.2 文件与任务构成

| 任务 | 文件 | 样本数 |
|---|---:|---:|
| 懂推荐 | 4 | 19,204 |
| 懂物料 | 7 | 10,384 |
| 懂用户 | 1 | 2,892 |
| 合计 | **12** | **32,480** |

全目录有 32,335 个精确唯一行和 145 个精确重复行。目录中没有单独的懂世界或 CEval 文件。

### 3.3 用途与限制

- 适合作为官方基准数据和格式参考。
- 用户历史、item 和目标已经写入文本，原始 `uid` 没有保留。
- 不能从该目录恢复真实独立用户数。
- 完整 SID 可以作为训练时可观测 item 标识，但不能据此恢复原始 `pid`。

## 4. Explorer 上游数据：`/home/lyc/Data/Explorer_LLM_Rec_Competition`

### 4.1 定位

这是竞赛相关的上游结构化数据包，包含匿名用户行为、内容 ID、SID、caption、tag 和通用任务。它不是现成 SFT 问答集，必须先进行表连接、字段筛选和模板转换。

上游 join 规则是 `(domain, pid)`。`pid` 是哈希后的 `int64`，用户表不含 `uid`，每一行表示一个匿名用户样本。

### 4.2 数据表

| 表 | Parquet 文件数 | 行数 | 作用 |
|---|---:|---:|---|
| `OneReason_UserProfile` | 10 | 500,000 | 匿名用户的商品、视频、直播和广告行为序列 |
| `OneReason_Pid2Sid` | 198 | 35,914,095 | `(domain,pid) -> [s_a,s_b,s_c]` |
| `OneReason_Pid2Caption` | 136 | 21,061,327 | `(domain,pid) -> caption` |
| `OneReason_Pid2Tag` | 31 | 5,417,279 | `(domain,pid) -> tag_lv3` |
| `OneReason_General` | 158 | 152,005 | 通用文本任务 |
| 合计 | **533** | 不应跨表相加为“样本数” | 各表表示不同对象和映射关系 |

SID 的 domain 前缀映射为：

| domain | SID 前缀 |
|---|---|
| `video/video` | `<|video_begin|>` |
| `video/ad` | `<|ad_begin|>` |
| `goods` | `<|prod_begin|>` |
| `live` | `<|living_begin|>` |

字段和序列对齐规则以 `/home/lyc/Data/Explorer_LLM_Rec_Competition/README.md` 为准。

### 4.3 用途与限制

- 可以生成远大于官方 SFT 的懂物料、懂用户和懂推荐数据。
- 同一 `pid` 可能在不同表或分片中重复出现，生成前需要明确去重键。
- caption、SID 和用户行为来自不同表，join miss 必须显式统计。
- 上游表规模大不等于能够全量、等权用于一次 SFT。

## 5. 社区 baseline：`/home/lyc/Data/HF_kuaishou-llmrec-sft-baseline-0.91`

### 5.1 定位

这是 Hugging Face 社区发布者整理的 SFT baseline，不是官方原始目录。其 README 声称使用 OneReason-0.8B、LoRA 和该数据集取得 0.9107 榜单分数；该分数属于发布者提供的信息，不是本地重新评测结果。

### 5.2 任务构成

| 任务 | 样本数 |
|---|---:|
| 懂推荐 | 18,651 |
| 懂物料 | 9,684 |
| 懂用户 | 2,792 |
| 通识 CEval | 1,578 |
| 合计 | **32,705** |

全部 32,705 行均为精确唯一行。

### 5.3 数据配方

根据其 README，该 baseline 对比赛 `comp_sft` 做了以下处理：

1. 精确行去重。
2. 特殊字符和长度过滤。
3. 对懂推荐中同 prompt、相同 think 的重复样本保留一条 filled-think，其余改成 no-think。
4. 保留懂物料和懂用户中的 think trace。
5. 加入 1,578 条 CEval。
6. 使用固定随机种子打乱。

本地逐行核对结果：该 baseline 与 `/home/lyc/Data/dataset` 有 18,854 个完全相同的唯一行。这个数字小于两边的比赛样本数，原因包括清洗、去重、think/no-think 改写以及可能的数据版本差异。因此：

- HF baseline 不是官方目录的简单复制。
- HF baseline 也不是官方目录的严格超集。
- 将两者直接拼接会产生一部分精确重复和大量语义近重复。

具体配置见：

- `/home/lyc/Data/HF_kuaishou-llmrec-sft-baseline-0.91/README.md`
- `/home/lyc/Data/HF_kuaishou-llmrec-sft-baseline-0.91/hyperparameters.json`

## 6. 本项目扩充 SFT：`/home/lyc/Data/KSLLM4REC_Explorer_SFT`

### 6.1 定位

这是 `KSLLM4REC` 根据 Explorer Parquet 生成的平台格式 SFT。生成任务已经完成并通过 `final_validation_report.json` 中的全部验收条件。

主上传数据位于 `upload_parts/`，每个分片同时受以下上限约束：

- 文件大小不超过 500,000,000 bytes。
- 行数不超过 1,900,000。

### 6.2 任务构成

| 生成类别 | 分片数 | 样本数 | 生成依据 |
|---|---:|---:|---|
| `dongwuliao_text_to_sid` | 31 | 19,701,886 | `Pid2Caption + Pid2Sid`，描述 -> SID |
| `dongwuliao_sid_to_text` | 31 | 19,701,886 | `Pid2Caption + Pid2Sid`，SID -> 描述 |
| `dongyonghu_timeline_to_sid` | 6 | 474,717 | `UserProfile + Pid2Sid`，时间线 -> 目标 SID 数组 |
| `dongtuijian_multidomain_to_target` | 4 | 474,717 | `UserProfile + Pid2Sid`，多域历史 -> 分场景目标 |
| 合计 | **72** | **40,353,206** | 平台上传格式 JSONL |

物料双向样本合计 39,403,772 条，占主上传数据的约 97.65%。懂用户和懂推荐各占约 1.18%。

### 6.3 实际输入边界

生成过程实际使用：

- `OneReason_Pid2Sid`
- `OneReason_Pid2Caption`
- `OneReason_UserProfile`

生成过程明确不使用：

- `OneReason_Pid2Tag`
- `OneReason_General`

`/home/lyc/Data/dataset` 没有被合并进生成记录。生成脚本只检查该目录存在、记录其路径并验证输入文件未被修改；主记录流来自 Explorer 三张上游表。

### 6.4 生成统计

- SID 索引输入并成功索引 35,914,095 行。
- caption 输入 21,061,327 行；每个方向生成 19,701,886 条，跳过 13,766 条空 caption 和 1,345,675 条重复记录。
- UserProfile 输入 500,000 行；懂用户和懂推荐各生成 474,717 条，各跳过 25,283 条无有效目标的用户行。
- 所有主上传样本使用空 think 前缀，不生成伪推理。
- 输出 manifest 记录的生成耗时为 3,477.867 秒。

### 6.5 `SFT_demo.jsonl`

根目录另有 `SFT_demo.jsonl`，包含 795 条通用新闻摘要样本。它是独立保留的 demo，不属于 `upload_parts/` 中的 40,353,206 条主数据，也不应被误计入四类生成结果。

主要证据：

- `/home/lyc/REC_PROJECTS/KSLLM4REC/spec.md`
- `/home/lyc/Data/KSLLM4REC_Explorer_SFT/final_manifest.json`
- `/home/lyc/Data/KSLLM4REC_Explorer_SFT/final_validation_report.json`

## 7. 训练使用时的关键风险

### 7.1 官方 SFT 与 HF baseline 存在重叠

两者来自相近的比赛 SFT 来源，但 HF baseline 做过清洗和改写。混合前需要确定去重层级：精确行、相同 prompt、相同最终答案或相同 SID 历史。仅做精确行去重不能消除 think/no-think 改写产生的语义重复。

### 7.2 Explorer 扩充 SFT 的任务比例极不平衡

如果全量等权训练，约 97.65% 的样本来自懂物料，懂用户和懂推荐的梯度会被显著稀释。是否采样、按任务加权或分阶段训练需要单独设计，本文不确定具体比例。

### 7.3 四千万行不等于四千万独立 item

描述到 SID 和 SID 到描述是同一批 caption/SID 对的两个方向；两者不能相加解释为独立 item 数。重复 caption 和重复 `(domain,pid)` 已在生成阶段按既定规则过滤，但 SID 本身仍可能对应多条描述。

### 7.4 SFT 行不是独立用户

官方 SFT 和 HF baseline 不保留 `uid`。Explorer `UserProfile` 也明确将每一行定义为匿名用户样本，而非可跨行归并的自然人 ID。因此训练样本行数只能表示监督记录数。

### 7.5 think/no-think 分布不同

HF baseline 同时包含 filled-think 和 no-think。`KSLLM4REC_Explorer_SFT` 主上传数据统一使用空 think，不包含人工或模型生成的推理过程。混合训练时需要把 `enable_thinking` 和评测的 thinking/non-thinking 双分支一起纳入实验设计。

## 8. 推荐的数据集命名

在平台 `data_source` 或实验日志中，建议使用不会混淆来源的稳定名称：

| 数据范围 | 建议名称 |
|---|---|
| 官方 32,480 条 SFT | `official_comp_sft` |
| HF 0.9107 baseline | `hf_baseline_091` |
| Explorer 描述 -> SID | `explorer_text_to_sid` |
| Explorer SID -> 描述 | `explorer_sid_to_text` |
| Explorer 用户时间线 | `explorer_user_timeline` |
| Explorer 多域推荐 | `explorer_recommendation` |

不要把所有 Explorer 分片注册成同一个无法区分任务的 `data_source`，否则无法按任务监控 loss 或实施采样策略。

## 9. 验证依据

本文数字通过以下只读方式交叉核对：

- JSONL 使用逐行 JSON 解析和 `wc -l` 核对。
- Explorer 表使用 Parquet metadata 汇总文件数、行数和 schema。
- HF 配方和得分读取本地 README 与 `hyperparameters.json`。
- Explorer 扩充数据读取 `final_manifest.json`，并与 `final_validation_report.json` 的逐分片行数核对。
- 官方 SFT 与 HF baseline 的重叠按规范化后的 `(system,prompt,response)` 三元组集合计算。

文档只描述当前本地快照。数据目录内容、平台规则或上游版本发生变化后，应重新执行统计并更新最后核对日期。
