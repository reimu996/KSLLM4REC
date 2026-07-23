# OneReason RLOO Temperature 1.0 vs 1.2 诊断

## why

本实验要回答一个单一问题：在固定最终 RLOO Epoch 2 权重时，把在线采样温度从 `1.0` 提高到 `1.2`，能否增加同组候选的 reward 差异和 SID 多样性，同时不降低命中正确 SID 的能力。

它不是训练实验。两个温度都不执行反向传播、优化器更新、GT 注入、anchor loss 或奖励条件重采样。实验结果只能决定 `1.2` 是否值得进入后续训练 Spec，不能自动启动训练。

## 固定对象

- 权重：`artifacts/rloo/runs/frontier_sft_epoch2_lora64_ref_free_g16_rloo_anchor_2epoch/epoch_002`
- `adapter_model.safetensors` SHA256：`aeebde52b58efd3685a2925de9b66163d40a91e62467fac7e233b68e11c8fdb2`
- 数据：固定 calibration ID 文件中的前 512 个推荐组，保持文件内顺序
- 合法 SID：固定 905,469 叶子的 Frontier 前缀树
- 每组候选数：`G=16`
- 执行块：候选 `0..7` 与 `8..15`
- 随机种子：候选 `(group_id, candidate_index)` 使用 `rollout|42|3|group_id|candidate_index` 的 SHA256 前 8 字节
- 最大输出：32 token
- reward：`exact=1.0, same_ab=0.40, same_a=0.15, same_domain=0.01, other_domain=0.0`

## A/B 的有且仅有一个本质差异

判定层级：固定模型上的在线 rollout 分布。

固定项：模型、prompt、GT SID 集合、合法前缀树、候选编号、候选 seed、G、chunk、reward 和重放算法完全相同。

有且仅有 1 个本质差异：合法动作 logits 在 softmax 前分别除以 `1.0` 或 `1.2`。采样与同臂重放都使用该臂自己的温度。

## 一组具体例子

输入：同一个 `group_id=g1`，16 个候选编号为 `0..15`，Epoch 索引固定为 3。

```text
seed[0] = uint64_be(SHA256("rollout|42|3|g1|0")[:8])
...
seed[15] = uint64_be(SHA256("rollout|42|3|g1|15")[:8])

T=1.0: p100 = softmax(legal_logits / 1.0)
T=1.2: p120 = softmax(legal_logits / 1.2)
```

两臂读取相同随机数流，但概率区间不同，因此可以选到不同 token。每一步只能从前缀树给出的合法 token 集合采样，所以最终 16 条输出都必须是合法 SID。

## 指标定义

- `informative_rloo_group_rate`：16 个 reward 不全相等的组数 / 组数。
- `uniform_tier_group_rates.same_domain`：16 个候选全落在 `same_domain` 档的组数 / 组数。
- `mean_unique_sids_per_group`：每组不同 SID 数量的算术平均。
- `duplicate_slot_rate`：所有组的 `(16 - 不同 SID 数)` 之和 / 所有候选槽位数。
- `any_exact_group_rate`：至少一个候选命中任一 GT SID 的组数 / 组数。
- `average_group_max_reward`：每组 16 个 reward 最大值的算术平均。
- `average_legal_action_entropy_nats`：每个真正分支位置上，完整合法动作概率分布的熵，再按决策位置等权平均。单一合法 token 的位置不计入。
- `max_abs_sample_replay_logp_difference`：采样时所选 token 的 log-prob 与同温度重放所得 log-prob 的最大绝对差。

所有 rate 同时保留分子计数，防止只看百分比误判。

## 预先声明的结论门槛

只有以下条件全部成立，`comparison.json` 才写 `decision=promising`：

1. informative RLOO 组率至少增加 5 个百分点。
2. 每组平均不同 SID 至少增加 0.5。
3. 全组 `same_domain` 的比例至少下降 5 个百分点。
4. 至少一个 exact 的组率不下降。
5. 平均组内最高 reward 不下降。
6. `other_domain` 候选槽位率最多增加 2 个百分点。
7. 两臂采样/重放 log-prob 差不超过 `1e-5`。
8. 所有候选合法且所有数值有限。

## 执行关系

执行关系是串行且进程隔离：先在新进程加载一次模型跑 `T=1.0`，进程退出；再在另一个新进程重新加载相同模型跑 `T=1.2`。两臂不会共享模型对象或 CUDA 状态。

```bash
scripts/rloo/frontier_temperature_ab/test_cpu.sh
scripts/rloo/frontier_temperature_ab/run_smoke.sh
scripts/rloo/frontier_temperature_ab/run_full.sh
scripts/rloo/frontier_temperature_ab/verify.sh
```

完整串行入口：

```bash
scripts/rloo/frontier_temperature_ab/run_all.sh
```

## 输出

正式原始结果位于 `operation_logs/rloo/frontier_epoch2_temperature_t100_t120_v1/`，该目录由 `.gitignore` 排除。核心文件为：

- `t100_audit.jsonl`、`t100_summary.json`
- `t120_audit.jsonl`、`t120_summary.json`
- `comparison.json`
- `verification.json`
- `first8_determinism.json`
- `runtime.log`

仓库只提交配置、实现、脚本、测试和最终 Markdown 摘要，不提交大体积逐候选原始日志。
