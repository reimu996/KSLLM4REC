# OneReason Frontier r64 G16 RLOO

## 目标

本实验要解决旧 GRPO 的两个问题：强制把 GT 塞进 rollout 会改变在线采样分布；组内奖励全部相同时，策略梯度为零。训练因此使用纯在线 G=16 rollout，不注入 GT；奖励有差异时使用 RLOO，奖励全相同但未全部命中时使用 GT 集合锚点。

## 冻结输入

- base model：`/home/lyc/models/OneReason-0.8B-pretrain-competition`
- 初始 r64 adapter：`artifacts/sft/platform_exports/frontier_feedbackcore_listwise_invariant_epoch2_lora64_20260722/extracted`
- 数据源：`/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/train.jsonl`
- 推荐组：`artifacts/grpo/data/recommend_groups_frontier_v1/groups.jsonl`
- 合法 SID 前缀树：`artifacts/grpo/catalog/frontier_all_system_prompt_response_sids_v1`
- 固定 probe：`artifacts/grpo/data/frontier_probe_v1/fixed_probe_1024.jsonl`

这三份训练侧派生输入与上一版 Frontier GRPO 完全相同：17,016 个 prompt 组、30,465 条去重正例边、905,469 个合法 SID。变化只有初始 adapter 和训练算法。

关键 SHA256：

| 对象 | SHA256 |
| --- | --- |
| base weights | `28e66d2ec528473d335ede2b3faa08eddc53eb8ec93747a449e5e7ec812ede90` |
| initial adapter | `c31f1e67cd46b02a27a1ed44136eccb1f83b7055f247c84d4a6924cafa4c1d70` |
| adapter config | `041e3f79eaa6acfa181612838c03e275f0fa7883e0b286c82fbb6f164714731e` |
| tokenizer | `cd4d15f596979aecbc11ba12668cb21c9fb8452ce1235cd9d7878cc950170421` |
| source JSONL | `9e3465cedbdaf784ab4720699649c76c256414e2c2089efcd7657f63bae7449a` |
| groups JSONL | `a2571d97e4ac231d61d1f468bc23b8e65bebfd5ad7b1a2dd566f1bfd86a9658e` |
| trie manifest | `f51414b13a8982605ebe1f7fec75929245db60c854e3860d3b6c1625a3f33e4a` |
| fixed probe | `86e050a00506efbaef0e49997ed138462aaddc322eba9fc66227a25654a75e5e` |

## 训练规则

每个 prompt 实时采样 16 个候选，固定分成 `0..7` 和 `8..15` 两块。采样只能沿合法 SID 前缀树前进；单子节点直接追加 token，概率记为 1，不运行模型也不消耗随机数。分支节点只在合法 token 上做 FP32 softmax。

五档奖励：

```text
exact         1.00
same_ab       0.40
same_a        0.15
same_domain   0.01
other_domain  0.00
```

对于候选 `i`，RLOO advantage 为：

```text
A_i = reward_i - mean(其他 15 个 reward)
    = 16/15 * (reward_i - mean(16 个 reward))
```

不除以标准差。奖励不全相同时只计算 RLOO；16 个奖励全为 1 时跳过；其他“16 个奖励完全相同”情况只计算 GT 集合锚点。GT 只作为只读 reward/anchor 目标，不进入 rollout 候选。

每连续 8 个源 prompt 构成一个训练窗口。必须先完成该窗口全部 8 组 rollout，才允许一次参数更新。两个分支分别在窗口内求均值：

```text
window_loss = mean(rloo_group_losses) + lambda0 * mean(anchor_group_losses)
```

`lambda0` 由初始模型在固定 512 组上的 FP32 全局梯度范数校准：

```text
lambda0 = min(0.05, 0.10 * rloo_grad_norm / anchor_grad_norm)
```

校准不更新参数，完成后重新加载干净的初始模型。

## 参数

- epochs：2
- G：16
- rollout/loss/GT chunk：8
- LoRA：继续训练现有 `r=64, alpha=64` adapter；运行期 dropout 为 0
- learning rate：`5e-6`
- warmup：前 128 个源窗口
- scheduler：按固定 4,254 个源窗口计算 cosine；跳过窗口也推进调度位置
- AdamW：`betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0`
- max grad norm：1
- cutoff：16,384
- bf16、TF32、gradient checkpointing：开启
- reference model、KL、PPO clipping、GT 注入、reward z-score：全部禁用

## 执行

唯一运行环境：`/home/lyc/miniconda3/envs/onereason_lora_sft`。不安装或升级依赖。

```bash
cd /home/lyc/REC_PROJECTS/KSLLM4REC
scripts/rloo/frontier_lora64_g16/prepare.sh
scripts/rloo/frontier_lora64_g16/test_cpu.sh
scripts/rloo/frontier_lora64_g16/run_gates.sh
scripts/rloo/frontier_lora64_g16/run_epoch0_probe.sh
scripts/rloo/frontier_lora64_g16/run_full.sh
scripts/rloo/frontier_lora64_g16/run_trained_probes.sh
scripts/rloo/frontier_lora64_g16/verify.sh
```

也可在实现已提交后执行 `scripts/rloo/frontier_lora64_g16/run_all.sh`。

## 硬门禁

正式训练必须加载同一运行签名和同一 `lambda0` 的六份报告：结构、校准、采样/重放概率一致性、32 条最长 prompt 显存、512 组训练信号、256 组端到端耗时。

- sampled/replayed decision log-prob 最大绝对差：`<= 1e-5`
- RTX 4090 peak reserved：`<= 20 GiB`
- RLOO group rate：`>= 25%`
- 时间投影：`1.2 * seconds / 256 * 34032 / 3600 <= 72 h`
- 所有候选合法，reward/loss/gradient 有限

## 产物

- 两轮 adapter：`artifacts/rloo/runs/frontier_sft_epoch2_lora64_ref_free_g16_rloo_anchor_2epoch/epoch_001` 和 `epoch_002`
- 恢复点：同一 run 目录下 `recovery/`
- 每组审计：`train_audit.jsonl`
- 每窗口进度：`train_progress.jsonl`
- 门禁、probe、最终验证：`operation_logs/rloo/frontier_sft_epoch2_lora64_g16_v1/`

正式验证要求 epoch 0/1/2 均完成固定 1,024 条 beam-16 合法约束 probe，并逐行复算训练 reward、RLOO branch、调度位置、恢复游标与 adapter SHA。
