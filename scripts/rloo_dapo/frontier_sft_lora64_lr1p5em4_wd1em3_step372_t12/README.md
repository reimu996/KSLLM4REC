# RLOO-DAPO SFT372 T1.2 K=1 KV-cache 运行入口

## why

这组脚本要解决两个问题：第一，确保训练只能继承指定的 step-372 LoRA，不能被调用者遗留的环境变量悄悄改到另一份配置、目录或 GPU；第二，确保五个门禁全部成功之前，正式门禁目录始终不存在，失败证据与可用于训练的正式证据不会混在一起。

## 冻结起点

- adapter SHA256：`699826c7a276b463f7fe8195a0ee6297083d109a6afe23f161f0985b65684ecb`
- adapter config SHA256：`55969781f9f855ca35ad5133cb5d0db50f6f8466ce7b078603f26258523e5363`
- tokenizer SHA256：`cd4d15f596979aecbc11ba12668cb21c9fb8452ce1235cd9d7878cc950170421`
- tokenizer config SHA256：`0fde6acb74ecabcd1976fc0ea878a5f3e77b72fd51db3c7a6ffce25f570491f0`

源 adapter 记录的 LoRA dropout 是 `0.05`；RL 运行时关闭全部 dropout，后续保存的 adapter 记录 `0.0`。SFT optimizer、scheduler、RNG 和 global step 均不恢复。

## 固定运行身份

`CONFIG`、`LOG_ROOT`、`GATE_ROOT`、`PILOT_DIR`、`RUN_DIR`、`TRAIN_LOG`、`CPU_REPORT`、`CPU_LOG`、`DEVICE` 和 `CUDA_VISIBLE_DEVICES=0` 都由 `common.sh` 固定；同名环境变量不会覆盖它们。只有 `PYTHON` 允许调用者覆盖，默认值为 `/home/lyc/miniconda3/envs/onereason_lora_sft/bin/python`。GPU 门禁与正式训练还会逐字段核对设备名、UUID、显存字节数和 compute capability。

## 命令

```bash
scripts/rloo_dapo/frontier_sft_lora64_lr1p5em4_wd1em3_step372_t12/test_cpu.sh
scripts/rloo_dapo/frontier_sft_lora64_lr1p5em4_wd1em3_step372_t12/run_gates.sh
scripts/rloo_dapo/frontier_sft_lora64_lr1p5em4_wd1em3_step372_t12/run_full.sh
scripts/rloo_dapo/frontier_sft_lora64_lr1p5em4_wd1em3_step372_t12/status.sh
scripts/rloo_dapo/frontier_sft_lora64_lr1p5em4_wd1em3_step372_t12/verify.sh
```

`test_cpu.sh` 依次运行 `tests/rloo_dapo`、`tests/rloo`、`tests/sft`、`tests/grpo`、`tests/orpo`，完整 stdout/stderr 固定写入 `LOG_ROOT/cpu_tests.log`，并生成绑定当前 runtime signature 和日志 SHA256 的 `LOG_ROOT/cpu_gate.json`。测试源码或运行代码变化后，旧报告立即失效；GPU 门禁、正式训练和最终验证都会重新检查它。

## 门禁发布

`run_gates.sh` 开始时要求正式 `GATE_ROOT`、`PILOT_DIR` 和 `${PILOT_DIR}-dense-baseline` 全部不存在。报告和 pilot 先写入 `LOG_ROOT/.gates.staging.<UTC时间>.<PID>`：

1. 五个 gate CLI 都返回成功；
2. 脚本重新计算当前 runtime signature，并一次校验五份报告都为 `passed=true`、都属于该 signature；
3. optimized pilot 首窗口不超过 600 秒，并相对 dense baseline 至少加速 1.2 倍；
4. 连续两窗口与“一窗口后退出、恢复再跑一窗口”的 SID、reward、loss、ratio、adapter 和训练状态逐字节一致；
5. pilot、dense baseline、resume-check 三个目录的完整文件集合、大小和 SHA256 都写入报告并在开训前重算；
6. 三个 pilot 目录移到固定路径，并把 `pilot.json` 内的 staging 路径改成正式路径；
7. 整个报告目录通过同文件系统目录 rename，一次发布为固定 `GATE_ROOT`。

任一步失败或进程被中断，正式 `GATE_ROOT` 均不作为成功证据保留；所有已生成报告、日志和 pilot 被集中保留到独立的 `LOG_ROOT/gates.failed.<UTC时间>.<PID>` 供审计。当前 gate CLI 允许 staging `--output` / `--pilot-dir`；如果将来 CLI 把输出路径也收紧为固定正式路径，必须先给 CLI 增加显式 staging 支持，不能退回“直接写正式目录”。

## 正式训练互斥与恢复

正式训练必须等上述五份新报告在同一 runtime signature 下通过，且 pilot 的初始策略 SHA 等于冻结 adapter SHA。`run_full.sh` 使用 `flock` 独占 `LOG_ROOT/full_train.lock`，第二个并发训练会立即拒绝；训练 stdout/stderr 用 `tee -a` 追加到固定 `TRAIN_LOG`。重复运行只会从完整且 signature 匹配的 RLOO checkpoint 恢复。

`status.sh` 是只读、单次状态检查：它只读取训练锁、`windows.jsonl` 最后一个完整 JSON 行、`recovery/latest.json`、`run_summary.json` 和当前 invocation 的训练日志尾部，不扫描完整 group 日志。报告明确区分 `running`、`finalizing`、`complete`、`incomplete`、`failed`、`stalled` 和 `error`；只有 1064 个窗口、4256 次更新、完整 summary 与当前 invocation 的退出码 0 同时成立，才会给出 `success=true`。正常训练时按低频计划调用；检测到进程退出、15 分钟无日志进展、Traceback、CUDA OOM 或系统 Kill 时返回非零，随后切换到高频诊断。该纯监控脚本不参与训练行为，因此明确排除在 runtime code fingerprint 之外。
