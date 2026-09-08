# Qwen3.8-27B DRRQR Ascend 穿刺交付索引

本目录记录 DRRQR 0.1.4 在 Qwen3.8-27B、vLLM-Ascend 0.23.0、4×Ascend NPU（宿主 NPU4–7、容器内0–3）、BF16、TP4 环境上的正式同机 A/B 实验。实验已关闭，三档候选均拒绝；该结论不等价于否定论文算法，只约束本次插件、模型、硬件和测试协议。

## 最终数据

| 配置 | 实际剪枝率 | 吞吐 tok/s | 吞吐变化 | TTFT变化 | TPOT变化 | LongBench均值 | 精度变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0% | 120.661 | — | — | — | 45.407% | — |
| Dk=104 | 18.75% | 117.347 | -2.75% | +4.01% | +2.46% | 19.948% | -25.459 pp |
| Dk=88 | 31.25% | 118.315 | -1.94% | +2.82% | +1.79% | 14.436% | -30.971 pp |
| Dk=64 | 50% | 130.678 | +8.30% | -6.53% | -8.15% | 35.958% | -9.449 pp |

性能每档3轮，每轮40/40请求、每请求1024输出tokens；LongBench每档3轮，每轮127/127样本、零失败。TTFT/TPOT负值表示延迟降低。

## 已证实

- 插件源码、远端main与实验身份均锁定为 commit `d26dd8cde240fb578442f108620e7bb33ef703c4`。
- 实验wheel SHA256为 `014200a9c32f0b05fd0ea328f8664de43849a80aca2fa7a9d6cb46299aaa3d40`。
- 四个worker均加载插件，48个线性注意力层均有计划，Ascend GDN prefill/decode热路径均有激活证据。
- Dk=64同时改善吞吐、TTFT和TPOT，但LongBench下降9.449个百分点，未通过精度门禁。
- Dk=104和Dk=88性能略退化且精度严重下降；正确答案集合跨轮稳定性显著低于baseline与Dk=64。
- 最终独立审计为 `ok=true, errors=[]`；所有本实验容器已停止，资源释放审计通过。

## 不得越界推断

- 目前不能把20%/30%的非单调精度异常归因于DRRQR理论。
- 还没有证明论文参考实现、PyTorch参考路径与Ascend kernel逐层数值等价。
- 小剪枝率无端到端收益可能来自插件固定开销或算子shape效率，但当前没有profiling证据证明具体占比。
- 因没有候选同时通过性能和LongBench门禁，本实验没有运行GSM8K或native profiling。

## 成功经验

1. 基线与treatment使用独立容器，固定模型、镜像、TP、请求集合和输出长度。
2. 先锁定源码commit与wheel哈希，再启动服务；不能先pull旧插件后以本地新代码解释结果。
3. “插件安装成功”不是激活证据；必须在所有worker上证明目标层、计划以及prefill/decode热路径。
4. 性能与精度均进行三轮重复，并保存逐请求明细、完成数、失败数和哈希绑定。
5. 使用门禁控制昂贵阶段：LongBench不过则不扩展GSM8K，无唯一合格候选则不做profiling。
6. 任务结束后执行独立证据审计和资源释放审计。

## 失败经验与陷阱

1. 宿主NPU4–7应显式映射为容器内0–3；混用宿主编号与容器编号会导致初始化失败。
2. 控制器等待超时不等于容器内任务失败。Dk=88曾发生执行器超时，但原容器仍运行；核验身份后复用，避免盲目重启污染数据。
3. Qwen3.8检查点声明的 `Qwen3_5ForConditionalGeneration/qwen3_5_text` 是运行时ABI，不是换用了Qwen3.5权重。
4. 理论FLOPs下降不保证端到端性能单调；固定开销与硬件shape门槛可能吞掉小比例剪枝收益。
5. 精度非单调且跨轮正确集不稳定时，必须停止参数扫描并进入数值对齐，不得用故事补足因果。

## 基线复用规则

黄金基线可以复用，但每批实验至少执行轻量baseline anchor。模型或模型哈希、镜像、vLLM/vLLM-Ascend、CANN/驱动/固件、NPU映射、TP、启动参数、数据集、生成参数、benchmark代码任一变化，必须重跑完整基线。跨日、机器负载异常或anchor超出预设容差时也必须重跑。

## 下一步

新建独立实验身份，分别对Dk=128、104、88、64执行论文参考实现、PyTorch参考路径和Ascend实际kernel的逐层数值对齐，定位第一个偏差层。不得续写或重解释本次已关闭实验。

## 文件导航

- `PIERCING_REPORT.md` / `PIERCING_REPORT.html`：完整穿刺报告。
- `HANDOFF.md`：历史、资源约束和恢复说明。
- `experiment.json`、`evidence.json`、`three-arm-quality-evidence.json`：实验协议与结构化证据。
- `decision.json`：正式决策。
- `knowledge-proposals.json`：可提升为维护者知识的候选经验。
- `incident-prune30-executor-timeout.json`：超时事件与恢复过程。
- `final-evidence-audit.json`、`audit_final.py`：独立审计结果与审计程序。
- `task.json`：最终任务状态及精确恢复约束。

## 精确复核

```bash
cd /cache/cch/drror_vllm_ascend
git rev-parse HEAD
python3 docs/experiments/qwen38-drrqr-20260908/audit_final.py
python3 -m json.tool docs/experiments/qwen38-drrqr-20260908/final-evidence-audit.json >/dev/null
```

审计脚本依赖本机保留的原始实验产物目录；仓库内JSON与报告保存汇总、哈希绑定和决策。若迁移到其他主机，只能审阅已提交证据，不能声称重新完成原始产物审计。
