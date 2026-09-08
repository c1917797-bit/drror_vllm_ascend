# Qwen3.8-27B DRRQR Ascend 穿刺报告

> 算法：Strong RRQR / DRRQR state reduction
> 论文：*The Key to State Reduction in Linear Attention: A Rank-based Perspective*
> 模型：Qwen3.8-27B BF16 混合注意力模型
> 设备：4× Ascend 910B4，仅宿主 NPU4-7
> 运行时：vLLM-Ascend 0.23.0，TP4
> 结论日期：2026-09-08

## 1. 最终结论

本次完成了可审计的同机 baseline/treatment 穿刺，但当前三个 DRRQR
候选均不满足“性能提升且精度可接受”的联合门禁，结论为 **不采用**。

| 配置 | 实际削减率 | 吞吐变化 | TTFT 变化 | TPOT 变化 | LongBench-v2 均值 | 相对基线 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Dk128 | 0% | — | — | — | 45.406824% | — |
| Dk104 | 18.75% | -2.746264% | +4.0149% | +2.4621% | 19.947507% | -25.459318 pp |
| Dk88 | 31.25% | -1.943667% | +2.8229% | +1.7928% | 14.435696% | -30.971129 pp |
| Dk64 | 50% | +8.301873% | -6.527346% | -8.152862% | 35.958005% | -9.448819 pp |

Dk104 和 Dk88 同时损失性能和精度。Dk64 获得约 8.30% 吞吐提升，但
LongBench-v2 稳定下降约 9.45 个百分点，不是精度无损加速。

## 2. 身份与边界

- 模型：`/cache/austinov/Qwen3.8-27B`
- 模型配置 SHA256：
  `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`
- 权重索引 SHA256：
  `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`
- 18/18 权重分片，总计 55,563,006,776 bytes
- 混合注意力拓扑：64 层，其中 48 层 linear attention、16 层 full attention
- 原始 GDN key/value head dimension：128/128
- `Qwen3_5ForConditionalGeneration`、`qwen3_5_text` 是该 Qwen3.8
  权重自身声明的运行时 ABI 名称，不代表替换成了 Qwen3.5 模型。
- 论文 PDF：
  `/cache/cch/papers/2602.04852v2/2602.04852v2.pdf`
- 官方代码：
  `/cache/cch/LinearAttentionPruning`
- 官方代码 commit：
  `919d8667d951c385e08510bc1267c2e7049a4f56`
- 只使用宿主 NPU4-7；容器内映射为 NPU0-3。宿主 NPU0-3 未检查、未操作。
- CEval 按用户要求排除。

## 3. Step 0：插件与运行环境

DRRQR 以独立 monkeypatch Python 包交付，不侵入 baseline 容器：

- 仓库：`/cache/cch/drror_vllm_ascend`
- 远端：`github.com/c1917797-bit/drror_vllm_ascend`
- 当前本地与 `origin/main`：
  `d26dd8cde240fb578442f108620e7bb33ef703c4`
- 插件版本：0.1.4
- wheel：
  `/cache/cch/state-reduction-qwen38-drrqr-plugin-tp4-20260907/plugin-artifact-aligned-v5/drror_vllm_ascend_plugin-0.1.4-py3-none-any.whl`
- wheel SHA256：
  `014200a9c32f0b05fd0ea328f8664de43849a80aca2fa7a9d6cb46299aaa3d40`
- 源码与独立 wheel 环境均通过 30/30 测试。

可靠拉取与安装命令：

```bash
cd /cache/cch/drror_vllm_ascend
git fetch --prune origin
test "$(git rev-parse origin/main)" = +  "d26dd8cde240fb578442f108620e7bb33ef703c4"
git switch main
git pull --ff-only origin main
test "$(git rev-parse HEAD)" = +  "d26dd8cde240fb578442f108620e7bb33ef703c4"
git status --short
pip install --no-deps -e .
```

## 4. Step 1：独立 baseline

baseline 容器
`qwen38_drrqr_baseline_exact_host4_7_tp4_20260907_v6`
只安装同版本 vLLM-Ascend，不安装 DRRQR 插件。其保存的实际启动核心参数为：

```bash
docker run --name qwen38_drrqr_baseline_exact_host4_7_tp4_20260907_v6 \
  --cpuset-cpus 0-95 \
  -p 127.0.0.1:8227:8227 \
  --device=/dev/davinci4:/dev/davinci0 \
  --device=/dev/davinci5:/dev/davinci1 \
  --device=/dev/davinci6:/dev/davinci2 \
  --device=/dev/davinci7:/dev/davinci3 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /cache/austinov/Qwen3.8-27B:/cache/austinov/Qwen3.8-27B:ro \
  quay.io/ascend/vllm-ascend:v0.23.0 \
  vllm serve /cache/austinov/Qwen3.8-27B \
    --host 0.0.0.0 --port 8227 --served-model-name qwen38 \
    --tensor-parallel-size 4 --max-num-seqs 16 \
    --max-model-len 41984 --max-num-batched-tokens 16384 \
    --kv-cache-memory=14495514624 --trust-remote-code \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true}'
```

完整驱动和 CANN 只读挂载以保存的 container inspect 为准。baseline 的插件缺席、
模型列表 HTTP 200、确定性 smoke 和资源释放均通过审计。

### Baseline 性能

同事成功案例的性能 ruler 被复用：40 请求、并发 16、输入长度 361、
固定输出 1024、2 次 warmup、3 次正式重复。

| Run | output tok/s | TTFT ms | TPOT ms | duration s |
|---:|---:|---:|---:|---:|
| 1 | 117.343093 | 42317.222010 | 78.957545 | 349.061876 |
| 2 | 122.326739 | 38885.543035 | 76.761132 | 334.840938 |
| 3 | 122.312316 | 38884.594205 | 76.777863 | 334.880422 |
| mean | 120.660716 | 40029.119750 | 77.498847 | 339.594412 |

每轮均为 40/40 成功且严格输出 40,960 tokens。

### Baseline 精度

LongBench-v2 固定 127 题，三轮分别为：

- 57/127 = 44.881890%
- 59/127 = 46.456693%
- 57/127 = 44.881890%
- 均值 45.406824%，样本标准差 0.909213 pp

## 5. Step 2：校准、捕获与 DRRQR plan

校准数据来自提供的 LongBench-v2 完整文件，不与 127 题评测集重合：

- 完整源：503 行
- 评测集：127 行
- 候选非评测行：376 行
- 最终校准：16 个不重合样本，每个截断到 2048 tokens
- 捕获：3072/3072 tensors
- 覆盖：48/48 linear-attention 层；16 个 full-attention 层保持不变

这是一项明确标注的 Ascend adaptation；因未提供论文的 FineWeb-Edu
校准语料，不能宣称 paper-exact reproduction。

硬件对 float32 recurrent-state 行有 32-byte 写入对齐要求。名义 Dk102 和
Dk89 分别产生 408-byte、356-byte 行，不能运行。没有关闭断言或伪造结果，
而是建立相邻的硬件可运行档：

| Dk | 实际削减率 | plan SHA256 |
|---:|---:|---|
| 104 | 18.75% | `5bc3eabe4f98df880b28a287e28fe3c0fc9a8036b120c85f211d3cf41f7f9830` |
| 88 | 31.25% | `c75468decf7752388f3e6beb57b207bb6d426aa69b8aaf588431ceb22f7d6daf` |
| 64 | 50% | `3824e60fb9137fa1a1b134bff4594495014625583fdd62fdf5b9bc92f39b8bea` |

三份 plan 在所有 48 层上是嵌套的：Dk64 ⊂ Dk88 ⊂ Dk104。

## 6. Step 3：确认插件事实加载

每个 treatment 在 benchmark 前都必须通过：

1. 包与 entrypoint 可发现；
2. monkeypatch 已安装；
3. 96 个目标 Q/K/conv 权重变换完整；
4. 四个 TP worker 均出现 `model_configured`、
   `weight_load_complete`、`worker_dispatch_verified`；
5. 实际 Ascend `npu_recurrent_gated_delta_rule` decode 热路径使用目标 Dk；
6. prefill 和 decode smoke 成功。

Dk104、Dk88、Dk64 均通过这些门禁。因此精度结果不能解释为“插件没有加载”。

## 7. Step 4：Treatment 性能

| Arm | 吞吐三轮 tok/s | 均值 | 相对 baseline |
|---|---|---:|---:|
| Dk104 | 见原始 JSON 三轮 | 117.347055 | -2.746264% |
| Dk88 | 115.214308, 119.903224, 119.828889 | 118.315474 | -1.943667% |
| Dk64 | 126.682785 等三轮 | 130.677815 | +8.301873% |

Dk104、Dk88 没有端到端性能收益。Dk64 的 TTFT、TPOT 和吞吐方向均有利，
但必须继续通过精度门禁。

执行入口：

```bash
cd /cache/cch/.worktrees/WS-20260904-qwen38-dense-paper-to-ascend
python3 .agents/workstreams/WS-20260904-qwen38-dense-paper-to-ascend/state_reduction_qwen38_plugin_runner.py --phase self-test
python3 .agents/workstreams/WS-20260904-qwen38-dense-paper-to-ascend/state_reduction_qwen38_plugin_runner.py --phase performance --arm prune20_aligned
python3 .agents/workstreams/WS-20260904-qwen38-dense-paper-to-ascend/state_reduction_qwen38_plugin_runner.py --phase performance --arm prune30_aligned
python3 .agents/workstreams/WS-20260904-qwen38-dense-paper-to-ascend/state_reduction_qwen38_plugin_runner.py --phase performance --arm prune50
```

## 8. Step 5：Treatment LongBench-v2

| Arm | Run 1 | Run 2 | Run 3 | Mean | SD |
|---|---:|---:|---:|---:|---:|
| Dk104 | 20.472441% | 21.259843% | 18.110236% | 19.947507% | 1.639107 pp |
| Dk88 | 10.236220% | 16.535433% | 16.535433% | 14.435696% | 3.636852 pp |
| Dk64 | 35.433071% | 33.858268% | 38.582677% | 35.958005% | 2.405552 pp |

所有 9 个 treatment 质量运行均为 127/127 请求成功、0 failure。

执行入口：

```bash
python3 .agents/workstreams/WS-20260904-qwen38-dense-paper-to-ascend/state_reduction_qwen38_plugin_runner.py --phase longbenchv2 --arm prune20_aligned
python3 .agents/workstreams/WS-20260904-qwen38-dense-paper-to-ascend/state_reduction_qwen38_plugin_runner.py --phase longbenchv2 --arm prune30_aligned
python3 .agents/workstreams/WS-20260904-qwen38-dense-paper-to-ascend/state_reduction_qwen38_plugin_runner.py --phase longbenchv2 --arm prune50
```

## 9. 为什么没有继续 GSM8K 和 profiling

预注册门禁规定：

- 只有 LongBench-v2 穿刺成功的候选才补测 GSM8K；
- 只有同时通过性能与质量的唯一 winner 才做 native profiling。

三个候选均未通过联合门禁。继续运行 GSM8K 或 profiling 既不会使当前候选
变成可采用，也会扩大算力消耗。因此状态被明确记录为
`not-run-by-preregistered-gate`，不是“遗漏”或“伪造通过”。

## 10. 非单调质量异常

Dk104/Dk88 的精度比更激进的 Dk64 更差，不能用“剪得越少越安全”解释。
逐题正确集合的三轮 Jaccard 为：

| Arm | 三组 pairwise Jaccard |
|---|---|
| Baseline | 0.8413, 0.9000, 0.9016 |
| Dk104 | 0.2619, 0.2250, 0.2500 |
| Dk88 | 0.0968, 0.1724, 0.1667 |
| Dk64 | 0.8333, 0.7736, 0.7692 |

这证明 Dk104/Dk88 在当前 serving path 上具有强烈的重复间不稳定性；它不证明
根因一定是 DRRQR，也不证明一定是 Ascend kernel。后续若继续研究，必须新建
实验身份，先做 Dk128/Dk104/Dk88/Dk64 的直接数值一致性与 kernel parity，
再决定是否修改算法或 runtime。

## 11. 与同事 HYPIC 成功案例的对比

| 项目 | 同事 HYPIC 案例 | 本次 DRRQR |
|---|---|---|
| 目标 | 位置无关缓存加速 | linear-attention recurrent state 降维 |
| 模型 | Qwen3.5-27B | Qwen3.8-27B BF16 |
| 卡数 | 2× Ascend 910C | 4× Ascend 910B4 |
| 集成 | 独立可编辑安装插件 | 独立 monkeypatch wheel/可编辑安装 |
| 激活证明 | runtime hooks/evidence | 四 worker + 96 权重变换 + prefill/decode hot path |
| baseline/treatment | 独立配置 | 独立容器，baseline 不装插件 |
| 性能重复 | 固定 ruler | 同样三轮固定输出 ruler |
| 精度 | 成功案例通过 | 三档均失败 |
| 结论 | 可报告收益 | 当前配置不采用 |

成功案例最有价值的不是某组环境变量，而是“独立容器、真实热路径证据、固定
A/B ruler、重复测量、失败留痕、资源释放”这套方法。本次完整复用了该方法，
因此即使结论为负，也能回答收益多少以及为什么不能采用。

## 12. 失败与恢复经验

1. 名义 Dk102/Dk89 在 benchmark 前因 AscendC 行对齐不兼容，归类为实现兼容
   incident，不能算算法负结果。
2. Dk88 的首次 launcher 使用了错误的 30 秒 child timeout；operation
   `exec-0000000000001039` 终止时容器仍在。核验精确容器后由
   `exec-0000000000001043` 复用并完成，没有盲删容器或重跑已接受数据。
3. 长作业必须使用足够的 child timeout，并同时检查 operation 与 container
   生命周期。
4. 不能因为插件测试通过就宣称算法生效；必须有每个 worker 的真实硬件热路径
   证据。
5. 不能因为某一档性能提升就跳过质量门禁。

## 13. 最终审计与交接

独立审计命令：

```bash
cd /cache/cch/.worktrees/WS-20260904-qwen38-dense-paper-to-ascend
python3 .agents/workstreams/WS-20260904-qwen38-dense-paper-to-ascend/audit_qwen38_drrqr_final.py
```

结果：

- `ok=true`
- `errors=[]`
- 审计文件：
  `/cache/cch/state-reduction-qwen38-drrqr-plugin-tp4-20260907/final-evidence-audit.json`
- 本实验标签下运行容器：0
- NPU4-7 和端口 8227：已由各阶段 release audit 证明释放

当前实验已经终止，没有需要恢复的 operation。精确只读检查：

```bash
cat /cache/cch/state-reduction-qwen38-drrqr-plugin-tp4-20260907/final-evidence-audit.json
cat /cache/cch/.worktrees/WS-20260904-qwen38-dense-paper-to-ascend/.agents/workstreams/WS-20260904-qwen38-dense-paper-to-ascend/decision-state-reduction-qwen38-drrqr-plugin-tp4-20260907.json
cat /cache/cch/.worktrees/WS-20260904-qwen38-dense-paper-to-ascend/.agents/workstreams/WS-20260904-qwen38-dense-paper-to-ascend/evidence-qwen38-drrqr-three-arm-quality-20260908.json
```

不要在当前 experiment identity 下删除证据后重跑。若研究非单调异常，应创建
新 identity、重新冻结 baseline 与 test contract，并继续保持 NPU0-3 隔离。
