# Qwen3.8-27B DRRQR piercing commands

This runbook adapts the proven HYPIC command structure while keeping only
commands that are meaningful for DRRQR. It is a template, not evidence that an
experiment has run.

## 0. Bind exact resources

Fill these from the target host. Do not guess card numbers or paths:

```bash
# Reused candidate from the HYPIC run. Keep it only if the Qwen3.8 gate below
# passes; otherwise select the Qwen3.8 image documented by the target
# vLLM-Ascend commit and record its digest.
export IMAGE='m.daocloud.io/quay.io/ascend/vllm-ascend:v0.23.0rc1-a3-openeuler'
export BASELINE_CONTAINER='drror_qwen38_baseline'
export TREATMENT_CONTAINER='drror_qwen38_treatment'
export MODEL_PATH='/cache/cch/Qwen3.8-27B'
export PLUGIN_PATH='/cache/cch/drror_vllm_ascend'
export BENCH_PATH='/cache/cch/benchmark-3.1-20260119-master'
export RESULT_ROOT='/cache/cch/experiments/qwen38-drrqr'

# Operator-selected lane. Re-prove ownership immediately before use.
export HOST_NPUS='4,5,6,7'
export CONTAINER_VISIBLE_NPUS='0,1,2,3'
export HCCL_PORT_RANGE='<reserved-port-start>-<reserved-port-end>'

test -f "${MODEL_PATH}/config.json"
test -f "${MODEL_PATH}/model.safetensors.index.json"
test -d "${PLUGIN_PATH}/.git"
npu-smi info
sha256sum "${MODEL_PATH}/config.json" \
  "${MODEL_PATH}/model.safetensors.index.json"
```

The colleague's HYPIC experiment used two Ascend 910C devices. This is a fresh
four-card identity selected by the operator: host NPU4-7, container-visible
0-3, TP4. Historical two-card measurements are context only.

## 1. Image and isolated baseline container

Reuse the proven image and driver mounts, but expose only assigned devices:

```bash
docker pull "${IMAGE}"

docker run -itd \
  --name "${BASELINE_CONTAINER}" \
  --privileged=true \
  --network=host \
  --shm-size=800g \
  --device=/dev/davinci4:/dev/davinci0 \
  --device=/dev/davinci5:/dev/davinci1 \
  --device=/dev/davinci6:/dev/davinci2 \
  --device=/dev/davinci7:/dev/davinci3 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/vnpu.cfg:/etc/vnpu.cfg \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /cache:/cache \
  "${IMAGE}" bash

docker exec "${BASELINE_CONTAINER}" npu-smi info
```

## 2. Verify baseline runtime (do not install DRRQR)

```bash
docker exec -it "${BASELINE_CONTAINER}" bash

# The remaining commands run inside the container; host shell variables are
# not inherited by an interactive docker exec.
export MODEL_PATH='/cache/cch/Qwen3.8-27B'
export PLUGIN_PATH='/cache/cch/drror_vllm_ascend'
export BENCH_PATH='/cache/cch/benchmark-3.1-20260119-master'
export RESULT_ROOT='/cache/cch/experiments/qwen38-drrqr'
export HOST_NPUS='4,5,6,7'
export CONTAINER_VISIBLE_NPUS='0,1,2,3'
export HCCL_PORT_RANGE='<reserved-port-start>-<reserved-port-end>'

python - <<'PY'
import importlib.metadata as md
print("vllm", md.version("vllm"))
print("vllm-ascend", md.version("vllm-ascend"))
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
print("Qwen3.8 architecture interface", Qwen3_5Model)
PY

python - <<'PY'
import importlib.metadata as md
try:
    md.version("drror-vllm-ascend-plugin")
except md.PackageNotFoundError:
    pass
else:
    raise AssertionError("baseline container must not install DRRQR")
matches = [ep for ep in md.entry_points(group="vllm.general_plugins")
           if ep.name == "drror_vllm_ascend"]
assert not matches, matches
print("baseline confirmed: DRRQR package and entry point are absent")
PY
```

## 3. Shared environment

Use the same settings for baseline and every treatment:

```bash
export ASCEND_RT_VISIBLE_DEVICES="${CONTAINER_VISIBLE_NPUS}"
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_PORT_RANGE}"
export VLLM_ASCEND_DRRQR_STRICT=1
export VLLM_ASCEND_DRRQR_REQUIRE_RUNTIME_HOOKS=1
```

Do not enable prefix caching in the first A/B. It is not part of DRRQR and
would introduce a second treatment variable.

## 4. Baseline service

```bash
export VLLM_ASCEND_DRRQR_ENABLE=0
unset VLLM_ASCEND_DRRQR_CAPTURE_ENABLE
unset VLLM_ASCEND_DRRQR_PLAN_PATH
unset VLLM_ASCEND_DRRQR_PLAN_SHA256
mkdir -p "${RESULT_ROOT}/baseline"

vllm serve "${MODEL_PATH}" \
  --served-model-name qwen3.8 \
  --host 0.0.0.0 \
  --port 6666 \
  --data-parallel-size 1 \
  --tensor-parallel-size 4 \
  --max-model-len 40960 \
  --max-num-batched-tokens 40960 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_capture_sizes":[1,4,8,12,16,24,32,48,56,64,72,84,96,108,112,128,160,172,196,200,212,232,272,288,312,328,344,360,384,400,416,432,448,480,512],"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --trust-remote-code \
  --async-scheduling 2>&1 | tee "${RESULT_ROOT}/baseline/service.log"
```

After all baseline smoke, performance, and accuracy repetitions are complete,
archive the raw results and stop the baseline container:

~~~bash
docker stop "${BASELINE_CONTAINER}"
npu-smi info
~~~

Do not start treatment until NPU4-7 are idle. Keep the stopped baseline
container until its image ID, package list, logs, raw rows, and hashes have
been archived.

## 5. Create and install the isolated treatment container

Use the same image and mounts, but a different container. It must never run at
the same time as the baseline container:

~~~bash
docker run -itd \
  --name "${TREATMENT_CONTAINER}" \
  --privileged=true \
  --network=host \
  --shm-size=800g \
  --device=/dev/davinci4:/dev/davinci0 \
  --device=/dev/davinci5:/dev/davinci1 \
  --device=/dev/davinci6:/dev/davinci2 \
  --device=/dev/davinci7:/dev/davinci3 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/vnpu.cfg:/etc/vnpu.cfg \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /cache:/cache \
  "${IMAGE}" bash

docker exec "${TREATMENT_CONTAINER}" npu-smi info
docker exec -it "${TREATMENT_CONTAINER}" bash

cd /cache/cch/drror_vllm_ascend
git pull --ff-only origin main
pip install --no-deps -e .

python - <<'PY'
import importlib.metadata as md
print("plugin-version", md.version("drror-vllm-ascend-plugin"))
matches = [ep for ep in md.entry_points(group="vllm.general_plugins")
           if ep.name == "drror_vllm_ascend"]
assert len(matches) == 1, matches
print("entry-point", matches[0])
PY
~~~

Record and compare the two containers' image IDs, vLLM and vLLM Ascend
versions, driver/CANN versions, model metadata hashes, dataset hashes, and
benchmark commit. Apart from the DRRQR package and arm-specific variables,
any mismatch invalidates the A/B until reconciled.

## 6. Treatment services

The official dimension rule is
`floor(128 * (1 - pruning_ratio))`: 20%, 30%, and 50% yield 102, 89, and 64.
Each plan must use the same calibration set. This command remains a template
until plan preparation and the plan files pass their tests.

```bash
export ARM='prune20'                 # prune20, prune30, or prune50
export TARGET_DK='102'               # 102, 89, or 64
export VLLM_ASCEND_DRRQR_ENABLE=1
export VLLM_ASCEND_DRRQR_PLAN_PATH="${RESULT_ROOT}/plans/${ARM}.json"
export VLLM_ASCEND_DRRQR_PLAN_SHA256="$(sha256sum "${VLLM_ASCEND_DRRQR_PLAN_PATH}" | awk '{print $1}')"
export VLLM_ASCEND_DRRQR_EVIDENCE_FILE="${RESULT_ROOT}/${ARM}/evidence.jsonl"
mkdir -p "${RESULT_ROOT}/${ARM}"
: > "${VLLM_ASCEND_DRRQR_EVIDENCE_FILE}"

vllm serve "${MODEL_PATH}" \
  --served-model-name qwen3.8 \
  --host 0.0.0.0 \
  --port 6666 \
  --data-parallel-size 1 \
  --tensor-parallel-size 4 \
  --max-model-len 40960 \
  --max-num-batched-tokens 40960 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching \
  --hf-overrides "{\"text_config\":{\"linear_key_head_dim\":${TARGET_DK}}}" \
  --compilation-config '{"cudagraph_capture_sizes":[1,4,8,12,16,24,32,48,56,64,72,84,96,108,112,128,160,172,196,200,212,232,272,288,312,328,344,360,384,400,416,432,448,480,512],"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --trust-remote-code \
  --async-scheduling 2>&1 | tee "${RESULT_ROOT}/${ARM}/service.log"
```

Never reuse one process for another arm. Stop the service, confirm the devices
are idle, and restart with the next immutable plan.

## 7. Smoke request

```bash
curl http://127.0.0.1:6666/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8",
    "prompt": "Please reproduce verbatim the opening sentence of the United States Declaration of Independence (1776), starting with \"When in the Course of human events\".",
    "max_tokens": 100,
    "temperature": 0
  }'
```

HTTP 200 is only a smoke gate, not accuracy or performance evidence.

For a treatment, the smoke gate is incomplete until the independent TP4
activation audit passes:

```bash
drror-audit-activation \
  --evidence "${VLLM_ASCEND_DRRQR_EVIDENCE_FILE}" \
  --plan-sha256 "${VLLM_ASCEND_DRRQR_PLAN_SHA256}" \
  --target-head-k-dim "${TARGET_DK}" \
  --expected-workers 4 \
  --output "${RESULT_ROOT}/${ARM}/activation-audit.json"
```

Required worker events are model configuration, complete 96-tensor weight
loading, and an actual reduced GDN hot-path event on four distinct worker PIDs.

## 8. Performance test

Reuse the general fixed-token workload: 32K input, 1K output, 40 requests,
concurrency 16. Do not use `benchmark_prefix_coexistence.py`; relocated PIC and
PIC evidence measure HYPIC rather than DRRQR.

```bash
cd /cache/cch/q3_q4_project/aisbench_auto_tools_prefix_base

# Configure once per arm:
# DATASET_PATH="/home/dataset_base"
# WORK_PATH="/cache/cch/benchmark-3.1-20260119-master"
# MODEL_NAME="qwen3.8"
# HOST_IP="127.0.0.1"
# HOST_PORT="6666"

mkdir -p /home/dataset_base
python3 aisbench_test.py \
  --input_len 32768 \
  --output_len 1024 \
  --data_num 40 \
  --concurrency 16
```

Run warmup first, then at least three measured repetitions for every arm.
Archive raw per-request output, not only one aggregate.

## 9. Accuracy test

Reuse ais_bench 3.1.0. In its service adapter set the Qwen3.8 path/name, local
port 6666, batch size 32, temperature 0, and thinking disabled. Then run:

```bash
cd /cache/cch/benchmark-3.1-20260119-master
pip install -e ./

cd ais_bench/datasets
cp -r /cache/hhs-aisbench/ais_bench/datasets/LongBench-v2 ./

cd /cache/cch/benchmark-3.1-20260119-master
ais_bench --models vllm_api_general_chat --datasets longbenchv2_gen --debug --dump-eval-details
ais_bench --models vllm_api_general_chat --datasets longbenchv2_gen --debug --dump-eval-details
ais_bench --models vllm_api_general_chat --datasets longbenchv2_gen --debug --dump-eval-details
```

NIAH+ and GSM8K must use the same local copies and commands for all arms.
C-Eval is excluded by the current experiment decision. Hash every dataset copy.

## 10. Arm sequence and evidence

The order is fixed: finish and freeze baseline in the plugin-free baseline
container, stop it and prove NPU4-7 are idle, then create the treatment
container and run prune20, prune30, and prune50.

For each arm:

1. Start a fresh service, wait for health, and run the smoke request.
2. Run performance warmup and three measured repetitions.
3. Stop the service and confirm the devices are idle.
4. Restart the same arm and run each selected accuracy dataset three times.
5. Archive service log, evidence JSONL, raw requests, environment, commits,
   hashes, and before/after `npu-smi` snapshots.

The HYPIC results in the reference document are historical comparison data,
not a DRRQR baseline. They must not be reported as DRRQR measurements.
