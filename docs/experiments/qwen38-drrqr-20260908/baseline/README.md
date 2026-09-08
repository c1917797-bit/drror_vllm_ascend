# Baseline data

This directory contains the accepted no-plugin baseline for the Qwen3.8-27B DRRQR piercing experiment.

## Performance

Each measured run completed 40/40 requests, with 1,024 output tokens per request and 40,960 output tokens per run.

| Run | Duration (s) | Output throughput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 349.061876 | 117.343093 | 42317.222010 | 78.957545 |
| 2 | 334.840938 | 122.326739 | 38885.543035 | 76.761132 |
| 3 | 334.880422 | 122.312316 | 38884.594205 | 76.777863 |
| Arithmetic mean | 339.594412 | 120.660716 | 40029.119750 | 77.498847 |
| Sample SD | 8.199088 | 2.873155 | 1981.554740 | 1.263297 |

TTFT and TPOT are first averaged across the 40 requests within each run. The reported baseline is then the arithmetic mean of the three run-level means. Run 1 was not preregistered as warm-up, so it remains part of the accepted baseline.

## LongBench quality

Each run completed 127/127 samples with zero request failures.

| Run | Correct | Samples | Accuracy |
| --- | ---: | ---: | ---: |
| 1 | 57 | 127 | 44.881890% |
| 2 | 59 | 127 | 46.456693% |
| 3 | 57 | 127 | 44.881890% |
| Arithmetic mean | 57.667 | 127 per run | 45.406824% |

The sample standard deviation of the three accuracy values is 0.909213 percentage points.

## Runtime identity

- Model: `/cache/austinov/Qwen3.8-27B`
- Runtime: vLLM-Ascend 0.23.0
- Precision: BF16
- Tensor parallelism: TP=4
- Host devices: NPU4-7, mapped to container-visible NPU0-3
- DRRQR plugin: not installed and not loaded

The full request-level LongBench JSONL files remain in immutable host storage because they total approximately 1.5 MB. Their hashes and binding checks are recorded in the summary and audit files included here.
