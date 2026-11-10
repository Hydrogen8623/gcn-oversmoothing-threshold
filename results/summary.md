# Experiment Summary

Aggregation key: `(k, layers)`.

| k | layers | test_acc (mean ± std) | n_seeds | missing_seeds | noise_scales | run_ids |
|---:|---:|---:|---:|---|---|---|
| 4 | 1 | 0.994 ± 0.009 | 5 | None | 0.1, 0.5, 1.0 | k4_L1_s0_noise0.1; k4_L1_s0_noise0.5; k4_L1_s0_noise1.0; k4_L1_s1_noise0.1; k4_L1_s1_noise0.5; k4_L1_s1_noise1.0; k4_L1_s2_noise0.1; k4_L1_s2_noise0.5; k4_L1_s2_noise1.0; k4_L1_s3_noise0.1; k4_L1_s3_noise0.5; k4_L1_s3_noise1.0; k4_L1_s4_noise0.1; k4_L1_s4_noise0.5; k4_L1_s4_noise1.0 |
| 4 | 2 | 0.999 ± 0.001 | 5 | None | 0.1, 0.5, 1.0 | k4_L2_s0_noise0.1; k4_L2_s0_noise0.5; k4_L2_s0_noise1.0; k4_L2_s1_noise0.1; k4_L2_s1_noise0.5; k4_L2_s1_noise1.0; k4_L2_s2_noise0.1; k4_L2_s2_noise0.5; k4_L2_s2_noise1.0; k4_L2_s3_noise0.1; k4_L2_s3_noise0.5; k4_L2_s3_noise1.0; k4_L2_s4_noise0.1; k4_L2_s4_noise0.5; k4_L2_s4_noise1.0 |
| 4 | 4 | 0.999 ± 0.001 | 5 | None | 0.1, 0.5, 1.0 | k4_L4_s0_noise0.1; k4_L4_s0_noise0.5; k4_L4_s0_noise1.0; k4_L4_s1_noise0.1; k4_L4_s1_noise0.5; k4_L4_s1_noise1.0; k4_L4_s2_noise0.1; k4_L4_s2_noise0.5; k4_L4_s2_noise1.0; k4_L4_s3_noise0.1; k4_L4_s3_noise0.5; k4_L4_s3_noise1.0; k4_L4_s4_noise0.1; k4_L4_s4_noise0.5; k4_L4_s4_noise1.0 |

## Notes

- `n_seeds` counts distinct seeds among completed runs with valid test accuracy.
- `missing_seeds` is checked against the expected seed set: 0, 1, 2, 3, 4.
- Standard deviation uses the sample standard deviation (ddof=1).
- Groups follow the requested `(k, layers)` key. If multiple noise scales exist, their runs are included in the same group; see `noise_scales` and `run_ids`.
- Failed or incomplete runs are retained in `summary.csv`, but excluded from accuracy aggregation.
