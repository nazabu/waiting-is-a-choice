# Automated benchmark summary

## Minimal scope, 10 trials (n=256)

| Kernel | Median mean_us | Mean of means | Trial std | p95 mean_us |
|---|---:|---:|---:|---:|
| `serial_two_phase_stream` | 7.7030 | 7.8384 | 0.4067 | 8.9667 |
| `two_kernels_sync` | 14.5079 | 16.8309 | 7.2003 | 37.0321 |
| `fused_pipeline` | 9.3185 | 12.4340 | 7.5920 | 33.1821 |

Headline (median-of-trials): two-kernel/fused = **1.557x**, serial/fused = **0.827x**.

## Size sweep (single run each)

| n | serial_us | two_kernel_us | fused_us | two/fused | serial/fused |
|---:|---:|---:|---:|---:|---:|
| 256 | 7.9044 | 16.6601 | 24.7824 | 0.672x | 0.319x |
| 1024 | 11.6517 | 16.9947 | 10.8214 | 1.570x | 1.077x |
| 2048 | 7.6707 | 13.5937 | 10.7790 | 1.261x | 0.712x |
