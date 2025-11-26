# Issue: Tests fail for external contributors due to incomplete skip logic

## Summary

When setting up a local development environment as an external contributor, several tests fail due to missing or incomplete skip logic for:
1. S3 bucket access (403/AccessDenied errors)
2. Optional dependencies (liger-kernel)
3. Multi-GPU requirements

## Problem Details

### 1. S3 Tests - Missing skip for AccessDenied (403)

**Affected tests:**
- `src/test/io_test.py::test_s3_functionality`
- `src/test/data/mixes_test.py::test_olmoe_mix`
- `src/test/data/mixes_test.py::test_dolma17_mix`
- `src/test/data/mixes_test.py::test_v3_small_ppl_validation_mix`
- `src/test/train/checkpoint_test.py::test_checkpointer_with_remote_s3_dir`
- `src/test/train/checkpoint_test.py::test_async_checkpointer_with_remote_s3_dir`
- `src/test/distributed/checkpoint/filesystem_test.py::test_save_and_load_remotely_to_s3_with_dtensors`

**Current behavior:** Tests catch `NoCredentialsError` but not `ClientError` with 403/AccessDenied. When no credentials are configured, botocore falls back to anonymous/unsigned requests, which return 403 Forbidden from private S3 buckets (rather than raising `NoCredentialsError`).

**Error seen:**
```
olmo_core.exceptions.OLMoNetworkError: '_s3_file_size' failed 3 attempts with: An error occurred (403) when calling the HeadObject operation: Forbidden
```

**Fix:** Also catch `ClientError` (for 403/AccessDenied) and `OLMoNetworkError` (which wraps the error after retries).

### 2. liger-kernel Test - Missing skip decorator

**Affected test:**
- `src/test/nn/lm_head_test.py::test_lm_head_fused_linear_loss`

**Current behavior:** Test has `@requires_gpu` but no skip for missing liger-kernel.

**Error seen:**
```
RuntimeError: 'fused_linear_cross_entropy_loss' requires liger-kernel
```

**Fix:** Add `@pytest.mark.skipif(not _has_liger_kernel(), reason="Requires liger-kernel")`.

### 3. Multi-GPU Test - Missing @requires_multi_gpu decorator

**Affected test:**
- `src/test/distributed/checkpoint_test.py::test_load_checkpoint_with_missing_keys`

**Current behavior:** Test spawns 2 distributed processes but lacks `@requires_multi_gpu` marker.

**Error seen:**
```
torch.AcceleratorError: CUDA error: initialization error
```

**Fix:** Add `@requires_multi_gpu` decorator.

## Environment

- Python 3.11
- PyTorch 2.8.0+cu128
- Single GPU (RTX 3090)
- No access to Ai2 internal S3 buckets

## Proposed Changes

1. Update S3 test exception handling to skip on 403/AccessDenied
2. Add liger-kernel availability check and skip decorator
3. Add `@requires_multi_gpu` to distributed checkpoint test

## Notes

- The S3 mixes tests (`test_olmoe_mix`, etc.) are read-only and could potentially use public URLs if training data is available elsewhere (e.g., HuggingFace). However, these tests specifically verify internal S3 path structure, so skipping is the appropriate fix.
- There may be additional tests with similar issues hidden beneath currently-skipped tests. Further investigation may be needed.

## Next Steps

1. Open GitHub issue describing the problem
2. Submit PR with fixes (referencing the issue)
3. Mention on Discord for visibility: "Submitted first PR #X to fix test skips for external contributors. Happy to adjust based on feedback!"
