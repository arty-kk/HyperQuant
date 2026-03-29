# Resident benchmark

## Meta

- **page_size**: 64
- **rank**: 1
- **bits**: 3
- **group_size**: 128
- **hot_pages**: 8
- **concurrent_sessions**: 8
- **active_window_tokens**: 256
- **runtime_value_bytes**: 2
- **slice_iterations**: 1

## online_vector_stream

| candidate | resident/session bytes | artifact/session bytes | ratio vs baseline |
|---|---:|---:|---:|
| baseline_full_resident | 4194304 | 4194304 | 1.0000 |
| vector_codec_full_envelope | 884832 | 846826 | 0.2110 |
| resident_tier | 159943 | 2383700 | 0.0381 |
| context_codec_full_envelope | 2322746 | 2227686 | 0.5538 |

> Tiered route `resident_tier`: resident/session = 159943 bytes, artifact/session = 2383700 bytes, slice read = 4.574 ms.

## structured_long_context

| candidate | resident/session bytes | artifact/session bytes | ratio vs baseline |
|---|---:|---:|---:|
| baseline_full_resident | 4194304 | 4194304 | 1.0000 |
| vector_codec_full_envelope | 884832 | 366689 | 0.2110 |
| resident_tier | 159901 | 416248 | 0.0381 |
| context_codec_full_envelope | 182074 | 167646 | 0.0434 |

> Tiered route `resident_tier`: resident/session = 159901 bytes, artifact/session = 416248 bytes, slice read = 3.367 ms.

## mixed_long_context

| candidate | resident/session bytes | artifact/session bytes | ratio vs baseline |
|---|---:|---:|---:|
| baseline_full_resident | 8388608 | 8388608 | 1.0000 |
| vector_codec_full_envelope | 1769568 | 1374440 | 0.2109 |
| resident_tier | 252008 | 3181922 | 0.0300 |
| context_codec_full_envelope | 3797178 | 3585551 | 0.4527 |

> Tiered route `resident_tier`: resident/session = 252008 bytes, artifact/session = 3181922 bytes, slice read = 2.955 ms.
