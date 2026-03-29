# Route benchmark

## Meta

- **bits**: 3
- **group_size**: 128
- **iterations**: 1
- **warmup**: 0
- **page_size**: 64
- **vector_count**: 16384
- **vector_dim**: 128
- **structured_tokens**: 4096
- **mixed_tokens**: 8192
- **prefer_native_fwht**: True
- **residual_topk**: 1
- **native_fwht**: {'available': True, 'path': '/tmp/hyperquant_native/libhyperquant_fwht.so', 'error': None}

## online_vector_stream

| codec | ratio | rms | cosine | encode ms | decode ms |
|---|---:|---:|---:|---:|---:|
| dense_rotation_baseline | 9.907 | 0.133961 | 0.985330 | 799.980 | 188.928 |
| vector_codec | 9.906 | 0.133764 | 0.985375 | 309.685 | 110.635 |

> Highest compression ratio: `dense_rotation_baseline`. Fastest encode path: `vector_codec`. All codecs were measured on the same synthetic workload with the same harness.

## structured_long_context

| codec | ratio | rms | cosine | encode ms | decode ms |
|---|---:|---:|---:|---:|---:|
| dense_rotation_baseline | 25.854 | 0.237217 | 0.985321 | 497.233 | 200.258 |
| vector_codec | 25.796 | 0.236864 | 0.985424 | 54.423 | 19.978 |
| context_codec | 38.712 | 0.006031 | 0.999991 | 21.499 | 2.763 |

> Highest compression ratio: `context_codec`. Fastest encode path: `context_codec`. All codecs were measured on the same synthetic workload with the same harness.

## mixed_long_context

| codec | ratio | rms | cosine | encode ms | decode ms |
|---|---:|---:|---:|---:|---:|
| dense_rotation_baseline | 12.024 | 0.314323 | 0.985371 | 502.318 | 291.767 |
| vector_codec | 11.993 | 0.314872 | 0.985323 | 119.426 | 38.580 |
| context_codec | 4.523 | 0.007299 | 0.999991 | 97.862 | 4.108 |

> Highest compression ratio: `dense_rotation_baseline`. Fastest encode path: `context_codec`. All codecs were measured on the same synthetic workload with the same harness.
