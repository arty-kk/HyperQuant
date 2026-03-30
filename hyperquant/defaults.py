# Copyright 2026 Сацук Артём Венедиктович (Satsuk Artem)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

VECTOR_BITS_DEFAULT = 3
VECTOR_GROUP_SIZE_DEFAULT = 128
VECTOR_ROTATION_SEED_DEFAULT = 17
VECTOR_RESIDUAL_TOPK_DEFAULT = 1
VECTOR_PREFER_NATIVE_FWHT_DEFAULT = True

CONTEXT_PAGE_SIZE_DEFAULT = 64
CONTEXT_RANK_DEFAULT = 1
CONTEXT_PREFIX_KEEP_VECTORS_DEFAULT = 32
CONTEXT_SUFFIX_KEEP_VECTORS_DEFAULT = 64
CONTEXT_LOW_RANK_ERROR_THRESHOLD_DEFAULT = 0.03
CONTEXT_REF_ROUND_DECIMALS_DEFAULT = 3
CONTEXT_ENABLE_PAGE_REF_DEFAULT = True
CONTEXT_PAGE_REF_REL_RMS_THRESHOLD_DEFAULT = 0.005
CONTEXT_ENABLE_INT8_FALLBACK_DEFAULT = True
CONTEXT_TRY_INT8_FOR_PROTECTED_DEFAULT = True
CONTEXT_INT8_REL_RMS_THRESHOLD_DEFAULT = 0.01
CONTEXT_INT8_MAX_ABS_THRESHOLD_DEFAULT = 0.05

RESIDENT_HOT_PAGES_DEFAULT = 8
RESIDENT_ALLOW_VECTOR_FOR_PROTECTED_DEFAULT = False
RESIDENT_CONCURRENT_SESSIONS_DEFAULT = 8
RESIDENT_ACTIVE_WINDOW_TOKENS_DEFAULT = 256
RESIDENT_RUNTIME_VALUE_BYTES_DEFAULT = 2
