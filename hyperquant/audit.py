# SPDX-FileCopyrightText: 2026 Сацук Артём Венедиктович (Satsuk Artem)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from .benchmark import stats_to_pretty_json
from .bundle import CodebookBundle
from .codebook_codec import CodebookCodec
from .guarantee import GuaranteeMode, ContextGuaranteeProfile
from .context_codec import ContextCodec, ContextCompressionStats, ContextCodecConfig


@dataclass
class AuditArtifacts:
    stats: ContextCompressionStats
    markdown: str

    def to_json(self) -> str:
        return stats_to_pretty_json(self.stats)


def audit_context_input(
    array: np.ndarray,
    *,
    context_config: ContextCodecConfig,
    guarantee_profile: ContextGuaranteeProfile | None = None,
    bundle_path: str | Path | None = None,
    protected_vector_indices: list[int] | None = None,
) -> AuditArtifacts:
    context = ContextCodec(context_config)
    _, context_stats = context.compress(
        array,
        protected_vector_indices=protected_vector_indices,
        guarantee_profile=guarantee_profile,
        guarantee_mode=GuaranteeMode.ALLOW_BEST_EFFORT,
    )

    conservative_block = "not run"
    if bundle_path is not None:
        bundle = CodebookBundle.load(bundle_path)
        conservative = CodebookCodec(bundle)
        _, conservative_stats = conservative.compress(array, protected_vector_indices=protected_vector_indices)
        conservative_block = stats_to_pretty_json(conservative_stats)

    guarantee_block = "not configured"
    if guarantee_profile is not None:
        guarantee_block = "PASS" if context_stats.guarantee_passed else "FAIL"

    contour_block = context_stats.contour_details if context_stats.contour_details is not None else {"contour": context_stats.contour}
    verdict = (
        "The input fits the `context_structured` contour and can be served through the guaranteed path."
        if context_stats.contour_supported and context_stats.guarantee_passed is not False
        else "The input does not fit the `context_structured` contour; route it to the conservative codebook path or reject it in fail-closed mode."
    )

    markdown = f"""# HyperQuant Audit

## Context path

```json
{stats_to_pretty_json(context_stats)}
```

## Explicit contour

```json
{json.dumps(contour_block, indent=2, ensure_ascii=False, sort_keys=True)}
```

## Routing decision

- Recommended route: `{context_stats.route_recommendation}`
- Supported contour: `{context_stats.contour}`
- Reasons: `{context_stats.contour_reasons}`

## Guarantee

- Status: {guarantee_block}
- Profile: {guarantee_profile.to_dict() if guarantee_profile is not None else 'not configured'}

## Conservative baseline

```json
{conservative_block}
```

## Verdict

{verdict}
"""
    return AuditArtifacts(stats=context_stats, markdown=markdown)
