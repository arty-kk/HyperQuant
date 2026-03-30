# SPDX-FileCopyrightText: 2026 Сацук Артём Венедиктович (Satsuk Artem)
# SPDX-License-Identifier: Apache-2.0

__version__ = "1.0.0"

from .audit import AuditArtifacts, audit_context_input
from .bundle import CodebookBundle
from .codebook_codec import CodebookEnvelope, CodebookStats, CodebookCodec
from .config import CodebookConfig, CompressionConfig, CompressionMode
from .contour import ContourAnalysis, ContourThresholds, ProductContour, analyze_context_contour
from .route_benchmark import RouteBenchmarkArtifacts, run_route_benchmark
from .guarantee import (
    ContourViolation,
    GuaranteeMode,
    GuaranteeOutcome,
    GuaranteeViolation,
    ContextGuaranteeProfile,
)
from .live_data import (
    LiveDataProfile,
    generate_mixed_long_context,
    generate_online_vector_stream,
    generate_structured_long_context,
)
from .context_codec import (
    ContextEnvelope,
    ContextCompressionStats,
    ContextCodecConfig,
    ContextPageMode,
    ContextCodec,
)
from .resident_tier import (
    ResidentTierConfig,
    ResidentBenchmarkArtifacts,
    ResidentPlan,
    ResidentPlanner,
    ResidentAccessReport,
    ResidentPageDescriptor,
    ResidentPageMode,
    ResidentTierStore,
    ResidentTierManifest,
    ResidentTierStats,
    build_resident_store,
    run_resident_benchmark,
)

from .vector_codec import (
    DenseRotationBaseline,
    VectorCodec,
    RotationKind,
    RotatedScalarCodec,
    RotatedScalarConfig,
    RotatedScalarEnvelope,
    RotatedScalarStats,
)

__all__ = [
    "__version__",
    "AuditArtifacts",
    "audit_context_input",
    "CodebookBundle",
    "CodebookEnvelope",
    "CodebookStats",
    "CodebookCodec",
    "CodebookConfig",
    "CompressionConfig",
    "CompressionMode",
    "ContourAnalysis",
    "ContourThresholds",
    "ProductContour",
    "analyze_context_contour",
    "RouteBenchmarkArtifacts",
    "run_route_benchmark",
    "LiveDataProfile",
    "generate_online_vector_stream",
    "generate_structured_long_context",
    "generate_mixed_long_context",
    "GuaranteeMode",
    "GuaranteeOutcome",
    "GuaranteeViolation",
    "ContourViolation",
    "ContextGuaranteeProfile",
    "ContextEnvelope",
    "ContextCompressionStats",
    "ContextCodecConfig",
    "ContextPageMode",
    "ContextCodec",
    "ResidentTierConfig",
    "ResidentBenchmarkArtifacts",
    "ResidentPlan",
    "ResidentPlanner",
    "ResidentAccessReport",
    "ResidentPageDescriptor",
    "ResidentPageMode",
    "ResidentTierStore",
    "ResidentTierManifest",
    "ResidentTierStats",
    "build_resident_store",
    "run_resident_benchmark",
    "RotationKind",
    "RotatedScalarConfig",
    "RotatedScalarEnvelope",
    "RotatedScalarStats",
    "RotatedScalarCodec",
    "DenseRotationBaseline",
    "VectorCodec",
]
