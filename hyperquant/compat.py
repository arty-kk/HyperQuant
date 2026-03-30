# SPDX-FileCopyrightText: 2026 Сацук Артём Венедиктович (Satsuk Artem)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum

try:  # Python 3.11+
    from enum import StrEnum as _StrEnum
except ImportError:  # pragma: no cover - Python 3.10 fallback
    class _StrEnum(str, Enum):
        """Compatibility fallback for enum.StrEnum on Python 3.10."""

        def __str__(self) -> str:
            return str(self.value)

StrEnum = _StrEnum

__all__ = ["StrEnum"]
