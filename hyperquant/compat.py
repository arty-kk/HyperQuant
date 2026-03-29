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
