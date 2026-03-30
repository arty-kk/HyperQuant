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

from hyperquant.cli import build_parser


def _subcommand_parser(name: str):
    parser = build_parser()
    subparsers_action = next(action for action in parser._actions if action.dest == "command")  # noqa: SLF001
    return subparsers_action.choices[name]


def test_context_decompress_file_help_has_only_decode_inputs() -> None:
    parser = _subcommand_parser("context-decompress-file")
    help_text = parser.format_help()
    assert "--input" in help_text
    assert "--output" in help_text
    assert "--page-size" not in help_text
    assert "--rank" not in help_text
    assert "--low-rank-error-threshold" not in help_text
