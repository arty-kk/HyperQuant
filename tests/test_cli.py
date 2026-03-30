# SPDX-FileCopyrightText: 2026 Сацук Артём Венедиктович (Satsuk Artem)
# SPDX-License-Identifier: Apache-2.0

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
