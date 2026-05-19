# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Structured feedback from the big-model critic to the small-model worker."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class Severity(str, Enum):
    CRITICAL = "critical"  # Blocks correct functioning or a required feature is missing
    MAJOR = "major"        # User-visible problem or significant missing piece
    MINOR = "minor"        # Polish, style, or nice-to-have


@dataclass
class ImprovementItem:
    issue: str
    fix: str
    severity: Severity = Severity.MAJOR


@dataclass
class ImprovementMap:
    items: List[ImprovementItem] = field(default_factory=list)
    score: int = 5          # 1–10; 9–10 means ship it
    summary: str = ""

    @property
    def is_satisfied(self) -> bool:
        """True when the critic considers the output good enough."""
        has_critical = any(i.severity == Severity.CRITICAL for i in self.items)
        return self.score >= 8 and not has_critical

    def render(self) -> str:
        """Human-readable summary printed between iterations."""
        lines = [
            f"{'='*60}",
            f"  REVIEW — Score: {self.score}/10",
            f"  {self.summary}",
            f"{'='*60}",
        ]
        if not self.items:
            lines.append("  No issues found.")
        else:
            _tags = {Severity.CRITICAL: "[CRITICAL]", Severity.MAJOR: "[MAJOR]", Severity.MINOR: "[MINOR]"}
            for item in self.items:
                tag = _tags.get(item.severity, "[MAJOR]")
                lines.append(f"\n{tag} {item.issue}")
                lines.append(f"     -> FIX: {item.fix}")
        lines.append("")
        return "\n".join(lines)

    def to_prompt_injection(self) -> str:
        """Formats the map for injection into the worker's next prompt.

        Returns an empty string when the output is already satisfied so the
        caller can safely append it without checking first.
        """
        if self.is_satisfied:
            return ""

        lines = [
            "\n=== CODE REVIEW — REQUIRED IMPROVEMENTS ===",
            f"Score: {self.score}/10  (need ≥ 8 with no critical issues to pass)",
            f"{self.summary}",
            "",
            "You MUST fix every issue below before finishing. "
            "The existing files are already in the project directory — "
            "edit them in place, do not start over.",
            "",
        ]
        for item in self.items:
            prefix = "[CRITICAL] " if item.severity == Severity.CRITICAL else ""
            lines.append(f"• {prefix}{item.issue}")
            lines.append(f"  → {item.fix}")
        lines.append("\nAddress all issues above, then provide your final answer.")
        return "\n".join(lines)
