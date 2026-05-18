# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Bundler: converts a RefinementResult into a reusable agent profile.

After the refinement loop satisfies the critic, the bundler:
  1. Extracts the recurring mistake patterns from all ImprovementMaps.
  2. Writes a YAML profile (model + system-prompt addendum + key lessons).
  3. Writes a SKILL.md the agent can load on the next task so it starts smarter.
"""

from __future__ import annotations

import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from gaia.refine.improvement_map import ImprovementMap, Severity


@dataclass
class BundleProfile:
    """Portable description of what made the refinement succeed."""

    use_case: str
    model_id: str
    language: str
    project_type: str
    lessons: List[str] = field(default_factory=list)   # Top recurring fixes
    system_prompt_addendum: str = ""                    # Injected before every task

    def to_dict(self) -> dict:
        return {
            "use_case": self.use_case,
            "model_id": self.model_id,
            "language": self.language,
            "project_type": self.project_type,
            "lessons": self.lessons,
            "system_prompt_addendum": self.system_prompt_addendum,
        }


class Bundler:
    """Packages a successful refinement run into files the agent can reload next time.

    Args:
        use_case: Human-readable label (e.g. "web-dev", "email-triage").
        model_id: Lemonade model that was used as the worker.
        language: Language passed to CodeAgent.
        project_type: Project type passed to CodeAgent.
        output_dir: Where to write the bundle files (defaults to ~/.gaia/bundles/<use_case>/).
    """

    def __init__(
        self,
        use_case: str,
        model_id: str,
        language: str = "typescript",
        project_type: str = "frontend",
        output_dir: Optional[str] = None,
    ):
        self.use_case = use_case
        self.model_id = model_id
        self.language = language
        self.project_type = project_type

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path.home() / ".gaia" / "bundles" / use_case
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def package(self, maps: List[ImprovementMap]) -> BundleProfile:
        """Extract lessons from all ImprovementMaps and write bundle files.

        Args:
            maps: All ImprovementMaps produced during the refinement run
                  (including intermediate iterations, not just the final one).

        Returns:
            BundleProfile with the distilled lessons.
        """
        lessons = self._extract_lessons(maps)
        addendum = self._build_system_prompt_addendum(lessons)
        profile = BundleProfile(
            use_case=self.use_case,
            model_id=self.model_id,
            language=self.language,
            project_type=self.project_type,
            lessons=lessons,
            system_prompt_addendum=addendum,
        )
        self._write_yaml(profile)
        self._write_skill_md(profile)
        print(f"✅  Bundle written to {self.output_dir}")
        return profile

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_lessons(self, maps: List[ImprovementMap], top_n: int = 8) -> List[str]:
        """Return the most frequently recurring fix descriptions across all maps."""
        fix_counter: Counter = Counter()
        for m in maps:
            for item in m.items:
                # Normalise whitespace to improve dedup
                fix_counter[item.fix.strip()] += (
                    2 if item.severity == Severity.CRITICAL else 1
                )
        return [fix for fix, _ in fix_counter.most_common(top_n)]

    def _build_system_prompt_addendum(self, lessons: List[str]) -> str:
        if not lessons:
            return ""
        lines = [
            f"=== {self.use_case.upper()} LESSONS (learned from prior runs) ===",
            "Apply every rule below without being asked:",
            "",
        ]
        for lesson in lessons:
            lines.append(f"• {lesson}")
        return "\n".join(lines)

    def _write_yaml(self, profile: BundleProfile) -> None:
        path = self.output_dir / "profile.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(profile.to_dict(), f, allow_unicode=True, sort_keys=False)

    def _write_skill_md(self, profile: BundleProfile) -> None:
        """Write a SKILL.md the agent can read at the start of the next task."""
        slug = self.use_case.lower().replace(" ", "-")
        lessons_md = "\n".join(f"- {l}" for l in profile.lessons)
        content = textwrap.dedent(f"""\
            ---
            name: {slug}-bundle
            description: >
              Accumulated lessons for {self.use_case} tasks with model {self.model_id}.
              Load this skill before starting any {self.use_case} task to avoid known pitfalls.
            ---

            # {self.use_case} Bundle

            ## Model
            `{self.model_id}` via Lemonade (language: {self.language}, project_type: {self.project_type})

            ## Key lessons (most impactful first)

            {lessons_md}

            ## How to use

            When starting a new {self.use_case} task, begin your system prompt with the
            contents of `profile.yaml → system_prompt_addendum` so the model applies
            these lessons from the first attempt rather than learning them mid-run.
        """)
        path = self.output_dir / "SKILL.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
