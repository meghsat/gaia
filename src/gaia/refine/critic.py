# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Big-model critic: evaluates worker output and produces an ImprovementMap.

Both worker and critic run on Lemonade simultaneously — the small model on the
NPU and the big model on the iGPU/CPU.  Lemonade routes each request to the
correct execution unit based on the model ID, so both are available in parallel
with no swapping or loading delay between iterations.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from gaia.refine.improvement_map import ImprovementItem, ImprovementMap, Severity

logger = logging.getLogger(__name__)

# Default big model used as the critic. The user can override via --critic-model.
DEFAULT_CRITIC_MODEL = "Qwen3.6-35B-A3B-GGUF"

_SYSTEM_PROMPT = """\
You are a senior software engineer reviewing code produced by a junior AI assistant.
Your job is to evaluate the output against the original task and return a structured
improvement map so the assistant can fix the issues in the next iteration.

The worker output is structured in two sections:
1. WORKER REASONING & DESIGN CONTEXT — the assistant's design decisions, wireframe
   interpretations, and iteration notes written before and during coding. Read this
   first to understand WHY choices were made.
2. GENERATED CODE FILES — the actual HTML/CSS/JS produced.

Rules:
- Read the reasoning context before judging the code — distinguish a bad decision
  from a good decision that was implemented poorly.
- If a design choice in the context is itself wrong (misread wireframe, wrong color),
  flag the reasoning error, not just the code symptom.
- Be specific: reference exact file names, component names, CSS properties, function names.
- Only list issues that are actually present — do not hallucinate problems.
- If the output fully satisfies the task, return an empty items list with score 9 or 10.
- Respond ONLY with valid JSON. No markdown fences. No text before or after the JSON object.

Response format:
{
  "score": <integer 1-10>,
  "summary": "<one sentence overall assessment>",
  "items": [
    {
      "issue": "<what is wrong or missing>",
      "fix": "<exactly what to change — be specific>",
      "severity": "critical|major|minor"
    }
  ]
}

Scoring:
  9-10  Excellent — every requirement met, ship it
  7-8   Good — minor polish needed
  5-6   Functional but significant gaps
  1-4   Major problems, needs rework

Severity:
  critical  Core requirement missing or functionality broken
  major     User-visible problem or important missing piece
  minor     Style, polish, or nice-to-have
"""


class Critic:
    """Reviews worker output using a big model served by Lemonade.

    Args:
        model: Lemonade model ID for the critic (should be a larger model than the worker).
        base_url: Lemonade server base URL (defaults to http://localhost:13305/api/v1).
    """

    def __init__(
        self,
        model: str = DEFAULT_CRITIC_MODEL,
        base_url: Optional[str] = None,
    ):
        from gaia.llm.providers.lemonade import LemonadeProvider

        self._model = model
        self._provider = LemonadeProvider(
            model=model,
            base_url=base_url,
            system_prompt=_SYSTEM_PROMPT,
        )

    def evaluate(
        self,
        task: str,
        output: str,
        image_path: Optional[str] = None,
        image_paths: Optional[list[str]] = None,
    ) -> ImprovementMap:
        """Score the worker output against the original task.

        Args:
            task: The original task description given to the worker.
            output: The worker's response / generated code summary.
            image_path: Single UI reference image (legacy — use image_paths instead).
            image_paths: List of UI wireframe images. All are sent to the critic
                         in one call so it can compare each page against its reference.
                         Only used when the critic model supports vision.

        Returns:
            ImprovementMap with per-item feedback and an overall score.
        """
        logger.info("Critic requesting model swap → %s", self._model)

        # Merge single + multi into one list, deduplicate, keep only existing files
        all_paths: list[str] = []
        for p in ([image_path] if image_path else []) + (image_paths or []):
            if p and Path(p).exists() and p not in all_paths:
                all_paths.append(p)

        if all_paths:
            logger.info("Vision review with %d wireframe image(s): %s",
                        len(all_paths), [Path(p).name for p in all_paths])
            return self._evaluate_with_images(task, output, all_paths)
        return self._evaluate_text(task, output)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _user_message(self, task: str, output: str, image_note: str = "") -> str:
        parts = [
            f"=== ORIGINAL TASK ===\n{task}",
            "",
        ]
        if image_note:
            parts.append(image_note)
            parts.append("")
        parts += [
            f"=== WORKER OUTPUT ===\n{output}",
        ]
        return "\n".join(parts)

    def _evaluate_text(self, task: str, output: str) -> ImprovementMap:
        user_msg = self._user_message(task, output)
        raw = self._provider.chat(
            messages=[{"role": "user", "content": user_msg}],
            model=self._model,
            max_tokens=16000,
            temperature=0.1,
        )
        return self._parse(raw if isinstance(raw, str) else str(raw))

    def _evaluate_with_images(
        self, task: str, output: str, image_paths: list[str]
    ) -> ImprovementMap:
        """Vision-based review against one or more wireframe images.

        All wireframes are sent in a single call so the critic can cross-reference
        each generated page against its reference design.  Falls back to text-only
        review if the model doesn't support vision or if all image reads fail.
        """
        try:
            images: list[bytes] = []
            names: list[str] = []
            for p in image_paths:
                try:
                    images.append(Path(p).read_bytes())
                    names.append(Path(p).name)
                except Exception as exc:
                    logger.warning("Could not read wireframe image %s: %s", p, exc)

            if not images:
                logger.warning("No wireframe images could be read — falling back to text-only.")
                return self._evaluate_text(task, output)

            image_note = (
                f"{len(images)} UI wireframe image(s) provided for comparison: "
                f"{', '.join(names)}.\n"
                "For each generated page, find the matching wireframe and compare:\n"
                "- Layout structure (grid, sidebar, hero placement)\n"
                "- Component presence and position (navbar items, buttons, cards, forms)\n"
                "- Visual hierarchy (font sizes, spacing, color contrast)\n"
                "Flag any generated page that deviates from its wireframe reference."
            )
            user_msg = self._user_message(task, output, image_note)

            # LemonadeProvider.vision() delegates to VLMClient.
            # Passing multiple images requires a multimodal GGUF — raises if text-only.
            raw = self._provider.vision(
                images=images,
                prompt=user_msg,
            )
            return self._parse(raw if isinstance(raw, str) else str(raw))

        except Exception as exc:
            logger.warning(
                "Vision-based critique failed (%s) — falling back to text-only review. "
                "Load a multimodal GGUF as the critic model to enable image comparison.",
                exc,
            )
            return self._evaluate_text(task, output)

    def _parse(self, text: str) -> ImprovementMap:
        """Extract ImprovementMap from the model's JSON response."""
        # Strip thinking tokens (<think>...</think>) that Qwen3 emits
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            logger.warning("Critic returned non-JSON response: %.200s", text)
            return ImprovementMap(
                items=[
                    ImprovementItem(
                        issue="Could not parse critic response",
                        fix="Check Lemonade logs and ensure the critic model follows JSON instructions",
                        severity=Severity.MAJOR,
                    )
                ],
                score=5,
                summary="Review parsing failed — manual inspection needed",
            )

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error in critic response: %s", exc)
            return ImprovementMap(score=5, summary="JSON decode error in critic response")

        items = []
        for raw in data.get("items", []):
            try:
                severity = Severity(raw.get("severity", "major"))
            except ValueError:
                severity = Severity.MAJOR
            items.append(
                ImprovementItem(
                    issue=raw.get("issue", ""),
                    fix=raw.get("fix", ""),
                    severity=severity,
                )
            )

        return ImprovementMap(
            items=items,
            score=int(data.get("score", 5)),
            summary=data.get("summary", ""),
        )
