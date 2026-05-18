# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Per-task session tracking for the refinement review service."""

from __future__ import annotations

import datetime
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from gaia.refine.improvement_map import ImprovementMap

# All review logs land under ~/.gaia/sessions/<session_id>/
_SESSIONS_DIR = Path.home() / ".gaia" / "sessions"


@dataclass
class Session:
    """Tracks one end-to-end refinement task: multiple Hermes iterations against one task."""

    session_id: str
    task: str
    assets_dir: Optional[str]
    critic_model: str
    max_iterations: int
    bundle_name: Optional[str]

    # Populated during the run
    iterations: int = 0
    maps: List[ImprovementMap] = field(default_factory=list)
    final_score: int = 0
    satisfied: bool = False
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def record(self, improvement_map: ImprovementMap) -> None:
        self.maps.append(improvement_map)
        self.iterations += 1
        self.final_score = improvement_map.score
        self.satisfied = improvement_map.is_satisfied
        if self.satisfied:
            self.finished_at = time.time()
        self._persist(improvement_map)

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------

    @property
    def session_dir(self) -> Path:
        return _SESSIONS_DIR / self.session_id

    def _persist(self, improvement_map: ImprovementMap) -> None:
        """Write the latest iteration result to ~/.gaia/sessions/<session_id>/."""
        try:
            d = self.session_dir
            d.mkdir(parents=True, exist_ok=True)

            # ── session.json (created once, updated each iteration) ──
            session_meta = {
                "session_id": self.session_id,
                "task": self.task,
                "critic_model": self.critic_model,
                "assets_dir": self.assets_dir,
                "max_iterations": self.max_iterations,
                "bundle_name": self.bundle_name,
                "started_at": datetime.datetime.fromtimestamp(self.started_at).isoformat(),
                "iterations_completed": self.iterations,
                "final_score": self.final_score,
                "satisfied": self.satisfied,
            }
            (d / "session.json").write_text(json.dumps(session_meta, indent=2))

            # ── iteration-N.json (raw review data) ──
            items_data = [
                {"issue": it.issue, "fix": it.fix, "severity": it.severity.value}
                for it in improvement_map.items
            ]
            review_data = {
                "iteration": self.iterations,
                "timestamp": datetime.datetime.now().isoformat(),
                "score": improvement_map.score,
                "summary": improvement_map.summary,
                "is_satisfied": improvement_map.is_satisfied,
                "items": items_data,
                "improvement_map_text": improvement_map.to_prompt_injection(),
            }
            (d / f"iteration-{self.iterations}.json").write_text(
                json.dumps(review_data, indent=2)
            )

            # ── iteration-N.md (human-readable) ──
            _write_iteration_md(d, self.iterations, review_data)

            # ── REVIEW_LOG.md (running summary across all iterations) ──
            _append_review_log(d, self.iterations, review_data, self.task)

        except Exception as exc:
            # Persistence is best-effort — never crash the server
            import logging
            logging.getLogger(__name__).warning("Failed to persist review: %s", exc)

    def status(self) -> dict:
        return {
            "session_id": self.session_id,
            "task": self.task[:120],
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "final_score": self.final_score,
            "satisfied": self.satisfied,
            "elapsed_s": round(time.time() - self.started_at, 1),
        }


# ---------------------------------------------------------------------------
# File-writing helpers (module-level so Session dataclass stays clean)
# ---------------------------------------------------------------------------

def _write_iteration_md(session_dir: Path, iteration: int, data: dict) -> None:
    icons = {"critical": "🔴", "major": "🟡", "minor": "🔵"}
    satisfied = data["is_satisfied"]
    lines = [
        f"# GAIA Review — Iteration {iteration}",
        "",
        f"| | |",
        f"|---|---|",
        f"| **Date** | {data['timestamp']} |",
        f"| **Score** | **{data['score']}/10** |",
        f"| **Status** | {'✅ SATISFIED' if satisfied else '🔄 NEEDS WORK'} |",
        "",
        "## Summary",
        "",
        data["summary"],
        "",
    ]
    items = data.get("items", [])
    if items:
        lines += [f"## Issues ({len(items)} total)", ""]
        for i, item in enumerate(items, 1):
            icon = icons.get(item["severity"], "•")
            lines += [
                f"### {icon} Issue {i} — [{item['severity'].upper()}]",
                "",
                f"**Problem:** {item['issue']}",
                "",
                f"**Fix:** {item['fix']}",
                "",
            ]
    else:
        lines += ["## Issues", "", "None — output satisfies all requirements.", ""]

    map_text = data.get("improvement_map_text", "")
    if map_text:
        lines += ["## Improvement Map (paste into Hermes)", "", "```", map_text, "```", ""]

    (session_dir / f"iteration-{iteration}.md").write_text("\n".join(lines))


def _append_review_log(session_dir: Path, iteration: int, data: dict, task: str) -> None:
    log_path = session_dir / "REVIEW_LOG.md"
    icons = {"critical": "🔴", "major": "🟡", "minor": "🔵"}

    if not log_path.exists():
        log_path.write_text(
            f"# GAIA Review Log\n\n"
            f"**Session dir:** `{session_dir}`  \n"
            f"**Task:** {task[:300]}{'...' if len(task) > 300 else ''}  \n\n"
            f"---\n"
        )

    satisfied = data["is_satisfied"]
    status = "✅ SATISFIED" if satisfied else "🔄 NEEDS WORK"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n## Iteration {iteration} — Score {data['score']}/10 — {status}\n\n")
        f.write(f"_{data['timestamp']}_\n\n")
        f.write(f"**{data['summary']}**\n\n")
        items = data.get("items", [])
        if items:
            for item in items:
                icon = icons.get(item["severity"], "•")
                f.write(f"- {icon} `{item['severity'].upper()}` {item['issue']}\n")
        else:
            f.write("No issues.\n")
        f.write(f"\n[Full review →](iteration-{iteration}.md)  \n\n---\n")


class SessionStore:
    """Thread-safe in-memory store for active refinement sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(
        self,
        task: str,
        critic_model: str,
        assets_dir: Optional[str] = None,
        max_iterations: int = 5,
        bundle_name: Optional[str] = None,
    ) -> Session:
        session_id = str(uuid.uuid4())[:8]
        session = Session(
            session_id=session_id,
            task=task,
            assets_dir=assets_dir,
            critic_model=critic_model,
            max_iterations=max_iterations,
            bundle_name=bundle_name,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(session_id)

    def all_statuses(self) -> list:
        with self._lock:
            return [s.status() for s in self._sessions.values()]
