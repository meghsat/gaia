# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
GAIA Refinement Review Service.

Hermes (or any agent) calls POST /review after each coding iteration.
GAIA evaluates the output with the big model and returns a structured
improvement map.  The calling agent implements the fixes and calls /review
again until is_satisfied=true.

Start the server:
    gaia refine serve [--port 8899] [--critic-model Qwen3.6-35B-A3B-GGUF]

Hermes calls it like:
    curl -s -X POST http://localhost:8899/review \\
         -H "Content-Type: application/json" \\
         -d '{"task":"...", "output":"...", "session_id":"abc123"}'
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from gaia.refine.critic import Critic, DEFAULT_CRITIC_MODEL
from gaia.refine.session import SessionStore

logger = logging.getLogger(__name__)

app = FastAPI(title="GAIA Refinement Review Service", version="1.0.0")

# Module-level singletons — populated in create_app() before uvicorn starts.
_critic: Optional[Critic] = None
_store: SessionStore = SessionStore()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    task: str
    assets_dir: Optional[str] = None
    max_iterations: int = 5
    bundle_name: Optional[str] = None


class StartResponse(BaseModel):
    session_id: str
    message: str


class ReviewRequest(BaseModel):
    session_id: str
    output: str                    # Hermes's current code / answer
    image_path: Optional[str] = None  # optional UI reference image path


class ReviewDirRequest(BaseModel):
    session_id: str
    directory: str                 # WSL or Windows path to the output directory
    image_path: Optional[str] = None  # optional UI wireframe image for vision review
    max_bytes_per_file: int = 12_000  # truncation limit per file


class ReviewResponse(BaseModel):
    session_id: str
    iteration: int
    score: int                     # 1–10
    summary: str
    is_satisfied: bool
    items: list                    # list of {issue, fix, severity}
    improvement_map_text: str      # human-readable, ready to paste into Hermes prompt
    message: str                   # top-level status message for the agent


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "critic_model": _critic._model if _critic else None}


@app.post("/session/start", response_model=StartResponse)
def start_session(req: StartRequest):
    """Create a new refinement session before the first Hermes iteration.

    Returns a session_id that Hermes must include in every /review call.
    """
    session = _store.create(
        task=req.task,
        critic_model=_critic._model if _critic else DEFAULT_CRITIC_MODEL,
        assets_dir=req.assets_dir,
        max_iterations=req.max_iterations,
        bundle_name=req.bundle_name,
    )
    logger.info("Session started: %s | task: %.80s", session.session_id, req.task)
    logger.info("Review logs will be saved to: %s", session.session_dir)
    return StartResponse(
        session_id=session.session_id,
        message=(
            f"Session {session.session_id} created. "
            f"Call POST /review with session_id='{session.session_id}' "
            "after each coding iteration."
        ),
    )


@app.post("/review", response_model=ReviewResponse)
def review(req: ReviewRequest):
    """Evaluate one Hermes coding iteration and return an improvement map.

    Hermes calls this after finishing each coding pass.  The response tells
    Hermes whether to stop (is_satisfied=true) or what to fix next.
    """
    session = _store.get(req.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown session_id '{req.session_id}'. Call POST /session/start first.",
        )

    if session.satisfied:
        return ReviewResponse(
            session_id=session.session_id,
            iteration=session.iterations,
            score=session.final_score,
            summary="Already satisfied — no further review needed.",
            is_satisfied=True,
            items=[],
            improvement_map_text="",
            message="✅ Task is already marked satisfied. You can stop.",
        )

    if session.iterations >= session.max_iterations:
        return ReviewResponse(
            session_id=session.session_id,
            iteration=session.iterations,
            score=session.final_score,
            summary="Maximum iterations reached.",
            is_satisfied=False,
            items=[],
            improvement_map_text="",
            message=(
                f"⚠️ Reached max iterations ({session.max_iterations}). "
                "Stopping. Bundle what you have."
            ),
        )

    if _critic is None:
        raise HTTPException(status_code=503, detail="Critic not initialised.")

    logger.info(
        "Reviewing session=%s iteration=%d", session.session_id, session.iterations + 1
    )
    logger.info(
        "Review will be saved → %s",
        session.session_dir / f"iteration-{session.iterations + 1}.md",
    )

    improvement_map = _critic.evaluate(
        task=session.task,
        output=req.output,
        image_path=req.image_path,
    )
    session.record(improvement_map)

    # Build the improvement map text Hermes pastes into its next prompt
    map_text = improvement_map.to_prompt_injection()

    # Trigger bundling when satisfied
    if improvement_map.is_satisfied and session.bundle_name:
        _trigger_bundle(session)

    items_json = [
        {
            "issue": item.issue,
            "fix": item.fix,
            "severity": item.severity.value,
        }
        for item in improvement_map.items
    ]

    if improvement_map.is_satisfied:
        msg = (
            f"✅ Score {improvement_map.score}/10 — satisfied after "
            f"{session.iterations} iteration(s). Task complete."
        )
    else:
        msg = (
            f"🔄 Score {improvement_map.score}/10 — not satisfied yet "
            f"(iteration {session.iterations}/{session.max_iterations}). "
            "Implement every fix in 'items' then call /review again."
        )

    return ReviewResponse(
        session_id=session.session_id,
        iteration=session.iterations,
        score=improvement_map.score,
        summary=improvement_map.summary,
        is_satisfied=improvement_map.is_satisfied,
        items=items_json,
        improvement_map_text=map_text,
        message=msg,
    )


@app.post("/review-dir", response_model=ReviewResponse)
def review_dir(req: ReviewDirRequest):
    """Evaluate one Hermes coding iteration by reading files directly from disk.

    Hermes sends only the session_id and the output directory path.
    GAIA reads every .html/.css/.js file from that directory server-side,
    so the critic always sees the actual code — not a text summary.

    WSL paths (/home/user/...) are automatically converted to the Windows
    UNC equivalent (\\\\wsl.localhost\\Ubuntu-24.04\\...) so the server
    can read WSL files without any special setup.

    Example (from Hermes in WSL):
        curl -s -X POST http://192.168.96.1:8899/review-dir \\
             -H "Content-Type: application/json" \\
             -d '{"session_id":"abc123","directory":"/home/amd/.hermes/multi-agent/website"}'
    """
    session = _store.get(req.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown session_id '{req.session_id}'. Call POST /session/start first.",
        )

    if session.satisfied:
        return ReviewResponse(
            session_id=session.session_id,
            iteration=session.iterations,
            score=session.final_score,
            summary="Already satisfied — no further review needed.",
            is_satisfied=True,
            items=[],
            improvement_map_text="",
            message="✅ Task is already marked satisfied. You can stop.",
        )

    if session.iterations >= session.max_iterations:
        return ReviewResponse(
            session_id=session.session_id,
            iteration=session.iterations,
            score=session.final_score,
            summary="Maximum iterations reached.",
            is_satisfied=False,
            items=[],
            improvement_map_text="",
            message=(
                f"⚠️ Reached max iterations ({session.max_iterations}). "
                "Stopping. Bundle what you have."
            ),
        )

    if _critic is None:
        raise HTTPException(status_code=503, detail="Critic not initialised.")

    # Collect actual file contents from the output directory
    output, files_read = _collect_dir_output(req.directory, req.max_bytes_per_file)

    # Auto-discover wireframe images from session's assets_dir (top-level PNGs only).
    # req.image_path is still honoured if explicitly provided; auto-discovery fills
    # the rest so the critic always sees all reference wireframes.
    wireframes = _discover_wireframes(session.assets_dir)
    logger.info(
        "review-dir session=%s iteration=%d | dir=%s | files=%d | chars=%d | wireframes=%d",
        session.session_id, session.iterations + 1,
        req.directory, files_read, len(output), len(wireframes),
    )

    improvement_map = _critic.evaluate(
        task=session.task,
        output=output,
        image_path=req.image_path,   # explicit override still respected
        image_paths=wireframes,       # all wireframes from assets_dir auto-added
    )
    session.record(improvement_map)

    map_text = improvement_map.to_prompt_injection()
    if improvement_map.is_satisfied and session.bundle_name:
        _trigger_bundle(session)

    items_json = [
        {"issue": item.issue, "fix": item.fix, "severity": item.severity.value}
        for item in improvement_map.items
    ]

    if improvement_map.is_satisfied:
        msg = (
            f"✅ Score {improvement_map.score}/10 — satisfied after "
            f"{session.iterations} iteration(s). Task complete."
        )
    else:
        msg = (
            f"🔄 Score {improvement_map.score}/10 — not satisfied yet "
            f"(iteration {session.iterations}/{session.max_iterations}). "
            "Implement every fix in 'items' then call /review-dir again."
        )

    return ReviewResponse(
        session_id=session.session_id,
        iteration=session.iterations,
        score=improvement_map.score,
        summary=improvement_map.summary,
        is_satisfied=improvement_map.is_satisfied,
        items=items_json,
        improvement_map_text=map_text,
        message=msg,
    )


@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Return the current status of a session."""
    session = _store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id '{session_id}'")
    return session.status()


@app.get("/sessions")
def list_sessions():
    """List all active sessions."""
    return _store.all_statuses()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_wireframes(assets_dir: Optional[str]) -> list[str]:
    """Return paths to wireframe PNGs/JPGs in the top level of assets_dir.

    Only includes files directly in assets_dir root — not in subdirectories
    (subdirs like website_images/ contain content images used inside the site,
    not reference wireframes for comparison).

    WSL paths are converted to Windows UNC paths automatically.
    """
    if not assets_dir:
        return []
    win_dir = _wsl_to_windows(assets_dir) if assets_dir.startswith("/") else assets_dir
    from pathlib import Path as _Path
    d = _Path(win_dir)
    if not d.exists():
        logger.warning("assets_dir not found for wireframe discovery: %s", win_dir)
        return []
    _IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
    wireframes = [
        str(p) for p in sorted(d.iterdir())
        if p.is_file() and p.suffix.lower() in _IMG_EXTS
    ]
    logger.info("Wireframes discovered in %s: %s", assets_dir,
                [p.split("\\")[-1] for p in wireframes])
    return wireframes


def _wsl_to_windows(path: str) -> str:
    """Convert a WSL absolute path to its Windows UNC equivalent.

    Uses `wslpath -w` (most reliable — handles any distro name and mountpoint)
    then falls back to manual UNC construction.

    /home/user/foo  →  \\\\wsl.localhost\\Ubuntu-24.04\\home\\user\\foo
    """
    import subprocess
    if not path.startswith("/"):
        return path  # already a Windows path
    # Primary: let WSL's own wslpath convert the path — authoritative
    try:
        result = subprocess.run(
            ["wsl.exe", "wslpath", "-w", path],
            capture_output=True, text=True, timeout=5,
        )
        converted = result.stdout.strip()
        if result.returncode == 0 and converted:
            logger.debug("wslpath: %s → %s", path, converted)
            return converted
    except Exception as exc:
        logger.debug("wslpath failed (%s) — falling back to manual UNC construction", exc)
    # Fallback: manual UNC construction (strip null bytes from --list output)
    try:
        list_result = subprocess.run(
            ["wsl.exe", "--list", "--quiet"],
            capture_output=True, text=True, timeout=5,
        )
        distros = [
            l.strip().replace("\x00", "")
            for l in list_result.stdout.splitlines()
            if l.strip().replace("\x00", "") and "docker" not in l.lower()
        ]
        distro = distros[0] if distros else "Ubuntu-24.04"
    except Exception:
        distro = "Ubuntu-24.04"
    win_path = path.replace("/", "\\")
    return f"\\\\wsl.localhost\\{distro}{win_path}"


def _collect_dir_output(directory: str, max_bytes_per_file: int = 12_000) -> tuple[str, int]:
    """Read every .html/.css/.js file from *directory* and return (output_text, file_count).

    Also reads any Markdown files from a `.gaia-context/` subdirectory and
    prepends them as reasoning context so the critic understands the design
    decisions and iteration notes behind the code.

    Accepts both WSL paths (/home/...) and Windows paths (C:\\...).
    WSL paths are automatically converted to \\\\wsl.localhost\\... UNC paths
    so the Windows-side GAIA server can read them directly.
    """
    from pathlib import Path as _Path

    win_dir = _wsl_to_windows(directory) if directory.startswith("/") else directory
    dir_path = _Path(win_dir)

    if not dir_path.exists():
        return f"[Directory not found: {directory} (resolved to {win_dir})]", 0

    parts: list[str] = [
        f"Directory: {directory}\n"
        f"Files collected by GAIA server from: {win_dir}\n"
    ]
    files_read = 0

    # ── Section 1: Worker reasoning & design context ──────────────────────
    # Read all .md files from .gaia-context/ (written by Hermes during build).
    # Placed first so the critic reads WHY before reviewing WHAT.
    context_dir = dir_path / ".gaia-context"
    context_files = sorted(context_dir.glob("*.md")) if context_dir.exists() else []
    if context_files:
        parts.append(
            f"\n\n{'#'*60}\n"
            f"# WORKER REASONING & DESIGN CONTEXT\n"
            f"# (written by Hermes during build — read before reviewing code)\n"
            f"{'#'*60}\n"
        )
        for p in context_files:
            parts.append(f"\n\n--- {p.name} ---\n")
            try:
                parts.append(p.read_text(encoding="utf-8", errors="replace")[:6_000])
            except Exception as exc:
                parts.append(f"[read error: {exc}]")
    else:
        parts.append("\n[No .gaia-context/ reasoning files found — critic reviewing code only]\n")

    # ── Section 2: Generated code files ──────────────────────────────────
    parts.append(
        f"\n\n{'#'*60}\n"
        f"# GENERATED CODE FILES\n"
        f"{'#'*60}\n"
    )
    _CODE_EXTENSIONS = {".html", ".css", ".js"}
    for p in sorted(dir_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in _CODE_EXTENSIONS:
            # Skip anything inside .gaia-context/
            if ".gaia-context" in p.parts:
                continue
            rel = p.relative_to(dir_path)
            parts.append(f"\n\n{'='*60}\nFILE: {rel}\n{'='*60}\n")
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                parts.append(content[:max_bytes_per_file])
                if len(content) > max_bytes_per_file:
                    parts.append(
                        f"\n... [{len(content) - max_bytes_per_file:,} bytes truncated]"
                    )
                files_read += 1
            except Exception as exc:
                parts.append(f"[read error: {exc}]")

    if files_read == 0:
        parts.append("\n[No .html/.css/.js files found in directory]")

    return "".join(parts), files_read


def _trigger_bundle(session) -> None:
    """Package the session's improvement maps into a reusable bundle."""
    try:
        from gaia.bundle import Bundler
        bundler = Bundler(
            use_case=session.bundle_name,
            model_id="lemonade",
            language="typescript",
        )
        bundler.package(session.maps)
        logger.info("Bundle written for session %s → %s", session.session_id, session.bundle_name)
    except Exception as exc:
        logger.warning("Bundling failed: %s", exc)


def create_app(critic_model: str = DEFAULT_CRITIC_MODEL) -> FastAPI:
    """Initialise the critic and return the FastAPI app (called by the CLI)."""
    global _critic
    _critic = Critic(model=critic_model)
    return app


def serve(host: str = "0.0.0.0", port: int = 8899, critic_model: str = DEFAULT_CRITIC_MODEL):
    """Start the review server (blocking)."""
    import uvicorn

    create_app(critic_model)
    logger.info("GAIA Refinement Review Service on http://%s:%d", host, port)
    logger.info("Critic model: %s", critic_model)
    uvicorn.run(app, host=host, port=port, log_level="info")
