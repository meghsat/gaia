# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
GAIA Refinement package.

Hermes is the coding worker; GAIA is the judge and bundler.

Usage (server mode — the normal path):
    gaia refine serve --critic-model Qwen3.6-35B-A3B-GGUF

Hermes then calls:
    POST http://localhost:8899/session/start   # once per task
    POST http://localhost:8899/review          # after each coding iteration
"""

from gaia.refine.improvement_map import ImprovementItem, ImprovementMap, Severity

__all__ = [
    "ImprovementMap",
    "ImprovementItem",
    "Severity",
]
