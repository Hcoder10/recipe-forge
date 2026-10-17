"""What a tool call can reach: the mission, the target machine, and the team's shared services."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from forge.tools.executor import Executor


@dataclass
class ToolContext:
    mission: Any                 # forge.agents.mission.MissionState
    executor: Executor
    role: str                    # lead | worker | verifier
    agent: str                   # e.g. worker-2
    log_event: Callable[..., None]
    services: Any                # forge.agents.mission.Services: brain, decisions, inbox, gate, bench, team
    scratch: dict = field(default_factory=dict)
