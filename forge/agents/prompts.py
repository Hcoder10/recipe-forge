"""Cache-stable prompt layout.

  [system]  fixed prefix: role + MANUAL + mission brief + experiment protocol   (byte-stable all mission)
  [user]    slowly refreshed context: shared facts view + learned summary       (changes every N turns)
  ...       history, appended only (assistant / tool / user event messages)
  [user]    volatile status: time and $ left, pending triggers, nudges          (last message only; not kept)
"""
from __future__ import annotations

import json


def mission_brief(m) -> str:
    rec = m.recipe
    serve = rec.get("serve", {})
    lines = [
        "MISSION BRIEF",
        f"mission: {m.id}",
        f"recipe: {m.recipe_id}  model: {rec.get('model')}  engine: {rec.get('engine', 'vllm')}",
        "Inspect the reference configuration through the authorized tools.",
    ]
    wl = rec.get("workload", {})
    lines += [
        "Preserve the reference workload and target distribution.",
        f"metric: {m.metric} (higher is better)",
        f"target: machine {m.machine.get('name')}, GPU class {m.gpu_class}",
        f"time box: {m.time_box_min:.0f} min; LLM budget: ${m.budget_usd:.2f}",
        f"team: lead, {m.n_workers} workers ({', '.join(m.worker_names)}), verifier",
        f"workspace on the target: {m.ws}",
        "lanes: arch (initial inspection), then ledger rows: attention, ssm, moe, gemm/quant, norm, sampler, "
        "launch-overhead; acceptance (tokens accepted per step, including drafter evaluation). Build "
        "sol_ledger() after arch_profile() and assign each worker one row or acceptance. Runtime, kernels, "
        "drafter, config, env and system describe implementation surfaces, not independent flag lanes.",
        "measurement: bench() runs the visible prompt set on a work GPU against a same-GPU reference. The gate "
        "runs its own hidden prompt set with paired timing and a lossless check against the reference; only a "
        "gate verdict counts.",
        "Follow the locked mission completion criteria; only the gate can establish completion.",
    ]
    return "\n".join(lines)


def fixed_prefix(role_text: str, manual: str, brief: str, protocol: str) -> str:
    return f"{role_text}\n\n{manual}\n\n{brief}\n\n{protocol}\n"


def context_block(facts_view: str, learned: str, extra: str = "") -> str:
    return ("Consult authorized evidence by identifier. Keep shared context and memory procedural; "
            "do not repeat findings, experiment outcomes or configuration values.")


def status_block(m, role: str, pending_text: str | None, nudges: list[str], extra: list[str]) -> str:
    left_min = m.minutes_left()
    lines = [f"STATUS: {left_min:.0f} of {m.time_box_min:.0f} min left; LLM ${m.meter.cost_usd:.2f} of "
             f"${m.budget_usd:.2f} used."]
    if pending_text is not None:
        lines.append(pending_text)
    lines += extra
    lines += nudges
    return "\n".join(lines)


SUMMARIZE = ("Summarize procedure and remaining work only. Keep evidence identifiers and file paths, "
             "but omit speed findings, experiment outcomes, configuration values and try-this advice. "
             "Keep the summary short.")
