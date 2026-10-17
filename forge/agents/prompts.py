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
        f"published serve args (the reference): {json.dumps(serve.get('args'))}",
    ]
    if serve.get("env"):
        lines.append(f"reference serve env: {json.dumps(serve.get('env'))}")
    if serve.get("source") or rec.get("source"):
        lines.append(f"source of the recipe: {serve.get('source') or rec.get('source')}")
    if rec.get("drafter"):
        lines.append(f"drafter in the recipe: {rec.get('drafter')}")
    wl = rec.get("workload", {})
    lines += [
        f"workload: {json.dumps({k: v for k, v in wl.items() if k not in ('bench_prompts', 'hidden')})}",
        f"metric: {m.metric} (higher is better)",
        f"target: machine {m.machine.get('name')}, GPU class {m.gpu_class}",
        f"time box: {m.time_box_min:.0f} min; LLM budget: ${m.budget_usd:.2f}",
        f"team: lead, {m.n_workers} workers ({', '.join(m.worker_names)}), verifier",
        f"workspace on the target: {m.ws}",
        "lanes: config (serve flags), env (environment variables), runtime (vLLM Python source), kernels "
        "(Triton/CUDA in the workspace), drafter (speculative decoding drafter), system (machine-level settings "
        "a job may change)",
        "measurement: bench() runs the visible prompt set on a work GPU against a same-GPU reference. The gate "
        "runs its own hidden prompt set with paired timing and a lossless check against the reference; only a "
        "gate verdict counts.",
        f"DONE CRITERIA (locked {m.done_hash[:12]}): success = {m.criteria['success']}. "
        f"termination = {m.criteria['termination']}.",
    ]
    return "\n".join(lines)


def fixed_prefix(role_text: str, manual: str, brief: str, protocol: str) -> str:
    return f"{role_text}\n\n{manual}\n\n{brief}\n\n{protocol}\n"


def context_block(facts_view: str, learned: str, extra: str = "") -> str:
    out = ["SHARED FACTS (from all missions; refreshed periodically):", facts_view]
    if learned:
        out += ["", "YOUR EARLIER WORK IN THIS MISSION (summary of compacted turns):", learned]
    if extra:
        out += ["", extra]
    return "\n".join(out)


def status_block(m, role: str, pending_text: str | None, nudges: list[str], extra: list[str]) -> str:
    left_min = m.minutes_left()
    lines = [f"STATUS: {left_min:.0f} of {m.time_box_min:.0f} min left; LLM ${m.meter.cost_usd:.2f} of "
             f"${m.budget_usd:.2f} used."]
    if pending_text is not None:
        lines.append(pending_text)
    lines += extra
    lines += nudges
    return "\n".join(lines)


SUMMARIZE = ("Summarize the conversation turns below for your own later use. Keep: every bench id with its "
             "change and delta_pct, bundle ids, gate request and gate ids with outcomes, fact ids, file edits "
             "made, errors that blocked you, and what remains to do. Drop reasoning chatter. At most 250 words, "
             "plain lines.")
