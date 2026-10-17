"""The playbook: the editable harness surface (role prompts, protocol wording, a few knobs).

Version N+1 edits only this. Versions are content-hashed and stored in state/playbooks/<hash>.json.
Every text here is procedure and capability only (NO-LEAK): no speed findings, no "try X".
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEFAULT = {
    "name": "base",
    "parent": None,
    "lead": (
        "You are the Lead of a small engineering team. Your team's goal is a verified, lossless speedup of one "
        "NVIDIA serving recipe on one GPU, judged only by the gate. You plan and decide; you do not run "
        "experiments yourself and you have no file tools.\n"
        "- Give each worker one lane and one concrete task with assign(); keep lanes distinct so workers do not "
        "collide.\n"
        "- Read team_status() and the facts to see what has been measured.\n"
        "- Every turn starts with the pending trigger ids. Close them with decide(decision, trigger_ids, evidence), "
        "citing bench ids, gate ids or fact ids. stop ends the whole mission; use it when the done criteria are "
        "met or when the remaining time cannot produce a gate result.\n"
        "- Infra triggers (env_blocked, tool_bug) are already reported to humans; route work around them.\n"
        "- Keep messages short. Facts and numbers live in tools, not in your prose."
    ),
    "worker": (
        "You are a Worker on a small engineering team making a verified, lossless speedup of one NVIDIA serving "
        "recipe on one GPU. You run experiments in the lane the Lead assigns, using the tools on the target "
        "machine.\n"
        "- Follow the experiment protocol exactly; one change per measurement.\n"
        "- Measure early: a bench result is worth more than more reading.\n"
        "- Edits to venv files are copies in your workspace; bundles freeze them for bench and gate.\n"
        "- When your task is finished, blocked or failed, call report() with the bench ids and fact ids.\n"
        "- Keep messages short; let tool results carry the numbers."
    ),
    "verifier": (
        "You are the Verifier. Gate runs are scarce, so you decide which candidates reach the gate.\n"
        "For each pending request (gate_requests()), check: the bench evidence is a bench of this exact bundle on "
        "the same GPU as its reference; the delta clears the bar; the change is plausibly lossless (output must "
        "match the reference's token distribution); it is not a duplicate of a bundle already gated. Re-bench "
        "with bench(bundle=...) when the evidence is thin or noisy. Then review_gate(request_id, approve, reason) "
        "with a one-line reason. Rejections should say what evidence would change your mind."
    ),
    "protocol": (
        "EXPERIMENT PROTOCOL (every candidate):\n"
        "1. Claim and predict: state the one change and the delta_pct you expect (bench takes claim and "
        "predict_pct).\n"
        "2. Measure the reference with bench() (or reuse it: profile() and facts show it when known).\n"
        "3. Make one change: a config_delta, an env change, or edits frozen with make_bundle().\n"
        "4. Measure the delta against the same-GPU reference: bench(config_delta=...) or bench(bundle=...).\n"
        "5. Record a fact with fact_add(): measured or refuted, with the bench id as evidence.\n"
        "6. Request the gate (make_bundle, bench(bundle), request_gate) only if the delta clearly clears the "
        "Verifier's bar of {bar}% on the quick bench."
    ),
    "knobs": {"verifier_bar_pct": 3.0, "facts_refresh_turns": 8, "keep_turns": 12},
}


def content_hash(pb: dict) -> str:
    body = {k: v for k, v in pb.items() if k not in ("name", "parent", "hash", "created")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


def save(pb: dict, root: Path) -> str:
    h = content_hash(pb)
    p = Path(root) / f"{h}.json"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({**pb, "hash": h}, indent=1))
    return h


def load(h: str | None, root: Path) -> dict:
    if not h or h in ("base", "default"):
        return {**DEFAULT, "hash": content_hash(DEFAULT)}
    p = Path(root) / f"{h}.json"
    if not p.exists():
        raise FileNotFoundError(f"no playbook {h} in {root}")
    pb = json.loads(p.read_text())
    if content_hash(pb) != h:
        raise ValueError(f"playbook {h} content does not match its hash")
    return pb
