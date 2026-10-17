"""Public-workload headline measurements; numerical reports are operator-only."""
from __future__ import annotations

import json
import math
import random
import uuid
from pathlib import Path, PurePosixPath

from forge.tools.registry import EnvBlocked, ToolError

NOTICE = "Headline measurement only; not a substitute for the gate. Hardware edition and runtime may differ."


def workload(recipe):
    """Build neutral public inputs without consulting any gate prompt source."""
    published = recipe.get("published")
    wl = dict(recipe.get("workload") or {})
    if published:
        for key in ("concurrency", "isl", "osl"):
            value = published.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("published workload lengths and concurrency must be positive integers")
        wl.update(concurrency=published["concurrency"], max_tokens=published["osl"],
                  isl=published["isl"], metric=published["metric"], chat=None)
    wl.update(ignore_eos=True, strict_fixed_length=True)
    prompts = None
    if published or wl.get("isl"):
        rng = random.Random(0)
        words = ("tree", "stone", "water", "cloud", "field", "river", "book", "light")
        prompts = [{"prompt": " ".join(rng.choice(words) for _ in range(int(wl["isl"]))),
                    "max_tokens": int(wl["max_tokens"])}
                   for _ in range(int(wl.get("concurrency", 1)))]
    return wl, prompts


def run_record(spec, work, gpu):
    """Called inside one scheduler grant; every server uses the sandbox bench path."""
    from forge.tools.benchjob import serve_and_measure
    reps = spec.get("reps", 3)
    if isinstance(reps, bool) or not isinstance(reps, int) or not 1 <= reps <= 20:
        raise ValueError("reps must be an integer in 1..20")
    wl, prompts = workload(spec["record_recipe"])
    spec = {**spec, "sandbox": True, "workload": wl, "n_prompts": None}
    if prompts is not None:
        path = work / "record-prompts.jsonl"
        path.write_text("".join(json.dumps(p) + "\n" for p in prompts), encoding="utf-8")
        spec["prompts"] = str(path)
    candidate = spec.get("candidate") or {}
    runs = []
    for rep in range(reps):
        for side in ("reference", "candidate"):
            c = candidate if side == "candidate" else {}
            result = serve_and_measure(spec, c.get("bundle_dir"), c.get("config_delta"),
                                       c.get("env"), f"record-{side}-{rep}", work)
            if (not result.get("fixed_length") or result.get("token_count_mismatch")
                    or not result.get("n")):
                raise ValueError("record run did not complete the fixed-length workload")
            row = {"side": side, "rep": rep + 1, "gpu": gpu,
                   "output_tok_s": result["output_tok_s"],
                   "output_tok_s_per_user": result["output_tok_s"] / int(wl.get("concurrency", 1)),
                   "decode_tok_s": result["decode_tok_s"], "ttft_ms_mean": result["ttft_ms_mean"]}
            if not all(math.isfinite(row[k]) for k in
                       ("output_tok_s", "output_tok_s_per_user", "decode_tok_s", "ttft_ms_mean")):
                raise ValueError("record run returned invalid metrics")
            runs.append(row)
    metrics = ("output_tok_s", "output_tok_s_per_user", "decode_tok_s", "ttft_ms_mean")
    means = {side: {k: sum(r[k] for r in runs if r["side"] == side) / reps for k in metrics}
             for side in ("reference", "candidate")}
    published = spec["record_recipe"].get("published")
    return {"ok": True, "mode": "record", "gpu": gpu, "runs": runs, "means": means,
            "published": published, "workload": {k: wl.get(k) for k in
                ("concurrency", "isl", "max_tokens", "metric", "ignore_eos")},
            "workload_source": "published" if published else "recipe workload (no published block)",
            "note": NOTICE}


def record_bench(ctx, bundle=None, config_delta=None, env=None, reps=3):
    """Persist the full report for operators and return a measurement-free receipt."""
    from forge.agents import events
    from forge.tools.bundle import normalize_delta
    if ctx.role not in ("lead", "verifier"):
        raise ToolError("record_bench is available to the Lead and Verifier only")
    if isinstance(reps, bool) or not isinstance(reps, int) or not 1 <= reps <= 20:
        raise ToolError("reps must be an integer in 1..20")
    if bundle and (config_delta or env):
        raise ToolError("bundle cannot be combined with config_delta or env")
    m, runner = ctx.mission, ctx.services.bench
    bundle_dir = None
    if bundle:
        if bundle not in m.bundles:
            raise ToolError("unknown bundle")
        bundle_dir = m.bundle_dir(bundle)
    rec = m.recipe
    try:
        workload(rec)
    except (ValueError, KeyError, TypeError):
        raise ToolError("recipe has no complete public workload metadata") from None
    record_id = "record-" + uuid.uuid4().hex
    work = str(PurePosixPath(runner.forge_root) / "state" / "recordbench" / record_id)
    spec = {"mode": "record", "reps": reps, "python": m.python,
            "recipe": {"model": rec["model"], "serve_args": rec["serve"]["args"],
                       "env": rec["serve"].get("env") or {},
                       "served_model_name": rec["serve"].get("served_model_name")},
            "record_recipe": {"published": rec.get("published"), "workload": rec["workload"]},
            "prompts": m.bench_prompts, "work_dir": work, "sandbox": True,
            "ref_cache_dir": str(PurePosixPath(runner.forge_root) / "state" / "refcache"),
            "cache_dir": str(PurePosixPath(runner.forge_root) / "state" / "benchcache"),
            "serve_cwd": getattr(m, "serve_cwd", None), "hf_home": getattr(m, "hf_home", None),
            "candidate": {"bundle_dir": bundle_dir, "config_delta": normalize_delta(config_delta) if config_delta else None,
                          "env": env}}
    try:
        spec_path = str(PurePosixPath(work) / "spec.json")
        runner.ex.write_text(spec_path, json.dumps(spec))
        result = runner.ex.run([runner.python, "-m", "forge.tools.benchjob", "--spec", spec_path],
                               cwd=runner.forge_root, env={"PYTHONPATH": runner.forge_root},
                               timeout_s=runner.timeout_s * 2 * reps, kind="bench", vram_gb=runner.vram_gb,
                               gpus=1, mission_id=m.id, label="record-bench")
        if result.infra_error or result.rc:
            raise ValueError("job failed")
        line = next(l for l in reversed(result.stdout.splitlines()) if l.startswith("BENCH_RESULT "))
        report = json.loads(line[len("BENCH_RESULT "):])
        if not report.get("ok") or report.get("mode") != "record":
            raise ValueError("invalid report")
    except Exception:
        # Do not leak server output, secrets or measurements through an error message.
        raise EnvBlocked("record measurement unavailable; no report was accepted") from None
    root = Path(m.bench_log).parent.parent.parent
    events.append(root / "records.jsonl", {"record_id": record_id, "mission_id": m.id,
                  "recipe_id": m.recipe_id, "bundle": bundle, **report})
    ctx.log_event("record_bench", record_id=record_id, by=ctx.agent)
    return {"record_id": record_id, "status": "recorded", "note": NOTICE}
