"""Append-only memory of measured changes; claims alone are never evidence."""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from pathlib import Path

from forge.agents import config, events
from forge.tools.bundle import args_to_dict, normalize_delta

_LOCK = threading.RLock()


def _path(name):
    return config.state_root() / "winlib" / (name + ".jsonl")


def _canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _values(value):
    if isinstance(value, dict):
        return {str(k): _values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_values(v) for v in value]
    if isinstance(value, str):
        try:
            return _values(json.loads(value))
        except (ValueError, TypeError):
            return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def _flags(delta, env):
    return {"config_delta": _values(normalize_delta(delta)), "env": _values(env or {})}


def arch_traits(recipe: dict, model_config: dict | None) -> dict:
    cfg = model_config or {}
    cfg = {**cfg, **(cfg.get("text_config") or {})}
    flags = args_to_dict((recipe.get("serve") or {}).get("args"))
    layers = cfg.get("num_hidden_layers", cfg.get("n_layer"))
    pattern = next((cfg[k] for k in ("hybrid_override_pattern", "layers_block_type", "layer_types")
                    if cfg.get(k)), [])
    if isinstance(pattern, str):
        pattern = list(pattern) if all(c in "M*-EAS" for c in pattern) else pattern.replace(",", " ").split()
    mix = {"ssm": 0, "attention": 0, "moe": 0}
    for block in pattern:
        block = str(block).lower()
        mix["ssm"] += int(block in ("m", "s") or "mamba" in block or "ssm" in block)
        mix["attention"] += int(block in ("*", "a") or "attention" in block or "attn" in block)
        mix["moe"] += int(block == "e" or "moe" in block)
    experts = next((cfg[k] for k in ("num_experts", "n_routed_experts", "num_local_experts")
                    if cfg.get(k)), None)
    if not pattern and layers:
        mix["ssm" if "mamba" in str(cfg.get("model_type", "")).lower() else "attention"] = layers
    if experts and not mix["moe"] and "hybrid_override_pattern" not in cfg:
        mix["moe"] = layers or len(pattern)
    spec = _values(flags.get("--speculative-config", recipe.get("speculative_config") or {}))
    spec = spec if isinstance(spec, dict) else {}
    drafter = recipe.get("drafter") or {}
    method = spec.get("method") or flags.get("--speculative-config.method")
    if isinstance(drafter, dict):
        method = method or drafter.get("method")
    architectures = cfg.get("architectures") or []
    return {
        "architecture": architectures[0] if architectures else None,
        "model_type": cfg.get("model_type"), "layer_mix": mix,
        "hidden_size": cfg.get("hidden_size", cfg.get("d_model")), "num_layers": layers,
        "kv_heads": cfg.get("num_key_value_heads", cfg.get("num_attention_heads")),
        "num_experts": experts, "num_experts_per_tok": cfg.get("num_experts_per_tok"),
        "quantization": (cfg.get("quantization_config") or {}).get("quant_method")
        or flags.get("--quantization") or recipe.get("quantization"),
        "drafter": bool(drafter or spec or flags.get("--speculative-config.model")),
        "drafter_method": method,
        "concurrency": (recipe.get("workload") or {}).get("concurrency", recipe.get("concurrency")),
        "gpu_class": recipe.get("gpu_class"),
    }


def _number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def _target_file(value):
    path = str(value).replace("\\", "/")
    if "vllm/" in path:
        return "vllm/" + path.split("vllm/", 1)[1]
    return re.sub(r"^(?:[ab]/|overlays/)", "", path)


def code_targets(files_or_targets, diffs=None):
    """Normalize overlay manifests and explicit targets; infer functions only from diff hunks."""
    if isinstance(files_or_targets, str):
        files_or_targets = [files_or_targets]
    if isinstance(files_or_targets, dict):
        if "file" in files_or_targets or "path" in files_or_targets:
            files_or_targets = [files_or_targets]
        else:
            files_or_targets = [{"file": k, **(v if isinstance(v, dict) else {})}
                                for k, v in files_or_targets.items()]
    by_file = {}
    for item in files_or_targets or []:
        item = {"file": item} if isinstance(item, str) else item
        path = _target_file(item.get("file") or item.get("path") or "")
        if not path:
            continue
        funcs = item.get("functions") or ([item["function"]] if item.get("function") else [])
        if isinstance(funcs, str):
            funcs = [funcs]
        by_file.setdefault(path, set()).update(funcs)
        if item.get("diff"):
            diffs = {**(diffs if isinstance(diffs, dict) else {}), path: item["diff"]}
    chunks = diffs.items() if isinstance(diffs, dict) else [(None, diffs or "")]
    for path, diff in chunks:
        current = _target_file(path) if path else None
        in_hunk = False
        for line in str(diff).splitlines():
            if line.startswith("+++ "):
                current = _target_file(line[4:].split("\t")[0])
                in_hunk = False
            elif line.startswith("@@"):
                in_hunk = True
            if in_hunk and current in by_file:
                match = re.search(r"\b(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", line)
                if match:
                    by_file[current].add(match[1])
    return [{"file": path, "functions": sorted(funcs)} for path, funcs in sorted(by_file.items())]


def match_code(files_or_targets, entries) -> str | None:
    """Require a shared file AND function; unknown functions never imply code equivalence."""
    wanted = code_targets(files_or_targets)
    for entry in entries:
        targets = code_targets(entry.get("targets") or (entry.get("change") or {}).get("targets"))
        if any(a["file"] == b["file"] and set(a["functions"]) & set(b["functions"])
               for a in wanted for b in targets):
            return entry["entry_id"]
    return None


def distill(mission_dir: Path, recipe: dict, model_config: dict | None, gpu_class: str) -> list[dict]:
    """Return newly appended entries. Repeated measurements collapse to their latest result."""
    mission_dir = Path(mission_dir)
    log = events.read(mission_dir / "events.jsonl")
    mid = next((r["mission_id"] for r in log if r.get("mission_id")), mission_dir.name)
    requests = events.read(mission_dir / "gate_requests.jsonl")
    traits = arch_traits(recipe, model_config)
    traits["gpu_class"] = gpu_class
    grouped = {}
    for bench in events.read(mission_dir / "benches.jsonl"):
        delta = _number((bench.get("result") or {}).get("delta_pct"))
        if delta is None:
            continue
        bundle = bench.get("bundle")
        info = next((r for r in requests if bundle and r.get("bundle") == bundle), {})
        manifest = info.get("manifest") or bench.get("manifest") or (bundle if isinstance(bundle, dict) else {})
        if not manifest and isinstance(bundle, str):
            bundle_path = mission_dir / "bundles" / bundle / "manifest.json"
            if bundle_path.is_file() and bundle_path.resolve().is_relative_to(mission_dir.resolve()):
                manifest = json.loads(bundle_path.read_text())
        change = _flags(bench.get("config_delta") or manifest.get("config_delta"),
                        bench.get("env") or manifest.get("env"))
        if bundle:
            change["bundle_hash"] = bench.get("bundle_hash") or info.get("hash") or (
                bundle.get("hash") if isinstance(bundle, dict) else bundle)
        files = bench.get("files") or manifest.get("files") or (
            list(manifest.get("overlays") or []) + list(manifest.get("kernels") or []))
        if files:
            change["files"] = sorted(files, key=_canon)
        if not any(change.values()):
            continue
        key = hashlib.sha256(_canon([mid, change]).encode()).hexdigest()
        gates = [r for r in log if r.get("kind") == "gate_result" and (
            (bench.get("bench_id") and r.get("bench_id") == bench["bench_id"])
            or (bundle and r.get("bundle") == bundle)
            or (info.get("id") and r.get("request_id") == info["id"]))]
        gate = next((r for r in reversed(gates) if r.get("verdict") in
                     ("win", "loss", "lossy", "gated_win", "gated_loss")), None)
        targets = code_targets(manifest.get("overlays") or files,
                               bench.get("diffs") or bench.get("diff") or manifest.get("diffs") or manifest.get("diff"))
        category = next((r.get("arch_profile_category") or
                         (r.get("arch_profile") or {}).get("category") for r in
                         [gate or {}, bench, bench.get("result") or {}]
                         if r.get("arch_profile_category") or isinstance(r.get("arch_profile"), dict)
                         and r["arch_profile"].get("category")), None)
        if category:
            for target in targets:
                target["category"] = category
        outcome = "quick_gain" if delta > 0 else "quick_loss"
        evidence = [bench["bench_id"]] if bench.get("bench_id") else []
        if gate:
            outcome = {"win": "gated_win", "loss": "gated_loss"}.get(gate["verdict"], gate["verdict"])
            delta = _number(gate.get("delta_pct"))
            if gate.get("gate_id"):
                evidence.append(gate["gate_id"])
        tasks = [r for r in log if r.get("kind") == "worker_task" and r.get("worker") == bench.get("by")
                 and r.get("t", 0) <= bench.get("t", float("inf"))]
        lane = bench.get("lane") or (tasks[-1].get("lane") if tasks else None) or "config"
        previous = grouped.get(key)
        if previous:
            evidence = list(dict.fromkeys(previous["outcome"]["evidence"] + evidence))
            if previous["outcome"]["kind"] in ("gated_win", "gated_loss", "lossy") and not gate:
                previous["outcome"]["evidence"] = evidence
                continue
        change["summary"] = bench.get("claim") or manifest.get("note") or _canon(change)
        grouped[key] = {"entry_id": key, "mission_id": mid, "recipe_id": recipe.get("id"),
                        "gpu_class": gpu_class, "traits": traits, "lane": lane, "change": change,
                        "outcome": {"kind": outcome, "delta_pct": delta, "evidence": evidence},
                        "by": bench.get("by", "unknown"), "targets": targets}
    with _LOCK:
        known = {r["entry_id"] for r in events.read(_path("entries"))}
        return [events.append(_path("entries"), row) for key, row in grouped.items() if key not in known]


def relevant(traits: dict, gpu_class: str, k: int = 12) -> list[dict]:
    def score(entry):
        other = entry.get("traits") or {}
        points = sum(weight for key, weight in (("architecture", 6), ("model_type", 2),
                     ("layer_mix", 4), ("quantization", 2), ("drafter", 2), ("drafter_method", 1))
                     if traits.get(key) is not None and traits.get(key) == other.get(key))
        points += 3 * bool(gpu_class and entry.get("gpu_class") == gpu_class)
        targets = entry.get("targets") or []
        files = {_target_file(f) for f in traits.get("model_files", [])}
        same_arch = bool(traits.get("architecture") and traits["architecture"] == other.get("architecture"))
        categories = {k for k, v in (traits.get("layer_mix") or {}).items() if v}
        if "ssm" in categories:
            categories.add("ssm/mamba")
        if "mlp" in categories:
            categories.add("dense gemm")
        points += 6 * any(_target_file(t.get("file", "")) in files or
                          same_arch and "/models/" in t.get("file", "") for t in targets)
        points += 4 * any(t.get("category") in categories for t in targets)
        a, b = _number(traits.get("concurrency")), _number(other.get("concurrency"))
        if a and b and a > 0 and b > 0:
            points += 2 / (1 + abs(math.log2(a / b)))
        outcome = entry.get("outcome") or {}
        points += {"gated_win": 4, "quick_gain": 2}.get(outcome.get("kind"), -2)
        return points, _number(outcome.get("delta_pct")) or 0, entry.get("t", 0)
    return sorted(events.read(_path("entries")), key=score, reverse=True)[:max(0, k)]


def render(entries) -> str:
    if not entries:
        return ""
    lines = ["WIN LIBRARY (measured by earlier missions; re-verify on this model)"]
    for entry in entries:
        traits, outcome = entry.get("traits", {}), entry.get("outcome", {})
        where = {k: traits[k] for k in ("architecture", "layer_mix", "quantization", "drafter",
                                       "concurrency") if traits.get(k) is not None}
        where["gpu"] = entry.get("gpu_class")
        # Agent-facing text is an index, never a historical speed finding or recommendation.
        line = (f"entry {entry.get('entry_id')} | {entry.get('lane')} | "
                f"targets {_canon(entry.get('targets') or [])[:400]} | "
                f"where {_canon(where)[:250]} | evidence {','.join(outcome.get('evidence', []))[:100]}")
        line = " ".join(line.split())
        # A conservative UTF-8 byte budget also bounds unusually token-dense text.
        if len(("\n".join(lines) + "\n" + line).encode("utf-8")) > 1500:
            break
        lines.append(line)
    return "\n".join(lines)


def match_entry(config_delta, env, entries) -> str | None:
    target = _flags(config_delta, env)
    if not any(target.values()):
        return None
    for entry in entries:
        change = entry.get("change") or {}
        # Flag equality cannot establish reuse of code edits.
        if change.get("files"):
            continue
        if target == _flags(change.get("config_delta"), change.get("env")):
            return entry["entry_id"]
    return None


def record_reuse(mission_id, entry_id, bench_id, delta_pct):
    delta = _number(delta_pct)
    if delta is None:
        return
    with _LOCK:
        rows = events.read(_path("reuse"))
        if any((r["mission_id"], r["entry_id"], r["bench_id"]) ==
               (mission_id, entry_id, bench_id) for r in rows):
            return
        events.append(_path("reuse"), {"mission_id": mission_id, "entry_id": entry_id,
                                      "bench_id": bench_id, "delta_pct": delta})


def reuse_stats() -> dict:
    """Rates: fraction of library entries tried, and fraction of reuse benches helped (>0)."""
    entries, reuse = events.read(_path("entries")), events.read(_path("reuse"))
    per_entry = {r["entry_id"]: {"tried": 0, "helped": 0} for r in entries}
    for row in reuse:
        counts = per_entry.setdefault(row["entry_id"], {"tried": 0, "helped": 0})
        counts["tried"] += 1
        counts["helped"] += int(row["delta_pct"] > 0)
    return {"per_entry": per_entry,
            "missions": len({r["mission_id"] for r in entries + reuse}), "entries": len(entries),
            "reuse_rate": sum(bool(per_entry[r["entry_id"]]["tried"]) for r in entries) / len(entries)
            if entries else 0.0,
            "help_rate": sum(r["delta_pct"] > 0 for r in reuse) / len(reuse) if reuse else 0.0}
