"""Recipe catalog: NVIDIA's published vLLM serve recipes for Nemotron models, one YAML per recipe.

Each YAML is a *base* recipe (model + NVIDIA's exact serve args + source). Its `variants` expand it into
the runnable recipes the fleet schedules, with ids like `nano-9b-v2:c8`:

    from forge.recipes import load_all, get
    for r in load_all():                 # expanded variants, sorted by id
        print(r["id"], r["workload"]["concurrency"], r["hardware"]["fits"])
    r = get("bolt-74b-lilicorr:c1", gpu_class="rtxpro6000_95g")

An expanded recipe is a plain dict (JSON-able) with the base fields plus:
    base_id, variant, workload {concurrency, max_tokens, temperature, metric, bench_prompts, chat},
    serve.args (published args with the GPU class's documented `hardware.adapt` applied, if any),
    serve.env (= vllm.env), serve.published_args (verbatim), serve.adapted (why, or None)
"""
from __future__ import annotations

import copy
import glob
import hashlib
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(HERE, "prompts")
DEFAULT_PROMPTS = "forge/recipes/prompts/public_v1.jsonl"   # relative to FORGE_ROOT
GPU_CLASSES = ("rtxpro6000_95g", "rtx5090_32g")


def load_base(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        r = yaml.safe_load(f)
    stem = os.path.splitext(os.path.basename(path))[0]
    if r.get("id") != stem:
        raise ValueError(f"{path}: id {r.get('id')!r} must equal the file stem {stem!r}")
    for k in ("model", "serve", "variants", "hardware"):
        if k not in r:
            raise ValueError(f"{path}: missing {k}")
    if not r["serve"].get("source"):
        raise ValueError(f"{path}: serve.source (the published page) is required")
    return r


def bases() -> list[dict]:
    return [load_base(p) for p in sorted(glob.glob(os.path.join(HERE, "*.yaml")))]


def apply_flags(args: list[str], set_flags: dict) -> list[str]:
    """Replace the value of each `--flag` in args (or append it). Values are strings."""
    out = list(args)
    for flag, val in (set_flags or {}).items():
        if flag in out:
            i = out.index(flag)
            if i + 1 < len(out) and not out[i + 1].startswith("--"):
                out[i + 1] = str(val)
            else:
                out.insert(i + 1, str(val))
        else:
            out += [flag, str(val)]
    return out


def expand(base: dict, gpu_class: str | None = None) -> list[dict]:
    out = []
    for v in base["variants"]:
        r = copy.deepcopy(base)
        r.pop("variants")
        r["base_id"] = base["id"]
        r["id"] = f"{base['id']}:{v['name']}"
        r["variant"] = v["name"]
        r["workload"] = {
            "concurrency": int(v["concurrency"]),
            "max_tokens": int(v.get("max_tokens", 512)),
            "temperature": 0.0,
            "metric": v.get("metric", "decode_tok_s" if int(v["concurrency"]) == 1 else "output_tok_s"),
            "bench_prompts": v.get("bench_prompts", DEFAULT_PROMPTS),
            "chat": base.get("chat") or {},
        }
        serve = r["serve"]
        serve["published_args"] = list(serve["args"])
        serve["env"] = dict((base.get("vllm") or {}).get("env") or {})
        serve["adapted"] = None
        adapt = ((base.get("hardware") or {}).get("adapt") or {}).get(gpu_class) if gpu_class else None
        if adapt:
            serve["args"] = apply_flags(serve["args"], adapt.get("set") or {})
            serve["adapted"] = {"gpu_class": gpu_class, "set": adapt.get("set"), "why": adapt.get("why")}
        if "--served-model-name" in serve["args"]:
            serve["served_model_name"] = serve["args"][serve["args"].index("--served-model-name") + 1]
        r["gpu_class"] = gpu_class
        r["recipe_sha256"] = recipe_hash(r)
        out.append(r)
    return out


def recipe_hash(r: dict) -> str:
    key = {k: r[k] for k in ("model", "revision", "drafter") if k in r}
    key["args"] = r["serve"]["args"]
    key["env"] = r["serve"].get("env")
    key["workload"] = {k: v for k, v in r["workload"].items() if k != "chat"}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def load_all(gpu_class: str | None = None, runnable_only: bool = False) -> list[dict]:
    out = []
    for b in bases():
        fits = (b.get("hardware") or {}).get("fits") or []
        if runnable_only and not fits:
            continue
        if gpu_class and runnable_only and gpu_class not in fits:
            continue
        out += expand(b, gpu_class)
    return sorted(out, key=lambda r: r["id"])


def get(recipe_id: str, gpu_class: str | None = None) -> dict:
    recipe_id = recipe_id.replace(".", ":", 1) if ":" not in recipe_id and "." in recipe_id else recipe_id
    base_id = recipe_id.split(":", 1)[0]
    path = os.path.join(HERE, base_id + ".yaml")
    if not os.path.exists(path):
        raise KeyError(f"no recipe {base_id!r}; known: {[b['id'] for b in bases()]}")
    for r in expand(load_base(path), gpu_class):
        if r["id"] == recipe_id or (r["base_id"] == recipe_id and r["variant"] == "c1"):
            return r
    raise KeyError(f"recipe {base_id} has no variant for {recipe_id!r}")


def fits(recipe: dict, gpu_class: str) -> bool:
    return gpu_class in ((recipe.get("hardware") or {}).get("fits") or [])


def load_prompts(path: str | None = None) -> list[dict]:
    path = path or os.path.join(PROMPTS_DIR, "public_v1.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
