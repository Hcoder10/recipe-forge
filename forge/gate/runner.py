"""One gate run: paired AB/BA reps, one timed pass per server, lossless checks, verdict.

Sides: A = the reference (the recipe exactly as published), B = the candidate (recipe + bundle).
Reference gates run the reference on both sides and feed the per-recipe null history.

Per rep r: order AB when r is even, BA when odd. Each server is started by the gate in its own scope,
warmed up on fixed warmup prompts, then timed once over rep r's hidden slice (no prompt repeats in a
server lifetime). After timing it teacher-forces the hidden probes, and the other side's timed outputs
when that side already ran in this pair. Time is end to end per request; tokens are the gate's own
re-tokenized count, checked against the server's count within 1%.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import threading
import time

from forge.gate import lossless, stats

ALPHA = 0.001
ALPHA_CONFIRM = 0.05
MIN_NULL_PAIRS = 8
TOKEN_TOL = 0.01
PROBE_TOPK = 5


@dataclasses.dataclass
class Workload:
    concurrency: int = 1
    max_tokens: int = 512
    metric: str = "decode_tok_s"
    prompts_per_rep: int = 8


class GateInvalid(Exception):
    """The run cannot be judged (machine not quiet, server failed, counts disagree...)."""


def _ids_from_tokens(tokens: list[str]) -> list[int]:
    return [int(str(t).split(":")[-1]) for t in tokens]


def timed_pass(server, model: str, prompts: list[list[int]], wl: Workload, retok) -> dict:
    """One pass, every prompt once. Returns per-request outputs and the metric."""
    results: list[dict | None] = [None] * len(prompts)
    errors: list[str] = []
    idx = iter(range(len(prompts)))
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                i = next(idx, None)
            if i is None:
                return
            try:
                r, dt = server.complete(model, prompts[i], wl.max_tokens, logprobs=1)
            except Exception as e:  # recorded; the pass becomes invalid
                errors.append(f"{type(e).__name__}: {str(e)[:200]}")
                return
            ch = r["choices"][0]
            lp = ch.get("logprobs") or {}
            out_ids = _ids_from_tokens(lp.get("tokens") or [])
            results[i] = {"dt": dt, "text": ch.get("text", ""), "ids": out_ids,
                          "self_lp": lp.get("token_logprobs") or [],
                          "reported": (r.get("usage") or {}).get("completion_tokens", len(out_ids))}

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, wl.concurrency))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    if errors or any(r is None for r in results):
        raise GateInvalid(f"timed pass failed: {errors[:3]}")
    for r in results:
        r["tokens"] = len(retok(r["text"]))
    tok = sum(r["tokens"] for r in results)
    rep = sum(r["reported"] for r in results)
    if rep == 0 or abs(tok - rep) > TOKEN_TOL * rep:
        raise GateInvalid(f"token count mismatch: gate counted {tok}, server reported {rep}")
    if wl.concurrency <= 1:
        value = tok / sum(r["dt"] for r in results)
    else:
        value = tok / wall
    return {"value": value, "tokens": tok, "reported": rep, "wall_s": wall, "outputs": results}


def teacher_force(server, model: str, seqs: list[list[int]], start: list[int]) -> list[list[dict[int, float]]]:
    """prompt_logprobs (top-K) for each sequence, from position start[i] on."""
    out = []
    for s, st in zip(seqs, start):
        r, _ = server.complete(model, s, 1, prompt_logprobs=PROBE_TOPK)
        plp = lossless.parse_prompt_logprobs(r["choices"][0].get("prompt_logprobs") or [])
        out.append(plp[st:len(s)])
    return out


class History:
    """Per-recipe null history: reference-vs-reference log gains and lossless comparisons."""

    def __init__(self, path: str):
        self.path = path
        try:
            with open(path) as f:
                self.d = json.load(f)
        except (OSError, ValueError):
            self.d = {"null_gains": [], "null_probe": [], "null_decode": [], "controls": []}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path + ".tmp", "w") as f:
            json.dump(self.d, f, indent=1)
        os.replace(self.path + ".tmp", self.path)

    @property
    def n_null(self) -> int:
        return len(self.d["null_gains"])


@dataclasses.dataclass
class RunCtx:
    """Everything a run needs from the outside, so tests can pass fakes.

    make_server(side, name) -> server with start() -> seconds, model_name(), complete(...), stop(),
    log_tail(), cleanup() and attribute `unit`. quiet() -> None or a reason string.
    """
    make_server: object
    chat_ids: object          # messages -> prompt token ids (reference tokenizer + chat template)
    retok: object             # text -> token ids (reference tokenizer)
    quiet: object
    history: History
    log: object = print
    watch: object = None      # (server) -> context manager with .foreign list


def run(kind: str, slices, wl: Workload, reps: int, ctx: RunCtx, confirm_index: int = 0) -> dict:
    """kind: reference | candidate | confirm. Returns the verdict dict (no prompt text in it)."""
    sides = ("A", "A") if kind == "reference" else ("A", "B")
    prompts = [slices.confirm(confirm_index, r, wl.prompts_per_rep) if kind == "confirm"
               else slices.rep(r, wl.prompts_per_rep) for r in range(reps)]
    warm = [ctx.chat_ids(it["messages"]) for it in slices.warmup()]
    probes = [ctx.chat_ids(it["messages"]) for it in slices.probes()]
    pairs, probe_rows, decode_rows, servers_meta = [], [], [], []
    for r in range(reps):
        order = [0, 1] if r % 2 == 0 else [1, 0]
        ids = [ctx.chat_ids(it["messages"]) for it in prompts[r]]
        got: dict[int, dict] = {}
        for pos in order:
            side = sides[pos]
            why = ctx.quiet()
            if why:
                raise GateInvalid(f"machine not quiet before rep {r}: {why}")
            name = f"r{r}{'ab'[pos]}"
            srv = ctx.make_server(side, name)
            meta = {"rep": r, "slot": "ab"[pos], "side": side}
            try:
                meta["startup_s"] = srv.start()
                model = srv.model_name()
                for w in warm:
                    srv.complete(model, w, min(64, wl.max_tokens))
                if ctx.watch:
                    with ctx.watch(srv) as wt:
                        res = timed_pass(srv, model, ids, wl, ctx.retok)
                    if wt.foreign:
                        raise GateInvalid(f"foreign GPU process during timed pass: {wt.foreign[:2]}")
                else:
                    res = timed_pass(srv, model, ids, wl, ctx.retok)
                res["probe"] = teacher_force(srv, model, probes, [1] * len(probes))
                other = 1 - pos
                if other in got:   # teacher-force the other side's timed outputs on this server
                    o = got[other]
                    seqs = [p + out["ids"] for p, out in zip(ids, o["outputs"])]
                    res["tf_other"] = teacher_force(srv, model, seqs, [len(p) for p in ids])
                got[pos] = res
                meta.update({"value": res["value"], "tokens": res["tokens"], "wall_s": round(res["wall_s"], 3)})
            except GateInvalid:
                raise
            except Exception as e:
                raise GateInvalid(f"server {name} ({side}) failed: {type(e).__name__}: {str(e)[:300]}") from e
            finally:
                meta["log_tail"] = srv.log_tail(64 * 1024)
                srv.stop()
                srv.cleanup()
                servers_meta.append(meta)
                ctx.log(f"rep {r} {name} side {side}: {meta.get('value')}")
        a, b = got[0], got[1]
        pairs.append({"rep": r, "order": "AB" if order == [0, 1] else "BA", "a": a["value"], "b": b["value"],
                      "log_gain": math.log(b["value"] / a["value"])})
        probe_rows.append(lossless.probe_compare(a["probe"], b["probe"]))
        first, second = (a, b) if order == [0, 1] else (b, a)
        decode_rows.append({"chosen_by": "b" if first is b else "a", **lossless.decode_regret(
            [o["ids"] for o in first["outputs"]], [o["self_lp"] for o in first["outputs"]], second["tf_other"])})
    gains = [p["log_gain"] for p in pairs]
    v = {"kind": kind, "reps": reps, "pairs": pairs, "probe": probe_rows, "decode": decode_rows,
         "servers": servers_meta, "workload": dataclasses.asdict(wl)}
    h = ctx.history
    if kind == "reference":
        v["verdict"] = "reference"
        v["null_pairs_before"] = h.n_null
        return v
    null = h.d["null_gains"]
    v["test"] = stats.paired_test(gains, null, ALPHA_CONFIRM if kind == "confirm" else ALPHA)
    # lossless: BA pairs judge the candidate's own decode choices under a reference that teacher-forced
    # them; AB pairs judge the reference's choices under the candidate. Both must sit in the null band.
    v["lossless"] = lossless.judge(probe_rows, decode_rows, lossless.band(h.d["null_probe"], h.d["null_decode"]))
    if h.n_null < MIN_NULL_PAIRS:
        v["verdict"], v["detail"] = "invalid", f"only {h.n_null} reference pairs (< {MIN_NULL_PAIRS})"
    elif not v["lossless"]["pass"]:
        v["verdict"], v["detail"] = "lossy", "; ".join(v["lossless"]["fails"][:3])
    elif v["test"]["win"]:
        v["verdict"], v["detail"] = "win", f"+{v['test']['gain_pct']:.2f}% > {v['test']['threshold_pct']:.2f}%"
    else:
        v["verdict"], v["detail"] = "loss", v["test"]["why"]
    v["delta_pct"] = (math.exp(sum(gains) / len(gains)) - 1) * 100
    return v


def record_reference(h: History, v: dict, gate_id: str) -> None:
    """Add a reference gate's pairs and lossless comparisons to the null history."""
    h.d["null_gains"] += [p["log_gain"] for p in v["pairs"]]
    h.d["null_probe"] += v["probe"]
    h.d["null_decode"] += v["decode"]
    h.d["controls"].append(gate_id)
    h.save()
