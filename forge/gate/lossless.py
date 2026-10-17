"""Lossless checks from teacher-forced logprobs.

Two signals, both compared with the reference's own run-to-run band (reference vs reference):

probe:   the server under test teacher-forces hidden probe prompts (prompt_logprobs, top-K). Against a
         reference server's probe run: top-1 agreement, mean KL(ref || test) over the reference top-K,
         and max regret (how much worse, under the reference, the test server's top-1 is).
decode:  a reference server teacher-forces the test server's timed outputs. Regret of every token the
         test server chose (reference top-1 logprob minus reference logprob of the chosen token), and
         the gap between the logprob the test server reported for its own token and the reference's.

Positions whose regret is under the near-tie floor never count as disagreements.
"""
from __future__ import annotations

import math

NEAR_TIE_NATS = 0.25


def parse_prompt_logprobs(plp: list) -> list[dict[int, float]]:
    """vLLM prompt_logprobs (list of None | {token_id: {logprob, rank, ...}}) -> list of {id: logprob}."""
    out = []
    for pos in plp or []:
        if not pos:
            out.append({})
            continue
        d = {}
        for k, v in pos.items():
            tid = int(str(k).split(":")[-1]) if not isinstance(k, int) else k
            d[tid] = float(v["logprob"] if isinstance(v, dict) else v)
        out.append(d)
    return out


def _top1(d: dict[int, float]) -> int | None:
    return max(d, key=d.get) if d else None


def probe_compare(ref: list[list[dict[int, float]]], test: list[list[dict[int, float]]]) -> dict:
    """Compare per-position top-K distributions of two probe runs over the same token sequences."""
    n = agree = flips = 0
    kl_sum = 0.0
    max_regret = 0.0
    for r_seq, t_seq in zip(ref, test):
        for r, t in zip(r_seq, t_seq):
            if not r or not t:
                continue
            n += 1
            r1, t1 = _top1(r), _top1(t)
            floor_r = min(r.values())
            regret = r[r1] - r.get(t1, floor_r)
            if r1 == t1:
                agree += 1
            elif regret > NEAR_TIE_NATS:
                flips += 1
            max_regret = max(max_regret, regret)
            floor_t = min(t.values())
            kl = 0.0
            for tok, lp in r.items():
                kl += math.exp(lp) * (lp - t.get(tok, floor_t))
            kl_sum += max(0.0, kl)
    return {"positions": n, "top1_agree": agree / n if n else None, "flips": flips,
            "flip_rate": flips / n if n else None, "kl_mean": kl_sum / n if n else None,
            "max_regret": max_regret}


def decode_regret(chosen: list[list[int]], self_lp: list[list[float | None]],
                  ref_tf: list[list[dict[int, float]]]) -> dict:
    """chosen[i]: output token ids of request i; self_lp[i]: the test server's logprob for each;
    ref_tf[i]: reference teacher-forced top-K at each output position (aligned to chosen[i])."""
    n = flips = 0
    max_regret = 0.0
    gap_sum = 0.0
    gap_n = 0
    for toks, lps, dists in zip(chosen, self_lp, ref_tf):
        for j, tok in enumerate(toks):
            if j >= len(dists) or not dists[j]:
                continue
            d = dists[j]
            n += 1
            r1 = _top1(d)
            lp_tok = d.get(tok)
            regret = d[r1] - (lp_tok if lp_tok is not None else min(d.values()))
            max_regret = max(max_regret, regret)
            if tok != r1 and regret > NEAR_TIE_NATS:
                flips += 1
            if lp_tok is not None and j < len(lps) and lps[j] is not None:
                gap_sum += abs(lps[j] - lp_tok)
                gap_n += 1
    return {"positions": n, "flips": flips, "flip_rate": flips / n if n else None, "max_regret": max_regret,
            "lp_gap_mean": gap_sum / gap_n if gap_n else None}


def band(null_probe: list[dict], null_decode: list[dict]) -> dict:
    """The reference's own run-to-run band from reference-vs-reference comparisons."""
    def worst(rows, key, fn=max, default=0.0):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return fn(vals) if vals else default
    return {"probe_top1_agree_min": worst(null_probe, "top1_agree", min, 1.0),
            "probe_kl_max": worst(null_probe, "kl_mean"),
            "probe_flip_rate_max": worst(null_probe, "flip_rate"),
            "decode_flip_rate_max": worst(null_decode, "flip_rate"),
            "decode_max_regret": worst(null_decode, "max_regret"),
            "decode_lp_gap_max": worst(null_decode, "lp_gap_mean"),
            "n_probe": len(null_probe), "n_decode": len(null_decode)}


def judge(probe_rows: list[dict], decode_rows: list[dict], b: dict) -> dict:
    """Pass iff every candidate comparison sits inside the band (with slack and the near-tie floor)."""
    fails = []
    lim = {"top1": b["probe_top1_agree_min"] - 0.01,
           "kl": max(3 * b["probe_kl_max"], 1e-3),
           "pflip": max(2 * b["probe_flip_rate_max"], 0.002),
           "dflip": max(2 * b["decode_flip_rate_max"], 0.005),
           "dregret": max(2 * b["decode_max_regret"], 4 * NEAR_TIE_NATS),
           "gap": max(3 * b["decode_lp_gap_max"], 0.02)}
    for r in probe_rows:
        if r["top1_agree"] is not None and r["top1_agree"] < lim["top1"]:
            fails.append(f"probe top-1 agreement {r['top1_agree']:.4f} < {lim['top1']:.4f}")
        if r["kl_mean"] is not None and r["kl_mean"] > lim["kl"]:
            fails.append(f"probe KL {r['kl_mean']:.5f} > {lim['kl']:.5f}")
        if r["flip_rate"] is not None and r["flip_rate"] > lim["pflip"]:
            fails.append(f"probe flip rate {r['flip_rate']:.4f} > {lim['pflip']:.4f}")
    for r in decode_rows:
        if r["flip_rate"] is not None and r["flip_rate"] > lim["dflip"]:
            fails.append(f"decode flip rate {r['flip_rate']:.4f} > {lim['dflip']:.4f}")
        if r["max_regret"] > lim["dregret"]:
            fails.append(f"decode max regret {r['max_regret']:.3f} > {lim['dregret']:.3f}")
        if r["lp_gap_mean"] is not None and r["lp_gap_mean"] > lim["gap"]:
            fails.append(f"decode logprob gap {r['lp_gap_mean']:.4f} > {lim['gap']:.4f}")
    if not probe_rows or not decode_rows:
        fails.append("missing lossless evidence (no probe or decode comparison)")
    return {"pass": not fails, "fails": fails[:10], "limits": lim}
