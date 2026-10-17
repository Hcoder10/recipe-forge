"""Pure decode roofline accounting; no benchmarks or GPU access.

Structures may extend arch_profile with context_length, head_dim, attention_heads,
intermediate_size, moe_intermediate_size, ssm_inner_size, ssm_state_size,
ssm_weight_params_per_layer, vocab_size and explicit storage byte widths.
Missing dimensions produce unknown rows, never invented measurements.
"""
from __future__ import annotations

import math

from forge.tools.registry import Tool, ToolError

# Approximate vendor peak specifications, decimal GB/s and TFLOP/s.
# RTX PRO dense BF16/FP32-accumulate peaks inferred from sparse FP4 / 16,
# using the Blackwell precision ratios in the architecture whitepaper.
# https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-6000-max-q/workstation-datasheet-blackwell-rtx-pro-6000-max-q-nvidia-3519233.pdf
# https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf
# https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
# H100 SXM: published sparse BF16 divided by two.
# https://www.nvidia.com/en-us/data-center/h100/
# B200: DGX per-GPU dense FP8 divided by two for BF16.
# https://www.nvidia.com/en-us/data-center/dgx-b200/
GPU_SPECS = {
    "rtx pro 6000 blackwell max-q": (1792, 3511 / 16),
    "rtx pro 6000 blackwell workstation": (1792, 4000 / 16),
    "rtx 5090": (1792, 209.5),
    "h100": (3350, 989.5),
    "b200": (8000, 2250),
}
CATEGORIES = ("attention", "ssm", "moe", "gemm/quant", "norm", "sampler", "launch-overhead")
ALIASES = {"ssm/mamba": "ssm", "dense gemm": "gemm/quant", "quant/dequant": "gemm/quant",
           "norm/act": "norm", "sampler/topk/softmax": "sampler", "launch gaps": "launch-overhead"}


def _number(value, field, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError("Structural dimensions and measurements must be numeric")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ToolError("Structural dimensions and measurements must be finite and nonnegative")
    return value


def _gpu_spec(gpu):
    if isinstance(gpu, dict):
        gpu = gpu.get("name")
    if not isinstance(gpu, str):
        raise ToolError("Supply a supported GPU name or profile the mission first")
    name = " ".join(gpu.lower().replace("_", " ").split())
    if "6000" in name and "blackwell" in name:
        key = "rtx pro 6000 blackwell max-q" if "max-q" in name else "rtx pro 6000 blackwell workstation"
        if "server" in name:
            raise ToolError("Supply a supported GPU variant")
    else:
        key = next((k for k in ("rtx 5090", "h100", "b200") if k in name), None)
    if key is None or (key == "h100" and any(x in name for x in ("pcie", "nvl"))):
        raise ToolError("Supply a supported GPU variant")
    return GPU_SPECS[key]


def sol_ledger(structure=None, batch=1, gpu=None):
    """Compute rows from a structure or an arch_profile result envelope."""
    if not isinstance(structure, dict):
        raise ToolError("Run arch_profile first or supply its structure with required dimensions")
    _number(batch, "batch", True)
    if int(batch) != batch:
        raise ToolError("Batch must be an integer")
    profile = structure if "structure" in structure else {}
    s = profile.get("structure", structure)
    if not isinstance(s, dict):
        raise ToolError("Structure must be an object")
    bandwidth, peak = _gpu_spec(gpu or profile.get("gpu") or s.get("gpu_class"))
    assumptions = [
        "Approximate vendor bandwidth and dense BF16 tensor peaks; H100 means SXM. Validate precision and variant.",
        "Roofline uses max(memory traffic / bandwidth, FLOPs / peak); no overlap or achieved efficiency is inferred.",
        "Weight traffic assumes cold weights once per step with reuse across the batch; cache residency is excluded.",
        "Dense and expert FFNs assume gated projections; KV traffic includes context reads and new-token writes.",
        "SSM recurrence excludes implementation-specific convolution and projection traffic unless supplied explicitly.",
        "Launch overhead has an ideal zero device-work floor, not an achievable host timing prediction.",
        "Unknown dimensions or absolute measurements remain null; percentages alone cannot establish step time.",
        "Gap percent is measured-minus-SOL divided by measured time; negative gaps require calibration review.",
    ]

    def n(key, default=None):
        value = s.get(key)
        return default if value is None else _number(value, key)

    mix = s.get("layer_mix") or {}
    if not isinstance(mix, dict):
        raise ToolError("Layer mix must be an object")
    counts = {k: _number(mix.get(k, 0), k) for k in ("attention", "ssm", "moe", "mlp")}
    h, layers = n("hidden_size"), n("num_layers", n("num_hidden_layers"))
    wb = n("weight_bytes", 2 if not s.get("quantization") else None)
    ab, kb, sb = n("activation_bytes", 2), n("kv_bytes", 2), n("ssm_state_bytes", 4)
    assumptions.append("Unspecified activation and KV storage use half precision; SSM state uses single precision. Supply actual widths.")
    if wb == 2 and "weight_bytes" not in s:
        assumptions.append("Unquantized weights default to half precision; quantized weights require an explicit byte width.")
    work = {k: None for k in CATEGORIES}
    for category, count in (("attention", counts["attention"]), ("ssm", counts["ssm"]),
                            ("moe", counts["moe"]), ("gemm/quant", counts["mlp"])):
        if count == 0 and mix:
            work[category] = (0, 0)
    # Attention projections are accounted here, avoiding duplication in GEMM.
    heads, hd, kv = n("attention_heads", n("num_attention_heads")), n("head_dim"), n("kv_heads", n("num_key_value_heads"))
    context = n("context_length")
    if hd is None and h is not None and heads:
        hd = h / heads
    if counts["attention"] and all(x is not None for x in (h, heads, hd, kv, context, wb)):
        params = counts["attention"] * (2 * h * heads * hd + 2 * h * kv * hd)
        traffic = params * wb + counts["attention"] * 2 * batch * (context + 1) * kv * hd * kb
        flops = 2 * batch * params + counts["attention"] * 4 * batch * heads * hd * context
        work["attention"] = traffic, flops
    inner = n("intermediate_size")
    if counts["mlp"] and all(x is not None for x in (h, inner, wb)):
        params = counts["mlp"] * 3 * h * inner
        work["gemm/quant"] = params * wb, 2 * batch * params
    experts, top = n("num_experts"), n("num_experts_per_tok")
    ei = n("moe_intermediate_size", inner)
    if counts["moe"] and all(x is not None for x in (h, experts, top, ei, wb)):
        if not experts or not top or top > experts:
            raise ToolError("Expert routing dimensions are inconsistent")
        active = n("active_experts_per_step", min(experts, batch * top))
        if not top <= active <= min(experts, batch * top):
            raise ToolError("Active expert count is inconsistent with batch routing")
        shared = n("num_shared_experts", 0)
        params = 3 * h * ei
        router = h * experts
        work["moe"] = (counts["moe"] * ((active + shared) * params + router) * wb,
                       counts["moe"] * 2 * batch * ((top + shared) * params + router))
        assumptions.append("Expert union defaults to distinct routes capped by expert count; supply active_experts_per_step for observed routing.")
    si, state, sw = n("ssm_inner_size"), n("ssm_state_size"), n("ssm_weight_params_per_layer")
    if counts["ssm"] and all(x is not None for x in (si, state, sw, wb)):
        work["ssm"] = (counts["ssm"] * (2 * batch * si * state * sb + sw * wb),
                       counts["ssm"] * batch * (5 * si * state + 2 * sw))
    if h is not None and layers is not None:
        norms = n("norms_per_layer", 2) * layers
        work["norm"] = norms * (h * wb + 2 * batch * h * ab) if wb is not None else None
        work["norm"] = None if work["norm"] is None else (work["norm"], norms * batch * h * 5)
    vocab = n("vocab_size")
    if vocab is not None and h is not None and wb is not None:
        work["sampler"] = h * vocab * wb + 2 * batch * vocab * ab, 2 * batch * h * vocab + 5 * batch * vocab
        assumptions.append("Sampler row includes the output projection and a logits read/write pass; embedding lookup is excluded.")
    work["launch-overhead"] = (0, 0)
    # Explicit per-category accounting permits unusual architectures and real kernel FLOP counts.
    overrides = s.get("category_work") or {}
    for cat, values in overrides.items():
        cat = ALIASES.get(cat, cat)
        if cat not in work or not isinstance(values, dict):
            raise ToolError("Category work must use supported ledger categories")
        work[cat] = (_number(values.get("bytes_per_step"), "bytes"),
                     _number(values.get("flops_per_step"), "flops"))
    measured = {}
    source = profile or s
    absolute = source.get("time_by_category_ms_per_step") or source.get("measured_ms_per_step") or {}
    if not isinstance(absolute, dict):
        raise ToolError("Category measurements must be an object")
    for cat, value in absolute.items():
        cat = ALIASES.get(cat, cat)
        measured[cat] = measured.get(cat, 0) + _number(value, "measurement")
    # Profiler percentages are normalized by summed category activity, NOT wall time.
    total = source.get("category_total_ms_per_step")
    if total is not None and not absolute:
        total = _number(total, "category total")
        for cat, pct in (source.get("time_by_category_pct") or {}).items():
            cat = ALIASES.get(cat, cat)
            measured[cat] = measured.get(cat, 0) + total * _number(pct, "percentage") / 100
    rows = []
    for cat, cost in work.items():
        sol = None if cost is None else max(cost[0] / (bandwidth * 1e9), cost[1] / (peak * 1e12)) * 1000
        m = measured.get(cat)
        gap = None if m is None or sol is None else m - sol
        row = {"category": cat, "sol_ms_per_step": sol, "gap_ms": gap,
               "gap_pct": None if gap is None or not m else 100 * gap / m}
        if m is not None:
            row["measured_ms_per_step"] = m
        if cost is not None:
            row.update(bytes_per_step=cost[0], flops_per_step=cost[1])
        else:
            assumptions.append("Supply missing structural dimensions for the " + cat + " row before ranking it.")
        rows.append(row)
    rows.sort(key=lambda row: (row["gap_ms"] is None, -(row["gap_ms"] or 0), row["category"]))
    return {"rows": rows, "assumptions": assumptions}


def t_sol_ledger(ctx, structure=None, batch=1, gpu=None):
    cached = getattr(ctx.mission, "_sol_arch_profile", None)
    return sol_ledger(structure if structure is not None else cached, batch,
                      gpu or (cached or {}).get("gpu") or getattr(ctx.mission, "gpu_class", None))


def register(reg):
    reg.add(Tool("sol_ledger", "Build a decode speed-of-light ledger from cached architecture or supplied structure; no GPU work.",
                 {"type": "object", "properties": {"structure": {"type": "object"},
                  "batch": {"type": "integer", "minimum": 1}, "gpu": {"type": "string"}}, "required": []},
                 t_sol_ledger, ("lead", "worker", "verifier"), order=4,
                 returns="category rows sorted by measured gap, with explicit assumptions and unknowns."))
