# Cosmos 3 Nano feasibility and unwired recipe

Research checked 2026-10-17. No model was downloaded or executed, no GPU was leased, and no serving compatibility or memory fit was experimentally established. The companion `forge/recipes/cosmos3-nano-video.yaml` is a research draft, not an executable gate configuration.

## Decision

Cosmos 3 Nano is an omni-model with autoregressive reasoning and diffusion generation. Video generation needs a video gate even when **vLLM-Omni** serves it. The present vLLM text gate cannot measure or establish losslessness for generated clips. A text-only reasoner recipe would measure a different task and is deliberately not substituted for the requested video workload.

NO-LEAK takes precedence over the request to reproduce published timing numbers: this document and recipe contain no measured speed findings, comparisons, or optimization advice. Published performance sections are not reproduced. This omission is not a claim that no published measurements exist. Their review must occur outside agent-visible context; the official research entry point is [NVIDIA's Cosmos 3 page](https://research.nvidia.com/labs/cosmos-lab/cosmos3/).

## Identity, access, and supported paths

- Exact first-party model: [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano), without a hyphen between Cosmos and 3. The [Hub API](https://huggingface.co/api/models/nvidia/Cosmos3-Nano) returned `private: false`, `gated: false`, and revision `e59a53c25979a090fa8706c9acc0c254a6e89b92`. The card and metadata were readable anonymously. No Hub license-click gate was required for those requests; an entire weights download was not tested. This does not establish access requirements for every auxiliary component.
- The [NVIDIA Hub search](https://huggingface.co/api/models?author=nvidia&search=Cosmos3-Nano&limit=30) also returned `nvidia/Cosmos3-Nano-Policy-DROID`. That policy checkpoint is not the selected video baseline; its gating/license and generation suitability were not inspected.
- License: [OpenMDW 1.1](https://openmdw.ai/license/1-1/), as linked by the [pinned model card](https://huggingface.co/nvidia/Cosmos3-Nano/blob/e59a53c25979a090fa8706c9acc0c254a6e89b92/README.md). Ungated access does not remove license obligations. No credentials were accessed.
- The card describes 16B parameters and separate autoregressive and diffusion towers; text output uses next-token decoding, while video uses iterative denoising. It lists Linux and BF16 as the tested precision. Other precisions are not established as equivalent baselines.
- The current first-party code is [NVIDIA/cosmos](https://github.com/NVIDIA/cosmos), with native inference in [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/inference.md). Older cosmos-predict/cosmos-reason projects should not be assumed to implement this checkpoint.
- Generation is documented through `Cosmos3OmniPipeline` in Diffusers and through vLLM-Omni. Plain vLLM and the Cosmos 3 Reasoner NIM provide reasoning paths; their text endpoints are not video-generation benchmarks. See [NVIDIA's backend guide](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/README.md).

## Official inference commands and fixed workload

The pinned model card's release container and H200 serving command are:

```sh
docker pull vllm/vllm-omni:cosmos3
vllm serve nvidia/Cosmos3-Nano \
  --omni \
  --host 0.0.0.0 \
  --port 8000 \
  --init-timeout 1800
```

These are source commands, not commands executed during this task. Before running, obtain a landlord lease, inherit its device and attribution environment, and apply its PyTorch memory-fraction setting in the process that owns the model. Do not replace the lease's device mapping with a hard-coded device. Resolve the container tag to a digest and ensure the server loads the pinned checkpoint rather than floating `main`.

The card's text-to-video request uses `POST /v1/videos/sync`, form fields, and `Accept: video/mp4`. Its JSON prompt and negative prompt are serialized from `assets/example_t2v_prompt.json` and `assets/negative_prompt.json`. The fixed example is **1280 x 720, 189 frames, 24 fps, 35 steps, guidance 6.0, maximum sequence length 4096, flow shift 10.0, seed 123**, with guardrails enabled and resolution/duration templates disabled. The draft preserves these settings. The adapter must stringify numeric form fields and JSON-encode `extra_params`; passing the YAML mapping directly as a form value is incorrect. Do not invoke an external prompt upsampler during a measurement: use immutable, already prepared prompt fixtures.

The same card gives this Diffusers generation path (imports and parameters retained; local output filename omitted):

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    enable_safety_checker=True,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0
)
result = pipe(
    prompt=json.dumps(json.load(open("assets/example_t2v_prompt.json"))),
    negative_prompt=json.dumps(json.load(open("assets/negative_prompt.json"))),
    num_frames=189, height=720, width=1280,
    num_inference_steps=35, guidance_scale=6.0,
    generator=torch.Generator(device="cuda").manual_seed(123),
)
```

`result.video` is exported at 24 fps in the source. This published example is not yet a controlled bench: add revision pins, lease enforcement, synchronized timing, and capture of pre-encoding output tensors. The [checkpoint scheduler configuration](https://huggingface.co/nvidia/Cosmos3-Nano/blob/e59a53c25979a090fa8706c9acc0c254a6e89b92/scheduler/scheduler_config.json) uses UniPC; its stored flow shift is 1.0, while the published example explicitly overrides it to 10.0. Preserve that override and the full scheduler configuration.

For completeness, the card's **reasoner-only** command uses the following flags; this does not generate video:

```sh
vllm serve nvidia/Cosmos3-Nano \
  --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --allowed-local-media-path / \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --port 8000
```

The source prefixes this with a single-device CUDA assignment; that prefix is intentionally omitted here because actual execution must inherit the landlord mapping. NVIDIA also documents a Reasoner NIM in its [launch article](https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3/). The current [backend cookbook](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/README.md#generator-nim) additionally documents a **Generator NIM**, with this exact container command:

```sh
docker run --runtime=nvidia --gpus all \
  --shm-size=32GB \
  --ulimit nofile=65536:65536 \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/cosmos3-generator:1.0.0
```

The variable names above are placeholders, not credentials. NIM uses NGC access rather than Hub authentication. This source command exposes all GPUs; any real launch must restrict container devices to its landlord lease. No NGC access, container pull, NIM model selection, or NIM profile fit was tested. The cookbook also lists a unified Cosmos3 NIM with explicit model type/variant selection; it should not be conflated with the standalone generator image.

The newer cookbook's vLLM-Omni command adds `--model-class-name Cosmos3OmniDiffusersPipeline` and `--allowed-local-media-path /` to the generation launch. The draft intentionally records the pinned model card's command, not a mixture of examples from different source versions. Resolving that runtime/version difference is an activation prerequisite.

## Metric and losslessness

Use **seconds per completed clip**, minimized, at the fixed workload above and concurrency one. Include prompt encoding, denoising, VAE decode, and the fixed safety checks. Define the boundary explicitly: an in-process tensor bench excludes model loading and file encoding; an HTTP bench includes request handling, transport, and MP4 encoding. Keep separate histories for those boundaries. Record startup and warmup separately, never inside the steady-state sample. Tokens per second is meaningful for the autoregressive reasoner, not for video generation.

Strict losslessness means identical finite output tensors before lossy video encoding, with matching shape, dtype, frame count, ordering, and color-space/range metadata. Use exact equality (and canonical tensor hashes), not just visual similarity or encoded MP4 byte equality. Fixed seeds alone do not ensure determinism: pin checkpoint and all component revisions, prompt bytes, negative prompt, RNG algorithm/device and initial noise, scheduler/timesteps, step count, guidance, precision, backend versions, hardware topology, VAE, and guardrail settings.

First establish reference-versus-reference repeatability. If exact equality is unattainable, a separately named tolerance mode may compare raw decoded tensors against a **predeclared** absolute/relative bound calibrated only on reference repeats. Its default remains zero; there is no validated tolerance yet. Reject missing outputs, shape changes, NaN/Inf, safety-filtered clips, and incomplete evidence. Do not call perceptual similarity or a changed sampler/step count lossless. MP4-only output cannot establish pre-encoding tensor equivalence; vLLM-Omni needs a private tensor capture hook, or the initial adapter must use Diffusers.

## Memory feasibility

The [transformer weight index](https://huggingface.co/nvidia/Cosmos3-Nano/blob/e59a53c25979a090fa8706c9acc0c254a6e89b92/transformer/diffusion_pytorch_model.safetensors.index.json) reports 30,347,338,240 bytes (about 28.26 GiB) for that component alone. This is not peak VRAM and does not include all encoders, VAE, safety models, activations, temporary buffers, or allocator reserve.

| GPU | Evidence and decision |
| --- | --- |
| RTX 5090, 32 GB | Full published BF16 video profile fit is **unverified**. Transformer weights alone leave little room; do not advertise this as a supported fit or infer feasibility from parameter count. |
| RTX PRO 6000, 96 GB | NVIDIA identifies this workstation class as a Nano target in its launch article. Exact peak memory and fit for the stated serving container and clip profile remain **unverified locally**. It is a candidate for validation, not an established fit. |
| Two GPUs | No local two-GPU execution was tested. Aggregate VRAM is not automatically usable; topology and backend sharding must be explicitly supported and frozen in the reference configuration. |

The native framework [FAQ](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/faq.md) lists **32 GB** for “Cosmos3-Nano (8B)”. This is published memory guidance, but its abbreviated parameter label differs from the 16B model card and it does not identify the exact clip profile, runtime, or peak-memory boundary. It is evidence for possible 32 GB operation, not proof that the full card's 720p vLLM-Omni configuration fits a 5090. The discrepancy remains unresolved; do not turn that row into an unconditional fit assertion.

Measure allocated and reserved peak memory in a leased run, including initialization and decode. Do not fill `hardware.fits` until that exact profile is validated. No GPU work was needed for this research task.

## Minimal bench and gate adapter proposal (not implemented)

Source review: `forge/gate/runner.py` defines `Workload`, `timed_pass`, `teacher_force`, and `run`; `forge/gate/lossless.py` defines `probe_compare`, `decode_regret`, `band`, and `judge`. The runner requests text completions with token IDs, retokenizes returned text, computes token rates, and teacher-forces log probabilities. Merely changing `Workload.metric` does not change those operations. The logprob band cannot compare video tensors. `run(kind, ...)` uses `kind` for reference/candidate/confirm; that argument must not be confused with the recipe's proposed `kind: video`.

Proposed follow-up files/functions (names are design proposals, not existing APIs):

| File | Functions and responsibility |
| --- | --- |
| `forge/gate/video_backend.py` (new) | `load_pipeline`, `generate`, `capture_tensor`, `close`: pinned Diffusers lifecycle first; later a vLLM-Omni backend with private raw-tensor capture. Enforce lease environment and memory fraction before model allocation. |
| `forge/gate/video_bench.py` (new) | `VideoWorkload`, `warmup`, `timed_clip`: recreate identical RNG/noise per fixture, synchronize every participating device before and after generation, use `time.perf_counter`, validate completion, and return outputs plus operator-private measurements. |
| `forge/gate/video_lossless.py` (new) | `canonicalize`, `compare_exact`, `calibrate_reference_band`, `judge`: structural and finite-value checks; strict tensor equality first; explicitly configured tolerance mode only. |
| `forge/gate/video_runner.py` (new) | `run_video`: paired AB/BA runs with identical hidden fixtures, reference repeats, quiet-machine/foreign-process checks, cleanup in `finally`, and distinct video history. For existing higher-is-better statistics, use `log(reference_seconds / candidate_seconds)`, not the text runner's rate expression. Reject nonpositive/nonfinite times. |
| `forge/recipes/__init__.py`, `forge/gate/service.py` (future edits) | Validate video workload and dispatch by recipe kind before text tokenization/server setup. Include all video settings, artifact hashes, timing boundary, runtime pins, and equality policy in identity/history keys. |
| `tests/core/test_video_gate.py` (new) | Standard-library `unittest`: fake pipeline/clock tests for synchronization, AB/BA fixture identity, equality failures, missing evidence, metric direction, lifecycle cleanup, controls, and no timing/log disclosure to agents. |

The bench loop is: load once per side; warm up on separate fixtures; poll controls; reset RNG; synchronize; start timer; generate and decode; synchronize; stop timer; capture canonical tensors; compare paired outputs; close only owned resources. Comparison and disk writes stay outside the tensor timing boundary. A two-worker setup must use separate leases and equivalent reference hardware, rather than allowing competing unaccounted workloads on a card.

Control integration must consume `<state_root>/missions/<id>/control.jsonl` incrementally as append-only rows `{t: float, by: "operator", action: "pause"|"resume"|"stop", reason: str}`. Preserve partial trailing rows for the next read. Pause stops admission of new clips; resume restarts admission; stop cancels or drains owned work and marks interrupted evidence invalid. Poll at request boundaries and, where supported, safe denoising-step callbacks. Never terminate another tenant's processes. Operator notes already exist at `operator_notes.jsonl` with `{t, by: "operator", to: "lead"|"worker-N"|"all", text}`; do not rewrite them or treat note text as control commands.

Raw timings, ratios, comparative verdicts, profiler traces, and server logs must stay in an operator-only store excluded from agent tools, prompts, and summaries. Current runner logging/result fields cannot be reused unchanged. Provide agents only schema/configuration validity and correctness status without performance verdicts or recommendations. Operator notes must pass the same information boundary before forwarding; operator authorship alone does not make speed-bearing text safe to expose.

Estimate: **2–3 engineering days** for a single-GPU Diffusers bench, strict equality gate, private results boundary, controls, and CPU unit tests; **another 1–2 days** for leased hardware validation and repeatability checks, assuming dependencies and access work. vLLM-Omni raw-tensor instrumentation or multi-GPU lifecycle support may add **2–4 days**. This is an implementation estimate, not a completed integration or a promised GPU execution time.

The draft preserves the catalog's required top-level fields but deliberately sets `variants: []` and `hardware.fits: []`. The loader currently does not enforce `kind` or `status`; empty variants are what prevent expansion, including direct `get()` lookup. All video settings live under `video_draft`, which existing execution code does not consume. Activation requires a separate adapter change and validation; nothing is wired by this task.

Existing test compatibility caveat: `tests/fleet/test_fleet.py::test_catalog_loads_and_cites_sources` assumes every base has nonempty variants. That assertion is incompatible with an intentionally unwired draft in this directory. The test is outside this task's allowed files and was not changed; its owner should explicitly distinguish drafts from runnable bases. Adding a dummy runnable text variant to satisfy it would be incorrect.
